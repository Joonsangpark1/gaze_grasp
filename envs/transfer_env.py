"""꽃 옮기기 + 능동 시야 환경 (교사용 특권 관측).

과제: 홀더(출발)의 꽃줄기를 목표 통에 옮긴다. 성공 = 통 바닥 touch 센서 > 0 이고 그 접촉이 줄기.
reward_mode="sparse" (기본, 11-15 사용자 결정): 에피소드 결과 하나만 보상, 행동·시선 정책이 공유.
    성공 +1 / 팔-카메라 충돌 −1 / 그 외 0.  시야 정보량·안 보임 비율은 보상이 아니라 측정 지표 (info).
reward_mode="dense" (이전 설계, 비교용): 아래 두 흐름.
보상은 두 흐름으로 나눠서 info["reward_streams"] = [task, view] 로 돌려준다 (critic 두 개용).
    task: 성공 +1 (종료) / 팔-카메라 충돌 -1 (종료)
    view: 매 스텝 λ·Φ(s') − c·[줄기가 두 눈 어디에도 안 보임]
          성공 시 + λ·Φ_max/(1-γ)        (남은 시야 몫 일시 지급 → 일찍 성공해도 손해 없음)
          충돌 시 − (λ·Φ_max + c)/(1-γ)   (최악: 계속 안 보이는 것보다도 나쁨)
          낙하 시 − c/(1-γ)               (앞으로 계속 안 보일 몫 → 일부러 끝내서 감점 피하기 방지)
환경 reward(스칼라) = task + view. 식과 값은 RECORD.md에 기록.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import gymnasium as gym
import mujoco
import numpy as np

from envs.kinematics import arm_reach, head_look_at, set_joint
from envs.scene import ARM_JOINTS, RIG_JOINTS, TABLE_Z, SceneConfig, build_model
from envs.info_view import InfoViewConfig, InfoViewScorer
from envs.view import ViewConfig, ViewScorer

ARM_REST = (0.0, -1.57, 1.57, 0.66, 0.0, 0.3)
GRIPPER_OPEN = 1.2


@dataclass
class EnvConfig:
    # 관측: "state" = 정답 상태 71차원 (디버깅용) / "image" = 양안 사진 + 자기 관절 (정책 입력)
    obs_mode: str = "state"
    fovea_px: int = 128                      # 중심와 이미지 (정사각, 화각 scene.fovea_fov_deg)
    periph_w: int = 128                      # 주변시 이미지 (전체 화각 축소)
    periph_h: int = 72
    control_hz: float = 20.0
    episode_steps: int = 400                 # 20 s
    # 행동 = 관절 목표 변화량 (정규화 [-1,1] × 아래 크기). 스텝당 = 속도 제한
    arm_delta: tuple = (0.05, 0.05, 0.05, 0.05, 0.05, 0.2)        # rad/step (gripper 0.2)
    rig_delta: tuple = (0.01, 0.05, 0.05, 0.05, 0.05, 0.05)       # rig_z m/step, 나머지 rad/step
    # 보상
    reward_mode: str = "sparse"              # "sparse" | "dense"
    gamma: float = 0.99                      # dense일 때 남은 시야 몫 계산용 (학습 γ와 같아야 함)
    view_lambda: float = 0.005
    not_visible_penalty: float = 0.005       # c: 줄기가 두 눈 어디에도 안 보이는 스텝마다
    phi_max: float = 1.0
    # 랜덤화
    holder_x: tuple = (0.38, 0.50)
    holder_y: tuple = (-0.08, 0.08)
    cup_x: tuple = (0.32, 0.48)
    cup_y: tuple = (-0.26, -0.16)
    min_holder_cup_dist: float = 0.14
    stem_k_scale: tuple = (0.7, 1.3)
    # 시작 상태 (커리큘럼 단계) 확률: "rest" = 처음 자세 / "near" = 그리퍼가 줄기 옆 (벌린 채)
    #   "grasped_*" 단계(줄기를 쥔 채 시작)는 스크립트 파지 상태 생성 후 추가 예정
    stage_probs: dict = field(default_factory=lambda: {"rest": 1.0})
    near_offset: tuple = (-0.07, 0.0, 0.0)   # 줄기 가운데 기준 TCP 위치
    rig_aim_at_start: bool = True            # 시작 때 카메라가 줄기를 대략 보고 있게
    stem_fall_z: float = TABLE_Z - 0.05      # 줄기가 테이블 밖으로 떨어지면 종료
    scene: SceneConfig = field(default_factory=SceneConfig)
    view: ViewConfig = field(default_factory=ViewConfig)
    info_view: InfoViewConfig = field(default_factory=InfoViewConfig)

    def to_dict(self):
        return asdict(self)


class TransferEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None):
        self.cfg = cfg = cfg or EnvConfig()
        self.render_mode = render_mode
        m = self.m = build_model(cfg.scene)
        self.d = mujoco.MjData(m)
        self.n_sub = max(1, round(1.0 / (cfg.control_hz * m.opt.timestep)))
        self.view = ViewScorer(m, cfg.scene, cfg.view)
        self.info_view = (InfoViewScorer(m, cfg.scene, cfg.info_view, dt=self.n_sub * m.opt.timestep)
                          if cfg.view.mode == "info" else None)
        self._view_diag = {}
        if self.info_view is not None:
            self.stem_targets = [i for i, t in enumerate(cfg.info_view.targets) if t.startswith("stem")]
        self._stem_seen = True

        names = ARM_JOINTS + RIG_JOINTS
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
        self.act_ids = np.array([m.actuator(n).id for n in names])
        rng = m.jnt_range[[m.joint(n).id for n in names]]
        self.t_lo, self.t_hi = rng[:, 0].copy(), rng[:, 1].copy()
        # 그리퍼 목표는 모터 명령 범위로 자름 (관절 한계 너머로 닫아 계속 조이게, scene.gripper_ctrl_min)
        self.t_lo[5] = m.actuator_ctrlrange[self.act_ids[5], 0]
        self.delta = np.array(cfg.arm_delta + cfg.rig_delta)

        self.stem_free = m.jnt_qposadr[m.joint("stem_free").id]
        self.stem_free_v = m.jnt_dofadr[m.joint("stem_free").id]
        self.stem_joints = [m.joint(f"stem_j{i}").id for i in range(1, cfg.scene.stem_segments)]
        self.stem_k0 = m.jnt_stiffness[self.stem_joints].copy()
        self.holder_bid, self.cup_bid = m.body("holder").id, m.body("cup").id
        self.touch_adr = m.sensor_adr[m.sensor("cup_touch").id]
        self.tcp_sid = m.site("tcp").id

        root = m.body_rootid
        geom_root = root[m.geom_bodyid]
        self.arm_geom = geom_root == m.body("base").id
        self.rig_geom = geom_root == m.body("rig_base").id
        self.stem_geom = geom_root == m.body("stem").id
        self.cup_floor = m.geom("cup_floor").id
        fingers = {m.body("gripper").id, m.body("moving_jaw_so101_v1").id}
        self.finger_fixed = (m.geom_bodyid == m.body("gripper").id) & (m.geom_contype != 0)
        self.finger_jaw = (m.geom_bodyid == m.body("moving_jaw_so101_v1").id) & (m.geom_contype != 0)
        del fingers

        # 보상 상수
        self.view_bonus = cfg.view_lambda * cfg.phi_max / (1.0 - cfg.gamma)
        self.pen_share = cfg.not_visible_penalty / (1.0 - cfg.gamma)

        obs_dim = len(self._obs(np.zeros(12)))
        self.state_dim = obs_dim
        self.proprio_dim = len(self._proprio(np.zeros(12)))
        self._fovea_r = self._periph_r = None
        if cfg.obs_mode == "image":
            fp, pw, ph = cfg.fovea_px, cfg.periph_w, cfg.periph_h
            self.observation_space = gym.spaces.Dict({
                "fovea": gym.spaces.Box(0, 255, (6, fp, fp), np.uint8),      # 왼눈 RGB + 오른눈 RGB
                "periph": gym.spaces.Box(0, 255, (6, ph, pw), np.uint8),
                "proprio": gym.spaces.Box(-np.inf, np.inf, (self.proprio_dim,), np.float32),
                "state": gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),   # 정책엔 안 씀 (critic 옵션·기록용)
            })
        else:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self._renderer = None
        self.stage_probs = dict(cfg.stage_probs)

    def set_stage_probs(self, probs: dict):
        """커리큘럼 관리자가 학습 중에 호출 (VectorEnv.call)."""
        self.stage_probs = dict(probs)

    # ------------------------------------------------------------------ 상태 읽기
    def _contacts(self):
        d, m = self.d, self.m
        n = d.ncon
        if n == 0:
            return False, False, False, False
        g1 = d.contact.geom1[:n]
        g2 = d.contact.geom2[:n]
        arm_rig = np.any((self.arm_geom[g1] & self.rig_geom[g2]) | (self.rig_geom[g1] & self.arm_geom[g2]))
        stem_cup = np.any((self.stem_geom[g1] & (g2 == self.cup_floor)) | (self.stem_geom[g2] & (g1 == self.cup_floor)))
        fix = np.any((self.finger_fixed[g1] & self.stem_geom[g2]) | (self.finger_fixed[g2] & self.stem_geom[g1]))
        jaw = np.any((self.finger_jaw[g1] & self.stem_geom[g2]) | (self.finger_jaw[g2] & self.stem_geom[g1]))
        return bool(arm_rig), bool(stem_cup), bool(fix), bool(jaw)

    def _obs(self, prev_action):
        d, m = self.d, self.m
        feats = self.view.eye_features(d)
        self._last_feats = feats
        stem_b = d.site("stem_bottom").xpos
        stem_m = d.site("stem_mid").xpos
        stem_t = d.site("stem_top").xpos
        tcp = d.site_xpos[self.tcp_sid]
        cup_floor = d.xpos[self.cup_bid] + np.array([0, 0, 0.004])
        _, _, fix, jaw = self._contacts()
        return np.concatenate([
            d.qpos[self.qadr], d.qvel[self.vadr],                 # 12 + 12
            stem_b, stem_m, stem_t, d.qvel[self.stem_free_v:self.stem_free_v + 3],   # 12
            tcp, stem_m - tcp,                                    # 6
            cup_floor - stem_b, d.xpos[self.holder_bid],          # 6
            feats.ravel(),                                        # 8
            [d.sensordata[self.touch_adr], float(fix), float(jaw)],   # 3
            prev_action,                                          # 12
        ]).astype(np.float32)

    def _proprio(self, prev_action):
        """자기 몸 감각: 팔·카메라 장치 관절 위치/속도, 손가락 접촉, 통 센서(촉각), 이전 행동."""
        d = self.d
        _, _, fix, jaw = self._contacts()
        return np.concatenate([
            d.qpos[self.qadr], d.qvel[self.vadr],
            [float(fix), float(jaw), d.sensordata[self.touch_adr]],
            prev_action,
        ]).astype(np.float32)

    def _images(self):
        cfg = self.cfg
        if self._fovea_r is None:
            self._fovea_r = mujoco.Renderer(self.m, cfg.fovea_px, cfg.fovea_px)
            self._periph_r = mujoco.Renderer(self.m, cfg.periph_h, cfg.periph_w)
        out = {}
        for key, r, cams in (("fovea", self._fovea_r, ("eye_left_fovea", "eye_right_fovea")),
                             ("periph", self._periph_r, ("eye_left", "eye_right"))):
            imgs = []
            for c in cams:
                r.update_scene(self.d, camera=c)
                imgs.append(r.render().transpose(2, 0, 1))     # (3, H, W)
            out[key] = np.concatenate(imgs, 0)
        return out

    def _make_obs(self, state_vec, prev_action):
        if self.cfg.obs_mode != "image":
            return state_vec
        return {**self._images(), "proprio": self._proprio(prev_action), "state": state_vec}

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg, m, d, r = self.cfg, self.m, self.d, self.np_random
        stage = (options or {}).get("start_stage")
        if stage is None:
            names = list(self.stage_probs)
            p = np.array([self.stage_probs[k] for k in names], dtype=np.float64)
            stage = names[r.choice(len(names), p=p / p.sum())]
        mujoco.mj_resetData(m, d)

        # 홀더 / 통 위치
        while True:
            hx, hy = r.uniform(*cfg.holder_x), r.uniform(*cfg.holder_y)
            cx, cy = r.uniform(*cfg.cup_x), r.uniform(*cfg.cup_y)
            if math.hypot(hx - cx, hy - cy) >= cfg.min_holder_cup_dist:
                break
        m.body_pos[self.holder_bid][:2] = hx, hy
        m.body_pos[self.cup_bid][:2] = cx, cy
        m.jnt_stiffness[self.stem_joints] = self.stem_k0 * r.uniform(*cfg.stem_k_scale)
        z0 = TABLE_Z + 0.002 + cfg.scene.stem_radius
        d.qpos[self.stem_free:self.stem_free + 7] = [hx, hy, z0, 1, 0, 0, 0]

        # 팔
        for n, v in zip(ARM_JOINTS, ARM_REST):
            set_joint(m, d, n, v)
        if stage == "near":
            mujoco.mj_kinematics(m, d)
            target = d.site("stem_mid").xpos + np.array(cfg.near_offset) + r.uniform(-0.02, 0.02, 3)
            arm_reach(m, d, target)
            set_joint(m, d, "gripper", GRIPPER_OPEN)

        # 카메라 장치: 기둥에서 줄기 쪽을 보는 높은 자세 + 잡음
        bx, by = cfg.scene.rig_base_xy
        set_joint(m, d, "rig_z", r.uniform(0.0, 0.1))
        set_joint(m, d, "rig_yaw", math.atan2(hy - by, hx - bx) + r.uniform(-0.2, 0.2))
        j1 = r.uniform(-0.25, 0.25)
        set_joint(m, d, "rig_j1", j1)
        set_joint(m, d, "rig_j2", -j1)
        if cfg.rig_aim_at_start:
            mujoco.mj_kinematics(m, d)
            head_look_at(m, d, d.site("stem_mid").xpos.copy())
        set_joint(m, d, "rig_j3", d.qpos[self.qadr[10]] + r.uniform(-0.2, 0.2))
        set_joint(m, d, "rig_j4", d.qpos[self.qadr[11]] + r.uniform(-0.2, 0.2))

        self.target = d.qpos[self.qadr].copy()
        d.ctrl[self.act_ids] = self.target
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d, nstep=20)   # 0.1 s 안정화
        self.t = 0
        self.prev_action = np.zeros(12)
        obs = self._obs(self.prev_action)
        if self.info_view is not None:
            self.info_view.reset_belief()
        self.phi = self._phi()
        obs = self._make_obs(obs, self.prev_action)
        self.stage = stage
        return obs, {"phi": self.phi}

    # ------------------------------------------------------------------ step
    def step(self, action):
        cfg, m, d = self.cfg, self.m, self.d
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.target = np.clip(self.target + a * self.delta, self.t_lo, self.t_hi)
        d.ctrl[self.act_ids] = self.target

        collided = False
        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)
            arm_rig, _, _, _ = self._contacts()
            if arm_rig:
                collided = True
                break
        self.t += 1
        self.prev_action = a

        obs = self._obs(a)
        phi = self._phi()
        obs = self._make_obs(obs, a)
        _, stem_cup, fix, jaw = self._contacts()
        touch = float(d.sensordata[self.touch_adr])
        success = (touch > 0.0) and stem_cup
        fell = d.qpos[self.stem_free + 2] < cfg.stem_fall_z

        not_seen = not self._stem_seen
        r_task, r_view = 0.0, cfg.view_lambda * phi - cfg.not_visible_penalty * not_seen
        if cfg.reward_mode == "sparse":
            r_task, r_view = (1.0 if success else -1.0 if collided else 0.0), 0.0
        elif collided:
            r_task, r_view = -1.0, -(self.view_bonus + self.pen_share)     # 최악
        elif success:
            r_task, r_view = 1.0, r_view + self.view_bonus
        elif fell:
            r_view = r_view - self.pen_share

        terminated = bool(collided or success or fell)
        truncated = (not terminated) and self.t >= cfg.episode_steps
        self.phi = phi
        info = {
            "reward_streams": np.array([r_task, r_view], dtype=np.float32),
            "phi": phi, "success": float(success), "collision": float(collided), "fell": float(fell),
            "finger_contact": float(fix and jaw),
            "stage_near": float(self.stage == "near"),
            "stem_not_visible": float(not_seen),
            **self._view_diag,
        }
        return obs, r_task + r_view, terminated, truncated, info

    def _phi(self) -> float:
        """시야 점수 Φ ∈ [0,1]. info 모드면 믿음을 한 스텝 갱신하고 진단값을 저장."""
        if self.info_view is None:
            self._stem_seen = bool(self._last_feats[:, 2].max() > 0)
            return self.view.phi(self.d, self._last_feats)
        res = self.info_view.update(self.d)
        self._stem_seen = bool(res["n_eyes_seen"][self.stem_targets].max() > 0)
        self._view_diag = {"cam_ypr": res["cam_ypr"], "phi_targets": res["phi_targets"],
                           "sigma_max_mm": res["sigma_max_mm"], "n_eyes_seen": res["n_eyes_seen"]}
        return res["phi"]

    # ------------------------------------------------------------------ render
    def render(self, camera="eye_left", width=640, height=360):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, height, width)
        self._renderer.update_scene(self.d, camera=camera)
        return self._renderer.render()

    def video_frame(self, text: str = "") -> np.ndarray:
        """영상 한 프레임 (800×360): 왼쪽 외부 시점 / 오른쪽 위 왼눈 중심와 / 오른쪽 아래 왼눈 주변시."""
        from PIL import Image, ImageDraw
        if getattr(self, "_video_r", None) is None:
            self._video_r = mujoco.Renderer(self.m, 360, 480)
            self._video_cam = mujoco.MjvCamera()
            self._video_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._video_cam.lookat[:] = [0.3, 0.05, 0.55]
            self._video_cam.azimuth, self._video_cam.elevation, self._video_cam.distance = 215, -25, 1.5
        self._video_r.update_scene(self.d, camera=self._video_cam)
        side = self._video_r.render()
        imgs = self._images()
        fov = Image.fromarray(imgs["fovea"][:3].transpose(1, 2, 0)).resize((180, 180), Image.NEAREST)
        per = Image.fromarray(imgs["periph"][:3].transpose(1, 2, 0)).resize((320, 180), Image.NEAREST)
        canvas = Image.new("RGB", (800, 360), (20, 20, 20))
        canvas.paste(Image.fromarray(side), (0, 0))
        canvas.paste(fov, (480 + 70, 0))
        canvas.paste(per, (480, 180))
        dr = ImageDraw.Draw(canvas)
        dr.text((485, 4), "fovea L", fill=(255, 255, 0))
        dr.text((485, 184), "periphery L", fill=(255, 255, 0))
        if text:
            dr.rectangle([0, 0, 8 + 6 * len(text), 16], fill=(0, 0, 0))
            dr.text((4, 2), text, fill=(255, 255, 255))
        return np.asarray(canvas)

    def close(self):
        if getattr(self, "_video_r", None) is not None:
            self._video_r.close()
            self._video_r = None
        for r in (self._renderer, self._fovea_r, self._periph_r):
            if r is not None:
                r.close()
        self._renderer = self._fovea_r = self._periph_r = None

"""정보량 시야 보상 — 카메라 기하로 계산한 위치 불확실성을 고정(월드) 좌표계에서 누적.

보상 전용 (정책 입력은 이미지). 렌더 없이 투영 기하만 사용.

대상 j (줄기 점들, 통) 마다 "보는 방향 칸"별 정보 행렬 Λ_{j,b} 를 유지 (고정 좌표계 기준 누적):
  보는 방향 = 대상 → 두 눈 중점 방향의 월드 yaw·pitch, bin_deg 칸으로 나눔.
  같은 칸에서 반복 측정은 독립이 아니므로 더하지 않고 더 좋은 것(trace 큰 쪽)만 남김
  → 가만히 쳐다보기만 하면 정보가 늘지 않고, 새 각도에서 봐야 늘어남 (운동 시차 / 삼각측량).
  1) 잊기:  Λ_{j,b} ← Λ_{j,b} · exp(−dt/τ_mem) · exp(−(‖Δp_j‖/v_ref)²)   (시간이 지나거나 대상이 움직이면 감소)
  2) 측정:  눈 k에 보이면 카메라 좌표 측정 정보
            σ_x = σ_y = Z·σ_ang,                         σ_ang = σ_px / f_eff  (중심와/주변시)
            σ_z = Z²·√2·σ_ang / B   (두 눈 모두 보임: 양안 시차)
            σ_z = ∞                  (한 눈만: 깊이 정보 없음 → 방향 정보만)
            월드로 회전: Λ_meas = R · diag(1/σ_x², 1/σ_y², 1/σ_z²) · Rᵀ,  R = 카메라 yaw·pitch·roll (월드 기준)
            칸 갱신:  Λ_{j,b} ← argmax_trace(Λ_{j,b}, Λ_meas)
  3) 합치기:  Σ_j = (Λ_prior + Σ_b Λ_{j,b})⁻¹,   Λ_prior = I/σ_prior²
     정보량:  I_j = −½ log det Σ_j
            정규화  Φ_j = clip((I_j − I_min)/(I_max − I_min), 0, 1)
            I_min = σ_prior 등방,  I_max = σ_best 등방
  Φ = 대상 평균 ∈ [0, 1]   (Φ_max = 1 → 남은 시야 몫 공식 그대로)

눈의 해상도는 정책 입력과 맞춤: 중심와 = 좁은 화각 카메라 (fovea_px, fovea_fov_deg),
주변시 = 전체 화각 축소 (periph_px 폭, 장면 FOV). 대상이 중심와 원뿔 안이면 중심와 해상도.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class InfoViewConfig:
    targets: tuple = ("stem_bottom", "stem_mid", "stem_top", "cup")   # site 이름 또는 "cup"(통 바닥 중심)
    sigma_px: float = 0.5              # 정책 입력 이미지에서의 픽셀 위치 오차
    fovea_px: int = 128                # 중심와 이미지 폭 (px)
    fovea_fov_deg: float = 14.0        # 중심와 화각
    periph_px: int = 128               # 주변시 이미지 폭 (px)
    min_width_px: float = 1.0          # 이 폭보다 작게 보이면 감지 못 함 (해당 이미지 해상도 기준)
    target_size_m: float = 0.008       # 감지 판정용 대상 크기 (줄기 지름)
    sigma_prior: float = 0.10          # 리셋 때 위치 불확실성 (m)
    sigma_best: float = 0.0005         # 정규화 상한 (m)
    bin_deg: float = 15.0              # 보는 방향 칸 크기 (월드 yaw·pitch)
    tau_mem: float = 3.0               # 기억 시간상수 (s)
    v_ref: float = 0.005               # 대상이 한 스텝에 이만큼 움직이면 정보 e^-1로 감소 (m)
    use_occlusion: bool = False        # True면 광선으로 가림 판정 (가리면 측정 없음)
    extra: dict = field(default_factory=dict)


def ypr_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """카메라 회전행렬(열 = 카메라 x오른쪽, y위, z뒤) → 월드 기준 yaw, pitch, roll (rad).

    광축 = −z열. yaw = 광축의 수평 방향, pitch = 광축의 위쪽 각(+위), roll = 광축을 축으로 한 화면 기울기.
    """
    fwd = -R[:, 2]
    up = R[:, 1]
    yaw = math.atan2(fwd[1], fwd[0])
    pitch = math.asin(max(-1.0, min(1.0, fwd[2])))
    # roll: 광축에 수직이고 수평인 기준 '오른쪽' 대비 카메라 x축의 기울기
    right_ref = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(right_ref)
    if n < 1e-6:
        return yaw, pitch, 0.0
    right_ref /= n
    up_ref = np.cross(right_ref, fwd)
    roll = math.atan2(float(R[:, 0] @ up_ref), float(R[:, 0] @ right_ref))
    del up
    return yaw, pitch, roll


class InfoViewScorer:
    def __init__(self, model: mujoco.MjModel, scene_cfg, cfg: InfoViewConfig | None = None, dt: float = 0.05):
        self.m = model
        self.cfg = cfg = cfg or InfoViewConfig()
        self.dt = dt
        self.cam_ids = [model.camera(n).id for n in ("eye_left", "eye_right")]
        self.B = max(scene_cfg.baseline, 1e-6)
        self.half_fov = math.radians(scene_cfg.eye_hfov_deg) / 2           # 수평 반화각
        self.half_fov_v = math.radians(scene_cfg.eye_fovy_deg) / 2
        self.fovea_half = math.radians(cfg.fovea_fov_deg) / 2
        self.f_fovea = (cfg.fovea_px / 2) / math.tan(self.fovea_half)
        self.f_periph = (cfg.periph_px / 2) / math.tan(self.half_fov)
        self.cup_bid = model.body("cup").id
        self.site_ids = {t: model.site(t).id for t in cfg.targets if t != "cup"}
        self.I_min = -1.5 * math.log(cfg.sigma_prior**2)       # −½ log det(σ² I₃) = −1.5 log σ²
        self.I_max = -1.5 * math.log(cfg.sigma_best**2)
        self.reset_belief()

    # ------------------------------------------------------------------
    def reset_belief(self):
        self.bins = [dict() for _ in self.cfg.targets]       # 대상별 {(yaw칸, pitch칸): Λ 3×3}
        self.prev_pos = None
        self.Sigma = np.tile(np.eye(3) * self.cfg.sigma_prior**2, (len(self.cfg.targets), 1, 1))

    def _target_pos(self, d) -> np.ndarray:
        out = []
        for t in self.cfg.targets:
            out.append(d.xpos[self.cup_bid] + np.array([0, 0, 0.004]) if t == "cup" else d.site_xpos[self.site_ids[t]])
        return np.array(out)

    def _visible(self, d, cid, p) -> tuple[bool, float, float]:
        """눈 cid에서 점 p가 감지되는지. (감지, Z, 그 영역의 f) 반환."""
        R = d.cam_xmat[cid].reshape(3, 3)
        c = R.T @ (p - d.cam_xpos[cid])
        Z = -c[2]
        if Z <= 1e-3:
            return False, Z, 0.0
        ax, ay = math.atan2(c[0], Z), math.atan2(c[1], Z)
        if abs(ax) > self.half_fov or abs(ay) > self.half_fov_v:
            return False, Z, 0.0
        in_fovea = math.hypot(ax, ay) <= self.fovea_half
        f = self.f_fovea if in_fovea else self.f_periph
        if f * self.cfg.target_size_m / Z < self.cfg.min_width_px:
            return False, Z, f
        if self.cfg.use_occlusion:
            geomid = np.zeros(1, np.int32)
            vec = p - d.cam_xpos[cid]
            dist = mujoco.mj_ray(self.m, d, d.cam_xpos[cid], vec / np.linalg.norm(vec), None, 1, -1, geomid)
            if 0 <= dist < np.linalg.norm(vec) - 0.01:
                return False, Z, f
        return True, Z, f

    def update(self, d) -> dict:
        """한 제어 스텝 뒤 호출. 믿음 갱신 후 Φ와 진단값 반환."""
        cfg = self.cfg
        pos = self._target_pos(d)
        vel = np.zeros(len(pos)) if self.prev_pos is None else np.linalg.norm(pos - self.prev_pos, axis=1)
        self.prev_pos = pos.copy()
        Rs = [d.cam_xmat[c].reshape(3, 3) for c in self.cam_ids]

        n_vis = np.zeros(len(pos), int)
        decay_t = math.exp(-self.dt / cfg.tau_mem)
        lam_prior = np.eye(3) / cfg.sigma_prior**2
        eye_mid = (d.cam_xpos[self.cam_ids[0]] + d.cam_xpos[self.cam_ids[1]]) / 2
        R = Rs[0]                                  # 두 눈 평행 → 같은 회전
        bin_rad = math.radians(cfg.bin_deg)
        for j, p in enumerate(pos):
            decay = decay_t * math.exp(-(vel[j] / cfg.v_ref) ** 2)
            bins = self.bins[j]
            for k in bins:
                bins[k] = bins[k] * decay
            vis = [self._visible(d, c, p) for c in self.cam_ids]
            seen = [v for v in vis if v[0]]
            n_vis[j] = len(seen)
            if seen:
                Z = float(np.mean([v[1] for v in seen]))
                f = max(v[2] for v in seen)
                s_ang = cfg.sigma_px / f
                s_lat = Z * s_ang
                info_cam = np.array([1 / s_lat**2, 1 / s_lat**2, 0.0])
                if len(seen) == 2:
                    s_z = Z**2 * math.sqrt(2) * s_ang / self.B
                    info_cam[2] = 1 / s_z**2
                Lam = R @ np.diag(info_cam) @ R.T
                v = eye_mid - p
                key = (int(math.floor(math.atan2(v[1], v[0]) / bin_rad)),
                       int(math.floor(math.atan2(v[2], math.hypot(v[0], v[1])) / bin_rad)))
                old = bins.get(key)
                if old is None or np.trace(Lam) > np.trace(old):
                    bins[key] = Lam
            total = lam_prior + (sum(bins.values()) if bins else 0)
            self.Sigma[j] = np.linalg.inv(total)

        sign, logdet = np.linalg.slogdet(self.Sigma)
        I = -0.5 * logdet
        phi_j = np.clip((I - self.I_min) / (self.I_max - self.I_min), 0.0, 1.0)
        yaw, pitch, roll = ypr_from_R(Rs[0])
        sig_axes = np.sqrt(np.clip(np.linalg.eigvalsh(self.Sigma), 0, None))   # (n, 3) 오름차순
        return {
            "phi": float(phi_j.mean()),
            "phi_targets": phi_j,
            "n_eyes_seen": n_vis,
            "sigma_max_mm": sig_axes[:, 2] * 1000,
            "sigma_min_mm": sig_axes[:, 0] * 1000,
            "cam_ypr": np.array([yaw, pitch, roll]),
        }

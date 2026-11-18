"""스크립트 시연 (prior data) — RLPD식 50:50 샘플링용.

팔: 웨이포인트 + IK (envs/kinematics.arm_reach, 별도 MjData에서 계산) → 관절 목표 차이를 [-1,1] 행동으로 역산
    접근(줄기 가운데 8 cm 앞) → 줄기 가운데 → 쥐기 → 들기 → 통 위 → 내리기 → 놓기
시선(카메라 6축): 균등 무작위 [-1,1] (11-16 사용자 결정 — 시선 정답을 시연에 넣지 않음)
보상·종료는 env 그대로 (sparse). 성공/실패 에피소드 모두 저장.

저장 (npz, 에피소드를 시간 순으로 이어 붙임, N=1 격자와 같은 형식):
    fovea/periph/proprio/state (T, ...) = 각 스텝의 관측 o_t
    act, rew, term, done, first (T,)
    final_* (E, ...) + final_t (E,) = 끝난 스텝의 진짜 다음 관측
실행: python -m scripts.make_demos --episodes 200 --out demos/prior_v1.npz [--obs-mode state]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.kinematics import arm_reach  # noqa: E402
from envs.scene import TABLE_Z  # noqa: E402
from envs.transfer_env import EnvConfig, TransferEnv  # noqa: E402

GRIP_CLOSE, GRIP_OPEN = -1.0, 1.0
# 넓은 패드(scene.gripper_pads, 11-17 B안): TCP = 고정 패드 접촉면(고무 겉면) 중심, 면 법선 = TCP 로컬 +x (움직이는 패드 쪽).
# 면은 평평하고 폭 40 mm(로컬 y) × 길이 59 mm(로컬 z)라 줄기를 면 가운데에, 고정 면에서 반지름 + 4 mm 띄워서 접근
# (패드 사이 간격: 그리퍼 q=+0.2에서 9 mm, −0.12에서 0 → 8 mm 줄기는 q≈+0.18에서 막혀 모터가 계속 조임)
# (이전 Menagerie 손가락용 값: [STEM_R + 0.010, 0, −0.020] — 얇은 판/작은 구라 손끝 쪽 좁은 틈에 물려야 했음)
STEM_R = 0.004
GRASP_LOCAL = np.array([STEM_R + 0.004, 0.0, 0.0])
WIGGLE, WIGGLE_T = 0.0, 8          # 뽑을 때 흔들기: 행동 진폭, 주기(스텝) → 관절 목표 ±0.038 rad


def _scaled(diff, speed):
    """관절 목표 차이를 같은 비율로 줄여 모든 관절이 동시에 도착 (관절별 clip이면 경로가 휨)."""
    m = np.abs(diff).max()
    return diff * (speed / m) if m > speed else diff


class ScriptedArm:
    """웨이포인트 순서대로 TCP를 옮기는 팔 스크립트. 행동 6차원 (팔 5 + 그리퍼)."""

    def __init__(self, env: TransferEnv, rng: np.random.Generator):
        self.env, self.rng = env, rng
        self.ds = mujoco.MjData(env.m)
        self.phase, self.goal, self.hold = 0, None, 0
        d = env.d
        # 시작 자세에선 집게가 위아래로 닫힘 → wrist_roll 90° 돌려 수평으로 닫히게 (세로 줄기를 옆에서 쥠)
        self.roll = float(np.clip(d.qpos[env.qadr[4]] + np.pi / 2, env.t_lo[4], env.t_hi[4]))
        # 쥘 점 = 마디 한가운데. stem_mid는 마디 사이 볼 조인트 위치라, 거기를 쥐면 위아래 마디가 조인트 둘레로 돌아
        # 홀더를 빠져나오는 순간 빠짐 (11-17: 거의 수평으로 쥐어도 바닥이 z≈0.47~0.50에서 매번 놓침)
        seg = env.cfg.scene.stem_length / env.cfg.scene.stem_segments
        mid = d.site("stem_mid").xpos.copy() + np.array([0.0, 0.0, 0.5 * seg])
        self.grasp_off = mid[2] - d.site("stem_bottom").xpos[2]        # 쥔 점 ~ 줄기 바닥 높이
        cup = env.d.xpos[env.cup_bid]
        jit = lambda s: rng.uniform(-s, s, 3)
        # "move" = 쥘 점 목표 / "bottom" = 줄기 바닥 목표 (줄기가 집게 안에서 기울어지므로 바닥 위치로 매 스텝 보정, 속도 절반)
        # 쥐는 자세를 먼저 풀고(roll 90° 가지에서 시작), 접근 시작점은 그 자세의 접근축(TCP 로컬 z, 팔 쪽)으로 5 cm 뒤
        grasp_pt = mid + jit(0.003)
        seed = env.target[:5].copy()
        seed[4] = self.roll
        # 숙임은 닿는 범위에서 수평에 가장 가깝게: 들린 채 쥐면 세로 줄기가 패드를 비스듬히 지나서, 뽑는 동안 물린 점이
        # 손끝 쪽으로 미끄러져 패드 끝(z −0.024)을 벗어남 (11-17, 홀더를 빠져나오는 순간 놓침). 수평이면 줄기 ⟂ 패드 길이
        self.az_ref = None
        free = self._ik(grasp_pt, seed=seed)
        az_free = float(self.ds.site_xmat[env.tcp_sid].reshape(3, 3)[2, 2])
        self.q_grasp = free
        for cand in np.linspace(0.0, az_free, 6)[:-1] if az_free > 0 else []:
            self.az_ref = float(cand)
            q = self._ik(grasp_pt, seed=free)
            R = self.ds.site_xmat[env.tcp_sid].reshape(3, 3)
            reach = np.linalg.norm(self.ds.site_xpos[env.tcp_sid] + R @ GRASP_LOCAL - grasp_pt)
            if reach < 0.002 and abs(R[2, 2] - cand) < 0.03:
                self.q_grasp = q
                break
        self.az_ref = None
        self._ik(grasp_pt, seed=self.q_grasp)       # self.ds를 고른 자세로
        az = self.ds.site_xmat[env.tcp_sid].reshape(3, 3)[:, 2].copy()
        self.az_ref = float(az[2])          # 이후 모든 자세에서 그리퍼 숙임을 쥘 때 값으로 유지 (들 때 줄기 기울지 않게)
        self.waypoints = [
            ("move", grasp_pt + 0.05 * az + jit(0.003), GRIP_OPEN),
            ("move", grasp_pt, GRIP_OPEN),
            ("grip", None, GRIP_CLOSE),
            *[("lift", mid + np.array([0, 0, dz]), GRIP_CLOSE) for dz in (0.02, 0.04, 0.07, 0.14)],   # 홀더(6 cm)에서 천천히 곧게 뽑기
            ("bottom", np.array([cup[0], cup[1], TABLE_Z + 0.10 + 0.04]), GRIP_CLOSE),   # 통 벽보다 4 cm 위
            ("bottom", np.array([cup[0], cup[1], TABLE_Z + 0.03]), GRIP_CLOSE),
            ("grip", None, GRIP_OPEN),
        ]

    def _ik(self, target, seed=None, iters=200, damp=0.03, w_ori=0.05):
        """쥘 점(TCP + R·GRASP_LOCAL)을 target에 + 닫힘축(로컬 x) 수평 + 접근축(로컬 z) 숙임 = az_ref. 팔 관절 5개 DLS.

        닫힘축이 수평이면 패드 면(로컬 y-z)에 세로 줄기가 놓여 숙임과 무관하게 선접촉.
        위치만 맞추면 들 때 손목 숙임이 바뀌어 쥔 줄기가 15°까지 기욺 → 숙임을 쥘 때 값으로 고정.
        완전 수평 접근은 이 높이에서 shoulder_lift 한계라 못 닿음 (az_ref=None이면 숙임 자유).
        """
        e, ds, m = self.env, self.ds, self.env.m
        ds.qpos[:] = e.d.qpos
        ds.qpos[e.qadr[:6]] = e.target[:6]
        if seed is None:   # 접근 중엔 쥐는 자세에서, 쥔 뒤엔 현재 관절 목표에서 시작 (같은 해 가지 유지)
            seed = self.q_grasp if self.phase < 2 else e.target[:5]
        ds.qpos[e.qadr[:5]] = seed
        dofs = e.vadr[:5]
        bid = m.site_bodyid[e.tcp_sid]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, ds)
            mujoco.mj_comPos(m, ds)
            R = ds.site_xmat[e.tcp_sid].reshape(3, 3)
            g = ds.site_xpos[e.tcp_sid] + R @ GRASP_LOCAL
            ax, az = R[:, 0], R[:, 2]
            ori = [ax[2]] if self.az_ref is None else [ax[2], az[2] - self.az_ref]
            err = np.concatenate([target - g, -w_ori * np.array(ori)])
            if np.abs(err).max() < 5e-4:
                break
            mujoco.mj_jac(m, ds, jacp, jacr, g, bid)
            Jw = jacr[:, dofs]
            rows = [jacp[:, dofs], w_ori * np.cross(Jw.T, ax)[:, 2]]      # d(ax_z)/dq = (ω×ax)_z
            if self.az_ref is not None:
                rows.append(w_ori * np.cross(Jw.T, az)[:, 2])
            J = np.vstack(rows)
            dq = J.T @ np.linalg.solve(J @ J.T + damp ** 2 * np.eye(len(err)), err)
            q = ds.qpos[e.qadr[:5]] + dq
            ds.qpos[e.qadr[:5]] = np.clip(q, e.t_lo[:5], e.t_hi[:5])
        mujoco.mj_kinematics(m, ds)
        self.tcp_goal = ds.site_xpos[e.tcp_sid].copy()
        return ds.qpos[e.qadr[:5]].copy()

    def act(self):
        e = self.env
        a = np.zeros(6)
        if self.phase >= len(self.waypoints):
            a[5] = GRIP_OPEN
            return a
        kind, tgt, grip = self.waypoints[self.phase]
        a[5] = grip
        if kind == "grip":
            self.hold += 1
            if self.hold >= 10:
                self.phase, self.hold = self.phase + 1, 0
            return a
        speed = 0.25 if kind == "lift" else 1.0
        if kind == "bottom":
            d = e.d
            bot = d.site("stem_bottom").xpos
            grasp_now = d.site_xpos[e.tcp_sid] + d.site_xmat[e.tcp_sid].reshape(3, 3) @ GRASP_LOCAL
            err_b = tgt - bot
            if self.goal is None or self.hold % 5 == 0:
                self.goal = self._ik(grasp_now + err_b)
            speed = 0.5
            a[:5] = _scaled((self.goal - e.target[:5]) / e.delta[:5], speed)
            self.hold += 1
            if np.linalg.norm(err_b[:2]) < 0.008 and abs(err_b[2]) < 0.015 or self.hold > 120:
                self.phase, self.goal, self.hold = self.phase + 1, None, 0
            return a
        # 관절 목표에 도착해도 중력 때문에 팔이 덜 뻗음 → 실제 쥘 점 오차만큼 IK 목표를 누적 보정
        if self.goal is None:
            self.cmd = tgt.copy()
            self.goal = self._ik(self.cmd)
        diff = (self.goal - e.target[:5]) / e.delta[:5]
        a[:5] = _scaled(diff, speed)
        if kind == "lift":
            # 홀더 구멍(틈 2.5 mm)에 걸린 줄기를 흔들며 뽑기: pan·wrist_flex에 위상 90° 차이 진동 (관절 목표 진폭 ≈ WIGGLE·0.05·T/2π)
            self.t_wig = getattr(self, "t_wig", 0) + 1
            w = 2 * np.pi * self.t_wig / WIGGLE_T
            a[0] = np.clip(a[0] + WIGGLE * np.sin(w), -1, 1)
            a[3] = np.clip(a[3] + WIGGLE * np.cos(w), -1, 1)
        d = e.d
        grasp_now = d.site_xpos[e.tcp_sid] + d.site_xmat[e.tcp_sid].reshape(3, 3) @ GRASP_LOCAL
        err = tgt - grasp_now
        if np.abs(diff).max() < 0.05:
            self.hold += 1
            if np.linalg.norm(err) < 0.004 or self.hold > 40:
                self.phase, self.goal, self.hold = self.phase + 1, None, 0
            elif self.hold % 5 == 0:
                self.cmd = self.cmd + err
                self.goal = self._ik(self.cmd)
        return a


def run(n_episodes, out, obs_mode, stage_probs, seed, eye_scale):
    cfg = EnvConfig(obs_mode=obs_mode, stage_probs=stage_probs)
    env = TransferEnv(cfg)
    rng = np.random.default_rng(seed)
    keys = ("fovea", "periph", "proprio", "state") if obs_mode == "image" else ("state",)
    buf = {k: [] for k in keys + ("act", "rew", "term", "done", "first")}
    finals = {k: [] for k in keys}
    final_t, outcomes, lens = [], [], []
    t0 = time.time()
    for ep in range(n_episodes):
        o, _ = env.reset(seed=int(rng.integers(1 << 30)))
        o = o if obs_mode == "image" else {"state": o}
        arm = ScriptedArm(env, rng)
        for t in range(cfg.episode_steps):
            a = np.concatenate([arm.act(), eye_scale * rng.uniform(-1, 1, 6)]).astype(np.float32)
            o2, r, term, trunc, info = env.step(a)
            o2 = o2 if obs_mode == "image" else {"state": o2}
            for k in keys:
                buf[k].append(o[k])
            buf["act"].append(a); buf["rew"].append(r); buf["term"].append(term)
            buf["done"].append(term or trunc); buf["first"].append(t == 0)
            if term or trunc:
                for k in keys:
                    finals[k].append(o2[k])
                final_t.append(len(buf["act"]) - 1)
                outcomes.append("success" if info["success"] else "collision" if info["collision"]
                                else "fell" if info["fell"] else "timeout")
                lens.append(t + 1)
                break
            o = o2
        if (ep + 1) % 10 == 0:
            oc = np.array(outcomes)
            print(f"[{ep + 1}/{n_episodes}] succ {np.mean(oc == 'success'):.2f} coll {np.mean(oc == 'collision'):.2f} "
                  f"fell {np.mean(oc == 'fell'):.2f} timeout {np.mean(oc == 'timeout'):.2f} "
                  f"len {np.mean(lens):.0f} | {len(buf['act'])} 전이, {time.time() - t0:.0f} s", flush=True)
    env.close()
    data = {k: np.asarray(v) for k, v in buf.items()}
    data.update({f"final_{k}": np.asarray(v) for k, v in finals.items()})
    data["final_t"] = np.asarray(final_t)
    data["outcome"] = np.asarray(outcomes)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **data)
        print(f"saved {out}: {len(data['act'])} 전이, {len(outcomes)} 에피소드", flush=True)
    return outcomes, lens


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", default="")
    ap.add_argument("--obs-mode", default="image", choices=["image", "state"])
    ap.add_argument("--near", type=float, default=1.0, help="near 시작 비율 (나머지 rest)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eye-scale", type=float, default=1.0, help="시선 무작위 행동 크기")
    args = ap.parse_args()
    probs = {"near": args.near, "rest": 1.0 - args.near} if args.near < 1.0 else {"near": 1.0}
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    run(args.episodes, args.out, args.obs_mode, probs, args.seed, args.eye_scale)

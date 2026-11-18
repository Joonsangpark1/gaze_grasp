"""장면 설계 확인용 렌더: 외부 시점 3개 + 양안 시점, 자세 2개 → analysis/rig_*.png

팔은 책상 고정, 카메라 장치는 원통 관절(rig_z, rig_yaw) + 회전 4개(rig_j1~j4).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.scene import SceneConfig, build_model  # noqa: E402

ARM_REST = {"shoulder_pan": 0.0, "shoulder_lift": -1.57, "elbow_flex": 1.57,
            "wrist_flex": 0.66, "wrist_roll": 0.0, "gripper": 0.3}
ARM_IK_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


def qadr(m, name):
    return m.jnt_qposadr[m.joint(name).id]


def set_joint(m, d, name, val):
    jid = m.joint(name).id
    d.qpos[m.jnt_qposadr[jid]] = np.clip(val, *m.jnt_range[jid]) if m.jnt_limited[jid] else val


def look_at(m, d, target, iters=60, gain=0.7):
    """머리 j3(yaw)/j4(pitch)만으로 target을 양안 중점의 광축에 맞춤. 남은 각도 오차(deg) 반환."""
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        cl, cr = d.cam("eye_left"), d.cam("eye_right")
        c = (cl.xpos + cr.xpos) / 2
        R = cl.xmat.reshape(3, 3)            # 카메라: x=오른쪽, y=위, -z=광축
        v = R.T @ (target - c)
        ex = math.atan2(v[0], -v[2])          # 오른쪽(+) 오차
        ey = math.atan2(v[1], -v[2])          # 위(+) 오차
        set_joint(m, d, "rig_j3", d.qpos[qadr(m, "rig_j3")] - gain * ex)
        set_joint(m, d, "rig_j4", d.qpos[qadr(m, "rig_j4")] - gain * ey)
    mujoco.mj_forward(m, d)
    return math.degrees(math.hypot(ex, ey))


def arm_reach(m, d, target, iters=200, damp=0.05):
    """팔 TCP 위치만 맞추는 DLS IK (보여주기용)."""
    sid = m.site("tcp").id
    dofs = [m.jnt_dofadr[m.joint(n).id] for n in ARM_IK_JOINTS]
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = target - d.site_xpos[sid]
        if np.linalg.norm(err) < 1e-3:
            break
        mujoco.mj_jacSite(m, d, jacp, None, sid)
        J = jacp[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(3), err)
        for n, step in zip(ARM_IK_JOINTS, dq):
            set_joint(m, d, n, d.qpos[qadr(m, n)] + step)
    mujoco.mj_forward(m, d)
    return float(np.linalg.norm(target - d.site_xpos[sid]))


def free_cam(lookat, az, el, dist):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.azimuth, c.elevation, c.distance = az, el, dist
    return c


def label(img, text):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    dr.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def render_pose(m, d, lines):
    W, Hh = 640, 480
    r = mujoco.Renderer(m, height=Hh, width=W)
    tiles = []
    for name, cam in [
        ("side (from right)", free_cam([0.25, 0.1, 0.6], 90, -12, 2.0)),
        ("front 3/4", free_cam([0.25, 0.1, 0.6], 215, -22, 1.9)),
        ("top", free_cam([0.3, 0.1, 0.5], 90, -89, 1.8)),
    ]:
        r.update_scene(d, camera=cam)
        tiles.append(label(r.render(), name))
    re = mujoco.Renderer(m, height=360, width=640)   # 눈은 1280x720 → 표시용 절반
    eyes = []
    for cam in ("eye_left", "eye_right"):
        re.update_scene(d, camera=cam)
        img = re.render().copy()
        img[178:182, 316:324] = (255, 0, 0)  # 화면 중앙
        img[176:184, 318:322] = (255, 0, 0)
        eyes.append(label(img, cam))
    r.close(); re.close()
    top = np.concatenate(tiles, axis=1)
    pad = np.full((360, top.shape[1] - 1280, 3), 255, np.uint8)
    sheet = np.concatenate([top, np.concatenate(eyes + [pad], axis=1)], axis=0)
    im = Image.fromarray(sheet)
    dr = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        dr.text((1300, 500 + 18 * i), line, fill=(0, 0, 0))
    return im


def main():
    cfg = SceneConfig()
    m = build_model(cfg)
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    bx, by = cfg.rig_base_xy
    sx, sy = cfg.stem_pos
    yaw_to_stem = math.atan2(sy - by, sx - bx)
    poses = {
        # 멀리서 작업공간 전체 보기
        "far": dict(rig={"rig_z": 0.1, "rig_yaw": yaw_to_stem, "rig_j1": math.radians(-10),
                         "rig_j2": math.radians(10)}, arm=None),
        # 줄기 옆으로 다가가서 보기 + 팔이 줄기 6cm 앞까지 접근
        "near": dict(rig={"rig_z": -0.05, "rig_yaw": yaw_to_stem + math.radians(25),
                          "rig_j1": math.radians(50), "rig_j2": math.radians(-50)},
                     arm=np.array([sx - 0.06, sy, cfg.stem_length / 2 + 0.41])),
    }
    for tag, p in poses.items():
        d = mujoco.MjData(m)
        for k, v in {**ARM_REST, **p["rig"]}.items():
            set_joint(m, d, k, v)
        mujoco.mj_forward(m, d)
        ik_err = arm_reach(m, d, p["arm"]) if p["arm"] is not None else float("nan")
        stem_mid = d.site("stem_mid").xpos.copy()
        resid = look_at(m, d, stem_mid)
        g = d.site("gaze_origin").xpos
        deg = {k: math.degrees(d.qpos[qadr(m, k)]) for k in ("rig_yaw", "rig_j1", "rig_j2", "rig_j3", "rig_j4")}
        lines = [
            f"[{tag}] rig_z={d.qpos[qadr(m, 'rig_z')]:+.2f}m  yaw={deg['rig_yaw']:+.0f}  "
            f"j1={deg['rig_j1']:+.0f}  j2={deg['rig_j2']:+.0f}  j3={deg['rig_j3']:+.0f}  j4={deg['rig_j4']:+.0f} deg",
            f"eye pos=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f})  eye->stem={np.linalg.norm(stem_mid - g):.3f}m  "
            f"gaze err={resid:.1f}deg",
            f"stem width ~{cfg.eye_focal_px * 2 * cfg.stem_radius / np.linalg.norm(stem_mid - g):.1f}px @1280  "
            f"arm IK err={ik_err * 100:.1f}cm",
        ]
        print("\n".join(lines))
        render_pose(m, d, lines).save(out / f"rig_{tag}.png")


if __name__ == "__main__":
    main()

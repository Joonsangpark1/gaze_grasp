"""줄기 휨 + 목표 통 센서 확인.

1) 홀더에 꽂힌 채 2초: 기울기 / 머리 처짐
2) 바닥 마디를 수평으로 잡고 있을 때 (그리퍼로 들고 옮기는 상황): 마디별 휨 각도, 머리 처짐
3) 목표 통 위에서 놓기: touch 센서 값 (닿기 전 0 → 닿으면 > 0)
→ analysis/stem_check.png
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
from envs.scene import TABLE_Z, SceneConfig, build_model, build_spec  # noqa: E402

ARM_REST = {"shoulder_pan": 0.0, "shoulder_lift": -1.57, "elbow_flex": 1.57,
            "wrist_flex": 0.66, "wrist_roll": 0.0, "gripper": 0.3}


def init(m, cfg):
    d = mujoco.MjData(m)
    for k, v in ARM_REST.items():
        d.qpos[m.jnt_qposadr[m.joint(k).id]] = v
        d.ctrl[m.actuator(k).id] = v
    # 카메라 장치는 비켜 둠
    d.ctrl[m.actuator("rig_j1").id] = d.qpos[m.jnt_qposadr[m.joint("rig_j1").id]] = -0.3
    mujoco.mj_forward(m, d)
    return d


def bend_angles(m, d, n):
    """각 ball joint의 회전 각도 (deg)."""
    out = []
    for i in range(1, n):
        q = d.qpos[m.jnt_qposadr[m.joint(f"stem_j{i}").id]: m.jnt_qposadr[m.joint(f"stem_j{i}").id] + 4]
        out.append(math.degrees(2 * math.acos(min(1.0, abs(q[0])))))
    return out


def cam(lookat, az, el, dist):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.azimuth, c.elevation, c.distance = az, el, dist
    return c


def snap(m, d, c, text):
    r = mujoco.Renderer(m, 400, 520)
    r.update_scene(d, camera=c)
    im = Image.fromarray(r.render())
    r.close()
    dr = ImageDraw.Draw(im)
    for i, t in enumerate(text):
        dr.rectangle([0, 18 * i, 8 + 6.2 * len(t), 18 * i + 17], fill=(0, 0, 0))
        dr.text((4, 18 * i + 3), t, fill=(255, 255, 255))
    return im


def main():
    cfg = SceneConfig()
    m = build_model(cfg)
    n, L = cfg.stem_segments, cfg.stem_length
    stem_mass = sum(m.body_mass[m.body(f"stem_b{i}").id] for i in range(1, n)) + m.body_mass[m.body("stem").id]
    print(f"stem: {n} segments, k={['%.3f' % k for k in cfg.stem_stiffness()]}, total mass {stem_mass * 1000:.1f} g")
    tiles = []

    # 1) 꽂힌 채 2초
    d = init(m, cfg)
    mujoco.mj_step(m, d, nstep=int(2.0 / m.opt.timestep))
    b, t = d.site("stem_bottom").xpos, d.site("stem_top").xpos
    lean = math.degrees(math.atan2(np.linalg.norm(t[:2] - b[:2]), t[2] - b[2]))
    ang = bend_angles(m, d, n)
    print(f"[1 upright] lean {lean:.2f} deg, joint bends {['%.2f' % a for a in ang]}")
    tiles.append(snap(m, d, cam([cfg.stem_pos[0], 0, TABLE_Z + 0.15], 90, -5, 0.6),
                      ["1) in holder, 2 s", f"lean {lean:.1f} deg"]))

    # 2) 바닥 마디를 수평으로 고정: freejoint를 지우고 월드에 고정
    #    (위치만 덮어쓰면 줄기 전체가 자유낙하해서 안 휘고, weld는 무른 구속이라 루트가 같이 기울어짐)
    root_pos = np.array([0.40, 0.20, TABLE_Z + 0.45])
    q_h = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_h, np.array([0.0, 1.0, 0.0]), math.pi / 2)  # 줄기 +z → +x
    spec = build_spec(cfg)
    spec.delete(spec.joint("stem_free"))
    spec.body("stem").pos, spec.body("stem").quat = root_pos, q_h
    m2 = spec.compile()
    d = init(m2, cfg)
    mujoco.mj_step(m2, d, nstep=int(3.0 / m2.opt.timestep))
    m_backup, m = m, m2
    root_axis = d.body("stem").xmat.reshape(3, 3)[:, 2]
    root_tilt = math.degrees(math.asin(-root_axis[2]))
    tip = d.site("stem_top").xpos
    straight_tip = root_pos + np.array([L, 0, 0])
    droop = straight_tip[2] - tip[2]
    ang = bend_angles(m, d, n)
    print(f"[2 horizontal, clamped at bottom] tip droop {droop * 100:.1f} cm "
          f"({math.degrees(math.atan2(droop, L)):.1f} deg), root tilt {root_tilt:.1f} deg, joint bends {['%.1f' % a for a in ang]}")
    tiles.append(snap(m, d, cam(root_pos + [L / 2, 0, -0.05], 90, 0, 0.55),
                      ["2) held horizontal at bottom", f"tip droop {droop * 100:.1f} cm",
                       "bends " + " ".join("%.0f" % a for a in ang) + " deg"]))
    m = m_backup

    # 3) 목표 통 위에서 세워 놓기
    d = init(m, cfg)
    cx, cy = cfg.cup_pos
    adr = m.jnt_qposadr[m.joint("stem_free").id]
    d.qpos[adr:adr + 3] = [cx, cy, TABLE_Z + 0.05]
    d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    sid = m.sensor("cup_touch").id
    mujoco.mj_forward(m, d)
    before = float(d.sensordata[m.sensor_adr[sid]])
    first_hit = None
    for k in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        v = float(d.sensordata[m.sensor_adr[sid]])
        if v > 0 and first_hit is None:
            first_hit = k * m.opt.timestep
    after = float(d.sensordata[m.sensor_adr[sid]])
    print(f"[3 cup] touch before {before:.3f} N, first contact at {first_hit} s, after 1.5 s {after:.3f} N")
    tiles.append(snap(m, d, cam([cx, cy, TABLE_Z + 0.12], 120, -15, 0.6),
                      ["3) dropped into target cup", f"touch {before:.2f} -> {after:.3f} N"]))

    # 4) 홀더 쪽 줄기에 touch 센서가 반응하지 않는지 (다른 통에 있을 땐 0)
    d = init(m, cfg)
    mujoco.mj_step(m, d, nstep=200)
    print(f"[4 stem still in source holder] cup touch {d.sensordata[m.sensor_adr[sid]]:.3f} N")

    sheet = Image.new("RGB", (520 * len(tiles), 400), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (520 * i, 0))
    sheet.save(ROOT / "analysis" / "stem_check.png")


if __name__ == "__main__":
    main()

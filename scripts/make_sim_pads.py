"""실물 그리퍼 패드(그리퍼_꽃병_설계/make_parts.py, flat)를 시뮬용 메시로 뽑기.

make_parts.py와 같은 계산으로 패드를 만들되, 출력용으로 눕히지 않고 원래 부품 좌표(STEP 로컬 = Menagerie 메시 로컬)로,
고무판(RUBBER_T)까지 포함한 접촉면 위치로 저장한다 (m 단위).
  pad_fixed_sim.stl  -> body "gripper"의 wrist_roll_follower 메시와 같은 pos/quat로 붙임
  pad_moving_sim.stl -> body "moving_jaw_so101_v1"의 moving_jaw 메시와 같은 pos/quat로 붙임

실행 (cad venv, trimesh/manifold3d 필요): C:\\Users\\A\\.venvs\\cad\\Scripts\\python scripts\\make_sim_pads.py
"""

import sys
from pathlib import Path

import numpy as np

DESIGN = Path(r"C:\Users\A\Desktop\로보틱스\그리퍼_꽃병_설계")
sys.path.insert(0, str(DESIGN))
import make_parts as mp  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "assets" / "so101" / "assets"

fixed_l = mp.load_step("Wrist_Roll_Follower_SO101")
moving_l = mp.load_step("Moving_Jaw_SO101")
fixed_finger = mp.inter(fixed_l.copy().apply_transform(mp.T_FIXED), mp.box(-80, 80, -80, 80, -200, -47))
x_face = fixed_finger.bounds[1, 0] + mp.MIN_FACE_T + mp.RUBBER_T
finger_l = mp.inter(moving_l, mp.box(-80, 80, -200, -24, -80, 80))
q_c = next(q for q in np.arange(-0.2, 0.0, 0.0025)
           if finger_l.copy().apply_transform(mp.t_moving(q)).bounds[0, 0] >= x_face + mp.RUBBER_T + mp.MIN_FACE_T)
T_c = mp.t_moving(q_c)
moving_finger = finger_l.copy().apply_transform(T_c)

# 고무 포함: 접촉면을 고무 겉면 위치(x_face)로
pf = mp.make_pad(fixed_finger, x_face, -1, False)
pm = mp.make_pad(moving_finger, x_face, +1, False)
pf_l = pf.copy().apply_transform(np.linalg.inv(mp.T_FIXED))
pm_l = pm.copy().apply_transform(np.linalg.inv(T_c))
for name, m in (("pad_fixed_sim.stl", pf_l), ("pad_moving_sim.stl", pm_l)):
    m = m.copy()
    m.apply_scale(0.001)
    m.export(OUT / name)
    print(name, "bounds (m)", np.round(m.bounds, 4).tolist(), "verts", len(m.vertices))
print(f"고무 겉면 x={x_face:.2f} mm (G 프레임), 패드끼리 닿는 각도 q={q_c:.4f} rad ({np.degrees(q_c):.1f}°)")

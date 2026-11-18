"""책상 고정 SO-101 + 독립 카메라 장치(양안) + 테이블 + 꽃줄기 장면.

손과 카메라는 분리돼 있다:
  - 팔: SO-101 베이스를 책상 위에 고정.
  - 카메라 장치: 바닥 기둥의 z축 원통 관절(z 이동 + z 회전) 1개 + 회전 관절 4개 → 양안 카메라.
    회전 관절 4개 기본값 = pitch(어깨) → pitch(팔꿈치) → yaw(머리) → pitch(머리).
    앞 두 pitch로 카메라 위치(앞뒤·높이), 머리 yaw/pitch로 바라보는 방향을 정한다.
    축은 SceneConfig.rig_axes 로 바꿀 수 있다.

MuJoCo는 Windows에서 한글 경로 파일을 못 열어서 (ParseXML: Error opening file),
XML·메시를 전부 파이썬에서 읽어 메모리 assets 로 넘긴다.

좌표: 바닥 z=0, 테이블은 +x 쪽. 오른쪽 = -y.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "so101"

TABLE_Z = 0.41
TABLE_FRONT_X = 0.15

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
RIG_JOINTS = ("rig_z", "rig_yaw", "rig_j1", "rig_j2", "rig_j3", "rig_j4")

_AXIS = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}


@dataclass
class SceneConfig:
    # --- 카메라 ---
    baseline: float = 0.065          # 눈 간격 (m). 0이면 두 카메라가 같은 위치
    eye_width: int = 1280
    eye_height: int = 720
    eye_hfov_deg: float = 70.0       # 1280px, FOV 70° → f≈914px
    eye_fwd: float = 0.03            # 머리 pitch 축에서 눈까지 앞 오프셋
    fovea_fov_deg: float = 14.0      # 중심와 카메라 화각 (정사각 이미지)
    # --- 카메라 장치 ---
    rig_base_xy: tuple[float, float] = (0.0, 0.25)   # 기둥 위치 (테이블 앞, 왼쪽)
    rig_column_h: float = 0.52       # 원통 관절 기준 기둥 높이 (rig_z=0일 때 j1 높이)
    rig_z_range: tuple[float, float] = (-0.15, 0.15)
    rig_yaw_range_deg: tuple[float, float] = (-120.0, 120.0)
    rig_link1: float = 0.28          # j1 → j2
    rig_link2: float = 0.15          # j2 → j3
    rig_axes: tuple[str, str, str, str] = ("y", "y", "z", "y")
    rig_ranges_deg: tuple[tuple[float, float], ...] = ((-30, 90), (-120, 120), (-90, 90), (-90, 90))
    # --- 팔 ---
    arm_base_xyz: tuple[float, float, float] = (TABLE_FRONT_X + 0.05, 0.0, TABLE_Z)
    arm_base_yaw_deg: float = 0.0    # 0 = +x(테이블 안쪽)를 향함
    # --- 꽃줄기 ---
    stem_radius: float = 0.004       # 지름 8mm
    finger_torsion: float = 0.03     # 손가락 패드 비틀림 마찰 (Menagerie 0.005 → 쥔 줄기가 쥔 축 둘레로 돌아 빠짐, 11-16 사용자: 올리기)
    # 그리퍼 모터 명령 하한 (rad). 관절 한계 −0.175와 같으면 다 닫힌 순간 목표=실제라 모터 힘 0 → 줄기를 못 조임.
    # 한계 너머로 명령하게 해서 실제 서보처럼 막혀도 계속 최대 힘(forcerange 2.94)으로 밀게 (11-17 사용자: A안)
    gripper_ctrl_min: float = -0.6
    gripper_pads: bool = True        # 실물 넓은 패드 (폭 40 mm, 고무 포함). False면 Menagerie 원래 손가락 (11-17 사용자: B안)
    pad_friction: float = 1.0        # 패드(고무) 미끄럼 마찰
    stem_length: float = 0.30        # reset 때 랜덤화 (env 쪽)
    stem_pos: tuple[float, float] = (0.45, 0.0)   # 테이블 위 홀더 위치 (x, y)
    stem_segments: int = 6           # 마디 수. 마디 사이 = ball joint
    stem_k_base: float = 1.8         # 바닥 쪽 첫 관절 강성 (N·m/rad)
    stem_k_tip: float = 0.09         # 머리 쪽 마지막 관절 강성. 그 사이는 기하급수로 감소
    stem_damping: float = 0.003      # 관절 감쇠 (N·m·s/rad)
    head_mass: float = 0.012         # 꽃 머리 질량 (kg)
    holder_hole_radius: float = 0.0065
    holder_depth: float = 0.06
    # --- 목표 통 (옮길 곳) ---
    cup_pos: tuple[float, float] = (0.40, -0.18)
    cup_radius: float = 0.035        # 안쪽 반지름
    cup_depth: float = 0.10
    extra: dict = field(default_factory=dict)

    def stem_stiffness(self) -> list[float]:
        """관절 i (바닥 → 머리) 강성. k_base에서 k_tip까지 기하급수 감소."""
        n = self.stem_segments - 1
        if n <= 1:
            return [self.stem_k_base] * n
        ratio = (self.stem_k_tip / self.stem_k_base) ** (1 / (n - 1))
        return [self.stem_k_base * ratio**i for i in range(n)]

    @property
    def eye_fovy_deg(self) -> float:
        t = math.tan(math.radians(self.eye_hfov_deg) / 2) * self.eye_height / self.eye_width
        return math.degrees(2 * math.atan(t))

    @property
    def eye_focal_px(self) -> float:
        return (self.eye_width / 2) / math.tan(math.radians(self.eye_hfov_deg) / 2)


def _read_assets() -> dict[str, bytes]:
    assets = {"so101.xml": (ASSET_DIR / "so101.xml").read_bytes()}
    for p in (ASSET_DIR / "assets").iterdir():
        assets[p.name] = p.read_bytes()
    return assets


def _ring_walls(prefix: str, r: float, d: float, n: int, wall_t: float, cls: str, shell_rgba: str,
                wall_rgba: str | None = None) -> str:
    """안쪽 반지름 r, 깊이 d의 빈 원통을 상자 n개로 근사 + 보이기용 반투명 겉면."""
    half_w = (r + wall_t) * math.tan(math.pi / n) + 0.001
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        cx, cy = (r + wall_t / 2) * math.cos(a), (r + wall_t / 2) * math.sin(a)
        out.append(
            f'<geom name="{prefix}_wall{i}" type="box" size="{wall_t/2} {half_w} {d/2}" '
            f'pos="{cx:.5f} {cy:.5f} {d/2}" euler="0 0 {a:.5f}" class="{cls}"'
            + (f' rgba="{wall_rgba}"/>' if wall_rgba else "/>")
        )
    out.append(
        f'<geom name="{prefix}_shell" type="cylinder" size="{r + wall_t} {d/2}" pos="0 0 {d/2}" '
        f'contype="0" conaffinity="0" rgba="{shell_rgba}"/>'
    )
    return "\n        ".join(out)


def _stem_xml(cfg: SceneConfig) -> str:
    """마디 사슬 줄기. 루트(맨 아래 마디)만 freejoint, 나머지는 ball joint로 이어짐.

    관절 강성은 바닥 → 머리로 갈수록 작아져서 머리 쪽이 더 잘 휜다.
    body 원점 = 각 마디의 아래 끝. 루트 원점 = 줄기 바닥점.
    """
    n, L, r = cfg.stem_segments, cfg.stem_length, cfg.stem_radius
    seg = L / n
    ks = cfg.stem_stiffness()
    sx, sy = cfg.stem_pos
    z0 = TABLE_Z + 0.002 + r          # 캡슐 아래 반구가 홀더 바닥에 닿게
    mid_i = n // 2

    def geom(i):
        return (f'<geom name="stem_seg{i}" type="capsule" fromto="0 0 0 0 0 {seg:.5f}" size="{r}" '
                f'class="stem"/>')

    inner = (f'<geom name="flower_head" type="sphere" size="0.025" pos="0 0 {seg + 0.02:.5f}" '
             f'rgba="0.9 0.3 0.45 1" contype="0" conaffinity="0" mass="{cfg.head_mass}"/>\n'
             f'<site name="stem_top" pos="0 0 {seg:.5f}" size="0.003"/>')
    for i in range(n - 1, 0, -1):
        extra = f'\n<site name="stem_mid" pos="0 0 0" size="0.003"/>' if i == mid_i else ""
        inner = (f'<body name="stem_b{i}" pos="0 0 {seg:.5f}">\n'
                 f'<joint name="stem_j{i}" type="ball" stiffness="{ks[i - 1]:.5f}" '
                 f'damping="{cfg.stem_damping}" armature="1e-5"/>\n'
                 f'{geom(i)}{extra}\n{inner}\n</body>')
    return (f'<body name="stem" pos="{sx} {sy} {z0:.5f}">\n'
            f'<freejoint name="stem_free"/>\n{geom(0)}\n'
            f'<site name="stem_bottom" pos="0 0 0" size="0.003"/>\n{inner}\n</body>')


def _rad_range(lo_hi) -> str:
    return f"{math.radians(lo_hi[0]):.5f} {math.radians(lo_hi[1]):.5f}"


def _rig_xml(cfg: SceneConfig) -> str:
    """기둥(원통 관절) → j1 → 링크1 → j2 → 링크2 → j3 → j4 → 양안 카메라."""
    a1, a2, a3, a4 = (_AXIS[a] for a in cfg.rig_axes)
    r1, r2, r3, r4 = (_rad_range(r) for r in cfg.rig_ranges_deg)
    bx, by = cfg.rig_base_xy
    h, L1, L2 = cfg.rig_column_h, cfg.rig_link1, cfg.rig_link2
    b2 = cfg.baseline / 2
    return f"""
    <body name="rig_base" pos="{bx} {by} 0">
      <geom name="rig_foot" type="cylinder" size="0.08 0.01" pos="0 0 0.01" rgba="0.3 0.3 0.33 1"/>
      <body name="rig_column" childclass="rig">
        <!-- z축 원통 관절 = z 이동 + z 회전 -->
        <joint name="rig_z" type="slide" axis="0 0 1" range="{cfg.rig_z_range[0]} {cfg.rig_z_range[1]}"/>
        <joint name="rig_yaw" type="hinge" axis="0 0 1" range="{_rad_range(cfg.rig_yaw_range_deg)}"/>
        <geom name="rig_column_geom" type="cylinder" size="0.025 {h / 2}" pos="0 0 {h / 2}" mass="1.0"
              contype="0" conaffinity="0"/>  <!-- 기둥은 보이기만 (z 이동 시 바닥 관통) -->
        <body name="rig_link1" pos="0 0 {h}">
          <joint name="rig_j1" axis="{a1}" range="{r1}"/>
          <geom name="rig_j1_motor" type="box" size="0.02 0.025 0.02" rgba="0.15 0.15 0.15 1" mass="0.06"/>
          <geom name="rig_link1_geom" type="capsule" fromto="0 0 0 0 0 {L1}" size="0.015" mass="0.15"/>
          <body name="rig_link2" pos="0 0 {L1}">
            <joint name="rig_j2" axis="{a2}" range="{r2}"/>
            <geom name="rig_j2_motor" type="box" size="0.02 0.025 0.02" rgba="0.15 0.15 0.15 1" mass="0.06"/>
            <geom name="rig_link2_geom" type="capsule" fromto="0 0 0 {L2} 0 0" size="0.012" mass="0.08"/>
            <body name="rig_head_yaw" pos="{L2} 0 0">
              <joint name="rig_j3" axis="{a3}" range="{r3}"/>
              <geom name="rig_j3_motor" type="box" size="0.02 0.02 0.015" rgba="0.15 0.15 0.15 1" mass="0.06"/>
              <body name="rig_head">
                <joint name="rig_j4" axis="{a4}" range="{r4}"/>
                <geom name="head_geom" type="box" size="0.02 {max(0.03, b2 + 0.02)} 0.018" pos="{cfg.eye_fwd - 0.02} 0 0"
                      mass="0.1" rgba="0.3 0.3 0.33 1"/>
                <site name="gaze_origin" pos="{cfg.eye_fwd} 0 0" size="0.004"/>
                <!-- MuJoCo 카메라는 자기 -z를 본다. xyaxes "0 -1 0 0 0 1" → 광축 = 머리 +x -->
                <camera name="eye_left" pos="{cfg.eye_fwd + 0.001} {b2} 0" xyaxes="0 -1 0 0 0 1"
                        fovy="{cfg.eye_fovy_deg:.4f}" resolution="{cfg.eye_width} {cfg.eye_height}"/>
                <camera name="eye_right" pos="{cfg.eye_fwd + 0.001} {-b2} 0" xyaxes="0 -1 0 0 0 1"
                        fovy="{cfg.eye_fovy_deg:.4f}" resolution="{cfg.eye_width} {cfg.eye_height}"/>
                <!-- 중심와: 같은 위치·방향, 좁은 화각 (정사각) -->
                <camera name="eye_left_fovea" pos="{cfg.eye_fwd + 0.001} {b2} 0" xyaxes="0 -1 0 0 0 1"
                        fovy="{cfg.fovea_fov_deg:.4f}"/>
                <camera name="eye_right_fovea" pos="{cfg.eye_fwd + 0.001} {-b2} 0" xyaxes="0 -1 0 0 0 1"
                        fovy="{cfg.fovea_fov_deg:.4f}"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def _world_xml(cfg: SceneConfig) -> str:
    sx, sy = cfg.stem_pos
    cx, cy = cfg.cup_pos
    ax, ay, az = cfg.arm_base_xyz
    return f"""
<mujoco model="gaze_grasp_scene">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" cone="elliptic" integrator="implicitfast"
          impratio="10" iterations="10" ls_iterations="20" noslip_iterations="3"/>
  <visual>
    <headlight diffuse="0.4 0.4 0.4" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <quality shadowsize="4096" offsamples="8"/>
    <global offwidth="{max(cfg.eye_width, 1280)}" offheight="{max(cfg.eye_height, 960)}"/>
  </visual>
  <default>
    <default class="rig">
      <geom rgba="0.75 0.75 0.78 1" condim="3"/>
      <joint type="hinge" damping="2" armature="0.01"/>
    </default>
    <default class="stem">
      <geom rgba="0.25 0.6 0.2 1" density="600" condim="4" friction="1.0 0.005 0.0001" priority="1"/>
    </default>
    <default class="holder">
      <geom rgba="0.55 0.45 0.35 1" condim="3" friction="0.6 0.005 0.0001"/>
    </default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.82" rgb2="0.72 0.72 0.72" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8"/>
    <material name="table" rgba="0.62 0.5 0.38 1"/>
  </asset>
  <worldbody>
    <light pos="0.5 0.5 2.5" dir="-0.2 -0.2 -1" diffuse="0.5 0.5 0.5" castshadow="true"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.01" material="floor"/>

    <!-- 테이블: 윗면 z={TABLE_Z}, 앞 모서리 x={TABLE_FRONT_X} -->
    <body name="table" pos="{TABLE_FRONT_X + 0.3} 0 {TABLE_Z / 2}">
      <geom name="table_top" type="box" size="0.3 0.5 {TABLE_Z / 2}" material="table"/>
    </body>

    <!-- 팔 장착점: 책상 위 고정 -->
    <site name="arm_mount" pos="{ax} {ay} {az}" euler="0 0 {math.radians(cfg.arm_base_yaw_deg)}" size="0.01"/>

    <!-- 홀더 (꽃병, 출발): 위치는 env reset 때 body_pos로 옮김 -->
    <body name="holder" pos="{sx} {sy} {TABLE_Z}">
        <geom name="holder_floor" type="cylinder" size="{cfg.holder_hole_radius + 0.004} 0.001" pos="0 0 0.001" class="holder"/>
        {_ring_walls("holder", cfg.holder_hole_radius, cfg.holder_depth, 8, 0.006, "holder", "0.55 0.45 0.35 0.35")}
    </body>

    <!-- 목표 통 (도착): 바닥에 touch 센서. 줄기가 바닥에 닿으면 성공 -->
    <body name="cup" pos="{cx} {cy} {TABLE_Z}">
        <geom name="cup_floor" type="cylinder" size="{cfg.cup_radius + 0.004} 0.002" pos="0 0 0.002" class="holder"
              rgba="0.3 0.45 0.7 1"/>
        {_ring_walls("cup", cfg.cup_radius, cfg.cup_depth, 12, 0.004, "holder", "0.3 0.45 0.7 0.3", "0.3 0.45 0.7 1")}
        <!-- touch 센서 영역: 통 바닥 바로 위 얇은 원판 -->
        <site name="cup_sensor" type="cylinder" size="{cfg.cup_radius} 0.006" pos="0 0 0.006" rgba="1 0.8 0 0.3"/>
    </body>

    <!-- 꽃줄기: 마디 사슬 (머리 쪽일수록 잘 휨). 홀더 구멍에 꽂혀 있어 건드리면 기울고, 잡으면 뽑힘 -->
    {_stem_xml(cfg)}
{_rig_xml(cfg)}
  </worldbody>

  <actuator>
    <!-- 토크 제한 없음: 설계 확인 후 토크 로그로 부품 선정 -->
    <position name="rig_z" joint="rig_z" kp="5000" kv="300"/>
    <position name="rig_yaw" joint="rig_yaw" kp="200" kv="20"/>
    <position name="rig_j1" joint="rig_j1" kp="200" kv="15"/>
    <position name="rig_j2" joint="rig_j2" kp="60" kv="5"/>
    <position name="rig_j3" joint="rig_j3" kp="20" kv="1"/>
    <position name="rig_j4" joint="rig_j4" kp="20" kv="1"/>
  </actuator>

  <sensor>
    <touch name="cup_touch" site="cup_sensor"/>
  </sensor>
</mujoco>
"""


def build_spec(cfg: SceneConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or SceneConfig()
    assets = _read_assets()
    spec = mujoco.MjSpec.from_string(_world_xml(cfg), assets)
    arm = mujoco.MjSpec.from_string(assets["so101.xml"].decode(), assets)
    # 손가락 접촉 비틀림 마찰 (줄기와 priority 같음 → 접촉 마찰은 둘 중 큰 값)
    for g in arm.geoms:
        if g.classname.name.startswith("collision_gripper"):
            g.friction[1] = cfg.finger_torsion
    if cfg.gripper_pads:
        # 실물과 같은 넓은 패드 (그리퍼_꽃병_설계 flat, 고무 1 mm 포함, scripts/make_sim_pads.py로 생성).
        # Menagerie 집게 충돌 형상(두께 1 mm 판 + 반지름 0.75 mm 구)은 가는 줄기를 면으로 못 물어 들 때 빠짐 (11-17 사용자: B안)
        # 메시는 원래 부품 좌표 → 원래 손가락 메시와 같은 pos/quat. 충돌은 볼록 껍질 (손가락을 감싼 블록, 접촉면은 평면)
        for body, mesh, pos, quat in (("gripper", "pad_fixed_sim", [0, -0.000218, 0.00095], [0, 1, 0, 0]),
                                      ("moving_jaw_so101_v1", "pad_moving_sim", [0, 0, 0.0189], [1, 0, 0, 0])):
            arm.add_mesh(name=mesh, file=f"{mesh}.stl")
            b = arm.body(body)
            col = b.add_geom(name=f"{mesh}_col", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, pos=pos, quat=quat)
            col.group, col.condim, col.priority, col.mass = 3, 6, 1, 0.0
            col.friction = [cfg.pad_friction, cfg.finger_torsion, 5e-4]
            col.solref = [0.01, 1]
            vis = b.add_geom(name=f"{mesh}_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, pos=pos, quat=quat)
            vis.contype = vis.conaffinity = 0
            vis.group, vis.mass, vis.rgba = 2, 0.0, [0.15, 0.15, 0.15, 1]
        # TCP = 고정 패드 접촉면(고무 겉면) 중심 (TCP는 항상 고정 그리퍼 중심). 면 법선 +x = 움직이는 집게 쪽
        tcp_pos = [-0.0049, 0.0, -0.0775]
    else:
        # TCP = 고정 집게 안쪽 패드(fixed_jaw_box4~7) 중심 (11-16 사용자: TCP는 항상 고정 그리퍼 중심).
        tcp_pos = np.mean([arm.geom(f"fixed_jaw_box{i}").pos for i in range(4, 8)], 0)
    arm.body("gripper").add_site(name="tcp", pos=tcp_pos, size=[0.004], group=3)
    arm.actuator("gripper").ctrlrange[0] = cfg.gripper_ctrl_min
    with warnings.catch_warnings():  # 빈 prefix 경고: 관절 이름을 so101 원래대로 유지하려고 일부러 씀
        warnings.simplefilter("ignore")
        spec.attach(arm, site=spec.site("arm_mount"), prefix="")
    # 팔 베이스는 책상에 고정 → 테이블과의 충돌 제외
    spec.add_exclude(bodyname1="table", bodyname2="base")
    spec.assets = assets   # attach 후 compile 때 메시를 다시 찾으므로 필요
    return spec


def build_model(cfg: SceneConfig | None = None) -> mujoco.MjModel:
    return build_spec(cfg).compile()

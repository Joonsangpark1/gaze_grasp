"""시작 자세 만들기용 간단한 IK (학습 중 제어에는 안 씀)."""

from __future__ import annotations

import math

import mujoco
import numpy as np

ARM_IK_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


def _qadr(m, name):
    return m.jnt_qposadr[m.joint(name).id]


def set_joint(m, d, name, val):
    jid = m.joint(name).id
    if m.jnt_limited[jid]:
        val = float(np.clip(val, *m.jnt_range[jid]))
    d.qpos[m.jnt_qposadr[jid]] = val


def arm_reach(m, d, target, iters=150, damp=0.05, tol=1e-3):
    """팔 TCP 위치만 target에 맞추는 DLS IK. 남은 오차(m) 반환. d.qpos를 바꾼다."""
    sid = m.site("tcp").id
    dofs = [m.jnt_dofadr[m.joint(n).id] for n in ARM_IK_JOINTS]
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        err = target - d.site_xpos[sid]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(m, d, jacp, None, sid)
        J = jacp[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(3), err)
        for n, step in zip(ARM_IK_JOINTS, dq):
            set_joint(m, d, n, d.qpos[_qadr(m, n)] + step)
    mujoco.mj_kinematics(m, d)
    return float(np.linalg.norm(target - d.site_xpos[sid]))


def head_look_at(m, d, target, iters=40, gain=0.7):
    """카메라 장치 머리 rig_j3/rig_j4만으로 target을 양안 중점의 광축에 맞춤. 남은 각도 오차(deg)."""
    cl, cr = m.camera("eye_left").id, m.camera("eye_right").id
    ex = ey = 0.0
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        c = (d.cam_xpos[cl] + d.cam_xpos[cr]) / 2
        R = d.cam_xmat[cl].reshape(3, 3)
        v = R.T @ (target - c)
        ex, ey = math.atan2(v[0], -v[2]), math.atan2(v[1], -v[2])
        set_joint(m, d, "rig_j3", d.qpos[_qadr(m, "rig_j3")] - gain * ex)
        set_joint(m, d, "rig_j4", d.qpos[_qadr(m, "rig_j4")] - gain * ey)
    return math.degrees(math.hypot(ex, ey))

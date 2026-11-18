"""시야 점수 Φ(s) ∈ [0, 1] — 기하 투영. mode="visible"(1단계 기본) / "oracle"(비교용).

렌더링 없이 줄기 위 점들을 양쪽 눈에 핀홀 투영해서 계산한다 (교사 PPO 대량 병렬용).
그리퍼가 줄기를 가리는 것은 계산하지 않음 = 가림으로 감점하지 않음 (계획서 결정).

눈 하나의 점수:
    visible  = 줄기 점들 중 화면 안 + 카메라 앞에 있는 비율
    center   = 1 - min(1, |초점 화면 위치 - 화면 중앙| / 화면 반대각선)
    size     = exp(-|ln(겉보기 폭 / 목표 폭)|)      (목표 폭과 같으면 1, 2배·절반이면 0.5)
    Φ_eye    = visible · center · size
Φ = 두 눈 평균.  Φ_max = 1.

이 모듈은 나중에 prior 유사도 보상 등으로 교체될 수 있다 (RECORD.md 참고).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ViewConfig:
    # "info"   : 카메라 기하로 계산한 위치 정보량, 고정 좌표계에서 누적 (envs/info_view.py)  ← 기본
    # "visible": 줄기가 시야 안에 있는 비율만
    # "oracle" : visible · center · size (사람이 정한 시점 기준, 비교용 정답지)
    mode: str = "info"
    target_width_px: float = 15.0    # oracle: 목표 겉보기 줄기 폭 (원본 해상도 기준)
    focus_site: str = "stem_mid"     # oracle: 화면 중앙에 둘 점


class ViewScorer:
    def __init__(self, model: mujoco.MjModel, scene_cfg, view_cfg: ViewConfig | None = None):
        self.m = model
        self.cfg = view_cfg or ViewConfig()
        self.cam_ids = [model.camera(n).id for n in ("eye_left", "eye_right")]
        self.W, self.H = scene_cfg.eye_width, scene_cfg.eye_height
        self.f = scene_cfg.eye_focal_px
        self.stem_r = scene_cfg.stem_radius
        n = scene_cfg.stem_segments
        # 줄기 위 표본점: 마디 경계 (stem body 원점들) + 꼭대기
        self.point_bodies = [model.body("stem").id] + [model.body(f"stem_b{i}").id for i in range(1, n)]
        self.top_site = model.site("stem_top").id
        self.focus_site = model.site(self.cfg.focus_site).id
        self.half_diag = 0.5 * float(np.hypot(self.W, self.H))

    def _points(self, d: mujoco.MjData) -> np.ndarray:
        pts = d.xpos[self.point_bodies]
        return np.vstack([pts, d.site_xpos[self.top_site][None]])

    def eye_features(self, d: mujoco.MjData) -> np.ndarray:
        """눈마다 (u, v, visible, width/target) → (2, 4). u, v는 화면 중앙 0, 가장자리 ±1."""
        pts = self._points(d)
        focus = d.site_xpos[self.focus_site]
        out = np.zeros((2, 4))
        for k, cid in enumerate(self.cam_ids):
            p, R = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
            # 카메라 좌표: x 오른쪽, y 위, -z 광축
            c = (pts - p) @ R
            z = -c[:, 2]
            front = z > 1e-3
            u = np.where(front, self.f * c[:, 0] / np.maximum(z, 1e-3), np.inf)
            v = np.where(front, self.f * c[:, 1] / np.maximum(z, 1e-3), np.inf)
            inside = front & (np.abs(u) <= self.W / 2) & (np.abs(v) <= self.H / 2)
            fc = (focus - p) @ R
            fz = max(-fc[2], 1e-3)
            fu, fv = self.f * fc[0] / fz, self.f * fc[1] / fz
            if -fc[2] <= 1e-3:  # 초점이 카메라 뒤
                fu, fv = self.W, self.H
            width = self.f * 2 * self.stem_r / fz
            out[k] = [np.clip(fu / (self.W / 2), -2, 2), np.clip(fv / (self.H / 2), -2, 2),
                      inside.mean(), width / self.cfg.target_width_px]
        return out

    def phi(self, d: mujoco.MjData, feats: np.ndarray | None = None) -> float:
        feats = self.eye_features(d) if feats is None else feats
        if self.cfg.mode == "visible":
            return float(feats[:, 2].mean())      # 두 눈 평균 visible
        scores = []
        for u, v, vis, wr in feats:
            dist_px = np.hypot(u * self.W / 2, v * self.H / 2)
            center = 1.0 - min(1.0, dist_px / self.half_diag)
            size = float(np.exp(-abs(np.log(max(wr, 1e-6)))))
            scores.append(vis * center * size)
        return float(np.mean(scores))

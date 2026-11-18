"""이미지 입력 SAC — 행동/시선 정책 교차 고정, sparse 보상 하나 공유.

정책 입력 (env obs_mode="image"):
    fovea  (6, 128, 128)  왼눈+오른눈 중심와 (화각 14°)   ← 두 눈을 채널로 겹침 (픽셀 단위 시차)
    periph (6,  72, 128)  왼눈+오른눈 주변시 (화각 70°)
    proprio (39)          팔·카메라 장치 관절 위치/속도, 손가락 접촉, 통 센서, 이전 행동
    → 최근 frame_stack 프레임을 채널로 쌓음
정답 상태 state(71)는 정책에 안 씀. critic_input="asym"일 때만 critic에 추가 (편향 없는 비대칭).

인코더 z = [E_fovea(fovea), E_periph(periph)] (각 50차원)
    학습: critic 손실 + 다음 프레임 특징 예측 (SPR식, 정답 상태 안 씀)
        L_pred = 2 − 2·cos( P(z_t, a_t), sg(Ē(o_{t+1})) ),   Ē = 인코더 EMA
critic (쌍둥이, 출력 1): Q_i([z, proprio, (state)], a)
    y = r + γ(1−d)(min_i Q̄_i(s', a') − α_arm·log π_arm − α_eye·log π_eye)
actor 2개 (입력 [sg(z), proprio]): 학습 중인 쪽 p만 갱신, 다른 쪽 q 고정
    L_p = E[α_p·log π_p − min_i Q_i(s, [a_p, a_q])]
실행: python -m agent.sac_image --run-name img_v0 --total-steps 2000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import AutoresetMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.transfer_env import EnvConfig, TransferEnv  # noqa: E402

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


@dataclass
class ImgSACConfig:
    run_name: str = "img_v0"
    seed: int = 1
    total_steps: int = 2_000_000
    num_envs: int = 8
    buffer_size: int = 100_000          # 전이 수 (이미지 1프레임 ≈ 153 KB → 약 15 GB, CPU RAM)
    frame_stack: int = 2
    batch_size: int = 256
    learning_starts: int = 10_000
    updates_per_vec_step: int = 4       # UTD = 4 / num_envs = 0.5
    gamma: float = 0.99
    tau: float = 0.01                   # critic·인코더 target EMA
    lr: float = 3e-4
    init_alpha: float = 0.1
    feat_dim: int = 50
    hidden: int = 256
    critic_input: str = "sym"           # "sym" (critic도 이미지 특징만) | "asym" (+ 정답 상태, 편향 없는 비대칭)
    pred_coef: float = 1.0              # 다음 프레임 특징 예측 손실 가중치
    aug_pad: int = 4                    # random shift 증강
    actor_update_every: int = 2
    alternate: bool = True
    phase_len_steps: int = 100_000
    first_phase: str = "arm"
    obs_norm_steps: int = 10_000
    curriculum: list = field(default_factory=lambda: [
        {"near": 1.0},
        {"near": 0.5, "rest": 0.5},
        {"rest": 1.0},
    ])
    curriculum_success: float = 0.6
    curriculum_window: int = 200
    # prior 시연 (scripts/make_demos.py npz). 배치 중 prior 비율 = start → end로 anneal_steps(env 스텝)에 걸쳐 줄임
    prior_path: str = ""                # 비면 안 씀. 예: demos/prior_v1.npz
    prior_start: float = 1.0            # 11-17 사용자: 100% → 0%, 100만 스텝
    prior_end: float = 0.0
    prior_anneal_steps: int = 1_000_000
    prior_shape: str = "linear"         # "linear" | "exp" (exp: start·(end/start)^t, end=0이면 1% 바닥 후 0)
    video_dir: str = ""                 # 비면 영상 안 남김. 예: ~/Desktop/박준상
    video_every_episodes: int = 200     # 학습 에피소드 이만큼 끝날 때마다 평가 에피소드 1개 녹화
    amp: bool = True                    # 인코더 bf16 (GPU 계산이 병목)
    log_every_vec_steps: int = 50
    save_every_vec_steps: int = 5000
    device: str = "cuda"


# ---------------------------------------------------------------------------- replay
class GridReplay:
    """env × 시간 격자에 프레임을 한 번만 저장. 프레임 쌓기·다음 관측은 인덱스로 복원.

    끝난 스텝의 진짜 다음 관측(final_obs)은 별도 dict에 저장 (autoreset 때문에 격자 다음 칸은 새 에피소드).
    """

    def __init__(self, cap, N, spaces, act_dim, k):
        self.Tc, self.N, self.k = max(2, cap // N), N, k
        T = self.Tc
        self.fov = np.zeros((T, N) + spaces["fovea"].shape, np.uint8)
        self.per = np.zeros((T, N) + spaces["periph"].shape, np.uint8)
        self.pro = np.zeros((T, N) + spaces["proprio"].shape, np.float32)
        self.sta = np.zeros((T, N) + spaces["state"].shape, np.float32)
        self.act = np.zeros((T, N, act_dim), np.float32)
        self.rew = np.zeros((T, N), np.float32)
        self.term = np.zeros((T, N), bool)
        self.done = np.zeros((T, N), bool)
        self.first = np.zeros((T, N), bool)
        self.final = {}
        self.ptr, self.filled = 0, 0

    def add(self, o, a, r, term, done, first, final_obs):
        t = self.ptr
        for key in [k for k in self.final if k[0] == t]:
            del self.final[key]
        self.fov[t], self.per[t], self.pro[t], self.sta[t] = o["fovea"], o["periph"], o["proprio"], o["state"]
        self.act[t], self.rew[t], self.term[t], self.done[t], self.first[t] = a, r, term, done, first
        for i, fo in final_obs.items():
            self.final[(t, i)] = fo
        self.ptr = (t + 1) % self.Tc
        self.filled = min(self.filled + 1, self.Tc)

    def _age(self, t):
        return (self.ptr - 1 - t) % self.Tc

    def _chain(self, t, i):
        """t에서 시작해 과거로 k프레임 (에피소드 시작·버퍼 끝에서 멈추고 반복). 오래된 것 먼저."""
        out, cur = [t], t
        for _ in range(self.k - 1):
            if not self.first[cur, i] and self._age(cur) + 1 <= self.filled - 1:
                cur = (cur - 1) % self.Tc
            out.append(cur)
        return out[::-1]

    def _gather(self, rows, cols, finals):
        """rows/cols: (B, k) 시간·env 인덱스. finals: {(b, j): obs dict} 대체 프레임."""
        fov = self.fov[rows, cols]
        per = self.per[rows, cols]
        pro = self.pro[rows[:, -1], cols[:, -1]].copy()
        sta = self.sta[rows[:, -1], cols[:, -1]].copy()
        for (b, j), fo in finals.items():
            fov[b, j], per[b, j] = fo["fovea"], fo["periph"]
            if j == self.k - 1:
                pro[b], sta[b] = fo["proprio"], fo["state"]
        B = rows.shape[0]
        return (fov.reshape(B, -1, *fov.shape[-2:]), per.reshape(B, -1, *per.shape[-2:]), pro, sta)

    def sample(self, B):
        ages = np.random.randint(1, self.filled, B)
        ts = (self.ptr - 1 - ages) % self.Tc
        envs = np.random.randint(0, self.N, B)
        r0 = np.array([self._chain(t, i) for t, i in zip(ts, envs)])
        r1, fin1 = [], {}
        for b, (t, i) in enumerate(zip(ts, envs)):
            if self.done[t, i]:
                chain = list(r0[b][1:]) + [t]           # 마지막 칸은 final_obs로 대체
                fin1[(b, self.k - 1)] = self.final[(t, i)]
            else:
                chain = self._chain((t + 1) % self.Tc, i)
            r1.append(chain)
        r1 = np.array(r1)
        cols = np.repeat(envs[:, None], self.k, 1)
        o = self._gather(r0, cols, {})
        o2 = self._gather(r1, cols, fin1)
        return o, self.act[ts, envs], self.rew[ts, envs], o2, self.term[ts, envs].astype(np.float32)


def load_prior(path, spaces, act_dim, k):
    """make_demos.py npz → env 1개짜리 GridReplay (에피소드를 시간 순으로 이어 붙인 격자, 꽉 찬 상태)."""
    z = np.load(path)
    T = len(z["act"])
    rb = GridReplay(T, 1, spaces, act_dim, k)
    rb.fov[:, 0], rb.per[:, 0] = z["fovea"], z["periph"]
    rb.pro[:, 0], rb.sta[:, 0] = z["proprio"], z["state"]
    rb.act[:, 0], rb.rew[:, 0] = z["act"], z["rew"]
    rb.term[:, 0], rb.done[:, 0], rb.first[:, 0] = z["term"], z["done"], z["first"]
    for j, t in enumerate(z["final_t"]):
        rb.final[(int(t), 0)] = {kk: z[f"final_{kk}"][j] for kk in ("fovea", "periph", "proprio", "state")}
    rb.ptr, rb.filled = 0, T   # 가장 마지막 칸(age 0 = T-1, 끝난 스텝)만 안 뽑힘
    oc = z["outcome"] if "outcome" in z.files else []
    print(f"prior 시연: {path}  {T} 전이, 에피소드 {int(z['first'].sum())}개, 성공 {int((oc == 'success').sum())}", flush=True)
    return rb


def prior_ratio(cfg, step):
    f = min(1.0, step / max(1, cfg.prior_anneal_steps))
    if cfg.prior_shape == "exp":
        end = max(cfg.prior_end, 0.01)
        r = cfg.prior_start * (end / cfg.prior_start) ** f
        return cfg.prior_end if f >= 1.0 else r
    return cfg.prior_start + (cfg.prior_end - cfg.prior_start) * f


def cat_batch(a, b):
    """GridReplay.sample 결과 두 개를 배치 축으로 이어 붙임."""
    (o_a, act_a, r_a, o2_a, d_a), (o_b, act_b, r_b, o2_b, d_b) = a, b
    return (tuple(np.concatenate([x, y]) for x, y in zip(o_a, o_b)), np.concatenate([act_a, act_b]),
            np.concatenate([r_a, r_b]), tuple(np.concatenate([x, y]) for x, y in zip(o2_a, o2_b)),
            np.concatenate([d_a, d_b]))


# ---------------------------------------------------------------------------- networks
class RandomShift(nn.Module):
    """DrQ-v2 random shift (가장자리 반복 패딩 후 무작위 이동)."""

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        if self.pad == 0:
            return x
        n, c, h, w = x.shape
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        eps_h, eps_w = 1.0 / (h + 2 * self.pad), 1.0 / (w + 2 * self.pad)
        ar_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h + 2 * self.pad, device=x.device)[:h]
        ar_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w + 2 * self.pad, device=x.device)[:w]
        base = torch.stack(torch.meshgrid(ar_h, ar_w, indexing="ij")[::-1], -1)[None].repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, (n, 1, 1, 2), device=x.device).float()
        shift *= torch.tensor([2.0 / (w + 2 * self.pad), 2.0 / (h + 2 * self.pad)], device=x.device)
        return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, in_ch, h, w, feat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1), nn.ReLU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_ch, h, w)).numel()
        self.head = nn.Sequential(nn.Linear(n, feat), nn.LayerNorm(feat), nn.Tanh())

    def forward(self, x):
        return self.head(self.conv(x / 255.0 - 0.5).flatten(1))


class Vision(nn.Module):
    """중심와 + 주변시 인코더 → z (2·feat)."""

    def __init__(self, fov_shape, per_shape, feat):
        super().__init__()
        self.fov = Encoder(fov_shape[0], fov_shape[1], fov_shape[2], feat)
        self.per = Encoder(per_shape[0], per_shape[1], per_shape[2], feat)

    def forward(self, fov, per):
        return torch.cat([self.fov(fov), self.per(per)], -1)


def mlp(i, o, hidden, ln=True):
    layers = []
    for a, b in ((i, hidden), (hidden, hidden)):
        layers += [nn.Linear(a, b)] + ([nn.LayerNorm(b)] if ln else []) + [nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(hidden, o))


class Critic(nn.Module):
    def __init__(self, in_dim, act_dim, hidden):
        super().__init__()
        self.q = nn.ModuleList([mlp(in_dim + act_dim, 1, hidden) for _ in range(2)])

    def forward(self, x, a):
        xa = torch.cat([x, a], -1)
        return torch.stack([q(xa).squeeze(-1) for q in self.q])     # (2, B)


class Actor(nn.Module):
    def __init__(self, in_dim, act_dim, hidden):
        super().__init__()
        self.body = mlp(in_dim, 2 * act_dim, hidden, ln=False)
        nn.init.uniform_(self.body[-1].weight, -1e-3, 1e-3)      # 처음 평균 행동 ≈ 0
        nn.init.zeros_(self.body[-1].bias)

    def sample(self, x):
        mean, log_std = self.body(x).chunk(2, -1)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp, torch.tanh(mean)


# ---------------------------------------------------------------------------- main
def make_env(env_cfg, seed):
    def thunk():
        env = TransferEnv(env_cfg)
        env.reset(seed=seed)
        return env
    return thunk


def main(cfg: ImgSACConfig, env_cfg: EnvConfig):
    env_cfg.obs_mode = "image"
    run_dir = ROOT / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"algo": "sac_image", "sac": asdict(cfg), "env": env_cfg.to_dict()}, indent=2, ensure_ascii=False,
        default=str), encoding="utf-8")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run_dir))
    log_csv = open(run_dir / "progress.csv", "w", encoding="utf-8")
    log_csv.write("vec_step,global_step,grad_steps,sps,phase,stage,success,collision,fell,ep_len,"
                  "phi_metric,not_visible_metric,q_loss,pred_loss,actor_loss,alpha_arm,alpha_eye,q_mean,prior_ratio\n")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    env_cfg.stage_probs = cfg.curriculum[0]
    # spawn: 부모가 CUDA를 건드린 뒤 fork하면 자식의 EGL 렌더러가 깨짐 (Offscreen framebuffer is not complete)
    envs = gym.vector.AsyncVectorEnv([make_env(env_cfg, cfg.seed + i) for i in range(cfg.num_envs)],
                                     autoreset_mode=AutoresetMode.SAME_STEP, context="spawn")
    N, k = cfg.num_envs, cfg.frame_stack
    sp = envs.single_observation_space
    act_dim = envs.single_action_space.shape[0]
    fov_shape = (sp["fovea"].shape[0] * k,) + sp["fovea"].shape[1:]
    per_shape = (sp["periph"].shape[0] * k,) + sp["periph"].shape[1:]
    P, S = sp["proprio"].shape[0], sp["state"].shape[0]
    zdim = 2 * cfg.feat_dim
    crit_in = zdim + P + (S if cfg.critic_input == "asym" else 0)

    vision = Vision(fov_shape, per_shape, cfg.feat_dim).to(device)
    vision_t = Vision(fov_shape, per_shape, cfg.feat_dim).to(device)
    vision_t.load_state_dict(vision.state_dict())
    critic = Critic(crit_in, act_dim, cfg.hidden).to(device)
    critic_t = Critic(crit_in, act_dim, cfg.hidden).to(device)
    critic_t.load_state_dict(critic.state_dict())
    predictor = mlp(zdim + act_dim, zdim, cfg.hidden).to(device)
    actors = {"arm": Actor(zdim + P, 6, cfg.hidden).to(device), "eye": Actor(zdim + P, 6, cfg.hidden).to(device)}
    q_opt = torch.optim.Adam(list(vision.parameters()) + list(critic.parameters()) + list(predictor.parameters()),
                             lr=cfg.lr)
    a_opts = {kk: torch.optim.Adam(v.parameters(), lr=cfg.lr) for kk, v in actors.items()}
    log_alphas = {kk: torch.tensor(np.log(cfg.init_alpha), device=device, requires_grad=True) for kk in actors}
    al_opts = {kk: torch.optim.Adam([v], lr=cfg.lr) for kk, v in log_alphas.items()}
    target_entropy = -6.0
    aug = RandomShift(cfg.aug_pad)

    # proprio / state 정규화 (처음 obs_norm_steps 통계로 고정)
    stats = {"proprio": [np.zeros(P), np.zeros(P)], "state": [np.zeros(S), np.zeros(S)]}
    norm = {"proprio": [torch.zeros(P, device=device), torch.ones(P, device=device)],
            "state": [torch.zeros(S, device=device), torch.ones(S, device=device)]}
    norm_n, norm_frozen = 0, False

    def nz(key, x):
        m, s = norm[key]
        return torch.clamp((x - m) / s, -10, 10)

    def to_t(x, dtype=torch.float32):
        return torch.as_tensor(x, device=device).to(dtype)

    torch.backends.cudnn.benchmark = True

    def encode(net, fov, per, augment):
        fov, per = to_t(fov), to_t(per)
        if augment:
            fov, per = aug(fov), aug(per)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
            z = net(fov, per)
        return z.float()

    def sample_joint(x):
        a1, l1, _ = actors["arm"].sample(x)
        a2, l2, _ = actors["eye"].sample(x)
        return torch.cat([a1, a2], -1), l1, l2

    def phase_at(step):
        if not cfg.alternate:
            return "both"
        other = "eye" if cfg.first_phase == "arm" else "arm"
        return cfg.first_phase if (step // cfg.phase_len_steps) % 2 == 0 else other

    rb = GridReplay(cfg.buffer_size, N, sp, act_dim, k)
    rb_prior = load_prior(cfg.prior_path, sp, act_dim, k) if cfg.prior_path else None
    p_ratio = prior_ratio(cfg, 0) if rb_prior is not None else 0.0

    # ---- 영상: 에피소드 묶음 구간마다 평가 에피소드 1개 (별도 env, 결정적 행동 = 평균) ----
    video_dir = Path(os.path.expanduser(cfg.video_dir)) / cfg.run_name if cfg.video_dir else None
    video_env = None
    if video_dir is not None:
        import imageio.v2 as imageio
        video_dir.mkdir(parents=True, exist_ok=True)
        video_env = gym.vector.AsyncVectorEnv([make_env(env_cfg, cfg.seed + 10_000)],
                                              autoreset_mode=AutoresetMode.SAME_STEP, context="spawn")

    def record_video(ep_lo, ep_hi):
        vo, _ = video_env.reset(seed=int(np.random.randint(1 << 30)))
        vs = {kk: deque([vo[kk]] * k, maxlen=k) for kk in ("fovea", "periph")}
        name = f"ep{ep_lo:06d}-{ep_hi:06d}_step{global_step:09d}"
        frames, outcome = [], "timeout"
        for t in range(env_cfg.episode_steps):
            frames.append(video_env.call("video_frame", f"{cfg.run_name} {name} t={t}")[0])
            with torch.no_grad():
                z = encode(vision, np.concatenate(list(vs["fovea"]), 1), np.concatenate(list(vs["periph"]), 1), False)
                x = torch.cat([z, nz("proprio", to_t(vo["proprio"]))], -1)
                a = torch.cat([actors["arm"].sample(x)[2], actors["eye"].sample(x)[2]], -1).cpu().numpy()
            vo, _, vterm, vtrunc, vinfo = video_env.step(a)
            if vterm[0] or vtrunc[0]:
                vf = vinfo["final_info"]
                outcome = "success" if vf["success"][0] else "collision" if vf["collision"][0] else \
                    "fell" if vf["fell"][0] else "timeout"
                break
            vs["fovea"].append(vo["fovea"])
            vs["periph"].append(vo["periph"])
        path = video_dir / f"{name}_{outcome}.mp4"
        imageio.mimwrite(str(path), frames, fps=int(env_cfg.control_hz), macro_block_size=1)
        print(f"  video saved: {path} ({len(frames)} frames, {outcome})", flush=True)

    n_episodes, next_video_at = 0, 0

    # 정책 쪽 프레임 쌓기
    o, _ = envs.reset(seed=cfg.seed)
    stack = {"fovea": deque([o["fovea"]] * k, maxlen=k), "periph": deque([o["periph"]] * k, maxlen=k)}
    first = np.ones(N, bool)

    win = deque(maxlen=cfg.curriculum_window)
    ep_len = np.zeros(N)
    ep_phi = np.zeros(N)
    ep_nv = np.zeros(N)
    stage_i, global_step, grad_steps, t0 = 0, 0, 0, time.time()
    q_loss = pred_loss = actor_loss = torch.tensor(0.0)
    q_mean = 0.0
    phase = phase_at(0)

    for vec_step in range(1, cfg.total_steps // N + 1):
        global_step += N
        if global_step <= cfg.learning_starts:
            act = np.random.uniform(-1, 1, (N, act_dim)).astype(np.float32)
        else:
            with torch.no_grad():
                fov = np.concatenate(list(stack["fovea"]), 1)
                per = np.concatenate(list(stack["periph"]), 1)
                z = encode(vision, fov, per, False)
                x = torch.cat([z, nz("proprio", to_t(o["proprio"]))], -1)
                act = sample_joint(x)[0].cpu().numpy()

        o2, r, term, trunc, info = envs.step(act)
        done = np.logical_or(term, trunc)
        fin = info.get("final_info", {})
        final_obs = {}
        phis = np.array(info.get("phi", np.zeros(N)), dtype=np.float64)
        nvis = np.array(info.get("stem_not_visible", np.zeros(N)), dtype=np.float64)
        for i in np.where(done)[0]:
            fo = info["final_obs"][i]
            final_obs[i] = {kk: np.asarray(fo[kk]) for kk in ("fovea", "periph", "proprio", "state")}
            phis[i], nvis[i] = fin["phi"][i], fin["stem_not_visible"][i]
        rb.add(o, act, r, term, done, first, final_obs)

        if not norm_frozen:
            for key in ("proprio", "state"):
                stats[key][0] += o[key].sum(0)
                stats[key][1] += (o[key].astype(np.float64) ** 2).sum(0)
            norm_n += N
            if global_step >= cfg.obs_norm_steps:
                for key in ("proprio", "state"):
                    mu = stats[key][0] / norm_n
                    sd = np.sqrt(np.maximum(stats[key][1] / norm_n - mu**2, 0)) + 1e-6
                    sd = np.where(sd < 1e-3, 1.0, sd)
                    norm[key][0].copy_(to_t(mu))
                    norm[key][1].copy_(to_t(sd))
                norm_frozen = True

        ep_len += 1
        ep_phi += phis
        ep_nv += nvis
        for i in np.where(done)[0]:
            win.append((float(fin["success"][i]), float(fin["collision"][i]), float(fin["fell"][i]),
                        ep_len[i], ep_phi[i] / ep_len[i], ep_nv[i] / ep_len[i]))
            ep_len[i] = ep_phi[i] = ep_nv[i] = 0
            n_episodes += 1
        if video_env is not None and n_episodes >= next_video_at:
            record_video(max(0, next_video_at - cfg.video_every_episodes), n_episodes)
            next_video_at += cfg.video_every_episodes

        # 프레임 쌓기 갱신: 끝난 env는 새 에피소드 첫 프레임으로 채움
        stack["fovea"].append(o2["fovea"])
        stack["periph"].append(o2["periph"])
        if done.any():
            for key in ("fovea", "periph"):
                arr = list(stack[key])
                for j in range(k):
                    arr[j] = arr[j].copy()
                    arr[j][done] = o2[key][done]
                stack[key] = deque(arr, maxlen=k)
        first = done.copy()
        o = o2

        new_phase = phase_at(global_step)
        if new_phase != phase:
            print(f"  -> phase {phase} -> {new_phase} at {global_step}", flush=True)
            phase = new_phase
        train_keys = ["arm", "eye"] if phase == "both" else [phase]

        if rb_prior is not None:
            p_ratio = prior_ratio(cfg, global_step)
        # ---- 학습 ----
        if global_step > cfg.learning_starts and norm_frozen and rb.filled > 2:
            for _ in range(cfg.updates_per_vec_step):
                grad_steps += 1
                n_prior = int(round(cfg.batch_size * p_ratio)) if rb_prior is not None else 0
                if n_prior >= cfg.batch_size:
                    batch = rb_prior.sample(cfg.batch_size)
                elif n_prior > 0:
                    batch = cat_batch(rb_prior.sample(n_prior), rb.sample(cfg.batch_size - n_prior))
                else:
                    batch = rb.sample(cfg.batch_size)
                (f0, p0, pr0, st0), ba, br, (f1, p1, pr1, st1), bd = batch
                ba, br, bd = to_t(ba), to_t(br), to_t(bd)
                pr0, pr1 = nz("proprio", to_t(pr0)), nz("proprio", to_t(pr1))
                st0, st1 = nz("state", to_t(st0)), nz("state", to_t(st1))
                al = {kk: v.exp().detach() for kk, v in log_alphas.items()}

                z0 = encode(vision, f0, p0, True)
                with torch.no_grad():
                    z1_t = encode(vision_t, f1, p1, True)       # 다음 관측 특징은 target 인코더로 한 번만
                    x1 = torch.cat([z1_t, pr1], -1)
                    a1, l_arm, l_eye = sample_joint(x1)
                    c1 = torch.cat([z1_t, pr1] + ([st1] if cfg.critic_input == "asym" else []), -1)
                    q1 = critic_t(c1, a1).min(0).values - al["arm"] * l_arm - al["eye"] * l_eye
                    y = br + cfg.gamma * (1 - bd) * q1
                c0 = torch.cat([z0, pr0] + ([st0] if cfg.critic_input == "asym" else []), -1)
                q = critic(c0, ba)
                q_loss = F.mse_loss(q, y.expand_as(q), reduction="none").mean(1).sum()
                pred = predictor(torch.cat([z0, ba], -1))
                pred_loss = (2 - 2 * F.cosine_similarity(pred, z1_t.detach(), dim=-1)).mean()
                q_opt.zero_grad()
                (q_loss + cfg.pred_coef * pred_loss).backward()
                q_opt.step()
                q_mean = q.mean().item()

                if grad_steps % cfg.actor_update_every == 0:
                    x0 = torch.cat([z0.detach(), pr0], -1)
                    c0d = c0.detach()
                    for kk in train_keys:
                        other = "eye" if kk == "arm" else "arm"
                        a_k, logp_k, _ = actors[kk].sample(x0)
                        with torch.no_grad():
                            a_o, _, _ = actors[other].sample(x0)
                        a_full = torch.cat([a_k, a_o], -1) if kk == "arm" else torch.cat([a_o, a_k], -1)
                        actor_loss = (al[kk] * logp_k - critic(c0d, a_full).min(0).values).mean()
                        a_opts[kk].zero_grad()
                        actor_loss.backward()
                        a_opts[kk].step()
                        al_loss = -(log_alphas[kk] * (logp_k.detach() + target_entropy)).mean()
                        al_opts[kk].zero_grad()
                        al_loss.backward()
                        al_opts[kk].step()

                with torch.no_grad():
                    for net, net_t in ((critic, critic_t), (vision, vision_t)):
                        for p_, pt in zip(net.parameters(), net_t.parameters()):
                            pt.mul_(1 - cfg.tau).add_(cfg.tau * p_)

        # ---- 기록 ----
        if vec_step % cfg.log_every_vec_steps == 0:
            sps = int(global_step / (time.time() - t0))
            if win:
                wa = np.array(win)
                st = dict(success=wa[:, 0].mean(), collision=wa[:, 1].mean(), fell=wa[:, 2].mean(),
                          ep_len=wa[:, 3].mean(), phi=wa[:, 4].mean(), not_visible=wa[:, 5].mean())
            else:
                st = dict(success=0, collision=0, fell=0, ep_len=0, phi=0, not_visible=0)
            alpha_a, alpha_e = log_alphas["arm"].exp().item(), log_alphas["eye"].exp().item()
            for kk, v in st.items():
                writer.add_scalar(f"episode/{kk}", v, global_step)
            for kk, v in dict(q_loss=q_loss.item(), pred_loss=pred_loss.item(), actor_loss=actor_loss.item(),
                              alpha_arm=alpha_a, alpha_eye=alpha_e, q_mean=q_mean, sps=sps, stage=stage_i, prior_ratio=p_ratio,
                              phase_eye=float(phase == "eye")).items():
                writer.add_scalar(f"train/{kk}", v, global_step)
            log_csv.write(f"{vec_step},{global_step},{grad_steps},{sps},{phase},{stage_i},{st['success']:.4f},"
                          f"{st['collision']:.4f},{st['fell']:.4f},{st['ep_len']:.1f},{st['phi']:.4f},"
                          f"{st['not_visible']:.4f},{q_loss.item():.5f},{pred_loss.item():.5f},"
                          f"{actor_loss.item():.5f},{alpha_a:.5f},{alpha_e:.5f},{q_mean:.4f},{p_ratio:.4f}\n")
            log_csv.flush()
            print(f"[{global_step}] {phase:4s} sps {sps} stage {stage_i} succ {st['success']:.2f} "
                  f"coll {st['collision']:.2f} len {st['ep_len']:.0f} | phi {st['phi']:.3f} "
                  f"unseen {st['not_visible']:.2f} | q {q_loss.item():.4f} pred {pred_loss.item():.3f} "
                  f"Q {q_mean:+.3f} a_arm/eye {alpha_a:.4f}/{alpha_e:.4f} prior {p_ratio:.2f}", flush=True)
            if (stage_i + 1 < len(cfg.curriculum) and len(win) == win.maxlen
                    and st["success"] >= cfg.curriculum_success):
                stage_i += 1
                envs.call("set_stage_probs", cfg.curriculum[stage_i])
                win.clear()
                print(f"  -> curriculum stage {stage_i}: {cfg.curriculum[stage_i]}", flush=True)

        if vec_step % cfg.save_every_vec_steps == 0:
            torch.save({"vision": vision.state_dict(), "critic": critic.state_dict(),
                        "actor_arm": actors["arm"].state_dict(), "actor_eye": actors["eye"].state_dict(),
                        "predictor": predictor.state_dict(), "norm": {kk: [t.cpu() for t in v] for kk, v in norm.items()},
                        "global_step": global_step, "stage": stage_i, "phase": phase},
                       run_dir / f"ckpt_{global_step:09d}.pt")

    envs.close()
    if video_env is not None:
        video_env.close()
    writer.close()
    log_csv.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="img_v0")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--buffer-size", type=int, default=None)
    ap.add_argument("--learning-starts", type=int, default=None)
    ap.add_argument("--obs-norm-steps", type=int, default=None)
    ap.add_argument("--phase-len-steps", type=int, default=None)
    ap.add_argument("--critic-input", choices=["sym", "asym"], default=None)
    ap.add_argument("--no-alternate", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--updates-per-vec-step", type=int, default=None)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--video-every-episodes", type=int, default=None)
    ap.add_argument("--prior-path", default=None)
    ap.add_argument("--prior-anneal-steps", type=int, default=None)
    ap.add_argument("--prior-shape", choices=["linear", "exp"], default=None)
    args = ap.parse_args()
    cfg, env_cfg = ImgSACConfig(), EnvConfig()
    cfg.run_name = args.run_name
    for kk in ("total_steps", "num_envs", "buffer_size", "learning_starts", "obs_norm_steps", "phase_len_steps",
               "critic_input", "seed", "updates_per_vec_step", "video_dir", "video_every_episodes",
               "prior_path", "prior_anneal_steps", "prior_shape"):
        if getattr(args, kk) is not None:
            setattr(cfg, kk, getattr(args, kk))
    if args.no_alternate:
        cfg.alternate = False
    os.environ.setdefault("MUJOCO_GL", "egl")
    main(cfg, env_cfg)

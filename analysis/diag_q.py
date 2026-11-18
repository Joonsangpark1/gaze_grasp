"""Q 과대평가 확인: 체크포인트의 critic Q(s_t, a_t)와 실제 할인 리턴 비교.

학습 때와 같은 확률적 행동(행동 정책)으로 에피소드를 돌리고 매 스텝 기록:
    q1, q2 (online critic, 증강 없음), r, log π_arm, log π_eye
비교 대상:
    G_t      = Σ_i γ^i r_{t+i}                                   (순수 과제 리턴, 에피소드 끝까지)
    G_soft_t = r_t + Σ_{i≥1} γ^i (r_{t+i} − α_arm·logπ_arm − α_eye·logπ_eye)   (critic이 배우는 soft 값)
    시간 초과로 끊긴 에피소드는 꼬리 γ^(T−t)·min Q(s_T, a_T)를 붙인 값도 따로 기록 (critic 자기 값이라 참고용)
실행: python -m analysis.diag_q --run img_sparse_v1 --ckpts 2480000 2560000 --alpha 0.0002 0.0003 --rounds 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.sac_image import Actor, Critic, Vision, make_env  # noqa: E402
from envs.transfer_env import EnvConfig  # noqa: E402


def load(run_dir, step, sp, k, device):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["sac"]
    ck = torch.load(run_dir / f"ckpt_{step:09d}.pt", map_location=device)
    fov_shape = (sp["fovea"].shape[0] * k,) + sp["fovea"].shape[1:]
    per_shape = (sp["periph"].shape[0] * k,) + sp["periph"].shape[1:]
    P, S = sp["proprio"].shape[0], sp["state"].shape[0]
    zdim = 2 * cfg["feat_dim"]
    crit_in = zdim + P + (S if cfg["critic_input"] == "asym" else 0)
    vision = Vision(fov_shape, per_shape, cfg["feat_dim"]).to(device)
    critic = Critic(crit_in, 12, cfg["hidden"]).to(device)
    arm, eye = Actor(zdim + P, 6, cfg["hidden"]).to(device), Actor(zdim + P, 6, cfg["hidden"]).to(device)
    vision.load_state_dict(ck["vision"]); critic.load_state_dict(ck["critic"])
    arm.load_state_dict(ck["actor_arm"]); eye.load_state_dict(ck["actor_eye"])
    norm = {kk: [t.to(device) for t in v] for kk, v in ck["norm"].items()}
    for net in (vision, critic, arm, eye):
        net.eval()
    return vision, critic, arm, eye, norm, cfg


def run(run_dir, step, alpha, rounds, n_envs, device, deterministic=False):
    env_cfg = EnvConfig()
    env_cfg.obs_mode = "image"
    env_cfg.stage_probs = {"near": 1.0}
    envs = gym.vector.AsyncVectorEnv([make_env(env_cfg, 777 + step % 1000 + i) for i in range(n_envs)],
                                     autoreset_mode=AutoresetMode.SAME_STEP, context="spawn")
    sp = envs.single_observation_space
    k = 2
    vision, critic, arm, eye, norm, cfg = load(run_dir, step, sp, k, device)
    gamma = cfg["gamma"]

    def nz(x):
        m, s = norm["proprio"]
        return torch.clamp((torch.as_tensor(x, device=device).float() - m) / s, -10, 10)

    @torch.no_grad()
    def qa(stk, pro):
        fov = torch.as_tensor(np.concatenate(list(stk["fovea"]), 1), device=device).float()
        per = torch.as_tensor(np.concatenate(list(stk["periph"]), 1), device=device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = vision(fov, per)
        x = torch.cat([z.float(), nz(pro)], -1)
        a1, l1, m1 = arm.sample(x)
        a2, l2, m2 = eye.sample(x)
        a = torch.cat([m1, m2], -1) if deterministic else torch.cat([a1, a2], -1)
        q = critic(x, a)
        return a.cpu().numpy(), q.cpu().numpy(), l1.cpu().numpy(), l2.cpu().numpy()

    episodes = []
    for rd in range(rounds):
        o, _ = envs.reset(seed=int(1000 * rd + step % 997))
        stk = {kk: deque([o[kk]] * k, maxlen=k) for kk in ("fovea", "periph")}
        traj = [dict(q=[], r=[], la=[], le=[]) for _ in range(n_envs)]
        active = np.ones(n_envs, bool)
        tail_q = [None] * n_envs
        outcome = [None] * n_envs
        while active.any():
            a, q, la, le = qa(stk, o["proprio"])
            o2, r, term, trunc, info = envs.step(a)
            for i in np.where(active)[0]:
                traj[i]["q"].append(q[:, i]); traj[i]["r"].append(r[i])
                traj[i]["la"].append(la[i]); traj[i]["le"].append(le[i])
            done = term | trunc
            for i in np.where(done & active)[0]:
                fi = info["final_info"]
                outcome[i] = ("success" if fi["success"][i] else "collision" if fi["collision"][i]
                              else "fell" if fi["fell"][i] else "timeout")
                if trunc[i] and not term[i]:
                    fo = info["final_obs"][i]
                    s1 = {kk: deque([np.asarray(stk[kk][-1][i:i + 1]), np.asarray(fo[kk])[None]], maxlen=k)
                          for kk in ("fovea", "periph")}
                    _, qT, _, _ = qa(s1, np.asarray(fo["proprio"])[None])
                    tail_q[i] = float(qT.min(0)[0])
                active[i] = False
            stk["fovea"].append(o2["fovea"]); stk["periph"].append(o2["periph"])
            o = o2
        for i in range(n_envs):
            t = traj[i]
            episodes.append(dict(q=np.array(t["q"]), r=np.array(t["r"]), la=np.array(t["la"]),
                                 le=np.array(t["le"]), outcome=outcome[i], tail_q=tail_q[i]))
    envs.close()

    rows = []
    for ep in episodes:
        T = len(ep["r"])
        ent = -(alpha[0] * ep["la"] + alpha[1] * ep["le"])
        G = np.zeros(T); Gs = np.zeros(T); Gt = np.zeros(T)
        g = gs = 0.0
        gt = ep["tail_q"] if ep["tail_q"] is not None else 0.0
        for t in reversed(range(T)):
            g = ep["r"][t] + gamma * g
            gt = ep["r"][t] + gamma * gt
            nxt = ent[t + 1] if t + 1 < T else 0.0
            gs = ep["r"][t] + gamma * (nxt + gs)
            G[t], Gs[t], Gt[t] = g, gs, gt
        qmin = ep["q"].min(1)
        rows.append(dict(outcome=ep["outcome"], T=T, q0=float(qmin[0]), qmean=float(qmin.mean()),
                         qmax=float(qmin.max()), q_tw_gap=float(np.abs(ep["q"][:, 0] - ep["q"][:, 1]).mean()),
                         G0=float(G[0]), Gmean=float(G.mean()), Gsoft_mean=float(Gs.mean()),
                         Gtail_mean=float(Gt.mean()), tail_q=ep["tail_q"],
                         logp_arm=float(ep["la"].mean()), logp_eye=float(ep["le"].mean()),
                         ent_term_mean=float(ent.mean())))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="img_sparse_v1")
    ap.add_argument("--ckpts", type=int, nargs="+", required=True)
    ap.add_argument("--alpha", type=float, nargs=2, default=[0.0002, 0.0003])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / "runs" / args.run
    out = {}
    for s in args.ckpts:
        rows = run(run_dir, s, args.alpha, args.rounds, args.num_envs, dev, args.deterministic)
        out[s] = rows
        print(f"\n=== ckpt {s} ({'det' if args.deterministic else 'stoch'}) ===")
        print(f"{'outcome':9s} {'T':>4s} {'Q(s0)':>7s} {'Qmean':>7s} {'Qmax':>7s} {'twin':>6s} "
              f"{'G0':>6s} {'Gmean':>6s} {'Gsoft':>7s} {'Gtail':>7s} {'tailQ':>7s} {'lpA':>6s} {'lpE':>6s}")
        for r in rows:
            tq = f"{r['tail_q']:+7.3f}" if r["tail_q"] is not None else "      -"
            print(f"{r['outcome']:9s} {r['T']:4d} {r['q0']:+7.3f} {r['qmean']:+7.3f} {r['qmax']:+7.3f} "
                  f"{r['q_tw_gap']:6.3f} {r['G0']:+6.3f} {r['Gmean']:+6.3f} {r['Gsoft_mean']:+7.3f} "
                  f"{r['Gtail_mean']:+7.3f} {tq} {r['logp_arm']:6.1f} {r['logp_eye']:6.1f}")
        qm = np.mean([r["qmean"] for r in rows]); gm = np.mean([r["Gmean"] for r in rows])
        gs = np.mean([r["Gsoft_mean"] for r in rows])
        print(f"평균: Q {qm:+.3f} | G {gm:+.3f} | G_soft {gs:+.3f} | Q−G_soft {qm - gs:+.3f}")
    tag = "det" if args.deterministic else "stoch"
    (run_dir / f"diag_q_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

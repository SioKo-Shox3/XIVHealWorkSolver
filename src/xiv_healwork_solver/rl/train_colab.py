"""Masked PPO trainer for XIVHealWorkSolver (runnable on Google Colab GPU or locally).

Example Colab run (after uploading kit + timeline or cloning repo on VM):
    python -m xiv_healwork_solver.rl.train_colab \
        --kit data/kits/example_barrier_healer.yaml \
        --timeline data/timelines/example_savage_3min.yaml \
        --updates 80 --episodes-per-update 16 --out-dir outputs/rl_run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from xiv_healwork_solver.models import EncounterTimeline, PlayerKit
from xiv_healwork_solver.rl.env import FF14HealMitEnv
from xiv_healwork_solver.rl.model import SkillAttentionActorCritic


def _compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    dts: list[float],
    gamma_per_sec: float = 0.992,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time GAE using elapsed-seconds discount factor gamma_per_sec ** dt."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_non_terminal = 0.0 if dones[t] else 1.0
        next_value = 0.0 if (t == n - 1 or dones[t]) else values[t + 1]
        gamma_t = gamma_per_sec ** max(0.0, dts[t])
        delta = rewards[t] + gamma_t * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma_t * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


@torch.no_grad()
def evaluate_policy(
    model: SkillAttentionActorCritic,
    kit: PlayerKit,
    timeline: EncounterTimeline,
    device: torch.device,
    n_eval_pulls: int = 24,
) -> dict[str, Any]:
    """Evaluate across evenly spaced 3s server-tick phases + stochastic partner/damage rolls."""
    model.eval()
    env = FF14HealMitEnv(kit, timeline, seed=12345)
    survivals = 0
    rewards: list[float] = []
    fillers: list[float] = []
    gcd_heals: list[float] = []
    min_party: list[float] = []
    min_mt: list[float] = []
    min_st: list[float] = []
    best_log: list[dict[str, Any]] = []

    for i in range(n_eval_pulls):
        phase = (3.0 * i) / max(1, n_eval_pulls)
        obs = env.reset(deterministic=(i == 0), fixed_tick_phase=phase)
        ep_ret = 0.0
        done = False
        while not done:
            st = torch.from_numpy(obs["state"]).unsqueeze(0).to(device)
            sk = torch.from_numpy(obs["skills"]).unsqueeze(0).to(device)
            fu = torch.from_numpy(obs["future"]).unsqueeze(0).to(device)
            mk = torch.from_numpy(obs["mask"]).unsqueeze(0).to(device)
            act, _, _, _ = model.get_action_and_value(st, sk, fu, mk, deterministic=True)
            obs, r, done, info = env.step(int(act.item()), deterministic=(i == 0))
            ep_ret += r
        s = info["stats"]
        survivals += int(s["survived"] > 0.5)
        rewards.append(ep_ret)
        fillers.append(s["filler_gcds"])
        gcd_heals.append(s["gcd_heals"])
        min_party.append(s["min_party_hp_ratio"])
        min_mt.append(s["min_mt_hp_ratio"])
        min_st.append(s["min_st_hp_ratio"])
        if i == 0:
            best_log = list(env.action_log)

    return {
        "eval_pulls": n_eval_pulls,
        "survival_rate_pct": round((survivals / n_eval_pulls) * 100.0, 2),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "mean_filler_gcds": round(float(np.mean(fillers)), 1),
        "mean_gcd_heals": round(float(np.mean(gcd_heals)), 2),
        "worst_party_hp_pct": round(float(np.min(min_party)) * 100.0, 1),
        "worst_mt_hp_pct": round(float(np.min(min_mt)) * 100.0, 1),
        "worst_st_hp_pct": round(float(np.min(min_st)) * 100.0, 1),
        "deterministic_action_log": best_log,
    }


def train_masked_ppo(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    updates: int = 60,
    episodes_per_update: int = 12,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    clip_coef: float = 0.2,
    ent_coef: float = 0.015,
    vf_coef: float = 0.5,
    seed: int = 42,
    out_dir: str | Path = "outputs/rl_run",
) -> dict[str, Any]:
    """Train SkillAttentionActorCritic with Masked PPO on GPU (if available) or CPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RL] Device: {device} | Kit: {kit.label} | Timeline: {timeline.encounter_name}")

    env = FF14HealMitEnv(kit, timeline, seed=seed)
    model = SkillAttentionActorCritic(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    history: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for upd in range(1, updates + 1):
        model.eval()
        states_buf: list[np.ndarray] = []
        skills_buf: list[np.ndarray] = []
        futures_buf: list[np.ndarray] = []
        masks_buf: list[np.ndarray] = []
        actions_buf: list[int] = []
        logprobs_buf: list[float] = []
        rewards_buf: list[float] = []
        values_buf: list[float] = []
        dones_buf: list[bool] = []
        dts_buf: list[float] = []

        ep_survivals = 0
        ep_returns: list[float] = []

        with torch.no_grad():
            for _ in range(episodes_per_update):
                obs = env.reset()
                done = False
                ep_r = 0.0
                while not done:
                    st = torch.from_numpy(obs["state"]).unsqueeze(0).to(device)
                    sk = torch.from_numpy(obs["skills"]).unsqueeze(0).to(device)
                    fu = torch.from_numpy(obs["future"]).unsqueeze(0).to(device)
                    mk = torch.from_numpy(obs["mask"]).unsqueeze(0).to(device)
                    act, lp, _, val = model.get_action_and_value(st, sk, fu, mk)
                    act_i = int(act.item())
                    next_obs, r, done, info = env.step(act_i)

                    states_buf.append(obs["state"])
                    skills_buf.append(obs["skills"])
                    futures_buf.append(obs["future"])
                    masks_buf.append(obs["mask"])
                    actions_buf.append(act_i)
                    logprobs_buf.append(float(lp.item()))
                    rewards_buf.append(r)
                    values_buf.append(float(val.item()))
                    dones_buf.append(done)
                    dts_buf.append(float(info.get("dt", 0.5)))

                    obs = next_obs
                    ep_r += r
                ep_survivals += int(info["stats"]["survived"] > 0.5)
                ep_returns.append(ep_r)

        adv, ret = _compute_gae(rewards_buf, values_buf, dones_buf, dts_buf)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        b_states = torch.from_numpy(np.asarray(states_buf)).to(device)
        b_skills = torch.from_numpy(np.asarray(skills_buf)).to(device)
        b_futures = torch.from_numpy(np.asarray(futures_buf)).to(device)
        b_masks = torch.from_numpy(np.asarray(masks_buf)).to(device)
        b_actions = torch.tensor(actions_buf, dtype=torch.long, device=device)
        b_logprobs = torch.tensor(logprobs_buf, dtype=torch.float32, device=device)
        b_adv = torch.from_numpy(adv).to(device)
        b_ret = torch.from_numpy(ret).to(device)

        model.train()
        n_steps = len(actions_buf)
        idxs = np.arange(n_steps)
        for _ in range(ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n_steps, minibatch_size):
                mb = idxs[start : start + minibatch_size]
                _, new_lp, entropy, new_val = model.get_action_and_value(
                    b_states[mb], b_skills[mb], b_futures[mb], b_masks[mb], action=b_actions[mb]
                )
                ratio = (new_lp - b_logprobs[mb]).exp()
                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        train_surv = (ep_survivals / episodes_per_update) * 100.0
        mean_ret = float(np.mean(ep_returns))
        row = {
            "update": upd,
            "train_survival_pct": round(train_surv, 1),
            "train_mean_return": round(mean_ret, 2),
            "elapsed_s": round(time.perf_counter() - t_start, 1),
        }
        history.append(row)

        if upd == 1 or upd % max(1, updates // 10) == 0 or upd == updates:
            ev = evaluate_policy(model, kit, timeline, device, n_eval_pulls=18)
            score = ev["survival_rate_pct"] * 10.0 + ev["mean_reward"]
            print(
                f"[Upd {upd:3d}/{updates}] train_surv={train_surv:5.1f}% ret={mean_ret:6.2f} | "
                f"eval_surv={ev['survival_rate_pct']:5.1f}% filler={ev['mean_filler_gcds']:4.1f} "
                f"gcd_heal={ev['mean_gcd_heals']:4.2f} worst_party={ev['worst_party_hp_pct']:4.1f}% "
                f"worst_mt={ev['worst_mt_hp_pct']:4.1f}% ({row['elapsed_s']}s)"
            )
            if score > best_score:
                best_score = score
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "hidden_dim": hidden_dim,
                        "kit_label": kit.label,
                        "encounter_name": timeline.encounter_name,
                        "eval": ev,
                    },
                    out_path / "best_policy.pt",
                )

    final_eval = evaluate_policy(model, kit, timeline, device, n_eval_pulls=30)
    summary = {
        "device": str(device),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "updates": updates,
        "final_eval": final_eval,
        "history": history,
    }
    (out_path / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train skill-agnostic Masked PPO policy.")
    parser.add_argument("--kit", required=True, help="Path to PlayerKit YAML")
    parser.add_argument("--timeline", required=True, help="Path to EncounterTimeline YAML")
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--episodes-per-update", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/rl_run")
    args = parser.parse_args()

    kit = PlayerKit.from_yaml(args.kit)
    timeline = EncounterTimeline.from_yaml(args.timeline)
    train_masked_ppo(
        kit=kit,
        timeline=timeline,
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

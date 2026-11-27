"""Masked PPO trainer for XIVHealWorkSolver (runnable on Google Colab GPU or locally).

Example Colab run (after uploading kit + timeline or cloning repo on VM):
    python -m xiv_healwork_solver.rl.train_colab \
        --kit data/kits/example_barrier_healer.yaml \
        --timeline data/timelines/example_savage_3min.yaml \
        --updates 80 --episodes-per-update 16 --out-dir outputs/rl_run
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from xiv_healwork_solver.models import EncounterTimeline, GearSamplingRange, PlayerKit
from xiv_healwork_solver.rl.env import FF14HealMitEnv
from xiv_healwork_solver.rl.model import SkillAttentionActorCritic
from xiv_healwork_solver.sim.safety import (
    action_log_to_plan,
    harden_plan_to_100pct,
    maximize_dps_at_100pct_survival,
    stress_test_plan,
)

# Persistent per-process worker globals (initialized once per ProcessPoolExecutor worker)
_WORKER_MODEL: SkillAttentionActorCritic | None = None
_WORKER_ENVS: list[FF14HealMitEnv] = []


def _init_rollout_worker(
    hidden_dim: int,
    kit_dump: dict[str, Any],
    timeline_dumps: list[dict[str, Any]],
    gear_range_dump: dict[str, Any] | None,
    base_seed: int,
) -> None:
    global _WORKER_MODEL, _WORKER_ENVS
    torch.set_num_threads(1)
    _WORKER_MODEL = SkillAttentionActorCritic(hidden_dim=hidden_dim)
    _WORKER_MODEL.eval()
    kit = PlayerKit.model_validate(kit_dump)
    gear_range = GearSamplingRange.model_validate(gear_range_dump) if gear_range_dump is not None else None
    timelines = [EncounterTimeline.model_validate(td) for td in timeline_dumps]
    _WORKER_ENVS = [
        FF14HealMitEnv(kit, tl, gear_sampling=gear_range, seed=base_seed + idx * 101)
        for idx, tl in enumerate(timelines)
    ]


def _run_rollout_batch(
    state_dict_cpu: dict[str, torch.Tensor],
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    assert _WORKER_MODEL is not None and _WORKER_ENVS
    _WORKER_MODEL.load_state_dict(state_dict_cpu)
    _WORKER_MODEL.eval()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

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
        for ep_i in range(n_episodes):
            env = _WORKER_ENVS[int(rng.integers(0, len(_WORKER_ENVS)))]
            obs = env.reset(seed=int(seed + ep_i * 17))
            done = False
            ep_r = 0.0
            info: dict[str, Any] = {}
            while not done:
                st = torch.from_numpy(obs["state"]).unsqueeze(0)
                sk = torch.from_numpy(obs["skills"]).unsqueeze(0)
                fu = torch.from_numpy(obs["future"]).unsqueeze(0)
                mk = torch.from_numpy(obs["mask"]).unsqueeze(0)
                act, lp, _, val = _WORKER_MODEL.get_action_and_value(st, sk, fu, mk)
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
            ep_survivals += int(info.get("stats", {}).get("survived", 0.0) > 0.5)
            ep_returns.append(ep_r)

    return {
        "states": states_buf,
        "skills": skills_buf,
        "futures": futures_buf,
        "masks": masks_buf,
        "actions": actions_buf,
        "logprobs": logprobs_buf,
        "rewards": rewards_buf,
        "values": values_buf,
        "dones": dones_buf,
        "dts": dts_buf,
        "ep_survivals": ep_survivals,
        "ep_returns": ep_returns,
    }


def _compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    dts: list[float],
    gamma_per_sec: float = 0.9985,
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
    """Evaluate across evenly spaced 3s server-tick phases + damage rolls (runs on CPU to avoid 1-row CUDA launch overhead)."""
    if device.type == "cuda":
        eval_model = SkillAttentionActorCritic(hidden_dim=model.hidden_dim)
        eval_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
        eval_model.eval()
        eval_dev = torch.device("cpu")
    else:
        eval_model = model
        eval_model.eval()
        eval_dev = device

    env = FF14HealMitEnv(kit, timeline, seed=12345, expected_partners=True)
    survivals = 0
    rewards: list[float] = []
    fillers: list[float] = []
    gcd_heals: list[float] = []
    min_party: list[float] = []
    min_mt: list[float] = []
    min_st: list[float] = []
    overheals: list[float] = []
    best_log: list[dict[str, Any]] = []

    for i in range(n_eval_pulls):
        phase = (3.0 * i) / max(1, n_eval_pulls)
        obs = env.reset(deterministic=(i == 0), fixed_tick_phase=phase)
        ep_ret = 0.0
        done = False
        while not done:
            st = torch.from_numpy(obs["state"]).unsqueeze(0).to(eval_dev)
            sk = torch.from_numpy(obs["skills"]).unsqueeze(0).to(eval_dev)
            fu = torch.from_numpy(obs["future"]).unsqueeze(0).to(eval_dev)
            mk = torch.from_numpy(obs["mask"]).unsqueeze(0).to(eval_dev)
            act, _, _, _ = eval_model.get_action_and_value(st, sk, fu, mk, deterministic=True)
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
        raw_h = s.get("player_heal_raw", 0.0)
        eff_h = s.get("player_heal_eff", 0.0)
        overheals.append(((raw_h - eff_h) / raw_h) * 100.0 if raw_h > 1e-6 else 0.0)
        if i == 0:
            best_log = list(env.action_log)

    return {
        "eval_pulls": n_eval_pulls,
        "survival_rate_pct": round((survivals / n_eval_pulls) * 100.0, 2),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "mean_filler_gcds": round(float(np.mean(fillers)), 1),
        "mean_gcd_heals": round(float(np.mean(gcd_heals)), 2),
        "mean_overheal_pct": round(float(np.mean(overheals)), 1),
        "worst_party_hp_pct": round(float(np.min(min_party)) * 100.0, 1),
        "worst_mt_hp_pct": round(float(np.min(min_mt)) * 100.0, 1),
        "worst_st_hp_pct": round(float(np.min(min_st)) * 100.0, 1),
        "deterministic_action_log": best_log,
    }


def _collect_bc_dataset(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    plan_actions: list[dict[str, Any]],
    gear_range: GearSamplingRange | None,
    n_trajectories: int = 12,
    seed: int = 42,
    fixed_kit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate supervised (obs, expert_action) trajectories by rolling out a verified plan across tick phases."""
    # When `fixed_kit=True` (e.g. an iLvl-specific optimal teacher), jitter only slightly around that iLvl so
    # the policy sees the exact optimal casts for that iLvl tier across all 12 server-tick phases!
    eff_range: GearSamplingRange | None = None
    if not fixed_kit:
        eff_range = gear_range
    elif gear_range is not None:
        eff_range = GearSamplingRange(
            gcd_min=max(2.28, round(kit.gcd_s - 0.04, 2)),
            gcd_max=min(2.50, round(kit.gcd_s + 0.04, 2)),
            piety_min=max(440, kit.gear_stats.piety - 200),
            piety_max=min(1800, kit.gear_stats.piety + 300),
            tenacity_min=max(420, kit.gear_stats.tenacity - 200),
            tenacity_max=min(2200, kit.gear_stats.tenacity + 200),
            heal_scale_min=0.98,
            heal_scale_max=1.02,
            hp_scale_min=0.98,
            hp_scale_max=1.02,
        )
    env = FF14HealMitEnv(
        kit,
        timeline,
        gear_sampling=eff_range,
        raise_prob_per_pull=0.0,
        expected_partners=True,
        seed=seed,
    )
    idx_by_key: dict[tuple[str, str], int] = {}
    idx_by_skill: dict[str, int] = {}
    for entry in env.actions[2:]:
        assert entry.skill is not None
        idx_by_key[(entry.skill.id, entry.target_channel)] = entry.index
        idx_by_skill.setdefault(entry.skill.id, entry.index)

    scheduled = sorted(plan_actions, key=lambda a: (float(a["cast_at_s"]), 0 if a.get("is_gcd") else 1))
    states, skills, futures, masks, actions = [], [], [], [], []
    best_fallback: tuple[float, list, list, list, list, list] | None = None
    bc_survivals = 0

    for i in range(n_trajectories):
        phase = (3.0 * i) / max(1, n_trajectories)
        is_det = (i % 3 == 0)
        obs = env.reset(seed=seed + i * 29, deterministic=is_det, fixed_tick_phase=phase)
        ptr = 0
        done = False
        info: dict[str, Any] = {}
        ep_st, ep_sk, ep_fu, ep_mk, ep_ac = [], [], [], [], []
        while not done:
            t = env.t
            mask = obs["mask"]
            chosen = 0
            while ptr < len(scheduled) and float(scheduled[ptr]["cast_at_s"]) < t - 2.5:
                ptr += 1
            if ptr < len(scheduled) and float(scheduled[ptr]["cast_at_s"]) <= t + 0.5:
                act = scheduled[ptr]
                ch = str(act.get("target_channel", act.get("target", "party")))
                aidx = idx_by_key.get((act["skill_id"], ch), idx_by_skill.get(act["skill_id"], 0))
                if aidx >= 2 and mask[aidx]:
                    chosen = aidx
                    ptr += 1
            if chosen == 0 and mask[1]:
                imminent_gcd = (
                    ptr < len(scheduled)
                    and scheduled[ptr].get("is_gcd", False)
                    and float(scheduled[ptr]["cast_at_s"]) < t + env.kit.gcd_s - 1e-4
                )
                if not imminent_gcd:
                    chosen = 1

            ep_st.append(obs["state"])
            ep_sk.append(obs["skills"])
            ep_fu.append(obs["future"])
            ep_mk.append(mask)
            ep_ac.append(chosen)
            obs, _, done, info = env.step(chosen, deterministic=is_det)
        survived = info.get("stats", {}).get("survived", 0.0) > 0.5
        if survived:
            bc_survivals += 1
            states.extend(ep_st)
            skills.extend(ep_sk)
            futures.extend(ep_fu)
            masks.extend(ep_mk)
            actions.extend(ep_ac)
        elif best_fallback is None or env.t > best_fallback[0]:
            best_fallback = (env.t, ep_st, ep_sk, ep_fu, ep_mk, ep_ac)

    if bc_survivals == 0 and best_fallback is not None:
        print(f"[BC Collect] WARNING: 0/{n_trajectories} survived ({timeline.encounter_name}); falling back to longest pull ({best_fallback[0]:.1f}s)", flush=True)
        _, states, skills, futures, masks, actions = best_fallback
    else:
        print(
            f"[BC Collect] {timeline.encounter_name} (HP={kit.max_hp_party:.0f}): "
            f"kept_surviving={bc_survivals}/{n_trajectories} total_steps={len(actions)}",
            flush=True,
        )
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(skills, dtype=np.float32),
        np.asarray(futures, dtype=np.float32),
        np.asarray(masks, dtype=np.bool_),
        np.asarray(actions, dtype=np.int64),
    )


def _atomic_torch_save(payload: dict[str, Any], target_path: Path) -> None:
    """Write checkpoint to a sibling temporary file then atomically replace target_path."""
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)


def _evaluate_multi_gear(
    model: SkillAttentionActorCritic,
    kit: PlayerKit,
    timeline: EncounterTimeline,
    device: torch.device,
    gear_conditioned: bool,
    n_eval_pulls: int = 12,
    extra_timelines: list[EncounterTimeline] | None = None,
) -> tuple[dict[str, Any], float]:
    """Evaluate policy equally across every encounter and iLvl tier (Week-1 -> Mid -> BiS)."""
    all_tls = [timeline] + list(extra_timelines or [])
    base_ev = evaluate_policy(model, kit, timeline, device, n_eval_pulls=n_eval_pulls)
    if not gear_conditioned and len(all_tls) == 1:
        return base_ev, base_ev["survival_rate_pct"] * 10.0 + base_ev["mean_reward"]

    # Week-1 low-iLvl slow-GCD & BiS high-iLvl fast-GCD variants
    v_low = kit.with_gear_overrides(
        ilvl=kit.reference_ilvl - 10,
        gcd_s=min(2.50, round(kit.gcd_s + 0.08, 2)),
        piety=(max(440, kit.gear_stats.piety - 300) if kit.role == "healer" else None),
        tenacity=(max(420, kit.gear_stats.tenacity - 600) if kit.role != "healer" else None),
    )
    v_high = kit.with_gear_overrides(
        ilvl=kit.reference_ilvl + 10,
        gcd_s=max(2.30, round(kit.gcd_s - 0.08, 2)),
        piety=(min(1800, kit.gear_stats.piety + 400) if kit.role == "healer" else None),
        tenacity=(min(2200, kit.gear_stats.tenacity + 600) if kit.role != "healer" else None),
    )

    sub_pulls = max(4, n_eval_pulls // 2)
    enc_surv_means: list[float] = []
    enc_ret_means: list[float] = []
    per_enc: dict[str, float] = {}

    for idx, tl in enumerate(all_tls):
        ev_mid = base_ev if idx == 0 else evaluate_policy(model, kit, tl, device, n_eval_pulls=sub_pulls)
        s_vals = [ev_mid["survival_rate_pct"]]
        r_vals = [ev_mid["mean_reward"]]
        if gear_conditioned:
            ev_lo = evaluate_policy(model, v_low, tl, device, n_eval_pulls=sub_pulls)
            ev_hi = evaluate_policy(model, v_high, tl, device, n_eval_pulls=sub_pulls)
            s_vals.extend([ev_lo["survival_rate_pct"], ev_hi["survival_rate_pct"]])
            r_vals.extend([ev_lo["mean_reward"], ev_hi["mean_reward"]])
        m_s = float(np.mean(s_vals))
        m_r = float(np.mean(r_vals))
        enc_surv_means.append(m_s)
        enc_ret_means.append(m_r)
        per_enc[tl.encounter_name] = round(m_s, 1)
        if len(all_tls) > 1:
            print(f"    [Eval] {tl.encounter_name}: surv={s_vals} mean={m_s:.1f}%", flush=True)

    multi_surv = float(np.mean(enc_surv_means))
    multi_ret = float(np.mean(enc_ret_means))
    base_ev["multi_gear_survival_pct"] = round(multi_surv, 2)
    if len(all_tls) > 1:
        base_ev["per_encounter_survival_pct"] = per_enc
    return base_ev, multi_surv * 10.0 + multi_ret


def train_masked_ppo(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    updates: int = 40,
    episodes_per_update: int = 12,
    lr: float = 1e-4,
    hidden_dim: int = 128,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    clip_coef: float = 0.2,
    ent_coef: float = 0.003,
    vf_coef: float = 0.5,
    gear_conditioned: bool = True,
    bc_plan_actions: list[dict[str, Any]] | None = None,
    bc_epochs: int = 30,
    bc_skill_class_weight: float = 15.0,
    bc_kl_coef: float = 0.15,
    seed: int = 42,
    out_dir: str | Path = "outputs/rl_run",
    *,
    extra_timelines: list[EncounterTimeline] | None = None,
    extra_bc_plans: list[list[dict[str, Any]] | None] | None = None,
    bc_bundles: list[tuple[PlayerKit, EncounterTimeline, list[dict[str, Any]]]] | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    """Train gear/iLvl-conditioned SkillAttentionActorCritic with BC warm-start + KL-anchored Masked PPO.

    Supports multi-encounter training (`extra_timelines` + `extra_bc_plans`), explicit multi-iLvl BC
    teacher bundles (`bc_bundles`), and parallel CPU rollout workers (`num_workers > 1`).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gear_range = GearSamplingRange() if gear_conditioned else None
    extras = list(extra_timelines or [])
    extra_bcs = list(extra_bc_plans or [])
    if len(extra_bcs) > len(extras):
        raise ValueError(f"extra_bc_plans ({len(extra_bcs)}) exceeds extra_timelines ({len(extras)})")
    while len(extra_bcs) < len(extras):
        extra_bcs.append(None)
    all_timelines = [timeline] + extras
    all_bc_plans: list[list[dict[str, Any]] | None] = [bc_plan_actions] + extra_bcs
    has_any_bc = bool(bc_bundles) or any(bool(p) for p in all_bc_plans)
    print(
        f"[RL] Device: {device} | GearConditioned={gear_conditioned} | BC={has_any_bc} | "
        f"Workers={num_workers} | Encounters={len(all_timelines)} ({', '.join(t.encounter_name for t in all_timelines)}) | "
        f"Kit: {kit.label}",
        flush=True,
    )

    envs = [FF14HealMitEnv(kit, tl, gear_sampling=gear_range, seed=seed + idx * 97) for idx, tl in enumerate(all_timelines)]
    env = envs[0]
    model = SkillAttentionActorCritic(hidden_dim=hidden_dim).to(device)
    bc_anchor_model: SkillAttentionActorCritic | None = None
    bc_trained = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)

    # Optional class-weighted Behavior Cloning warm-start across all encounters / iLvl teacher bundles
    if has_any_bc and bc_epochs > 0:
        bc_sts, bc_sks, bc_fus, bc_mks, bc_acs = [], [], [], [], []
        if bc_bundles:
            for idx, (b_kit, b_tl, b_acts) in enumerate(bc_bundles):
                if not b_acts:
                    continue
                st_i, sk_i, fu_i, mk_i, ac_i = _collect_bc_dataset(
                    b_kit, b_tl, b_acts, gear_range, n_trajectories=12, seed=seed + idx * 31, fixed_kit=True
                )
                if len(ac_i) > 0:
                    bc_sts.append(st_i)
                    bc_sks.append(sk_i)
                    bc_fus.append(fu_i)
                    bc_mks.append(mk_i)
                    bc_acs.append(ac_i)
        else:
            n_traj_each = max(8, 12 // max(1, sum(1 for p in all_bc_plans if p)))
            for idx, (tl_i, bc_i) in enumerate(zip(all_timelines, all_bc_plans)):
                if not bc_i:
                    continue
                st_i, sk_i, fu_i, mk_i, ac_i = _collect_bc_dataset(
                    kit, tl_i, bc_i, gear_range, n_trajectories=n_traj_each, seed=seed + idx * 31
                )
                if len(ac_i) > 0:
                    bc_sts.append(st_i)
                    bc_sks.append(sk_i)
                    bc_fus.append(fu_i)
                    bc_mks.append(mk_i)
                    bc_acs.append(ac_i)

        if bc_acs:
            bc_st = np.concatenate(bc_sts, axis=0)
            bc_sk = np.concatenate(bc_sks, axis=0)
            bc_fu = np.concatenate(bc_fus, axis=0)
            bc_mk = np.concatenate(bc_mks, axis=0)
            bc_ac = np.concatenate(bc_acs, axis=0)

            t_st = torch.from_numpy(bc_st).to(device)
            t_sk = torch.from_numpy(bc_sk).to(device)
            t_fu = torch.from_numpy(bc_fu).to(device)
            t_mk = torch.from_numpy(bc_mk).to(device)
            t_ac = torch.from_numpy(bc_ac).to(device)
            cls_weights = torch.ones(env.n_actions, dtype=torch.float32, device=device)
            if env.n_actions > 2:
                cls_weights[2:] = bc_skill_class_weight
            ce_loss = nn.CrossEntropyLoss(weight=cls_weights)
            bc_opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            bc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(bc_opt, T_max=max(1, bc_epochs), eta_min=1e-4)
            model.train()
            n_bc = len(bc_ac)
            skill_samples = int((bc_ac >= 2).sum())
            bc_idxs = np.arange(n_bc)
            for ep in range(1, bc_epochs + 1):
                np.random.shuffle(bc_idxs)
                tot_loss = 0.0
                batches = 0
                for start in range(0, n_bc, minibatch_size):
                    mb = bc_idxs[start : start + minibatch_size]
                    logits, _ = model.forward(t_st[mb], t_sk[mb], t_fu[mb], t_mk[mb])
                    loss = ce_loss(logits, t_ac[mb])
                    bc_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    bc_opt.step()
                    tot_loss += float(loss.item())
                    batches += 1
                bc_sched.step()
                if ep == 1 or ep == bc_epochs or ep % 10 == 0:
                    print(
                        f"[BC Epoch {ep:2d}/{bc_epochs}] samples={n_bc} (skills={skill_samples}) "
                        f"weighted_loss={tot_loss / max(1, batches):.4f}",
                        flush=True,
                    )
            bc_trained = True
            if bc_kl_coef > 0.0:
                bc_anchor_model = SkillAttentionActorCritic(hidden_dim=hidden_dim).to(device)
                bc_anchor_model.load_state_dict(model.state_dict())
                bc_anchor_model.eval()
                for p in bc_anchor_model.parameters():
                    p.requires_grad_(False)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    history: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    # Evaluate BC warm-start checkpoint before PPO updates so best_policy.pt never regresses below BC
    if bc_trained:
        ev0, best_score = _evaluate_multi_gear(
            model, kit, timeline, device, gear_conditioned, n_eval_pulls=8, extra_timelines=extras
        )
        print(
            f"[BC  Eval  ] eval_surv={ev0['survival_rate_pct']:5.1f}% multi_gear={ev0.get('multi_gear_survival_pct', ev0['survival_rate_pct']):5.1f}% "
            f"filler={ev0['mean_filler_gcds']:4.1f} gcd_heal={ev0['mean_gcd_heals']:4.2f} oh={ev0['mean_overheal_pct']:4.1f}% "
            f"worst_party={ev0['worst_party_hp_pct']:4.1f}% worst_mt={ev0['worst_mt_hp_pct']:4.1f}% "
            f"enc={ev0.get('per_encounter_survival_pct', {})}",
            flush=True,
        )
        _atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "hidden_dim": hidden_dim,
                "kit_label": kit.label,
                "encounter_name": timeline.encounter_name,
                "trained_encounters": [t.encounter_name for t in all_timelines],
                "eval": ev0,
            },
            out_path / "best_policy.pt",
        )

    pool: ProcessPoolExecutor | None = None
    if num_workers > 1 and episodes_per_update >= 2:
        effective_workers = min(num_workers, episodes_per_update)
        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=ctx,
            initializer=_init_rollout_worker,
            initargs=(
                hidden_dim,
                kit.model_dump(mode="python"),
                [tl.model_dump(mode="python") for tl in all_timelines],
                gear_range.model_dump(mode="python") if gear_range is not None else None,
                seed + 5000,
            ),
        )

    try:
        train_rng = np.random.default_rng(seed + 1337)
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

            if pool is not None:
                state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                n_w = min(num_workers, episodes_per_update)
                base_n = episodes_per_update // n_w
                rem_n = episodes_per_update % n_w
                futs = []
                for wi in range(n_w):
                    cnt = base_n + (1 if wi < rem_n else 0)
                    if cnt > 0:
                        futs.append(pool.submit(_run_rollout_batch, state_cpu, cnt, seed + upd * 1000 + wi * 37))
                for fut in futs:
                    res = fut.result()
                    states_buf.extend(res["states"])
                    skills_buf.extend(res["skills"])
                    futures_buf.extend(res["futures"])
                    masks_buf.extend(res["masks"])
                    actions_buf.extend(res["actions"])
                    logprobs_buf.extend(res["logprobs"])
                    rewards_buf.extend(res["rewards"])
                    values_buf.extend(res["values"])
                    dones_buf.extend(res["dones"])
                    dts_buf.extend(res["dts"])
                    ep_survivals += int(res["ep_survivals"])
                    ep_returns.extend(res["ep_returns"])
            else:
                with torch.no_grad():
                    for _ in range(episodes_per_update):
                        cur_env = envs[int(train_rng.integers(0, len(envs)))]
                        obs = cur_env.reset()
                        done = False
                        ep_r = 0.0
                        info: dict[str, Any] = {}
                        while not done:
                            st = torch.from_numpy(obs["state"]).unsqueeze(0).to(device)
                            sk = torch.from_numpy(obs["skills"]).unsqueeze(0).to(device)
                            fu = torch.from_numpy(obs["future"]).unsqueeze(0).to(device)
                            mk = torch.from_numpy(obs["mask"]).unsqueeze(0).to(device)
                            act, lp, _, val = model.get_action_and_value(st, sk, fu, mk)
                            act_i = int(act.item())
                            next_obs, r, done, info = cur_env.step(act_i)

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

            # Warm-start the Value head on Update 1 after BC (even if bc_kl_coef==0) so an uncalibrated critic never shocks the Actor
            if upd == 1 and bc_trained:
                model.train()
                critic_opt = torch.optim.AdamW(model.value_head.parameters(), lr=1e-3)
                n_steps_w = len(actions_buf)
                idxs_w = np.arange(n_steps_w)
                for _ in range(10):
                    np.random.shuffle(idxs_w)
                    for start in range(0, n_steps_w, minibatch_size):
                        mb = idxs_w[start : start + minibatch_size]
                        _, val_pred = model.forward(b_states[mb], b_skills[mb], b_futures[mb], b_masks[mb])
                        c_loss = 0.5 * ((val_pred - b_ret[mb]) ** 2).mean()
                        critic_opt.zero_grad()
                        c_loss.backward()
                        nn.utils.clip_grad_norm_(model.value_head.parameters(), 1.0)
                        critic_opt.step()
                # Recompute calibrated values in minibatches (avoids GPU OOM on long multi-encounter rollouts)
                cal_vals_list: list[float] = []
                model.eval()
                with torch.no_grad():
                    for start in range(0, n_steps_w, minibatch_size):
                        end = min(n_steps_w, start + minibatch_size)
                        _, v_mb = model.forward(b_states[start:end], b_skills[start:end], b_futures[start:end], b_masks[start:end])
                        cal_vals_list.extend(v_mb.cpu().tolist())
                adv, ret = _compute_gae(rewards_buf, cal_vals_list, dones_buf, dts_buf)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                b_adv = torch.from_numpy(adv).to(device)
                b_ret = torch.from_numpy(ret).to(device)

            # Gentle KL decay spanning exact [1.0 -> 0.25] across updates
            kl_progress = ((upd - 1) / max(1, updates - 1)) if updates > 1 else 0.0
            kl_scale = max(0.25, 1.0 - 0.75 * kl_progress)
            cur_kl_coef = bc_kl_coef * kl_scale

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
                    kl_loss = torch.tensor(0.0, device=device)
                    if bc_anchor_model is not None and cur_kl_coef > 0.0:
                        with torch.no_grad():
                            ref_logits, _ = bc_anchor_model.forward(b_states[mb], b_skills[mb], b_futures[mb], b_masks[mb])
                            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                        cur_logits, _ = model.forward(b_states[mb], b_skills[mb], b_futures[mb], b_masks[mb])
                        cur_log_probs = torch.log_softmax(cur_logits, dim=-1)
                        cur_probs = cur_log_probs.exp()
                        kl_terms = torch.where(
                            b_masks[mb],
                            cur_probs * (cur_log_probs - ref_log_probs),
                            torch.zeros_like(cur_probs),
                        )
                        kl_loss = kl_terms.sum(dim=-1).mean()
                    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy.mean() + cur_kl_coef * kl_loss

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

            if upd == 1 or upd % max(1, updates // 12) == 0 or upd == updates:
                ev, score = _evaluate_multi_gear(
                    model, kit, timeline, device, gear_conditioned, n_eval_pulls=8, extra_timelines=extras
                )
                print(
                    f"[Upd {upd:3d}/{updates}] train_surv={train_surv:5.1f}% ret={mean_ret:6.2f} | "
                    f"eval_surv={ev['survival_rate_pct']:5.1f}% mg={ev.get('multi_gear_survival_pct', ev['survival_rate_pct']):5.1f}% "
                    f"filler={ev['mean_filler_gcds']:4.1f} gcd_heal={ev['mean_gcd_heals']:4.2f} "
                    f"oh={ev['mean_overheal_pct']:4.1f}% worst_party={ev['worst_party_hp_pct']:4.1f}% "
                    f"worst_mt={ev['worst_mt_hp_pct']:4.1f}% ({row['elapsed_s']}s)",
                    flush=True,
                )
                if score > best_score:
                    best_score = score
                    _atomic_torch_save(
                        {
                            "model_state_dict": model.state_dict(),
                            "hidden_dim": hidden_dim,
                            "kit_label": kit.label,
                            "encounter_name": timeline.encounter_name,
                            "trained_encounters": [t.encounter_name for t in all_timelines],
                            "eval": ev,
                        },
                        out_path / "best_policy.pt",
                    )
            else:
                print(
                    f"[Upd {upd:3d}/{updates}] train_surv={train_surv:5.1f}% ret={mean_ret:6.2f} ({row['elapsed_s']}s)",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    final_eval = evaluate_policy(model, kit, timeline, device, n_eval_pulls=24)
    summary = {
        "device": str(device),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "updates": updates,
        "trained_encounters": [t.encounter_name for t in all_timelines],
        "final_eval": final_eval,
        "history": history,
    }
    (out_path / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


@torch.no_grad()
def infer_with_gear(
    checkpoint_path: str | Path,
    kit: PlayerKit,
    timeline: EncounterTimeline,
    gcd_s: float | None = None,
    piety: int | None = None,
    tenacity: int | None = None,
    heal_scale: float | None = None,
    max_hp_party: float | None = None,
    max_hp_tank: float | None = None,
    n_eval_pulls: int = 12,
    guarantee_100pct: bool = True,
    worst_case_roll: float = 1.05,
    *,
    ilvl: int | None = None,
    min_hp_margin_ratio: float = 0.0,
) -> dict[str, Any]:
    """Run a trained gear/iLvl-conditioned checkpoint with custom runtime gear parameters (no retraining).

    When `guarantee_100pct=True`, stress-tests the inferred schedule across all 12 server-tick phases
    under worst-case `+5%` damage roll (`worst_case_roll=1.05`) and adds minimal-DPS-cost safety skills
    if any phase fails, then optimizes Expected DPS while maintaining strict 100.0% survival.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden_dim = int(ckpt.get("hidden_dim", 128))
    model = SkillAttentionActorCritic(hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    custom_kit = kit.with_gear_overrides(
        ilvl=ilvl,
        gcd_s=gcd_s,
        piety=piety,
        tenacity=tenacity,
        heal_scale=heal_scale,
        max_hp_party=max_hp_party,
        max_hp_tank=max_hp_tank,
    )

    ev = evaluate_policy(model, custom_kit, timeline, torch.device("cpu"), n_eval_pulls=n_eval_pulls)
    raw_plan = action_log_to_plan(custom_kit, timeline, ev["deterministic_action_log"], partner_trust=1.0)
    raw_stress = stress_test_plan(custom_kit, timeline, raw_plan, damage_roll=worst_case_roll, n_phases=12)
    ev["raw_worst_case_105_survival_pct"] = raw_stress.survival_rate_pct
    ev["effective_gear"] = {
        "ilvl": custom_kit.gear_stats.item_level,
        "gcd_s": custom_kit.gcd_s,
        "piety": custom_kit.gear_stats.piety,
        "max_hp_party": custom_kit.max_hp_party,
        "max_hp_tank": custom_kit.max_hp_tank,
        "heal_scale": custom_kit.gear_stats.heal_scale_multiplier,
    }

    if guarantee_100pct:
        hardened_plan, hardened_stress, repairs_added = harden_plan_to_100pct(
            custom_kit, timeline, raw_plan, damage_roll=worst_case_roll, n_phases=12
        )
        if hardened_stress.survival_rate_pct >= 100.0:
            hardened_plan, hardened_stress, dps_opt_stats = maximize_dps_at_100pct_survival(
                custom_kit,
                timeline,
                hardened_plan,
                damage_roll=worst_case_roll,
                n_phases=12,
                min_hp_margin_ratio=min_hp_margin_ratio,
            )
        else:
            dps_opt_stats = {"ghosts_removed": 0, "pruned": 0, "swapped": 0, "burst_shifted": 0}
        ev["hardened_repairs_added"] = repairs_added
        ev["dps_opt_stats"] = dps_opt_stats
        ev["verified_worst_case_105"] = hardened_stress.to_dict()
        ev["verified_plan"] = hardened_plan.model_dump(mode="json")
    return ev


def main() -> None:
    parser = argparse.ArgumentParser(description="Train skill-agnostic Masked PPO policy.")
    parser.add_argument("--kit", required=True, help="Path to PlayerKit YAML")
    parser.add_argument("--timeline", required=True, help="Path to primary EncounterTimeline YAML")
    parser.add_argument("--extra-timelines", nargs="*", default=[], help="Optional extra EncounterTimeline YAMLs")
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--episodes-per-update", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel CPU rollout workers")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bc-casts", default=None, help="Optional JSON file of expert casts for primary timeline")
    parser.add_argument("--extra-bc-casts", nargs="*", default=[], help="Optional JSON expert casts matching --extra-timelines")
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--bc-skill-weight", type=float, default=15.0)
    parser.add_argument("--bc-kl-coef", type=float, default=0.05)
    parser.add_argument("--no-gear-conditioned", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/rl_run")
    args = parser.parse_args()

    kit = PlayerKit.from_yaml(args.kit)
    timeline = EncounterTimeline.from_yaml(args.timeline)
    extra_tls = [EncounterTimeline.from_yaml(p) for p in args.extra_timelines]
    bc_actions = None
    if args.bc_casts:
        raw_bc = json.loads(Path(args.bc_casts).read_text(encoding="utf-8"))
        bc_actions = raw_bc["actions"] if isinstance(raw_bc, dict) and "actions" in raw_bc else raw_bc
    extra_bc: list[list[dict[str, Any]] | None] = []
    for p in args.extra_bc_casts:
        if p and Path(p).exists():
            rb = json.loads(Path(p).read_text(encoding="utf-8"))
            extra_bc.append(rb["actions"] if isinstance(rb, dict) and "actions" in rb else rb)
        else:
            extra_bc.append(None)
    while len(extra_bc) < len(extra_tls):
        extra_bc.append(None)
    train_masked_ppo(
        kit=kit,
        timeline=timeline,
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        gear_conditioned=not args.no_gear_conditioned,
        bc_plan_actions=bc_actions,
        bc_epochs=args.bc_epochs,
        bc_skill_class_weight=args.bc_skill_weight,
        bc_kl_coef=args.bc_kl_coef,
        extra_timelines=extra_tls or None,
        extra_bc_plans=extra_bc or None,
        num_workers=args.num_workers,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

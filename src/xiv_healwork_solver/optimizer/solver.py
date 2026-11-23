"""Skill-agnostic CP-SAT solver for FF14 mitigation and healing timelines.

Aligned with CombatEngine:
- Tracks continuous per-channel HP carryover across ALL affected channels (party / mt / st)
- Synthesizes AA planning checkpoints from `timeline.auto_attacks` so heavy AA phases
  (e.g. tank swaps / Ultimates) automatically receive mitigation & shields
- Enforces shield `shield_stack_group` mutual exclusion and cast-completion deadlines
- Credits Lucid Dreaming MP ticks progressively over time (never instantaneously at cast)
- Distinguishes pre-hit heals (capped at max HP before hit) from post-hit recovery heals
  and paces DoT damage by actual ticks landing before the next event
- Always runs chronological `replay_plan` verification through CombatEngine
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from xiv_healwork_solver.models import (
    DamageEvent,
    EncounterTimeline,
    EventEvaluation,
    LiveFightState,
    ObjectivePreset,
    OptimizationPlan,
    PartnerActionProb,
    PlannedAction,
    PlayerKit,
    SkillDef,
)
from xiv_healwork_solver.sim.engine import replay_plan

LOG_SCALE = 10_000
RATIO_SCALE = 10_000
RES_SCALE = 100
TANGENT_POINTS = (0.0, 0.12, 0.24, 0.38, 0.54, 0.72, 0.92)


@dataclass(frozen=True)
class PresetWeights:
    partner_trust: float
    dps_weight: int
    survival_slack_weight: int
    recovery_slack_weight: int
    reserve_breach_weight: int
    weave_overflow_weight: int
    movement_cast_weight: int
    overlap_weight: int
    extra_margin_bonus: int


PRESETS: dict[ObjectivePreset, PresetWeights] = {
    "balanced": PresetWeights(
        partner_trust=0.70,
        dps_weight=400,
        survival_slack_weight=25_000,
        recovery_slack_weight=40,
        reserve_breach_weight=2_500,
        weave_overflow_weight=1_200,
        movement_cast_weight=1_500,
        overlap_weight=350,
        extra_margin_bonus=2,
    ),
    "dps_greedy": PresetWeights(
        partner_trust=0.85,
        dps_weight=1_200,
        survival_slack_weight=25_000,
        recovery_slack_weight=25,
        reserve_breach_weight=2_000,
        weave_overflow_weight=1_800,
        movement_cast_weight=2_000,
        overlap_weight=500,
        extra_margin_bonus=0,
    ),
    "safe": PresetWeights(
        partner_trust=0.45,
        dps_weight=220,
        survival_slack_weight=35_000,
        recovery_slack_weight=90,
        reserve_breach_weight=3_500,
        weave_overflow_weight=900,
        movement_cast_weight=1_200,
        overlap_weight=300,
        extra_margin_bonus=6,
    ),
    "solo_carry": PresetWeights(
        partner_trust=0.0,
        dps_weight=180,
        survival_slack_weight=40_000,
        recovery_slack_weight=100,
        reserve_breach_weight=3_000,
        weave_overflow_weight=800,
        movement_cast_weight=1_000,
        overlap_weight=100,
        extra_margin_bonus=5,
    ),
}


def _neg_ln_one_minus(m: float) -> float:
    m_clamped = max(0.0, min(0.95, m))
    return -math.log(1.0 - m_clamped)


def _downtime_overlap(timeline: EncounterTimeline, t0: float, t1: float) -> float:
    total = 0.0
    for dw in timeline.downtime_windows:
        if not dw.boss_targetable:
            lo = max(t0, dw.start_s)
            hi = min(t1, dw.end_s)
            if hi > lo:
                total += hi - lo
    return total


def _is_post_hit_skill(sk: SkillDef) -> bool:
    """True for pure recovery / cleanse / MP skills that should fire AFTER damage lands."""
    has_pre_mit = (
        sk.mitigation_all > 0.0
        or sk.mitigation_physical > 0.0
        or sk.mitigation_magical > 0.0
        or sk.shield_value > 0.0
    )
    return not has_pre_mit


def _partner_event_summary(
    partner_actions: list[PartnerActionProb],
    damage_type: str,
    partner_trust: float,
) -> tuple[float, float, float, set[str]]:
    log_mit = 0.0
    shields = 0.0
    heals = 0.0
    active_groups: set[str] = set()
    used_groups_mit: dict[str, float] = {}

    for pa in partner_actions:
        eff_prob = pa.probability * partner_trust
        if eff_prob <= 0.0:
            continue
        mit = pa.effective_mitigation(damage_type)  # type: ignore[arg-type]
        if pa.stack_group:
            if pa.probability >= 0.5 and partner_trust > 0.0:
                active_groups.add(pa.stack_group)
            prev = used_groups_mit.get(pa.stack_group, 0.0)
            used_groups_mit[pa.stack_group] = max(prev, mit * eff_prob)
        elif mit > 0.0:
            log_mit += _neg_ln_one_minus(mit * eff_prob)
        shields += pa.shield_value * eff_prob
        heals += pa.heal_value * eff_prob

    for grp_mit in used_groups_mit.values():
        if grp_mit > 0.0:
            log_mit += _neg_ln_one_minus(grp_mit)

    return log_mit, shields, heals, active_groups


def _event_channels(ev: DamageEvent) -> tuple[str, ...]:
    if ev.target_scope == "party":
        return ("party", "mt", "st")
    if ev.target_scope == "both_tanks":
        return ("mt", "st")
    return (ev.target_scope,)


def _skill_target_channel(sk: SkillDef, ev: DamageEvent, kit: PlayerKit) -> str:
    if sk.target_scope == "boss_debuff":
        return "boss"
    if sk.target_scope == "party":
        return "party"
    if sk.target_scope == "self":
        return "mt" if (kit.role == "tank" and kit.player_is_mt) else ("st" if kit.role == "tank" else "party")
    if sk.target_scope in ("mt", "st"):
        return sk.target_scope
    if ev.target_scope in ("mt", "st"):
        return ev.target_scope
    return "mt"


def _skill_applies_to_channel(sk: SkillDef, ev: DamageEvent, ch: str, kit: PlayerKit) -> bool:
    tch = _skill_target_channel(sk, ev, kit)
    return tch in ("party", "boss") or tch == ch


def _build_planning_events(
    timeline: EncounterTimeline,
    start_time_s: float,
    horizon_s: float | None,
) -> list[DamageEvent]:
    """Combine explicit DamageEvents with synthesized AutoAttackSegment checkpoints (15s windows)."""
    events = [e for e in timeline.events if e.time_s >= start_time_s]
    aa_idx = 0
    for seg in timeline.auto_attacks:
        # Group consecutive AAs into <=15s windows so heavy AA phases can be planned & shielded
        aa_t = seg.start_s
        while aa_t < start_time_s - 1e-6:
            aa_t += seg.interval_s
        window_hits: list[float] = []
        while aa_t <= seg.end_s + 1e-6 and aa_t <= timeline.duration_s + 1e-6:
            if timeline.is_boss_targetable(aa_t):
                if window_hits and (aa_t - window_hits[0] >= 15.0 - 1e-6):
                    aa_idx += 1
                    events.append(
                        DamageEvent(
                            id=f"aa_chk_{aa_idx}_{seg.target}_{int(window_hits[0])}",
                            time_s=round(window_hits[0], 2),
                            name=f"Auto-Attacks ({seg.target.upper()} x{len(window_hits)})",
                            damage_type=seg.damage_type,
                            target_scope=seg.target,
                            raw_damage=round(seg.raw_damage_per_hit * len(window_hits), 1),
                            hit_count=len(window_hits),
                            hit_interval_s=seg.interval_s,
                            min_survival_hp_ratio=0.15,
                            recovery_target_ratio=0.75,
                        )
                    )
                    window_hits = []
                window_hits.append(aa_t)
            aa_t += seg.interval_s
        if window_hits:
            aa_idx += 1
            events.append(
                DamageEvent(
                    id=f"aa_chk_{aa_idx}_{seg.target}_{int(window_hits[0])}",
                    time_s=round(window_hits[0], 2),
                    name=f"Auto-Attacks ({seg.target.upper()} x{len(window_hits)})",
                    damage_type=seg.damage_type,
                    target_scope=seg.target,
                    raw_damage=round(seg.raw_damage_per_hit * len(window_hits), 1),
                    hit_count=len(window_hits),
                    hit_interval_s=seg.interval_s,
                    min_survival_hp_ratio=0.15,
                    recovery_target_ratio=0.75,
                )
            )

    if horizon_s is not None:
        events = [e for e in events if e.time_s <= start_time_s + horizon_s]
    events.sort(key=lambda e: e.time_s)
    return events


def solve_mitigation_plan(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    preset: ObjectivePreset = "balanced",
    partner_trust_override: float | None = None,
    time_limit_s: float = 10.0,
    live_state: LiveFightState | None = None,
    verify_with_replay: bool = True,
) -> OptimizationPlan:
    """Solve optimal mitigation & healing assignments with multi-channel HP carryover and replay verification."""
    t0 = time.perf_counter()
    weights = PRESETS[preset]
    partner_trust = (
        weights.partner_trust
        if partner_trust_override is None
        else max(0.0, min(1.0, partner_trust_override))
    )

    start_time_s = live_state.current_time_s if live_state is not None else 0.0
    horizon_s = live_state.horizon_s if live_state is not None else None
    events = _build_planning_events(timeline, start_time_s, horizon_s)

    skills = kit.skills
    n_events = len(events)
    n_skills = len(skills)
    heal_scale = kit.gear_stats.heal_scale_multiplier

    if n_events == 0:
        plan = OptimizationPlan(
            encounter_name=timeline.encounter_name,
            kit_label=kit.label,
            preset=preset,
            partner_trust=partner_trust,
            status="OPTIMAL",
            solve_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            objective_value=0.0,
            actions=[],
            evaluations=[],
        )
        if verify_with_replay:
            plan.replay = replay_plan(kit, timeline, plan, start_time_s=start_time_s)
        return plan

    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_e{e}_s{s}") for s in range(n_skills)] for e in range(n_events)]
    active = [[model.NewBoolVar(f"act_e{e}_s{s}") for s in range(n_skills)] for e in range(n_events)]

    # Precompute planned cast timestamps for every (e, s)
    cast_times: list[list[float]] = []
    for e, ev in enumerate(events):
        snap_t = ev.effective_snapshot_time()
        row: list[float] = []
        for s, sk in enumerate(skills):
            if _is_post_hit_skill(sk):
                ct = ev.time_s + 0.5
            else:
                lead = max(sk.lead_time_s, sk.cast_time_s)
                ct = max(start_time_s, snap_t - lead)
            row.append(ct)
        cast_times.append(row)

    # 1. Eligibility, cast-completion deadline, downtime boss lockouts, live cooldowns, and carryover
    for e, ev in enumerate(events):
        snap_t = ev.effective_snapshot_time()
        for s, sk in enumerate(skills):
            eligible = sk.applies_to_event(ev.target_scope, kit.role, kit.player_is_mt)
            cast_t = cast_times[e][s]
            if sk.needs_boss_target() and not timeline.is_boss_targetable(cast_t):
                eligible = False
            if not timeline.can_party_act(cast_t):
                eligible = False
            # Pre-hit mitigation/shield must finish casting BEFORE snapshot!
            if not _is_post_hit_skill(sk) and cast_t + sk.cast_time_s > snap_t + 1e-6:
                eligible = False
            if not eligible:
                model.Add(x[e][s] == 0)

            if live_state is not None and sk.id in live_state.cooldowns_remaining:
                ready_at = start_time_s + max(0.0, live_state.cooldowns_remaining[sk.id])
                if cast_t < ready_at - 1e-6:
                    model.Add(x[e][s] == 0)

            pre_active = False
            if live_state is not None and sk.id in live_state.active_buffs_remaining:
                if snap_t <= start_time_s + live_state.active_buffs_remaining[sk.id] + 1e-6:
                    pre_active = True

            if pre_active and eligible:
                model.Add(active[e][s] == 1)
                continue

            cover_sources = [x[e][s]]
            if sk.duration_s > sk.lead_time_s and not _is_post_hit_skill(sk):
                usable_span = sk.duration_s - max(sk.lead_time_s, sk.cast_time_s)
                for prev in range(e):
                    dt = snap_t - events[prev].effective_snapshot_time()
                    blocked_by_dt = sk.needs_boss_target() and _downtime_overlap(
                        timeline, events[prev].effective_snapshot_time(), snap_t
                    ) > 0
                    if (
                        not blocked_by_dt
                        and 0.0 < dt <= usable_span
                        and sk.applies_to_event(events[prev].target_scope, kit.role, kit.player_is_mt)
                    ):
                        if sk.target_scope in ("party", "boss_debuff") or events[prev].target_scope == ev.target_scope:
                            cover_sources.append(x[prev][s])

            if eligible:
                model.Add(active[e][s] <= sum(cover_sources))
                for src in cover_sources:
                    model.Add(active[e][s] >= src)
            else:
                model.Add(active[e][s] == 0)

    # 2. Cooldown & charge rolling-window constraints
    for s, sk in enumerate(skills):
        if sk.cooldown_s <= 0:
            continue
        for i in range(n_events):
            window = [
                j for j in range(i, n_events)
                if cast_times[j][s] - cast_times[i][s] < sk.cooldown_s - 1e-6
            ]
            if len(window) > sk.max_charges:
                model.Add(sum(x[j][s] for j in window) <= sk.max_charges)

        if sk.max_charges > 1:
            for i in range(n_events):
                elapsed = max(0.0, cast_times[i][s] - start_time_s)
                max_possible = sk.max_charges + int(elapsed // sk.cooldown_s)
                model.Add(sum(x[j][s] for j in range(i + 1)) <= max_possible)

    # 3. Stack-group mutual exclusion (both mitigation stack_group AND shield_stack_group!)
    mit_groups: dict[str, list[int]] = {}
    sh_groups: dict[str, list[int]] = {}
    for s, sk in enumerate(skills):
        if sk.stack_group:
            mit_groups.setdefault(sk.stack_group, []).append(s)
        if sk.shield_stack_group:
            sh_groups.setdefault(sk.shield_stack_group, []).append(s)
    for idxs in mit_groups.values():
        if len(idxs) > 1:
            for e in range(n_events):
                model.Add(sum(active[e][s] for s in idxs) <= 1)
    for idxs in sh_groups.values():
        if len(idxs) > 1:
            for e in range(n_events):
                model.Add(sum(x[e][s] for s in idxs) <= 1)

    objective_terms: list[cp_model.LinearExpr] = []

    # 4. Generic resource pool balance tracking (progressive Lucid ticks; no instant future credit!)
    resolved_pools = kit.resolved_resources()
    pool_after_vars: dict[str, list[cp_model.IntVar]] = {}
    filler_drain_per_s = kit.gear_stats.filler_gcd_mp_cost / kit.gcd_s if kit.gcd_s > 0 else 0.0

    for pool in resolved_pools:
        init_val = (
            live_state.current_resources[pool.name]
            if (live_state is not None and pool.name in live_state.current_resources)
            else pool.initial
        )
        init_i = int(round(max(0.0, min(pool.maximum, init_val)) * RES_SCALE))
        max_i = int(round(pool.maximum * RES_SCALE))
        reserve_i = int(round(min(pool.maximum, pool.min_reserve) * RES_SCALE))
        unit_scale = max(1, max_i // 10)

        prev_bal: cp_model.IntVar | None = None
        after_list: list[cp_model.IntVar] = []

        for e, ev in enumerate(events):
            prev_t = start_time_s if e == 0 else events[e - 1].time_s
            dt = max(0.0, ev.time_s - prev_t)
            dt_downtime = _downtime_overlap(timeline, prev_t, ev.time_s) if pool.name.lower() == "mp" else 0.0
            extra_saved_mp = filler_drain_per_s * dt_downtime
            delta_i = int(round((pool.regen_per_sec * dt + extra_saved_mp) * RES_SCALE))

            # Progressive MP ticks from earlier explicit MP-regen casts (e.g. Lucid Dreaming) landing in (prev_t, ev.time_s]
            lucid_tick_terms = []
            if pool.name.lower() == "mp":
                for prev_e in range(e):
                    for s, sk in enumerate(skills):
                        if sk.mp_regen_per_tick > 0 and sk.duration_s >= 3.0:
                            c_start = cast_times[prev_e][s]
                            c_end = c_start + sk.duration_s
                            lo = max(prev_t, c_start)
                            hi = min(ev.time_s, c_end)
                            if hi > lo:
                                ticks_in_win = int((hi - c_start) // 3.0) - int((lo - c_start) // 3.0)
                                if ticks_in_win > 0:
                                    lucid_tick_terms.append(
                                        int(round(sk.mp_regen_per_tick * ticks_in_win * RES_SCALE)) * x[prev_e][s]
                                    )

            bal_before = model.NewIntVar(0, max_i, f"res_{pool.name}_before_{e}")
            if e == 0:
                model.Add(bal_before == max(0, min(max_i, init_i + delta_i)))
            else:
                max_lucid_add = max_i
                low_bound = min(0, delta_i)
                high_bound = max(max_i, max_i + delta_i + max_lucid_add)
                raw_next = model.NewIntVar(low_bound, high_bound, f"res_{pool.name}_raw_{e}")
                model.Add(raw_next == prev_bal + delta_i + sum(lucid_tick_terms))  # type: ignore[operator]
                floored = model.NewIntVar(0, high_bound, f"res_{pool.name}_flr_{e}")
                model.AddMaxEquality(floored, [raw_next, 0])
                model.AddMinEquality(bal_before, [floored, max_i])

            spend_terms = []
            grant_terms = []
            for s, sk in enumerate(skills):
                if pool.name in sk.resource_costs:
                    adj_cost = sk.resource_costs[pool.name]
                    if pool.name.lower() == "mp" and sk.is_gcd and kit.gear_stats.filler_gcd_mp_cost > 0:
                        if timeline.is_boss_targetable(cast_times[e][s]):
                            adj_cost = max(0.0, adj_cost - kit.gear_stats.filler_gcd_mp_cost)
                    spend_terms.append(int(round(adj_cost * RES_SCALE)) * x[e][s])
                if pool.name in sk.resource_grants:
                    grant_terms.append(int(round(sk.resource_grants[pool.name] * RES_SCALE)) * x[e][s])

            if spend_terms:
                model.Add(bal_before - sum(spend_terms) >= 0)

            bal_after = model.NewIntVar(0, max_i, f"res_{pool.name}_after_{e}")
            net_after = model.NewIntVar(0, max_i * 2, f"res_{pool.name}_net_{e}")
            model.Add(net_after == bal_before - sum(spend_terms) + sum(grant_terms))
            model.AddMinEquality(bal_after, [net_after, max_i])

            if reserve_i > 0:
                breach = model.NewIntVar(0, reserve_i, f"res_{pool.name}_breach_{e}")
                model.Add(bal_after + breach >= reserve_i)
                breach_units = model.NewIntVar(0, 10, f"res_{pool.name}_bu_{e}")
                model.Add(unit_scale * breach_units >= breach)
                objective_terms.append(weights.reserve_breach_weight * breach_units)

            prev_bal = bal_after
            after_list.append(bal_after)

        pool_after_vars[pool.name] = after_list

    partner_by_event: dict[str, list[PartnerActionProb]] = {}
    for pa in timeline.partner_profile:
        partner_by_event.setdefault(pa.event_id, []).append(pa)

    # 5. Per-channel HP carryover across ALL affected channels (party, mt, st)
    max_hp_by_ch = {
        "party": int(round(kit.max_hp_party)),
        "mt": int(round(kit.max_hp_tank)),
        "st": int(round(kit.max_hp_tank)),
    }
    init_hp_by_ch = {
        "party": int(round(max_hp_by_ch["party"] * (live_state.current_party_hp_ratio if live_state else 1.0))),
        "mt": int(round(max_hp_by_ch["mt"] * ((live_state.current_mt_hp_ratio if (live_state and live_state.current_mt_hp_ratio is not None) else (live_state.current_party_hp_ratio if live_state else 1.0))))),
        "st": int(round(max_hp_by_ch["st"] * ((live_state.current_st_hp_ratio if (live_state and live_state.current_st_hp_ratio is not None) else (live_state.current_party_hp_ratio if live_state else 1.0))))),
    }

    prev_hp_var: dict[str, cp_model.IntVar | int] = dict(init_hp_by_ch)
    primary_hp_after_hit_vars: list[cp_model.IntVar] = []
    event_weave_budgets: list[int] = []

    for e, ev in enumerate(events):
        channels = _event_channels(ev)
        primary_ch = channels[0]
        raw_dmg = int(round(ev.raw_damage))
        dot_dmg = int(round(ev.dot_total_damage))
        dt_to_next = (events[e + 1].time_s - ev.time_s) if e + 1 < n_events else ev.dot_duration_s
        # Fraction of DoT ticks that actually land before the next event
        dot_ticks_total = max(1, int(round(ev.dot_duration_s / ev.dot_tick_interval_s)))
        dot_ticks_before_next = min(dot_ticks_total, int(max(0.0, dt_to_next) // ev.dot_tick_interval_s))
        dot_frac_before_next = (dot_ticks_before_next / dot_ticks_total) if dot_dmg > 0 else 0.0

        p_log_mit, p_shields, p_heals, p_groups = _partner_event_summary(
            partner_by_event.get(ev.id, []), ev.damage_type, partner_trust
        )
        p_log_i = int(round(p_log_mit * LOG_SCALE))

        for ch in channels:
            max_hp = max_hp_by_ch[ch]
            ch_scope = "mt" if ch == "mt" else ("st" if ch == "st" else "party")
            passive_mit = kit.passive_self_mitigation_for(ch_scope)  # type: ignore[arg-type]
            passive_log_i = int(round(_neg_ln_one_minus(passive_mit) * LOG_SCALE)) if passive_mit > 0 else 0

            self_log_terms = []
            for s, sk in enumerate(skills):
                if not _skill_applies_to_channel(sk, ev, ch, kit):
                    continue
                m = sk.effective_mitigation(ev.damage_type)
                if m > 0.0:
                    self_log_terms.append(int(round(_neg_ln_one_minus(m) * LOG_SCALE)) * active[e][s])

            total_log = model.NewIntVar(0, 3 * LOG_SCALE, f"logmit_{e}_{ch}")
            model.Add(total_log == passive_log_i + p_log_i + sum(self_log_terms))

            rem_ratio = model.NewIntVar(500, RATIO_SCALE, f"rem_ratio_{e}_{ch}")
            for lk in TANGENT_POINTS:
                fk = math.exp(-lk)
                intercept = int(round(RATIO_SCALE * fk * (1.0 + lk)))
                slope_num = int(round(fk * 1000))
                model.Add(1000 * rem_ratio >= 1000 * intercept - slope_num * total_log)

            if ev.mechanic_effect == "set_hp_to_1":
                eff_dmg = model.NewIntVar(0, max_hp, f"eff_dmg_{e}_{ch}")
                model.Add(eff_dmg == max_hp - 1)
            elif ev.mechanic_effect == "hp_percent_damage":
                pct_i = int(round(max_hp * ev.hp_percent_fraction))
                eff_dmg = model.NewIntVar(pct_i, pct_i, f"eff_dmg_{e}_{ch}")
            else:
                eff_dmg = model.NewIntVar(0, raw_dmg, f"eff_dmg_{e}_{ch}")
                model.Add(RATIO_SCALE * eff_dmg >= raw_dmg * rem_ratio)

            # Only charge DoT ticks that actually land before next event!
            dot_part_raw = int(round(dot_dmg * dot_frac_before_next))
            if dot_part_raw > 0:
                eff_dot = model.NewIntVar(0, dot_part_raw, f"eff_dot_{e}_{ch}")
                if ev.dot_snapshot_mitigation:
                    model.Add(RATIO_SCALE * eff_dot >= dot_part_raw * rem_ratio)
                else:
                    model.Add(eff_dot == dot_part_raw)
            else:
                eff_dot = model.NewIntVar(0, 0, f"eff_dot_{e}_{ch}")

            # Separate pre-hit heals (capped at max_hp BEFORE damage!) from post-hit heals & HoT ticks
            pre_heal_terms = []
            post_heal_terms = []
            shield_terms = []
            for s, sk in enumerate(skills):
                if not _skill_applies_to_channel(sk, ev, ch, kit):
                    continue
                if ev.mechanic_effect != "set_hp_to_1" and sk.shield_value > 0:
                    shield_terms.append(int(round(sk.shield_value * heal_scale)) * x[e][s])
                inst = sk.heal_instant * heal_scale
                if inst > 0:
                    if _is_post_hit_skill(sk):
                        post_heal_terms.append(int(round(inst)) * x[e][s])
                    else:
                        pre_heal_terms.append(int(round(inst)) * x[e][s])
                if sk.heal_hot_total > 0 and sk.duration_s > 0:
                    hot_ticks_tot = max(1, int(round(sk.duration_s / 3.0)))
                    hot_ticks_land = min(hot_ticks_tot, int(max(0.0, dt_to_next) // 3.0))
                    if hot_ticks_land > 0:
                        post_heal_terms.append(
                            int(round((sk.heal_hot_total * heal_scale) * (hot_ticks_land / hot_ticks_tot))) * x[e][s]
                        )

            total_shield = (0 if ev.mechanic_effect == "set_hp_to_1" else int(round(p_shields))) + sum(shield_terms)

            # Pre-hit HP (capped at max_hp so pre-hit heals at full HP do NOT act as post-hit recovery!)
            hp_pre_raw = model.NewIntVar(0, max_hp * 3, f"hp_pre_raw_{e}_{ch}")
            model.Add(hp_pre_raw == prev_hp_var[ch] + sum(pre_heal_terms))
            hp_before = model.NewIntVar(0, max_hp, f"hp_before_{e}_{ch}")
            model.AddMinEquality(hp_before, [hp_pre_raw, max_hp])

            hp_after_hit = model.NewIntVar(-raw_dmg * 2, max_hp * 2, f"hp_hit_{e}_{ch}")
            if ev.mechanic_effect == "set_hp_to_1":
                model.Add(hp_after_hit == 1)
            else:
                net_dmg = model.NewIntVar(0, raw_dmg + max_hp, f"net_dmg_{e}_{ch}")
                model.Add(net_dmg >= eff_dmg - total_shield)
                model.Add(hp_after_hit == hp_before - net_dmg)

            if ch == primary_ch:
                primary_hp_after_hit_vars.append(hp_after_hit)

            min_surv_hp = 1 if ev.mechanic_effect == "set_hp_to_1" else int(round(max_hp * ev.min_survival_hp_ratio))
            surv_slack = model.NewIntVar(0, raw_dmg * 2 + max_hp, f"surv_slack_{e}_{ch}")
            model.Add(hp_after_hit + surv_slack >= min_surv_hp)
            objective_terms.append(weights.survival_slack_weight * surv_slack)

            # Post-hit recovery & DoT carried to next event
            total_post_heal = int(round(p_heals)) + sum(post_heal_terms)
            hp_after_rec_raw = model.NewIntVar(-raw_dmg * 2, max_hp * 3, f"hp_rec_raw_{e}_{ch}")
            model.Add(hp_after_rec_raw == hp_after_hit - eff_dot + total_post_heal)
            hp_after_rec_lo = model.NewIntVar(1, max_hp * 3, f"hp_rec_lo_{e}_{ch}")
            model.AddMaxEquality(hp_after_rec_lo, [hp_after_rec_raw, 1])
            hp_after_rec = model.NewIntVar(1, max_hp, f"hp_rec_{e}_{ch}")
            model.AddMinEquality(hp_after_rec, [hp_after_rec_lo, max_hp])

            if ev.doom_full_heal_duration_s is not None:
                doom_slack = model.NewIntVar(0, max_hp, f"doom_slack_{e}_{ch}")
                model.Add(hp_after_rec + doom_slack >= max_hp)
                objective_terms.append(weights.survival_slack_weight * doom_slack)
            elif kit.role == "healer" and ch == primary_ch:
                target_rec_hp = int(round(max_hp * ev.recovery_target_ratio))
                rec_slack = model.NewIntVar(0, max_hp, f"rec_slack_{e}_{ch}")
                model.Add(hp_after_rec + rec_slack >= target_rec_hp)
                rec_slack_k = model.NewIntVar(0, max_hp // 1000 + 1, f"rec_slack_k_{e}_{ch}")
                model.Add(1000 * rec_slack_k >= rec_slack)
                objective_terms.append(weights.recovery_slack_weight * rec_slack_k)

            prev_hp_var[ch] = hp_after_rec

            if ch == primary_ch and weights.extra_margin_bonus > 0 and ev.mechanic_effect == "normal":
                eff_dmg_k = model.NewIntVar(0, raw_dmg // 1000 + 1, f"eff_dmg_k_{e}")
                model.Add(1000 * eff_dmg_k >= eff_dmg)
                objective_terms.append(weights.extra_margin_bonus * eff_dmg_k)

        # Esuna cleanse requirement
        if ev.esuna_required_count > 0:
            cleanse_terms = [x[e][s] for s, sk in enumerate(skills) if sk.cleanses_debuff]
            esuna_slack = model.NewIntVar(0, ev.esuna_required_count, f"esuna_slack_{e}")
            model.Add(sum(cleanse_terms) + esuna_slack >= ev.esuna_required_count)
            objective_terms.append(weights.survival_slack_weight * 100 * esuna_slack)

        # GCD / Weave slot budget
        prep_gcds = max(1, int(round(5.0 / kit.gcd_s)))
        per_gcd_slots = kit.weave_budget_at(ev.time_s)
        weave_budget = per_gcd_slots * prep_gcds
        event_weave_budgets.append(weave_budget)

        weave_used = sum(sk.weave_cost * x[e][s] for s, sk in enumerate(skills))
        weave_overflow = model.NewIntVar(0, 20, f"weave_ov_{e}")
        model.Add(weave_used <= weave_budget + weave_overflow)
        objective_terms.append(weights.weave_overflow_weight * weave_overflow)

        for s, sk in enumerate(skills):
            cast_t = cast_times[e][s]
            dw = timeline.downtime_at(cast_t)
            is_free_downtime = (
                dw is not None
                and not dw.boss_targetable
                and sk.is_gcd
                and cast_t + max(kit.gcd_s, sk.cast_time_s) <= dw.end_s + 1e-6
            )
            eff_dps_cost = 0.0 if is_free_downtime else sk.dps_cost
            dps_pen = int(round(eff_dps_cost * weights.dps_weight))
            if dps_pen > 0:
                objective_terms.append(dps_pen * x[e][s])
            if ev.requires_movement and sk.cast_time_s > 0.0:
                objective_terms.append(weights.movement_cast_weight * x[e][s])
            if sk.stack_group and sk.stack_group in p_groups:
                objective_terms.append(weights.overlap_weight * active[e][s])

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 4
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code)
    solve_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    actions: list[PlannedAction] = []
    evaluations: list[EventEvaluation] = []

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for e, ev in enumerate(events):
            used_skills: list[str] = []
            self_rem = 1.0
            self_shield = 0.0
            self_heal = 0.0
            weave_used_val = 0

            for s, sk in enumerate(skills):
                if solver.Value(x[e][s]) == 1:
                    used_skills.append(sk.name)
                    weave_used_val += sk.weave_cost
                    self_shield += sk.shield_value * heal_scale
                    self_heal += (sk.heal_instant + sk.heal_hot_total) * heal_scale
                    actions.append(
                        PlannedAction(
                            event_id=ev.id,
                            event_name=ev.name,
                            event_time_s=ev.time_s,
                            cast_at_s=round(cast_times[e][s], 2),
                            skill_id=sk.id,
                            skill_name=sk.name,
                            action_id=sk.action_id,
                            target_scope=sk.target_scope,
                            target_channel=_skill_target_channel(sk, ev, kit),  # type: ignore[arg-type]
                            is_gcd=sk.is_gcd,
                            weave_cost=sk.weave_cost,
                        )
                    )
                if solver.Value(active[e][s]) == 1:
                    self_rem *= 1.0 - sk.effective_mitigation(ev.damage_type)

            passive_mit = kit.passive_self_mitigation_for(ev.target_scope)
            passive_rem = 1.0 - passive_mit
            p_log_mit, p_shields, p_heals, _ = _partner_event_summary(
                partner_by_event.get(ev.id, []), ev.damage_type, partner_trust
            )
            partner_rem = math.exp(-p_log_mit)
            total_rem = passive_rem * self_rem * partner_rem
            max_hp_val = kit.max_hp_tank if ev.target_scope in ("mt", "st", "both_tanks") else kit.max_hp_party

            if ev.mechanic_effect == "set_hp_to_1":
                eff_dmg_val = max_hp_val - 1.0
            elif ev.mechanic_effect == "hp_percent_damage":
                eff_dmg_val = max_hp_val * ev.hp_percent_fraction
            else:
                eff_dmg_val = ev.raw_damage * total_rem

            total_shield_val = 0.0 if ev.mechanic_effect == "set_hp_to_1" else (self_shield + p_shields)
            total_heal_val = self_heal + p_heals
            net_hp = float(solver.Value(primary_hp_after_hit_vars[e]))
            margin_pct = (net_hp / max_hp_val) * 100.0

            res_snapshot = {
                pname: round(solver.Value(pvars[e]) / RES_SCALE, 1)
                for pname, pvars in pool_after_vars.items()
            }

            evaluations.append(
                EventEvaluation(
                    event_id=ev.id,
                    time_s=ev.time_s,
                    name=ev.name,
                    target_scope=ev.target_scope,
                    raw_damage=ev.raw_damage,
                    passive_gear_mitigation_pct=round(passive_mit * 100.0, 2),
                    self_mitigation_pct=round((1.0 - self_rem) * 100.0, 2),
                    partner_expected_mitigation_pct=round((1.0 - partner_rem) * 100.0, 2),
                    total_mitigation_pct=round((1.0 - total_rem) * 100.0, 2),
                    effective_damage=round(eff_dmg_val, 1),
                    total_shield=round(total_shield_val, 1),
                    total_heal=round(total_heal_val, 1),
                    net_hp_after_hit=round(net_hp, 1),
                    max_hp=max_hp_val,
                    survival_margin_pct=round(margin_pct, 2),
                    used_skills=used_skills,
                    weave_used=weave_used_val,
                    weave_budget=event_weave_budgets[e],
                    resource_after=res_snapshot,
                )
            )

    plan = OptimizationPlan(
        encounter_name=timeline.encounter_name,
        kit_label=kit.label,
        preset=preset,
        partner_trust=partner_trust,
        status=status_name,
        solve_time_ms=solve_ms,
        objective_value=float(solver.ObjectiveValue()) if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else -1.0,
        actions=actions,
        evaluations=evaluations,
    )

    if verify_with_replay and status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        init_ratios = None
        init_res = None
        init_cds = None
        if live_state is not None:
            init_ratios = {
                "party": live_state.current_party_hp_ratio,
                "mt": live_state.current_mt_hp_ratio if live_state.current_mt_hp_ratio is not None else live_state.current_party_hp_ratio,
                "st": live_state.current_st_hp_ratio if live_state.current_st_hp_ratio is not None else live_state.current_party_hp_ratio,
            }
            init_res = dict(live_state.current_resources)
            init_cds = dict(live_state.cooldowns_remaining)
        plan.replay = replay_plan(
            kit=kit,
            timeline=timeline,
            plan=plan,
            deterministic=True,
            start_time_s=start_time_s,
            initial_hp_ratios=init_ratios,
            initial_resources=init_res,
            initial_cooldowns=init_cds,
        )

    return plan


def replan_from_state(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    live_state: LiveFightState,
    preset: ObjectivePreset = "safe",
    partner_trust_override: float | None = None,
    time_limit_s: float = 0.3,
) -> OptimizationPlan:
    """Fast mid-fight replanner for recovering from Raises, slips, or missed cooldowns."""
    return solve_mitigation_plan(
        kit=kit,
        timeline=timeline,
        preset=preset,
        partner_trust_override=partner_trust_override,
        time_limit_s=time_limit_s,
        live_state=live_state,
    )

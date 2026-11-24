"""Worst-case (+5% damage roll x 12 server-tick phases) 100% survival verifier, hardener & DPS maximizer.

In FFXIV Savage/Ultimate raiding, the goal is STRICTLY:
  1. 100.0% survival (zero probabilistic wipes across worst-case +5% damage rolls and all 12 3s tick phases), AND
  2. Maximum Expected DPS (火力期待値) — prioritizing 0-potency oGCDs over damage-convertible resources
     (e.g. Aetherflow -> Energy Drain 100p) and GCD heals (e.g. Broil IV 310p), while avoiding 2-min burst windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xiv_healwork_solver.models import EncounterTimeline, OptimizationPlan, PlannedAction, PlayerKit, SkillDef
from xiv_healwork_solver.sim.engine import replay_plan


@dataclass
class StressTestReport:
    damage_roll: float
    total_phases: int
    survived_phases: int
    survival_rate_pct: float
    worst_min_hp_ratio: dict[str, float]
    worst_failure_time_s: float | None
    worst_failure_channel: str | None
    worst_deficit_hp: float
    mean_overheal_pct: float
    mean_filler_gcds: float
    mean_gcd_heals: float
    mean_expected_potency: float = 0.0
    mean_potency_lost: float = 0.0
    mean_dps_efficiency_pct: float = 100.0
    hp_triggers_fired: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "damage_roll": self.damage_roll,
            "total_phases": self.total_phases,
            "survived_phases": self.survived_phases,
            "survival_rate_pct": self.survival_rate_pct,
            "worst_min_hp_ratio": self.worst_min_hp_ratio,
            "worst_failure_time_s": self.worst_failure_time_s,
            "worst_failure_channel": self.worst_failure_channel,
            "worst_deficit_hp": self.worst_deficit_hp,
            "mean_overheal_pct": self.mean_overheal_pct,
            "mean_filler_gcds": self.mean_filler_gcds,
            "mean_gcd_heals": self.mean_gcd_heals,
            "mean_expected_potency": self.mean_expected_potency,
            "mean_potency_lost": self.mean_potency_lost,
            "mean_dps_efficiency_pct": self.mean_dps_efficiency_pct,
            "hp_triggers_fired": self.hp_triggers_fired,
        }


def action_log_to_plan(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    action_log: list[dict[str, Any]],
    partner_trust: float = 1.0,
    status: str = "rl_policy",
) -> OptimizationPlan:
    """Convert an RL deterministic_action_log into a replayable OptimizationPlan."""
    skills_by_id = {sk.id: sk for sk in kit.skills}
    events_sorted = sorted(timeline.events, key=lambda e: e.time_s)
    planned: list[PlannedAction] = []
    for i, item in enumerate(action_log):
        sid = str(item.get("skill_id", ""))
        sk = skills_by_id.get(sid)
        if sk is None:
            continue
        t = float(item.get("time_s", item.get("cast_at_s", 0.0)))
        ch = str(item.get("channel", item.get("target_channel", "party")))
        sc = "boss_debuff" if ch == "boss" else ch
        nxt = next((e for e in events_sorted if e.time_s >= t - 1.0), None)
        planned.append(
            PlannedAction(
                event_id=nxt.id if nxt is not None else f"t_{i}",
                event_name=nxt.name if nxt is not None else "Auto-Attack / Upkeep",
                event_time_s=nxt.time_s if nxt is not None else round(t + 2.0, 2),
                target_scope=sc,  # type: ignore[arg-type]
                cast_at_s=round(t, 2),
                skill_id=sk.id,
                skill_name=sk.name,
                target_channel=ch,  # type: ignore[arg-type]
                is_gcd=sk.is_gcd,
                weave_cost=sk.weave_cost,
            )
        )
    planned.sort(key=lambda a: (a.cast_at_s, 0 if a.is_gcd else 1))
    return OptimizationPlan(
        encounter_name=timeline.encounter_name,
        kit_label=kit.label,
        preset="balanced",
        status=status,
        objective_value=0.0,
        partner_trust=partner_trust,
        actions=planned,
        evaluations=[],
    )


def stress_test_plan(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    plan: OptimizationPlan,
    damage_roll: float = 1.05,
    n_phases: int = 12,
) -> StressTestReport:
    """Verify a plan across `n_phases` 3.0s server-tick phases under fixed `damage_roll` (default +5% worst-case).

    When `timeline.hp_trigger_mechanics` is non-empty, ALSO checks nominal (1.00x) and low (-5%, 0.95x)
    damage rolls so a lower roll that pushes HP across a lethal full-heal trigger can never pass uncertified.
    """
    rolls: tuple[float, ...] = (
        tuple(dict.fromkeys((0.95, 1.00, damage_roll)))
        if timeline.hp_trigger_mechanics
        else (damage_roll,)
    )
    total_runs = n_phases * len(rolls)
    survived = 0
    worst_hp = {"party": 1.0, "mt": 1.0, "st": 1.0}
    worst_fail_t: float | None = None
    worst_fail_ch: str | None = None
    worst_deficit = 0.0
    ohs: list[float] = []
    fillers: list[float] = []
    gcd_heals: list[float] = []
    pots: list[float] = []
    pot_losses: list[float] = []
    dps_effs: list[float] = []
    trig_max = 0

    for roll in rolls:
        for i in range(n_phases):
            phase = (3.0 * i) / max(1, n_phases)
            res = replay_plan(
                kit,
                timeline,
                plan,
                deterministic=True,
                fixed_tick_phase=phase,
                forced_damage_roll=roll,
            )
            if res.survived:
                survived += 1
            else:
                ft = res.first_death_time_s if res.first_death_time_s is not None else 0.0
                if worst_fail_t is None or ft < worst_fail_t or (abs(ft - worst_fail_t) < 1e-3 and res.deficit_hp > worst_deficit):
                    worst_fail_t = ft
                    worst_fail_ch = res.first_death_channel
                    worst_deficit = res.deficit_hp
            for ch in ("party", "mt", "st"):
                worst_hp[ch] = min(worst_hp[ch], res.min_hp_ratio.get(ch, 1.0))
            ohs.append(res.overheal_pct)
            fillers.append(float(res.filler_gcds))
            gcd_heals.append(float(res.gcd_heals))
            pots.append(res.expected_total_potency)
            pot_losses.append(res.expected_potency_lost)
            dps_effs.append(res.dps_efficiency_pct)
            trig_max = max(trig_max, res.hp_triggers_fired)

    n = max(1, total_runs)
    return StressTestReport(
        damage_roll=damage_roll,
        total_phases=total_runs,
        survived_phases=survived,
        survival_rate_pct=round((survived / n) * 100.0, 2),
        worst_min_hp_ratio={k: round(v, 4) for k, v in worst_hp.items()},
        worst_failure_time_s=worst_fail_t,
        worst_failure_channel=worst_fail_ch,
        worst_deficit_hp=round(worst_deficit, 1),
        mean_overheal_pct=round(sum(ohs) / n, 1),
        mean_filler_gcds=round(sum(fillers) / n, 1),
        mean_gcd_heals=round(sum(gcd_heals) / n, 2),
        mean_expected_potency=round(sum(pots) / n, 1),
        mean_potency_lost=round(sum(pot_losses) / n, 1),
        mean_dps_efficiency_pct=round(sum(dps_effs) / n, 2),
        hp_triggers_fired=trig_max,
    )


def _skill_helps_channel(sk: SkillDef, channel: str, kit: PlayerKit) -> tuple[bool, str]:
    """Return (helps, target_channel) for whether `sk` can mitigate/shield/heal `channel`."""
    has_def = (
        sk.mitigation_all > 0
        or sk.mitigation_physical > 0
        or sk.mitigation_magical > 0
        or sk.shield_value > 0
        or sk.shield_max_hp_ratio > 0
        or sk.heal_instant > 0
        or sk.heal_hot_total > 0
        or sk.delayed_heal_value > 0
    )
    if not has_def:
        return False, "party"
    if sk.target_scope == "boss_debuff":
        return True, "boss"
    if sk.target_scope == "party":
        return True, "party"
    if sk.target_scope == "single_ally":
        if channel in ("mt", "st"):
            return True, channel
        return False, "mt"
    if sk.target_scope == "self":
        self_ch = "mt" if kit.player_is_mt else "st"
        if kit.role == "tank" and channel == self_ch:
            return True, self_ch
        return False, self_ch
    if sk.target_scope == channel:
        return True, channel
    return False, "party"


def _effective_potency_loss(kit: PlayerKit, timeline: EncounterTimeline, sk: SkillDef, cast_t: float) -> float:
    dw = timeline.downtime_at(cast_t)
    is_free_dt = bool(
        sk.is_gcd
        and dw is not None
        and not dw.boss_targetable
        and cast_t + max(kit.gcd_s, sk.cast_time_s) <= dw.end_s + 1e-6
    )
    return kit.expected_skill_potency_loss(sk, cast_t, is_free_downtime=is_free_dt)


def harden_plan_to_100pct(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    plan: OptimizationPlan,
    damage_roll: float = 1.05,
    n_phases: int = 12,
    max_repairs: int = 18,
) -> tuple[OptimizationPlan, StressTestReport, int]:
    """Iteratively add lowest expected-potency-loss legal skills until `plan` achieves 100.0% survival
    across all `n_phases` server-tick phases under worst-case `damage_roll` (+5% default).
    """
    current = plan.model_copy(deep=True)
    report = stress_test_plan(kit, timeline, current, damage_roll=damage_roll, n_phases=n_phases)
    if report.survival_rate_pct >= 100.0:
        return current, report, 0

    events_sorted = sorted(timeline.events, key=lambda e: e.time_s)
    # Sort strictly by lowest base expected potency loss first (0p oGCDs -> 100p Aetherflow -> 310p GCD shields!)
    candidate_skills = sorted(
        kit.skills,
        key=lambda s: (kit.expected_skill_potency_loss(s, 0.0, False), 1 if s.is_gcd else 0, s.dps_cost, s.cooldown_s),
    )
    repairs = 0

    for _ in range(max_repairs):
        if report.survival_rate_pct >= 100.0 or report.worst_failure_time_s is None:
            break
        fail_t = report.worst_failure_time_s
        fail_ch = (report.worst_failure_channel or "party").split("(")[0]
        best_choice: tuple[tuple[int, float, float, float], OptimizationPlan, StressTestReport] | None = None

        offsets = (3.0, 5.5, 9.0, 14.0)
        for sk in candidate_skills:
            helps, t_ch = _skill_helps_channel(sk, fail_ch, kit)
            if not helps:
                continue
            for off in offsets:
                cast_t = round(max(0.5, fail_t - off), 2)
                if sk.needs_boss_target() and not timeline.is_boss_targetable(cast_t):
                    continue
                if any(a.skill_id == sk.id and abs(a.cast_at_s - cast_t) < 4.0 for a in current.actions):
                    continue
                nxt = next((e for e in events_sorted if e.time_s >= cast_t - 0.5), None)
                new_act = PlannedAction(
                    event_id=nxt.id if nxt is not None else f"safe_{repairs}",
                    event_name=nxt.name if nxt is not None else "Safety Buffer",
                    event_time_s=nxt.time_s if nxt is not None else round(fail_t, 2),
                    target_scope="boss_debuff" if t_ch == "boss" else t_ch,  # type: ignore[arg-type]
                    cast_at_s=cast_t,
                    skill_id=sk.id,
                    skill_name=sk.name,
                    target_channel=t_ch,  # type: ignore[arg-type]
                    is_gcd=sk.is_gcd,
                    weave_cost=sk.weave_cost,
                )
                trial_actions = sorted(current.actions + [new_act], key=lambda a: (a.cast_at_s, 0 if a.is_gcd else 1))
                trial_plan = current.model_copy(update={"actions": trial_actions})

                quick = replay_plan(
                    kit,
                    timeline,
                    trial_plan,
                    deterministic=True,
                    fixed_tick_phase=0.0,
                    forced_damage_roll=damage_roll,
                )
                if not quick.survived and quick.first_death_time_s is not None:
                    if quick.first_death_time_s <= fail_t + 0.05 and quick.deficit_hp >= report.worst_deficit_hp - 500.0:
                        continue

                trial_rep = stress_test_plan(kit, timeline, trial_plan, damage_roll=damage_roll, n_phases=n_phases)
                improved = (
                    trial_rep.survival_rate_pct > report.survival_rate_pct
                    or (
                        trial_rep.worst_failure_time_s is not None
                        and trial_rep.worst_failure_time_s > fail_t + 0.5
                    )
                    or (
                        trial_rep.worst_failure_time_s is not None
                        and abs(trial_rep.worst_failure_time_s - fail_t) <= 0.5
                        and trial_rep.worst_deficit_hp < report.worst_deficit_hp - 2_000.0
                    )
                )
                if not improved:
                    continue

                pot_loss = _effective_potency_loss(kit, timeline, sk, cast_t)
                cleared_here = 1 if (trial_rep.worst_failure_time_s is None or trial_rep.worst_failure_time_s > fail_t + 0.5) else 0
                progress_t = trial_rep.worst_failure_time_s if trial_rep.worst_failure_time_s is not None else (timeline.duration_s + 100.0)
                key = (
                    cleared_here,
                    trial_rep.survival_rate_pct,
                    progress_t,
                    -pot_loss - sk.dps_cost * 10.0,
                )
                if best_choice is None or key > best_choice[0]:
                    best_choice = (key, trial_plan, trial_rep)
                # Fast exit when a 0-potency oGCD already achieves 100% survival!
                if trial_rep.survival_rate_pct >= 100.0 and pot_loss <= 1e-6:
                    break
            if best_choice is not None and best_choice[2].survival_rate_pct >= 100.0:
                if _effective_potency_loss(kit, timeline, sk, max(0.5, fail_t - 3.0)) <= 1e-6:
                    break

        if best_choice is None:
            break
        _, current, report = best_choice
        repairs += 1

    if report.survival_rate_pct >= 100.0:
        current = current.model_copy(update={"status": "verified_100pct_worst_case"})
    return current, report, repairs


def maximize_dps_at_100pct_survival(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    plan: OptimizationPlan,
    damage_roll: float = 1.05,
    n_phases: int = 12,
    min_hp_margin_ratio: float = 0.0,
) -> tuple[OptimizationPlan, StressTestReport, dict[str, int]]:
    """Maximize Expected DPS (火力期待値) subject to STRICT 100.0% worst-case survival!

    Performs three exact DPS-improving passes while verifying all `n_phases` at `+5%` worst-case roll:
      1. Prune redundant high-potency-loss skills (310p GCD heals -> 100p Aetherflow -> redundant overheal casts).
      2. Downgrade/swap costly skills (`> 0p`) into cheaper unused skills (e.g. 310p GCD -> 0p fairy oGCD or 100p oGCD).
      3. Shift costly casts (`> 0p`) that fall inside 2-minute raid-buff burst windows slightly outside burst.
    """
    skills_by_id = {sk.id: sk for sk in kit.skills}
    current = plan.model_copy(deep=True)
    report = stress_test_plan(kit, timeline, current, damage_roll=damage_roll, n_phases=n_phases)
    stats = {"pruned": 0, "swapped": 0, "burst_shifted": 0}
    if report.survival_rate_pct < 100.0:
        return current, report, stats

    def _is_safe(rep: StressTestReport) -> bool:
        if rep.survival_rate_pct < 100.0:
            return False
        if min_hp_margin_ratio > 0.0:
            return min(rep.worst_min_hp_ratio.values()) >= min_hp_margin_ratio - 1e-4
        return True

    def _dps_score(rep: StressTestReport) -> tuple[float, float, float]:
        # `mean_expected_potency` is already exact net DPS potency (filler gained - resource potency lost),
        # so zero double-counting of displaced filler GCDs!
        return (rep.mean_expected_potency, -rep.mean_potency_lost, -rep.mean_overheal_pct)

    # Pass 1: Prune costly or redundant casts (highest potency loss first; also checks 0p oGCDs that may clip filler!)
    changed = True
    while changed:
        changed = False
        indexed = sorted(
            enumerate(current.actions),
            key=lambda kv: (
                _effective_potency_loss(kit, timeline, skills_by_id[kv[1].skill_id], kv[1].cast_at_s)
                if kv[1].skill_id in skills_by_id
                else 0.0,
                1 if kv[1].is_gcd else 0,
            ),
            reverse=True,
        )
        for idx, act in indexed:
            sk = skills_by_id.get(act.skill_id)
            if sk is None or sk.mp_regen_per_tick > 0:
                continue
            trial_actions = current.actions[:idx] + current.actions[idx + 1 :]
            trial_plan = current.model_copy(update={"actions": trial_actions})
            # Fast 1-phase pre-check before full multi-phase check
            q = replay_plan(kit, timeline, trial_plan, deterministic=True, fixed_tick_phase=0.0, forced_damage_roll=damage_roll)
            if not q.survived:
                continue
            trial_rep = stress_test_plan(kit, timeline, trial_plan, damage_roll=damage_roll, n_phases=n_phases)
            if _is_safe(trial_rep) and _dps_score(trial_rep) > _dps_score(report):
                current = trial_plan
                report = trial_rep
                stats["pruned"] += 1
                changed = True
                break

    # Pass 2: Downgrade remaining >0p skills to strictly cheaper unused oGCD skills (e.g. 310p GCD -> 0p oGCD)
    cheaper_pool = sorted(
        kit.skills,
        key=lambda s: (kit.expected_skill_potency_loss(s, 0.0, False), s.dps_cost),
    )
    for idx in range(len(current.actions)):
        act = current.actions[idx]
        sk_old = skills_by_id.get(act.skill_id)
        if sk_old is None:
            continue
        old_loss = _effective_potency_loss(kit, timeline, sk_old, act.cast_at_s)
        if old_loss <= 1e-6:
            continue
        for sk_new in cheaper_pool:
            if sk_new.id == sk_old.id:
                continue
            new_loss = _effective_potency_loss(kit, timeline, sk_new, act.cast_at_s)
            if new_loss >= old_loss - 1.0:
                continue
            helps, t_ch = _skill_helps_channel(sk_new, act.target_channel, kit)
            if not helps:
                continue
            if sk_new.needs_boss_target() and not timeline.is_boss_targetable(act.cast_at_s):
                continue
            new_act = act.model_copy(
                update={
                    "skill_id": sk_new.id,
                    "skill_name": sk_new.name,
                    "action_id": sk_new.action_id,
                    "target_channel": t_ch,
                    "target_scope": "boss_debuff" if t_ch == "boss" else t_ch,
                    "is_gcd": sk_new.is_gcd,
                    "weave_cost": sk_new.weave_cost,
                }
            )
            trial_actions = current.actions[:idx] + [new_act] + current.actions[idx + 1 :]
            trial_plan = current.model_copy(update={"actions": trial_actions})
            q = replay_plan(kit, timeline, trial_plan, deterministic=True, fixed_tick_phase=0.0, forced_damage_roll=damage_roll)
            if not q.survived:
                continue
            trial_rep = stress_test_plan(kit, timeline, trial_plan, damage_roll=damage_roll, n_phases=n_phases)
            if _is_safe(trial_rep) and _dps_score(trial_rep) > _dps_score(report):
                current = trial_plan
                report = trial_rep
                stats["swapped"] += 1
                break

    # Pass 3: Shift >0p skills sitting inside 2-min burst windows outside burst (testing both nearby offsets & exact burst edges!)
    for idx in range(len(current.actions)):
        act = current.actions[idx]
        sk = skills_by_id.get(act.skill_id)
        if sk is None or not kit.is_in_burst(act.cast_at_s):
            continue
        if _effective_potency_loss(kit, timeline, sk, act.cast_at_s) <= 1e-6:
            continue
        cycle = int((act.cast_at_s - kit.burst_offset_s) // kit.burst_interval_s)
        b_start = kit.burst_offset_s + cycle * kit.burst_interval_s
        b_end = b_start + kit.burst_duration_s
        candidate_times = [
            round(max(0.5, b_start - 0.5), 2),
            round(min(timeline.duration_s - 1.0, b_end + 0.5), 2),
            round(max(0.5, act.cast_at_s - 3.0), 2),
            round(min(timeline.duration_s - 1.0, act.cast_at_s + 3.0), 2),
        ]
        for new_t in dict.fromkeys(candidate_times):
            if kit.is_in_burst(new_t):
                continue
            new_act = act.model_copy(update={"cast_at_s": new_t})
            trial_actions = sorted(
                current.actions[:idx] + [new_act] + current.actions[idx + 1 :],
                key=lambda a: (a.cast_at_s, 0 if a.is_gcd else 1),
            )
            trial_plan = current.model_copy(update={"actions": trial_actions})
            trial_rep = stress_test_plan(kit, timeline, trial_plan, damage_roll=damage_roll, n_phases=n_phases)
            if _is_safe(trial_rep) and _dps_score(trial_rep) > _dps_score(report):
                current = trial_plan
                report = trial_rep
                stats["burst_shifted"] += 1
                break

    current = current.model_copy(update={"status": "verified_100pct_max_dps"})
    return current, report, stats

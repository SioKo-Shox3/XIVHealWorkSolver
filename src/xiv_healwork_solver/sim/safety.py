"""Worst-case (+5% damage roll x 12 server-tick phases) 100% survival verifier & minimal-DPS-loss hardener.

In FFXIV Savage/Ultimate raiding, a healwork that survives only probabilistically (e.g. 50-80%)
is unusable. This module verifies any candidate plan (from RL or CP-SAT) across all 12 server-tick
phases under worst-case +5% damage rolls and, if any phase fails, iteratively inserts the lowest-DPS-cost
legal defensive/healing action before the failure point until 100.0% worst-case survival is verified.
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
        # Attach to nearest upcoming event for readable sheet display
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
    """Verify a plan across `n_phases` 3.0s server-tick phases under fixed `damage_roll` (default +5% worst-case)."""
    survived = 0
    worst_hp = {"party": 1.0, "mt": 1.0, "st": 1.0}
    worst_fail_t: float | None = None
    worst_fail_ch: str | None = None
    worst_deficit = 0.0
    ohs: list[float] = []
    fillers: list[float] = []
    gcd_heals: list[float] = []

    for i in range(n_phases):
        phase = (3.0 * i) / max(1, n_phases)
        res = replay_plan(
            kit,
            timeline,
            plan,
            deterministic=True,
            fixed_tick_phase=phase,
            forced_damage_roll=damage_roll,
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

    return StressTestReport(
        damage_roll=damage_roll,
        total_phases=n_phases,
        survived_phases=survived,
        survival_rate_pct=round((survived / max(1, n_phases)) * 100.0, 2),
        worst_min_hp_ratio={k: round(v, 4) for k, v in worst_hp.items()},
        worst_failure_time_s=worst_fail_t,
        worst_failure_channel=worst_fail_ch,
        worst_deficit_hp=round(worst_deficit, 1),
        mean_overheal_pct=round(sum(ohs) / max(1, len(ohs)), 1),
        mean_filler_gcds=round(sum(fillers) / max(1, len(fillers)), 1),
        mean_gcd_heals=round(sum(gcd_heals) / max(1, len(gcd_heals)), 2),
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


def harden_plan_to_100pct(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    plan: OptimizationPlan,
    damage_roll: float = 1.05,
    n_phases: int = 12,
    max_repairs: int = 18,
) -> tuple[OptimizationPlan, StressTestReport, int]:
    """Iteratively add lowest-DPS-cost legal skills until `plan` achieves 100.0% survival across
    all `n_phases` server-tick phases under worst-case `damage_roll` (+5% default).

    Returns (hardened_plan, final_stress_report, repairs_added).
    """
    current = plan.model_copy(deep=True)
    report = stress_test_plan(kit, timeline, current, damage_roll=damage_roll, n_phases=n_phases)
    if report.survival_rate_pct >= 100.0:
        return current, report, 0

    events_sorted = sorted(timeline.events, key=lambda e: e.time_s)
    # Sort skills strictly by lowest DPS cost first (oGCDs before GCD shields!)
    candidate_skills = sorted(kit.skills, key=lambda s: (1 if s.is_gcd else 0, s.dps_cost, s.cooldown_s))
    repairs = 0

    for _ in range(max_repairs):
        if report.survival_rate_pct >= 100.0 or report.worst_failure_time_s is None:
            break
        fail_t = report.worst_failure_time_s
        fail_ch = (report.worst_failure_channel or "party").split("(")[0]
        # Evaluate candidate insertions quickly on the worst phase (phase=1.5) first, then confirm on all phases
        best_choice: tuple[tuple[int, float, float, float], OptimizationPlan, StressTestReport] | None = None

        # Candidate offsets before fail_t: immediate pre-hit + pre-mechanic window
        offsets = (3.0, 5.5, 9.0, 14.0)
        for sk in candidate_skills:
            helps, t_ch = _skill_helps_channel(sk, fail_ch, kit)
            if not helps:
                continue
            for off in offsets:
                cast_t = round(max(0.5, fail_t - off), 2)
                if sk.needs_boss_target() and not timeline.is_boss_targetable(cast_t):
                    continue
                # Avoid inserting an identical skill within 4s of an existing scheduled cast
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

                # Quick single-phase screen before full 12-phase check
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
                # Strict improvement check: higher survival rate, OR strictly later worst failure, OR strictly smaller deficit
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

                # Lexicographic preference: (cleared_this_fail_t, survival_rate, -gcd_cost, -dps_cost)
                cleared_here = 1 if (trial_rep.worst_failure_time_s is None or trial_rep.worst_failure_time_s > fail_t + 0.5) else 0
                progress_t = trial_rep.worst_failure_time_s if trial_rep.worst_failure_time_s is not None else (timeline.duration_s + 100.0)
                key = (
                    cleared_here,
                    trial_rep.survival_rate_pct,
                    progress_t,
                    -(10.0 if sk.is_gcd else 0.0) - sk.dps_cost,
                )
                if best_choice is None or key > best_choice[0]:
                    best_choice = (key, trial_plan, trial_rep)
                # Fast exit if an oGCD completely achieved 100% survival!
                if trial_rep.survival_rate_pct >= 100.0 and not sk.is_gcd:
                    break
            if best_choice is not None and best_choice[2].survival_rate_pct >= 100.0 and not sk.is_gcd:
                break

        if best_choice is None:
            break
        _, current, report = best_choice
        repairs += 1

    if report.survival_rate_pct >= 100.0:
        current = current.model_copy(update={"status": "verified_100pct_worst_case"})
    return current, report, repairs

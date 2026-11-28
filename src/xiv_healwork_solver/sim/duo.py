"""Mode B: True Dual-Actor (H1 + H2) Real-Time Combat Simulator & Joint Co-Optimizer.

Simulates BOTH healers (e.g. White Mage + Scholar, Astrologian + Scholar, White Mage + Sage, etc.)
simultaneously inside one shared CombatEngine timeline (shared HP, shared shields with exact
shield_stack_group overwrite rules, shared multiplicative mitigations with stack_group dedup,
and independent per-healer GCD / weave / cooldown / MP / Lily / Aetherflow / Addersgall budgets).

Co-optimizes BOTH healers together to achieve:
  1. Strict 100.0% worst-case (+5% damage roll x 12 server-tick phases) survival
  2. Zero dropped / illegal actions for BOTH H1 and H2
  3. Maximum COMBINED Expected DPS (H1 + H2 net expected potency) with minimal cross-healer overheal
  4. Healthy effective healing workload balance (no Chad-healing)
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xiv_healwork_solver.models import (
    EncounterTimeline,
    OptimizationPlan,
    PlannedAction,
    PlayerKit,
    ReplayEventSnapshot,
    SkillDef,
)
from xiv_healwork_solver.sim.engine import (
    EPS,
    PRIO_CAST_RESOLVE,
    CombatEngine,
    _ActiveBuff,
    _ScheduledItem,
    build_action_entries,
)
from xiv_healwork_solver.sim.partner_import import strip_cohealer_from_timeline
from xiv_healwork_solver.sim.safety import (
    _effective_potency_loss,
    _skill_helps_channel,
    action_log_to_plan,
    canonicalize_plan_actions,
)


@dataclass
class DuoReplayResult:
    survived: bool
    first_death_time_s: float | None
    first_death_channel: str | None
    deficit_hp: float
    min_hp_ratio: dict[str, float]
    h1_filler_gcds: int
    h2_filler_gcds: int
    h1_gcd_heals: int
    h2_gcd_heals: int
    h1_expected_potency: float
    h2_expected_potency: float
    combined_expected_potency: float
    h1_potency_lost: float
    h2_potency_lost: float
    combined_dps_efficiency_pct: float
    h1_overheal_pct: float
    h2_overheal_pct: float
    combined_overheal_pct: float
    h1_effective_heal_share_pct: float
    h2_effective_heal_share_pct: float
    h1_mp_min: float
    h2_mp_min: float
    h1_executed_actions: int
    h2_executed_actions: int
    h1_dropped_actions: int
    h2_dropped_actions: int
    hp_triggers_fired: int
    event_snapshots: list[ReplayEventSnapshot]


@dataclass
class DuoStressReport:
    damage_roll: float
    total_phases: int
    survived_phases: int
    survival_rate_pct: float
    worst_min_hp_ratio: dict[str, float]
    worst_failure_time_s: float | None
    worst_failure_channel: str | None
    worst_failure_phase: float
    worst_deficit_hp: float
    mean_h1_expected_potency: float
    mean_h2_expected_potency: float
    mean_combined_expected_potency: float
    mean_combined_potency_lost: float
    mean_combined_dps_efficiency_pct: float
    mean_h1_gcd_heals: float
    mean_h2_gcd_heals: float
    mean_h1_filler_gcds: float
    mean_h2_filler_gcds: float
    mean_h1_overheal_pct: float
    mean_h2_overheal_pct: float
    mean_combined_overheal_pct: float
    mean_h1_heal_share_pct: float
    mean_h2_heal_share_pct: float
    max_h1_dropped: int
    max_h2_dropped: int
    hp_triggers_fired: int

    @property
    def max_dropped_actions(self) -> int:
        return self.max_h1_dropped + self.max_h2_dropped

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
            "mean_h1_expected_potency": self.mean_h1_expected_potency,
            "mean_h2_expected_potency": self.mean_h2_expected_potency,
            "mean_combined_expected_potency": self.mean_combined_expected_potency,
            "mean_combined_potency_lost": self.mean_combined_potency_lost,
            "mean_combined_dps_efficiency_pct": self.mean_combined_dps_efficiency_pct,
            "mean_h1_gcd_heals": self.mean_h1_gcd_heals,
            "mean_h2_gcd_heals": self.mean_h2_gcd_heals,
            "mean_h1_filler_gcds": self.mean_h1_filler_gcds,
            "mean_h2_filler_gcds": self.mean_h2_filler_gcds,
            "mean_h1_overheal_pct": self.mean_h1_overheal_pct,
            "mean_h2_overheal_pct": self.mean_h2_overheal_pct,
            "mean_combined_overheal_pct": self.mean_combined_overheal_pct,
            "mean_h1_heal_share_pct": self.mean_h1_heal_share_pct,
            "mean_h2_heal_share_pct": self.mean_h2_heal_share_pct,
            "max_h1_dropped": self.max_h1_dropped,
            "max_h2_dropped": self.max_h2_dropped,
            "hp_triggers_fired": self.hp_triggers_fired,
        }


class DualHealerEngine(CombatEngine):
    """Exact real-time dual-healer simulator inheriting 100% of CombatEngine's boss/mechanic physics."""

    def __init__(
        self,
        kit_h1: PlayerKit,
        kit_h2: PlayerKit,
        timeline: EncounterTimeline,
        *,
        damage_variance: float = 0.0,
        partner_trust_scale: float = 1.0,
        forced_damage_roll: float | None = None,
        expected_partners: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            kit=kit_h1,
            timeline=timeline,
            damage_variance=damage_variance,
            raise_prob_per_pull=0.0,
            partner_trust_scale=partner_trust_scale,
            tank_passive_self_heal_per_tick=0.0,
            forced_damage_roll=forced_damage_roll,
            expected_partners=expected_partners,
            seed=seed,
        )
        self.h2_kit = kit_h2
        self.h2_actions = build_action_entries(kit_h2)
        self.h2_n_actions = len(self.h2_actions)
        self.h2_resolved_pools = {p.name: p for p in kit_h2.resolved_resources()}
        self.h2_heal_scale = kit_h2.gear_stats.heal_scale_multiplier
        self.h2_base_mp_per_3s_tick = (
            kit_h2.gear_stats.combat_mp_per_3s_tick() + kit_h2.gear_stats.lucid_dreaming_mp_per_sec * 3.0
        )
        # Ensure shared Max HP uses max across both kits (normally identical at same iLvl)
        self.max_hp["party"] = max(kit_h1.max_hp_party, kit_h2.max_hp_party)
        self.max_hp["mt"] = max(kit_h1.max_hp_tank, kit_h2.max_hp_tank)
        self.max_hp["st"] = max(kit_h1.max_hp_tank, kit_h2.max_hp_tank)

        self.h2_gcd_ready_at: float = 0.0
        self.h2_anim_lock_until: float = 0.0
        self.h2_casting_until: float = 0.0
        self.h2_weave_slots: int = kit_h2.base_weave_slots_per_gcd
        self.h2_resources: dict[str, float] = {}
        self.h2_skill_charges: dict[str, int] = {}
        self.h2_skill_next_charge_at: dict[str, float] = {}
        self.h2_stats: dict[str, float] = {}

    def _record_heal_attribution(self, heal_owner: str, raw_total: float, eff: float) -> None:
        if heal_owner == "h1":
            self.stats["player_heal_raw"] += raw_total
            self.stats["player_heal_eff"] += eff
        elif heal_owner == "h2":
            self.h2_stats["player_heal_raw"] += raw_total
            self.h2_stats["player_heal_eff"] += eff

    def reset(
        self,
        seed: int | None = None,
        deterministic: bool = True,
        fixed_tick_phase: float | None = None,
        initial_hp_ratios: dict[str, float] | None = None,
        initial_resources: dict[str, float] | None = None,
        initial_cooldowns: dict[str, float] | None = None,
        start_time_s: float = 0.0,
    ) -> None:
        self.h2_stats = {
            "filler_gcds": 0.0,
            "gcd_heals": 0.0,
            "downtime_free_gcds": 0.0,
            "ogcds_used": 0.0,
            "player_heal_raw": 0.0,
            "player_heal_eff": 0.0,
            "expected_potency_gained": 0.0,
            "expected_potency_lost": 0.0,
            "resource_potency_lost": 0.0,
            "mp_min": 10_000.0,
        }
        super().reset(
            seed=seed,
            deterministic=deterministic,
            fixed_tick_phase=fixed_tick_phase,
            initial_hp_ratios=initial_hp_ratios,
            initial_resources=initial_resources,
            initial_cooldowns=initial_cooldowns,
            start_time_s=start_time_s,
        )
        self.h2_gcd_ready_at = start_time_s
        self.h2_anim_lock_until = start_time_s
        self.h2_casting_until = start_time_s
        self.h2_weave_slots = self.h2_kit.weave_budget_at(start_time_s)
        self.h2_resources = {name: pool.initial for name, pool in self.h2_resolved_pools.items()}
        self.h2_stats["mp_min"] = self.h2_resources.get("mp", 10_000.0)
        self.h2_skill_charges = {sk.id: sk.max_charges for sk in self.h2_kit.skills}
        self.h2_skill_next_charge_at = {sk.id: math.inf for sk in self.h2_kit.skills}

        # Apply H2 pre-pull actions with exact parity to H1 (flat override_shield_value, mit, HoT, optional MP!)
        if start_time_s <= EPS and self.h2_kit.prepull_actions:
            skills_by_id = {sk.id: sk for sk in self.h2_kit.skills}
            for pp in sorted(self.h2_kit.prepull_actions, key=lambda a: a.offset_s):
                sk = skills_by_id.get(pp.skill_id)
                if sk is None:
                    continue
                sh_dur = max(0.0, sk.effective_shield_duration() + pp.offset_s)
                if pp.override_shield_value is not None:
                    sh_val = pp.override_shield_value
                elif sk.shield_max_hp_ratio > 0:
                    base_hp = self.max_hp[pp.target_channel] if pp.target_channel in self.max_hp else self.max_hp["party"]
                    sh_val = base_hp * sk.shield_max_hp_ratio
                else:
                    sh_val = sk.shield_value * self.h2_heal_scale
                if sh_val > 0 and sh_dur > EPS:
                    self._add_shield(pp.target_channel, sh_val, sh_dur, f"h2_prepull_{sk.id}", sk.shield_stack_group)
                rem_buff = max(0.0, sk.duration_s + pp.offset_s)
                hot_ticks = max(1, int(round(sk.duration_s / 3.0))) if sk.duration_s >= 3.0 else 1
                hot_per_tick = (sk.heal_hot_total * self.h2_heal_scale) / hot_ticks if sk.heal_hot_total > 0 else 0.0
                if rem_buff > EPS and (sk.mitigation_all > 0 or sk.mitigation_physical > 0 or sk.mitigation_magical > 0 or hot_per_tick > 0):
                    self.active_buffs.append(
                        _ActiveBuff(
                            skill_id=f"h2_prepull_{sk.id}",
                            stack_group=sk.stack_group,
                            target_channel=pp.target_channel,
                            expires_at=rem_buff,
                            mit_all=sk.mitigation_all,
                            mit_phys=sk.mitigation_physical,
                            mit_mag=sk.mitigation_magical,
                            hot_per_tick=hot_per_tick,
                            owner="h2",
                        )
                    )
                if pp.consumes_mp:
                    mp_cost = sk.resource_costs.get("mp", 0.0)
                    if mp_cost > 0 and "mp" in self.h2_resources:
                        self.h2_resources["mp"] = max(0.0, self.h2_resources["mp"] - mp_cost)
                        self.h2_stats["mp_min"] = min(self.h2_stats["mp_min"], self.h2_resources["mp"])
                if pp.consumes_cooldown and sk.cooldown_s > 0:
                    rem_cd = sk.cooldown_s + pp.offset_s
                    if rem_cd > EPS and self.h2_skill_charges[sk.id] > 0:
                        self.h2_skill_charges[sk.id] -= 1
                        self.h2_skill_next_charge_at[sk.id] = rem_cd

    def _update_continuous_state(self, new_t: float) -> None:
        dt = max(0.0, new_t - self.t)
        if dt > 0.0:
            for sk in self.h2_kit.skills:
                while self.h2_skill_charges[sk.id] < sk.max_charges and self.h2_skill_next_charge_at[sk.id] <= new_t + EPS:
                    charge_t = self.h2_skill_next_charge_at[sk.id]
                    self.h2_skill_charges[sk.id] += 1
                    if self.h2_skill_charges[sk.id] < sk.max_charges and sk.cooldown_s > 0:
                        self.h2_skill_next_charge_at[sk.id] = charge_t + sk.cooldown_s
                    else:
                        self.h2_skill_next_charge_at[sk.id] = math.inf
            for rname, pool in self.h2_resolved_pools.items():
                if rname.lower() == "mp" and self.h2_kit.role != "tank":
                    continue
                if pool.regen_per_sec > 0.0:
                    self.h2_resources[rname] = min(pool.maximum, self.h2_resources.get(rname, 0.0) + pool.regen_per_sec * dt)
        super()._update_continuous_state(new_t)

    def _process_item(self, item: _ScheduledItem) -> None:
        if item.kind == "server_tick":
            # H2 3s MP tick + H2 Lucid ticks + H2 Fairy/Pet auto-heal BEFORE super() schedules next tick
            if "mp" in self.h2_resources:
                mp_max = self.h2_resolved_pools["mp"].maximum
                extra_mp = sum(
                    b.mp_per_tick for b in self.active_buffs
                    if b.expires_at >= self.t - EPS and b.mp_per_tick > 0 and b.owner == "h2"
                )
                self.h2_resources["mp"] = min(mp_max, self.h2_resources["mp"] + self.h2_base_mp_per_3s_tick + extra_mp)
            if self.h2_kit.pet_auto_heal_per_3s_tick > 0:
                lowest_ch = min(("mt", "st", "party"), key=lambda c: self.hp[c] / self.max_hp[c])
                if self.hp[lowest_ch] < self.max_hp[lowest_ch] - 1.0:
                    self._apply_heal(
                        lowest_ch,
                        self.h2_kit.pet_auto_heal_per_3s_tick * self.h2_heal_scale,
                        is_player_heal=True,
                        single_channel_only=True,
                        heal_owner="h2",
                    )
        super()._process_item(item)

    def h2_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.h2_n_actions, dtype=np.bool_)
        mask[0] = True
        if not self.timeline.can_party_act(self.t) or self.t < self.h2_casting_until - EPS:
            return mask
        gcd_ready = self.t >= self.h2_gcd_ready_at - EPS
        anim_ready = self.t >= self.h2_anim_lock_until - EPS
        boss_ok = self.timeline.is_boss_targetable(self.t)
        filler_mp = self.h2_kit.gear_stats.filler_gcd_mp_cost
        if gcd_ready and anim_ready and boss_ok and self.h2_resources.get("mp", 10_000.0) >= filler_mp - EPS:
            mask[1] = True
        for entry in self.h2_actions[2:]:
            sk = entry.skill
            assert sk is not None
            if not anim_ready or self.h2_skill_charges[sk.id] <= 0:
                continue
            if sk.needs_boss_target() and not boss_ok:
                continue
            if sk.is_gcd:
                if not gcd_ready:
                    continue
            else:
                if self.h2_weave_slots < sk.weave_cost:
                    continue
            if any(self.h2_resources.get(rn, 0.0) < rc - EPS for rn, rc in sk.resource_costs.items()):
                continue
            mask[entry.index] = True
        return mask

    def execute_h2_action(self, action_idx: int) -> None:
        mask = self.h2_action_mask()
        if action_idx < 0 or action_idx >= self.h2_n_actions or not mask[action_idx]:
            return
        entry = self.h2_actions[action_idx]
        if entry.kind == "filler":
            self.h2_gcd_ready_at = self.t + self.h2_kit.gcd_s
            self.h2_anim_lock_until = self.t + 0.65
            self.h2_weave_slots = self.h2_kit.weave_budget_at(self.t)
            f_mp = self.h2_kit.gear_stats.filler_gcd_mp_cost
            if f_mp > 0 and "mp" in self.h2_resources:
                self.h2_resources["mp"] = max(0.0, self.h2_resources["mp"] - f_mp)
                self.h2_stats["mp_min"] = min(self.h2_stats["mp_min"], self.h2_resources["mp"])
            if self.h2_kit.on_attack_heal_target != "none" and self.h2_kit.on_attack_heal_value > 0:
                self._apply_heal(
                    self.h2_kit.on_attack_heal_target,
                    self.h2_kit.on_attack_heal_value * self.h2_heal_scale,
                    is_player_heal=True,
                    heal_owner="h2",
                )
            for rn, pool in self.h2_resolved_pools.items():
                if pool.gain_per_filler_gcd > 0:
                    self.h2_resources[rn] = min(pool.maximum, self.h2_resources.get(rn, 0.0) + pool.gain_per_filler_gcd)
            self.h2_stats["filler_gcds"] += 1.0
            bm = self.h2_kit.burst_dps_multiplier if self.h2_kit.is_in_burst(self.t) else 1.0
            self.h2_stats["expected_potency_gained"] += self.h2_kit.filler_gcd_potency * bm
        elif entry.kind == "skill":
            sk = entry.skill
            assert sk is not None
            was_full = self.h2_skill_charges[sk.id] == sk.max_charges
            self.h2_skill_charges[sk.id] -= 1
            if was_full and sk.cooldown_s > 0.0:
                self.h2_skill_next_charge_at[sk.id] = self.t + sk.cooldown_s
            for rn, rc in sk.resource_costs.items():
                self.h2_resources[rn] = max(0.0, self.h2_resources.get(rn, 0.0) - rc)
            for rn, rg in sk.resource_grants.items():
                rmax = self.h2_resolved_pools[rn].maximum if rn in self.h2_resolved_pools else 10_000.0
                self.h2_resources[rn] = min(rmax, self.h2_resources.get(rn, 0.0) + rg)
            if "mp" in self.h2_resources:
                self.h2_stats["mp_min"] = min(self.h2_stats["mp_min"], self.h2_resources["mp"])
            dw = self.timeline.downtime_at(self.t)
            is_free_dt = False
            if sk.is_gcd:
                self.h2_gcd_ready_at = self.t + max(self.h2_kit.gcd_s, sk.cast_time_s)
                self.h2_weave_slots = self.h2_kit.weave_budget_at(self.t)
                if dw is not None and not dw.boss_targetable and self.h2_gcd_ready_at <= dw.end_s + EPS:
                    is_free_dt = True
                    self.h2_stats["downtime_free_gcds"] += 1.0
                else:
                    self.h2_stats["gcd_heals"] += 1.0
            else:
                self.h2_weave_slots = max(0, self.h2_weave_slots - sk.weave_cost)
                self.h2_stats["ogcds_used"] += 1.0

            intr_p = self._cast_overlaps_movement(self.t, sk)
            interrupted = (intr_p >= 0.5) if (intr_p > 0.0 and self.deterministic) else (intr_p > 0.0 and float(self.rng.random()) < intr_p)
            if interrupted:
                self.h2_casting_until = self.t + sk.cast_time_s * 0.5
                self.h2_anim_lock_until = self.h2_casting_until
                self.h2_stats["cast_interrupts"] = self.h2_stats.get("cast_interrupts", 0.0) + 1.0
            elif sk.cast_time_s > 0.0:
                if self.t + sk.cast_time_s > self.timeline.duration_s + EPS:
                    self.h2_stats["cast_interrupts"] = self.h2_stats.get("cast_interrupts", 0.0) + 1.0
                else:
                    self.h2_casting_until = self.t + sk.cast_time_s
                    self.h2_anim_lock_until = self.h2_casting_until + 0.1
                    self._push(
                        self.h2_casting_until,
                        PRIO_CAST_RESOLVE,
                        "cast_resolve",
                        {"skill": sk, "channel": entry.target_channel, "actor": "h2", "heal_scale": self.h2_heal_scale},
                    )
            else:
                self.h2_anim_lock_until = self.t + sk.animation_lock_s
                self._apply_skill_effects(sk, entry.target_channel, owner="h2", heal_scale_override=self.h2_heal_scale)

            pot_loss = self.h2_kit.expected_skill_potency_loss(sk, self.t, is_free_downtime=is_free_dt)
            self.h2_stats["expected_potency_lost"] += pot_loss
            res_pot = sum(
                rc * self.h2_resolved_pools[rn].dps_potency_per_unit
                for rn, rc in sk.resource_costs.items()
                if rn in self.h2_resolved_pools and self.h2_resolved_pools[rn].dps_potency_per_unit > 0.0
            )
            self.h2_stats["resource_potency_lost"] += res_pot

    def next_duo_decision_time(self, max_wait_s: float = 1.0) -> float:
        candidates = [self.next_decision_time(max_wait_s=max_wait_s)]
        for cand in (self.h2_gcd_ready_at, self.h2_anim_lock_until, self.h2_casting_until):
            if cand > self.t + EPS:
                candidates.append(cand)
        return min(candidates)


def replay_duo_plan(
    kit_h1: PlayerKit,
    kit_h2: PlayerKit,
    timeline: EncounterTimeline,
    plan_h1: OptimizationPlan,
    plan_h2: OptimizationPlan,
    *,
    deterministic: bool = True,
    fixed_tick_phase: float = 1.5,
    forced_damage_roll: float | None = None,
) -> DuoReplayResult:
    """Chronologically replay H1 and H2 plans together through the shared DualHealerEngine."""
    eng = DualHealerEngine(
        kit_h1=kit_h1,
        kit_h2=kit_h2,
        timeline=timeline,
        damage_variance=0.0 if deterministic else 0.05,
        partner_trust_scale=min(plan_h1.partner_trust, plan_h2.partner_trust),
        forced_damage_roll=forced_damage_roll,
        expected_partners=True,
    )
    eng.reset(deterministic=deterministic, fixed_tick_phase=fixed_tick_phase)

    def _build_map(actions_list):
        by_key: dict[tuple[str, str], int] = {}
        by_sk: dict[str, int] = {}
        for e in actions_list[2:]:
            assert e.skill is not None
            by_key[(e.skill.id, e.target_channel)] = e.index
            by_sk.setdefault(e.skill.id, e.index)
        return by_key, by_sk

    h1_key, h1_sk = _build_map(eng.actions)
    h2_key, h2_sk = _build_map(eng.h2_actions)
    h1_skills_by_id = {s.id: s for s in kit_h1.skills}
    h2_skills_by_id = {s.id: s for s in kit_h2.skills}

    s1 = sorted(plan_h1.actions, key=lambda a: (a.cast_at_s, 0 if a.is_gcd else 1))
    s2 = sorted(plan_h2.actions, key=lambda a: (a.cast_at_s, 0 if a.is_gcd else 1))
    p1 = p2 = 0
    ex1 = ex2 = dr1 = dr2 = 0

    while eng.t < timeline.duration_s - EPS and not eng.died:
        acted = False
        # H1 scheduled actions
        while p1 < len(s1) and s1[p1].cast_at_s <= eng.t + EPS:
            act = s1[p1]
            aidx = h1_key.get((act.skill_id, act.target_channel), h1_sk.get(act.skill_id, 0))
            sk = h1_skills_by_id.get(act.skill_id)
            if aidx < 2 or sk is None or eng.t > act.cast_at_s + 2.5:
                p1 += 1
                dr1 += 1
                continue
            mask1 = eng.action_mask()
            if mask1[aidx]:
                eng.execute_action(aidx)
                p1 += 1
                ex1 += 1
                acted = True
            else:
                if sk.needs_boss_target() and not timeline.is_boss_targetable(eng.t):
                    p1 += 1
                    dr1 += 1
                    continue
                break

        if eng.died:
            break

        # H2 scheduled actions
        while p2 < len(s2) and s2[p2].cast_at_s <= eng.t + EPS:
            act = s2[p2]
            aidx = h2_key.get((act.skill_id, act.target_channel), h2_sk.get(act.skill_id, 0))
            sk = h2_skills_by_id.get(act.skill_id)
            if aidx < 2 or sk is None or eng.t > act.cast_at_s + 2.5:
                p2 += 1
                dr2 += 1
                continue
            mask2 = eng.h2_action_mask()
            if mask2[aidx]:
                eng.execute_h2_action(aidx)
                p2 += 1
                ex2 += 1
                acted = True
            else:
                if sk.needs_boss_target() and not timeline.is_boss_targetable(eng.t):
                    p2 += 1
                    dr2 += 1
                    continue
                break

        if eng.died:
            break

        # Filler attack GCDs for both H1 and H2 when ready and not colliding with imminent planned GCD
        m1 = eng.action_mask()
        if m1[1]:
            imm1 = p1 < len(s1) and s1[p1].is_gcd and s1[p1].cast_at_s < eng.t + kit_h1.gcd_s - EPS
            if not imm1:
                eng.execute_action(1)
                acted = True
        m2 = eng.h2_action_mask()
        if m2[1]:
            imm2 = p2 < len(s2) and s2[p2].is_gcd and s2[p2].cast_at_s < eng.t + kit_h2.gcd_s - EPS
            if not imm2:
                eng.execute_h2_action(1)
                acted = True

        nxt = eng.next_duo_decision_time(max_wait_s=1.0)
        if p1 < len(s1) and s1[p1].cast_at_s > eng.t + EPS:
            nxt = min(nxt, s1[p1].cast_at_s)
        if p2 < len(s2) and s2[p2].cast_at_s > eng.t + EPS:
            nxt = min(nxt, s2[p2].cast_at_s)
        if nxt <= eng.t + EPS and not acted:
            nxt = min(timeline.duration_s, eng.t + 0.05)
        eng.advance_to(nxt)

    if not eng.died:
        if p1 < len(s1):
            dr1 += len(s1) - p1
        if p2 < len(s2):
            dr2 += len(s2) - p2
    # Movement-interrupted or post-fight-end unresolved casts count as dropped/failed executions
    dr1 += int(eng.stats.get("cast_interrupts", 0.0))
    dr2 += int(eng.h2_stats.get("cast_interrupts", 0.0))

    def _pot_and_oh(st: dict[str, float]) -> tuple[float, float, float, float, float]:
        raw_h = st.get("player_heal_raw", 0.0)
        eff_h = st.get("player_heal_eff", 0.0)
        oh = ((raw_h - eff_h) / raw_h) * 100.0 if raw_h > EPS else 0.0
        net_p = max(0.0, st.get("expected_potency_gained", 0.0) - st.get("resource_potency_lost", 0.0))
        loss_p = st.get("expected_potency_lost", 0.0)
        return net_p, loss_p, oh, raw_h, eff_h

    h1_pot, h1_loss, h1_oh, h1_raw, h1_eff = _pot_and_oh(eng.stats)
    h2_pot, h2_loss, h2_oh, h2_raw, h2_eff = _pot_and_oh(eng.h2_stats)
    comb_pot = h1_pot + h2_pot
    comb_loss = h1_loss + h2_loss
    comb_dps_eff = (comb_pot / max(1.0, comb_pot + comb_loss)) * 100.0 if (comb_pot + comb_loss) > EPS else 100.0
    tot_raw = h1_raw + h2_raw
    tot_eff = h1_eff + h2_eff
    comb_oh = ((tot_raw - tot_eff) / tot_raw) * 100.0 if tot_raw > EPS else 0.0
    h1_share = (h1_eff / tot_eff) * 100.0 if tot_eff > EPS else 50.0
    h2_share = (h2_eff / tot_eff) * 100.0 if tot_eff > EPS else 50.0

    return DuoReplayResult(
        survived=not eng.died,
        first_death_time_s=eng.first_death_time_s,
        first_death_channel=eng.first_death_channel,
        deficit_hp=round(eng.max_deficit_hp, 1),
        min_hp_ratio={
            "party": round(eng.stats["min_party_hp_ratio"], 4),
            "mt": round(eng.stats["min_mt_hp_ratio"], 4),
            "st": round(eng.stats["min_st_hp_ratio"], 4),
        },
        h1_filler_gcds=int(eng.stats["filler_gcds"]),
        h2_filler_gcds=int(eng.h2_stats["filler_gcds"]),
        h1_gcd_heals=int(eng.stats["gcd_heals"]),
        h2_gcd_heals=int(eng.h2_stats["gcd_heals"]),
        h1_expected_potency=round(h1_pot, 1),
        h2_expected_potency=round(h2_pot, 1),
        combined_expected_potency=round(comb_pot, 1),
        h1_potency_lost=round(h1_loss, 1),
        h2_potency_lost=round(h2_loss, 1),
        combined_dps_efficiency_pct=round(comb_dps_eff, 2),
        h1_overheal_pct=round(h1_oh, 1),
        h2_overheal_pct=round(h2_oh, 1),
        combined_overheal_pct=round(comb_oh, 1),
        h1_effective_heal_share_pct=round(h1_share, 1),
        h2_effective_heal_share_pct=round(h2_share, 1),
        h1_mp_min=round(eng.stats["mp_min"], 1),
        h2_mp_min=round(eng.h2_stats["mp_min"], 1),
        h1_executed_actions=ex1,
        h2_executed_actions=ex2,
        h1_dropped_actions=dr1,
        h2_dropped_actions=dr2,
        hp_triggers_fired=int(eng.stats.get("hp_triggers_fired", 0.0)),
        event_snapshots=list(eng.event_snapshots),
    )


def stress_test_duo_plan(
    kit_h1: PlayerKit,
    kit_h2: PlayerKit,
    timeline: EncounterTimeline,
    plan_h1: OptimizationPlan,
    plan_h2: OptimizationPlan,
    *,
    damage_roll: float = 1.05,
    n_phases: int = 12,
) -> DuoStressReport:
    """Verify an H1+H2 duo plan across all server-tick phases under worst-case damage roll."""
    rolls: tuple[float, ...] = (
        tuple(dict.fromkeys((0.95, 1.00, damage_roll)))
        if timeline.hp_trigger_mechanics
        else (damage_roll,)
    )
    total = n_phases * len(rolls)
    surv = 0
    worst_hp = {"party": 1.0, "mt": 1.0, "st": 1.0}
    w_fail_t: float | None = None
    w_fail_ch: str | None = None
    w_fail_phase = 0.0
    w_def = 0.0
    h1_pots, h2_pots, c_pots, c_losses, c_effs = [], [], [], [], []
    h1_gcds, h2_gcds, h1_fil, h2_fil = [], [], [], []
    h1_ohs, h2_ohs, c_ohs, h1_sh, h2_sh = [], [], [], [], []
    m_dr1 = m_dr2 = trig_max = 0

    for roll in rolls:
        for i in range(n_phases):
            phase = (3.0 * i) / max(1, n_phases)
            res = replay_duo_plan(
                kit_h1, kit_h2, timeline, plan_h1, plan_h2,
                deterministic=True, fixed_tick_phase=phase, forced_damage_roll=roll,
            )
            if res.survived:
                surv += 1
            else:
                ft = res.first_death_time_s if res.first_death_time_s is not None else 0.0
                if w_fail_t is None or ft < w_fail_t or (abs(ft - w_fail_t) < 1e-3 and res.deficit_hp > w_def):
                    w_fail_t = ft
                    w_fail_ch = res.first_death_channel
                    w_fail_phase = phase
                    w_def = res.deficit_hp
            for ch in ("party", "mt", "st"):
                worst_hp[ch] = min(worst_hp[ch], res.min_hp_ratio.get(ch, 1.0))
            h1_pots.append(res.h1_expected_potency)
            h2_pots.append(res.h2_expected_potency)
            c_pots.append(res.combined_expected_potency)
            c_losses.append(res.h1_potency_lost + res.h2_potency_lost)
            c_effs.append(res.combined_dps_efficiency_pct)
            h1_gcds.append(float(res.h1_gcd_heals))
            h2_gcds.append(float(res.h2_gcd_heals))
            h1_fil.append(float(res.h1_filler_gcds))
            h2_fil.append(float(res.h2_filler_gcds))
            h1_ohs.append(res.h1_overheal_pct)
            h2_ohs.append(res.h2_overheal_pct)
            c_ohs.append(res.combined_overheal_pct)
            h1_sh.append(res.h1_effective_heal_share_pct)
            h2_sh.append(res.h2_effective_heal_share_pct)
            m_dr1 = max(m_dr1, res.h1_dropped_actions)
            m_dr2 = max(m_dr2, res.h2_dropped_actions)
            trig_max = max(trig_max, res.hp_triggers_fired)

    n = max(1, total)
    return DuoStressReport(
        damage_roll=damage_roll,
        total_phases=total,
        survived_phases=surv,
        survival_rate_pct=round((surv / n) * 100.0, 2),
        worst_min_hp_ratio={k: round(v, 4) for k, v in worst_hp.items()},
        worst_failure_time_s=w_fail_t,
        worst_failure_channel=w_fail_ch,
        worst_failure_phase=w_fail_phase,
        worst_deficit_hp=round(w_def, 1),
        mean_h1_expected_potency=round(sum(h1_pots) / n, 1),
        mean_h2_expected_potency=round(sum(h2_pots) / n, 1),
        mean_combined_expected_potency=round(sum(c_pots) / n, 1),
        mean_combined_potency_lost=round(sum(c_losses) / n, 1),
        mean_combined_dps_efficiency_pct=round(sum(c_effs) / n, 2),
        mean_h1_gcd_heals=round(sum(h1_gcds) / n, 2),
        mean_h2_gcd_heals=round(sum(h2_gcds) / n, 2),
        mean_h1_filler_gcds=round(sum(h1_fil) / n, 1),
        mean_h2_filler_gcds=round(sum(h2_fil) / n, 1),
        mean_h1_overheal_pct=round(sum(h1_ohs) / n, 1),
        mean_h2_overheal_pct=round(sum(h2_ohs) / n, 1),
        mean_combined_overheal_pct=round(sum(c_ohs) / n, 1),
        mean_h1_heal_share_pct=round(sum(h1_sh) / n, 1),
        mean_h2_heal_share_pct=round(sum(h2_sh) / n, 1),
        max_h1_dropped=m_dr1,
        max_h2_dropped=m_dr2,
        hp_triggers_fired=trig_max,
    )


def harden_duo_to_100pct(
    kit_h1: PlayerKit,
    kit_h2: PlayerKit,
    timeline: EncounterTimeline,
    plan_h1: OptimizationPlan,
    plan_h2: OptimizationPlan,
    *,
    damage_roll: float = 1.05,
    n_phases: int = 12,
    max_repairs: int = 90,
    min_hp_margin_ratio: float = 0.0,
    allowed_actors: tuple[str, ...] = ("h1", "h2"),
) -> tuple[OptimizationPlan, OptimizationPlan, DuoStressReport, int]:
    """Iteratively add the lowest combined-DPS-cost legal skill from `allowed_actors` until 100.0% survival
    (and optional `min_hp_margin_ratio` floor). Uses fast 2-phase screening before 12-phase certification!
    """
    cur1 = plan_h1.model_copy(deep=True)
    cur2 = plan_h2.model_copy(deep=True)
    rep = stress_test_duo_plan(kit_h1, kit_h2, timeline, cur1, cur2, damage_roll=damage_roll, n_phases=n_phases)

    def _goal_met(r: DuoStressReport) -> bool:
        if r.survival_rate_pct < 100.0:
            return False
        if min_hp_margin_ratio > 0.0 and not any(e.mechanic_effect == "set_hp_to_1" for e in timeline.events):
            return min(r.worst_min_hp_ratio.values()) >= min_hp_margin_ratio - 1e-4
        return True

    if _goal_met(rep):
        return cur1, cur2, rep, 0

    events_sorted = sorted(timeline.events, key=lambda e: e.time_s)
    pool: list[tuple[str, PlayerKit, SkillDef]] = []
    if "h1" in allowed_actors:
        pool.extend(("h1", kit_h1, sk) for sk in kit_h1.skills)
    if "h2" in allowed_actors:
        pool.extend(("h2", kit_h2, sk) for sk in kit_h2.skills)
    pool.sort(
        key=lambda item: (
            item[1].expected_skill_potency_loss(item[2], 0.0, False),
            1 if item[2].is_gcd else 0,
            item[2].dps_cost,
            item[2].cooldown_s,
        )
    )

    repairs = 0
    for _ in range(max_repairs):
        if _goal_met(rep):
            break
        # Identify either the earliest death point OR (when 100% survived) the tightest sub-margin event!
        margin_mode = rep.survival_rate_pct >= 100.0
        if margin_mode:
            snap_res = replay_duo_plan(
                kit_h1, kit_h2, timeline, cur1, cur2,
                deterministic=True, fixed_tick_phase=0.0, forced_damage_roll=damage_roll,
            )
            worst_ev_t = timeline.duration_s * 0.5
            worst_ev_ch = "party"
            worst_ev_r = 1.0
            for snap in snap_res.event_snapshots:
                for ch, max_v in (("party", max(kit_h1.max_hp_party, kit_h2.max_hp_party)), ("mt", max(kit_h1.max_hp_tank, kit_h2.max_hp_tank)), ("st", max(kit_h1.max_hp_tank, kit_h2.max_hp_tank))):
                    r_val = snap.hp_after.get(ch, max_v) / max_v
                    if r_val < worst_ev_r:
                        worst_ev_r = r_val
                        worst_ev_t = snap.time_s
                        worst_ev_ch = ch
            fail_t = worst_ev_t
            fail_ch = worst_ev_ch
            is_esuna = False
        else:
            if rep.worst_failure_time_s is None:
                break
            fail_t = rep.worst_failure_time_s
            raw_ch = rep.worst_failure_channel or "party"
            fail_ch = raw_ch.split("(")[0]
            is_esuna = "esuna_doom" in raw_ch

        # Fast screen on the EXACT failing phase (`rep.worst_failure_phase`), then certify top-5 on all `n_phases`!
        shortlist: list[tuple[tuple[float, float, float, float], OptimizationPlan, OptimizationPlan]] = []
        offsets = (1.5, 4.0, 8.0, 12.0) if not is_esuna else (4.0, 7.0)
        base_drops = rep.max_dropped_actions
        base_min_hp = min(rep.worst_min_hp_ratio.values())
        screen_phase = rep.worst_failure_phase if not margin_mode else 0.0

        for who, k_who, sk in pool:
            if is_esuna:
                if not sk.cleanses_debuff:
                    continue
                t_ch = fail_ch if fail_ch in ("mt", "st") else "party"
            else:
                helps, t_ch = _skill_helps_channel(sk, fail_ch, k_who)
                if not helps:
                    continue
            target_plan = cur1 if who == "h1" else cur2
            for off in offsets:
                cast_t = round(max(0.5, fail_t - off), 2)
                if sk.needs_boss_target() and not timeline.is_boss_targetable(cast_t):
                    continue
                if any(a.skill_id == sk.id and abs(a.cast_at_s - cast_t) < 2.5 for a in target_plan.actions):
                    continue
                nxt = next((e for e in events_sorted if e.time_s >= cast_t - 0.5), None)
                new_act = PlannedAction(
                    event_id=nxt.id if nxt is not None else f"duo_{repairs}",
                    event_name=nxt.name if nxt is not None else "Safety Buffer",
                    event_time_s=nxt.time_s if nxt is not None else round(fail_t, 2),
                    target_scope="boss_debuff" if t_ch == "boss" else t_ch,  # type: ignore[arg-type]
                    cast_at_s=cast_t,
                    skill_id=sk.id,
                    skill_name=sk.name,
                    action_id=sk.action_id,
                    target_channel=t_ch,  # type: ignore[arg-type]
                    is_gcd=sk.is_gcd,
                    weave_cost=sk.weave_cost,
                )
                trial_acts = sorted(target_plan.actions + [new_act], key=lambda a: (a.cast_at_s, 0 if a.is_gcd else 1))
                t1 = cur1.model_copy(update={"actions": trial_acts}) if who == "h1" else cur1
                t2 = cur2.model_copy(update={"actions": trial_acts}) if who == "h2" else cur2

                q0 = replay_duo_plan(
                    kit_h1, kit_h2, timeline, t1, t2,
                    deterministic=True, fixed_tick_phase=screen_phase, forced_damage_roll=damage_roll,
                )
                if (q0.h1_dropped_actions + q0.h2_dropped_actions) > base_drops:
                    continue
                if margin_mode:
                    q_min = min(q0.min_hp_ratio.values())
                    if not q0.survived or q_min <= base_min_hp + 0.002:
                        continue
                    pot_loss = _effective_potency_loss(k_who, timeline, sk, cast_t)
                    share_dev = abs(q0.h1_effective_heal_share_pct - 50.0) * 0.25 if len(allowed_actors) > 1 else 0.0
                    s_key = (1.0 if q_min >= min_hp_margin_ratio else 0.0, q_min, -pot_loss - sk.dps_cost * 10.0 - share_dev, 0.0)
                else:
                    q_prog = (timeline.duration_s + 100.0) if q0.survived else (q0.first_death_time_s or 0.0)
                    cleared = 1.0 if q_prog > fail_t + 0.5 else 0.0
                    if cleared == 0.0 and q0.deficit_hp >= rep.worst_deficit_hp - 1_500.0:
                        continue
                    pot_loss = _effective_potency_loss(k_who, timeline, sk, cast_t)
                    share_dev = abs(q0.h1_effective_heal_share_pct - 50.0) * 0.25 if len(allowed_actors) > 1 else 0.0
                    s_key = (cleared, q_prog, -q0.deficit_hp * 1e-4, -pot_loss - sk.dps_cost * 10.0 - share_dev)
                shortlist.append((s_key, t1, t2))

        if not shortlist:
            break
        shortlist.sort(key=lambda x: x[0], reverse=True)
        best_choice: tuple[tuple[float, float, float, float], OptimizationPlan, OptimizationPlan, DuoStressReport] | None = None
        checked_full = 0
        for s_key, t1, t2 in shortlist[:24]:
            t_rep = stress_test_duo_plan(kit_h1, kit_h2, timeline, t1, t2, damage_roll=damage_roll, n_phases=n_phases)
            if t_rep.max_dropped_actions > base_drops:
                continue
            checked_full += 1
            if margin_mode:
                if t_rep.survival_rate_pct < 100.0 or min(t_rep.worst_min_hp_ratio.values()) <= base_min_hp + 0.001:
                    continue
                key = (
                    1.0 if _goal_met(t_rep) else 0.0,
                    min(t_rep.worst_min_hp_ratio.values()),
                    t_rep.mean_combined_expected_potency,
                    -abs(t_rep.mean_h1_heal_share_pct - 50.0) if len(allowed_actors) > 1 else 0.0,
                )
            else:
                improved = (
                    t_rep.survival_rate_pct > rep.survival_rate_pct
                    or (t_rep.worst_failure_time_s is not None and t_rep.worst_failure_time_s > fail_t + 0.5)
                    or (
                        t_rep.worst_failure_time_s is not None
                        and abs(t_rep.worst_failure_time_s - fail_t) <= 0.5
                        and t_rep.worst_deficit_hp < rep.worst_deficit_hp - 1_500.0
                    )
                )
                if not improved:
                    continue
                cleared_here = 1.0 if (t_rep.worst_failure_time_s is None or t_rep.worst_failure_time_s > fail_t + 0.5) else 0.0
                prog_t = t_rep.worst_failure_time_s if t_rep.worst_failure_time_s is not None else (timeline.duration_s + 100.0)
                key = (cleared_here, t_rep.survival_rate_pct, prog_t, t_rep.mean_combined_expected_potency)
            if best_choice is None or key > best_choice[0]:
                best_choice = (key, t1, t2, t_rep)
            # Early exit once we have evaluated >=5 clean candidates AND already cleared this failure point cleanly!
            if checked_full >= 5 and best_choice is not None and best_choice[0][0] >= 1.0:
                break

        if best_choice is None:
            break
        _, cur1, cur2, rep = best_choice
        repairs += 1

    return cur1, cur2, rep, repairs


def maximize_duo_dps_at_100pct_survival(
    kit_h1: PlayerKit,
    kit_h2: PlayerKit,
    timeline: EncounterTimeline,
    plan_h1: OptimizationPlan,
    plan_h2: OptimizationPlan,
    *,
    damage_roll: float = 1.05,
    n_phases: int = 12,
    min_hp_margin_ratio: float = 0.05,
    allowed_actors: tuple[str, ...] = ("h1", "h2"),
) -> tuple[OptimizationPlan, OptimizationPlan, DuoStressReport, dict[str, int]]:
    """Co-optimize `allowed_actors` for maximum combined Expected DPS subject to strict 100.0% survival & 0 drops."""
    cur1 = canonicalize_plan_actions(kit_h1, timeline, plan_h1)
    cur2 = canonicalize_plan_actions(kit_h2, timeline, plan_h2)
    rep = stress_test_duo_plan(kit_h1, kit_h2, timeline, cur1, cur2, damage_roll=damage_roll, n_phases=n_phases)
    stats = {"ghosts_removed": 0, "pruned": 0, "cross_swapped": 0, "burst_shifted": 0}
    if rep.survival_rate_pct < 100.0:
        return cur1, cur2, rep, stats

    has_hp1 = any(e.mechanic_effect == "set_hp_to_1" for e in timeline.events)
    eff_margin = (
        min(min_hp_margin_ratio, min(rep.worst_min_hp_ratio.values()))
        if (min_hp_margin_ratio > 0.0 and not has_hp1)
        else 0.0
    )

    def _is_safe(r: DuoStressReport, max_drops: int = 0) -> bool:
        if r.survival_rate_pct < 100.0 or r.max_dropped_actions > max_drops:
            return False
        if eff_margin > 0.0:
            return min(r.worst_min_hp_ratio.values()) >= eff_margin - 1e-4
        return True

    def _score(r: DuoStressReport) -> tuple[float, float, float]:
        return (r.mean_combined_expected_potency, -r.mean_combined_potency_lost, -r.mean_combined_overheal_pct)

    # Pass 0: strip any ghost/dropped casts from allowed actors
    if rep.max_dropped_actions > 0:
        cleaned = True
        while cleaned and rep.max_dropped_actions > 0:
            cleaned = False
            for who in allowed_actors:
                acts = cur1.actions if who == "h1" else cur2.actions
                for idx in reversed(range(len(acts))):
                    t_acts = acts[:idx] + acts[idx + 1 :]
                    t1 = cur1.model_copy(update={"actions": t_acts}) if who == "h1" else cur1
                    t2 = cur2.model_copy(update={"actions": t_acts}) if who == "h2" else cur2
                    t_rep = stress_test_duo_plan(kit_h1, kit_h2, timeline, t1, t2, damage_roll=damage_roll, n_phases=n_phases)
                    if (
                        t_rep.survival_rate_pct >= 100.0
                        and t_rep.max_dropped_actions < rep.max_dropped_actions
                        and (eff_margin <= 0.0 or min(t_rep.worst_min_hp_ratio.values()) >= eff_margin - 1e-4)
                    ):
                        cur1, cur2, rep = t1, t2, t_rep
                        stats["ghosts_removed"] += 1
                        cleaned = True
                        break
                if cleaned:
                    break

    allowed_drops = rep.max_dropped_actions
    sk1_map = {s.id: s for s in kit_h1.skills}
    sk2_map = {s.id: s for s in kit_h2.skills}

    # Pass 1: Single priority-ordered prune pass (highest potency loss first, fast 2-phase pre-screen -> 12-phase verify)
    def _quick_pass(t1: OptimizationPlan, t2: OptimizationPlan) -> bool:
        for ph in (0.0, 1.5):
            q = replay_duo_plan(kit_h1, kit_h2, timeline, t1, t2, deterministic=True, fixed_tick_phase=ph, forced_damage_roll=damage_roll)
            if not q.survived or (q.h1_dropped_actions + q.h2_dropped_actions) > allowed_drops:
                return False
            if eff_margin > 0.0 and min(q.min_hp_ratio.values()) < eff_margin - 1e-4:
                return False
        return True

    prune_candidates: list[tuple[float, int, str, float, str]] = []
    if "h1" in allowed_actors:
        for act in cur1.actions:
            sk = sk1_map.get(act.skill_id)
            if sk is None or sk.mp_regen_per_tick > 0:
                continue
            loss = _effective_potency_loss(kit_h1, timeline, sk, act.cast_at_s)
            prune_candidates.append((loss, 1 if act.is_gcd else 0, "h1", act.cast_at_s, act.skill_id))
    if "h2" in allowed_actors:
        for act in cur2.actions:
            sk = sk2_map.get(act.skill_id)
            if sk is None or sk.mp_regen_per_tick > 0:
                continue
            loss = _effective_potency_loss(kit_h2, timeline, sk, act.cast_at_s)
            prune_candidates.append((loss, 1 if act.is_gcd else 0, "h2", act.cast_at_s, act.skill_id))
    prune_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for _, _, who, c_time, c_sid in prune_candidates:
        acts = cur1.actions if who == "h1" else cur2.actions
        idx = next((i for i, a in enumerate(acts) if a.skill_id == c_sid and abs(a.cast_at_s - c_time) < 1e-3), None)
        if idx is None:
            continue
        t_acts = acts[:idx] + acts[idx + 1 :]
        t1 = cur1.model_copy(update={"actions": t_acts}) if who == "h1" else cur1
        t2 = cur2.model_copy(update={"actions": t_acts}) if who == "h2" else cur2
        if not _quick_pass(t1, t2):
            continue
        t_rep = stress_test_duo_plan(kit_h1, kit_h2, timeline, t1, t2, damage_roll=damage_roll, n_phases=n_phases)
        if _is_safe(t_rep, allowed_drops) and _score(t_rep) > _score(rep):
            cur1, cur2, rep = t1, t2, t_rep
            allowed_drops = min(allowed_drops, rep.max_dropped_actions)
            stats["pruned"] += 1

    # Pass 2: Single-pass Cross-Healer Downgrade (replace >0p skill with strictly cheaper legal skill on any allowed actor)
    cheaper_all: list[tuple[str, PlayerKit, SkillDef]] = []
    if "h1" in allowed_actors:
        cheaper_all.extend(("h1", kit_h1, s) for s in kit_h1.skills if s.mp_regen_per_tick <= 0)
    if "h2" in allowed_actors:
        cheaper_all.extend(("h2", kit_h2, s) for s in kit_h2.skills if s.mp_regen_per_tick <= 0)
    cheaper_all.sort(key=lambda it: (it[1].expected_skill_potency_loss(it[2], 0.0, False), it[2].dps_cost))

    src_pairs = []
    if "h1" in allowed_actors:
        src_pairs.append(("h1", kit_h1, sk1_map))
    if "h2" in allowed_actors:
        src_pairs.append(("h2", kit_h2, sk2_map))
    for src_who, src_kit, src_map in src_pairs:
        # Snapshot costly actions by (cast_at_s, skill_id) so we iterate once without O(N^2) restarts
        costly = [
            (a.cast_at_s, a.skill_id)
            for a in (cur1.actions if src_who == "h1" else cur2.actions)
            if a.skill_id in src_map and _effective_potency_loss(src_kit, timeline, src_map[a.skill_id], a.cast_at_s) > 1e-6
        ]
        for c_time, c_sid in costly:
            src_acts = cur1.actions if src_who == "h1" else cur2.actions
            idx = next((i for i, a in enumerate(src_acts) if a.skill_id == c_sid and abs(a.cast_at_s - c_time) < 1e-3), None)
            if idx is None:
                continue
            act = src_acts[idx]
            sk_old = src_map[c_sid]
            old_loss = _effective_potency_loss(src_kit, timeline, sk_old, act.cast_at_s)
            for dst_who, dst_kit, sk_new in cheaper_all:
                if dst_who == src_who and sk_new.id == sk_old.id:
                    continue
                new_loss = _effective_potency_loss(dst_kit, timeline, sk_new, act.cast_at_s)
                if new_loss >= old_loss - 1.0:
                    continue
                helps, t_ch = _skill_helps_channel(sk_new, act.target_channel, dst_kit)
                if not helps or (sk_new.needs_boss_target() and not timeline.is_boss_targetable(act.cast_at_s)):
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
                if dst_who == src_who:
                    t_src = src_acts[:idx] + [new_act] + src_acts[idx + 1 :]
                    t1 = cur1.model_copy(update={"actions": t_src}) if src_who == "h1" else cur1
                    t2 = cur2.model_copy(update={"actions": t_src}) if src_who == "h2" else cur2
                else:
                    t_src = src_acts[:idx] + src_acts[idx + 1 :]
                    dst_acts = cur1.actions if dst_who == "h1" else cur2.actions
                    t_dst = sorted(dst_acts + [new_act], key=lambda a: (a.cast_at_s, 0 if a.is_gcd else 1))
                    t1 = cur1.model_copy(update={"actions": t_src if src_who == "h1" else t_dst})
                    t2 = cur2.model_copy(update={"actions": t_src if src_who == "h2" else t_dst})
                if not _quick_pass(t1, t2):
                    continue
                t_rep = stress_test_duo_plan(kit_h1, kit_h2, timeline, t1, t2, damage_roll=damage_roll, n_phases=n_phases)
                if _is_safe(t_rep, allowed_drops) and _score(t_rep) > _score(rep):
                    cur1, cur2, rep = t1, t2, t_rep
                    stats["cross_swapped"] += 1
                    break

    # Final canonicalization + exact replay certification
    f1 = canonicalize_plan_actions(kit_h1, timeline, cur1)
    f2 = canonicalize_plan_actions(kit_h2, timeline, cur2)
    f_rep = stress_test_duo_plan(kit_h1, kit_h2, timeline, f1, f2, damage_roll=damage_roll, n_phases=n_phases)
    margin_ok = (min_hp_margin_ratio <= 0.0 or has_hp1 or min(f_rep.worst_min_hp_ratio.values()) >= min_hp_margin_ratio - 1e-4)
    status = (
        "verified_100pct_duo_max_dps"
        if (f_rep.survival_rate_pct >= 100.0 and f_rep.max_dropped_actions == 0 and margin_ok)
        else "unverified_duo"
    )
    return f1.model_copy(update={"status": status}), f2.model_copy(update={"status": status}), f_rep, stats


def solve_duo_healwork(
    kit_h1: PlayerKit,
    kit_h2: PlayerKit,
    timeline: EncounterTimeline,
    *,
    seed_h1_actions: list[dict[str, Any]] | None = None,
    seed_h2_actions: list[dict[str, Any]] | None = None,
    damage_roll: float = 1.05,
    n_phases: int = 12,
    min_hp_margin_ratio: float = 0.05,
    max_repairs: int = 90,
    strip_averaged_cohealer: bool = True,
    tank_empirical_heal_retain_ratio: float = 0.55,
) -> tuple[OptimizationPlan, OptimizationPlan, DuoStressReport, EncounterTimeline, dict[str, int]]:
    """Mode B entry point: jointly solve & certify a unified H1 + H2 healwork sheet."""
    duo_tl = (
        strip_cohealer_from_timeline(timeline, tank_empirical_heal_retain_ratio=tank_empirical_heal_retain_ratio)
        if strip_averaged_cohealer
        else timeline.model_copy(deep=True)
    )

    def _default_seed(kit: PlayerKit, extra: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if extra is not None:
            return list(extra)
        out: list[dict[str, Any]] = []
        if any(s.id == "lucid_dreaming" for s in kit.skills):
            for t in range(30, int(duo_tl.duration_s), 60):
                out.append({"cast_at_s": float(t), "skill_id": "lucid_dreaming", "target_channel": "party"})
        # Seed core 0p DPS-positive AoE heals (e.g. WHM Assize 40s, AST Earthly Star 60s) respecting pre-pull CD!
        prepull_cd_end = {
            pp.skill_id: max(0.0, next((s.cooldown_s for s in kit.skills if s.id == pp.skill_id), 0.0) + pp.offset_s)
            for pp in kit.prepull_actions
            if pp.consumes_cooldown
        }
        for sk in kit.skills:
            if sk.id in ("assize", "earthly_star") and sk.cooldown_s >= 30.0:
                t = max(12.0, prepull_cd_end.get(sk.id, 0.0) + 2.0)
                while t < duo_tl.duration_s - 5.0:
                    out.append({"cast_at_s": round(t, 1), "skill_id": sk.id, "target_channel": "party"})
                    t += sk.cooldown_s + 0.5
        return out

    p1 = action_log_to_plan(kit_h1, duo_tl, _default_seed(kit_h1, seed_h1_actions), partner_trust=1.0, status="duo_seed")
    p2 = action_log_to_plan(kit_h2, duo_tl, _default_seed(kit_h2, seed_h2_actions), partner_trust=1.0, status="duo_seed")
    # Stage 1: Fast coarse hardening (4 phases) -> Stage 2: Exact full-phase certification & polish (`n_phases`)
    coarse_phases = min(4, n_phases)
    h1_plan, h2_plan, hrep, r_coarse = harden_duo_to_100pct(
        kit_h1, kit_h2, duo_tl, p1, p2,
        damage_roll=damage_roll, n_phases=coarse_phases, max_repairs=max_repairs,
        min_hp_margin_ratio=min_hp_margin_ratio,
    )
    repairs = r_coarse
    if n_phases > coarse_phases:
        h1_plan, h2_plan, hrep, r_fine = harden_duo_to_100pct(
            kit_h1, kit_h2, duo_tl, h1_plan, h2_plan,
            damage_roll=damage_roll, n_phases=n_phases, max_repairs=max(12, max_repairs - r_coarse),
            min_hp_margin_ratio=min_hp_margin_ratio,
        )
        repairs += r_fine
    if hrep.survival_rate_pct < 100.0:
        return h1_plan, h2_plan, hrep, duo_tl, {"repairs_added": repairs, "ghosts_removed": 0, "pruned": 0, "cross_swapped": 0, "burst_shifted": 0}

    o1, o2, orep, stats = maximize_duo_dps_at_100pct_survival(
        kit_h1, kit_h2, duo_tl, h1_plan, h2_plan,
        damage_roll=damage_roll, n_phases=n_phases, min_hp_margin_ratio=min_hp_margin_ratio,
    )
    stats = {"repairs_added": repairs, **stats}
    return o1, o2, orep, duo_tl, stats


def export_duo_markdown(
    kit_h1: PlayerKit,
    kit_h2: PlayerKit,
    timeline: EncounterTimeline,
    plan_h1: OptimizationPlan,
    plan_h2: OptimizationPlan,
    report: DuoStressReport,
    output_path: str | Path | None = None,
) -> str:
    """Render a side-by-side unified H1 + H2 Markdown healwork sheet."""
    ilvl_s = f"iLvl {kit_h1.gear_stats.item_level}" if kit_h1.gear_stats.item_level else "custom"
    lines = [
        f"# {timeline.encounter_name} — H1 + H2 ペア同時最適化ヒールワーク ({ilvl_s})",
        "",
        f"- **H1**: {kit_h1.label}",
        f"- **H2**: {kit_h2.label}",
        f"- **最悪+5%乱数 × {report.total_phases}位相 生存率**: **{report.survival_rate_pct:.1f}% ({report.survived_phases}/{report.total_phases})** | 不発: H1 `{report.max_h1_dropped}` / H2 `{report.max_h2_dropped}`",
        f"- **2人合計 火力期待値**: **`{report.mean_combined_expected_potency:,.1f}p`** (合計ロス `{report.mean_combined_potency_lost:,.1f}p` / **合計DPS効率 `{report.mean_combined_dps_efficiency_pct:.2f}%`**)",
        f"  - H1: `{report.mean_h1_expected_potency:,.1f}p` (攻撃GCD `{report.mean_h1_filler_gcds:.1f}` / GCDヒール `{report.mean_h1_gcd_heals:.1f}` / OH `{report.mean_h1_overheal_pct:.1f}%`)",
        f"  - H2: `{report.mean_h2_expected_potency:,.1f}p` (攻撃GCD `{report.mean_h2_filler_gcds:.1f}` / GCDヒール `{report.mean_h2_gcd_heals:.1f}` / OH `{report.mean_h2_overheal_pct:.1f}%`)",
        f"- **有効ヒール負担比率 (H1 : H2)**: **`{report.mean_h1_heal_share_pct:.1f}% : {report.mean_h2_heal_share_pct:.1f}%`** (全体OH `{report.mean_combined_overheal_pct:.1f}%`)",
        f"- **最低HP残量 (+5%最悪)**: `PT {report.worst_min_hp_ratio.get('party', 0)*100:.1f}%` / `MT {report.worst_min_hp_ratio.get('mt', 0)*100:.1f}%` / `ST {report.worst_min_hp_ratio.get('st', 0)*100:.1f}%`",
        "",
        "## H1 + H2 統合タイムライン（発動時刻順）",
        "",
        "| 発動時刻 | 対象ギミック | 担当 | スキル名 | 対象ch | 種別 |",
        "|---:|---|---|---|---|---|",
    ]
    merged: list[tuple[float, str, PlannedAction]] = [
        *[(a.cast_at_s, "H1", a) for a in plan_h1.actions],
        *[(a.cast_at_s, "H2", a) for a in plan_h2.actions],
    ]
    merged.sort(key=lambda x: (x[0], 0 if x[1] == "H1" else 1))
    for t, who, act in merged:
        kind = "GCD" if act.is_gcd else "oGCD"
        lines.append(
            f"| {t:6.2f}s | {act.event_name} ({act.event_time_s:.1f}s) | **{who}** | {act.skill_name} | `{act.target_channel}` | {kind} |"
        )
    md = "\n".join(lines) + "\n"
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(md, encoding="utf-8")
    return md


def export_duo_csv(
    plan_h1: OptimizationPlan,
    plan_h2: OptimizationPlan,
    output_path: str | Path | None = None,
) -> str:
    """Export unified H1 + H2 schedule as UTF-8 CSV."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["cast_at_s", "healer", "event_time_s", "event_name", "skill_id", "skill_name", "target_channel", "is_gcd"])
    merged: list[tuple[float, str, PlannedAction]] = [
        *[(a.cast_at_s, "H1", a) for a in plan_h1.actions],
        *[(a.cast_at_s, "H2", a) for a in plan_h2.actions],
    ]
    merged.sort(key=lambda x: (x[0], 0 if x[1] == "H1" else 1))
    for t, who, act in merged:
        w.writerow([t, who, act.event_time_s, act.event_name, act.skill_id, act.skill_name, act.target_channel, act.is_gcd])
    text = buf.getvalue()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8-sig")
    return text

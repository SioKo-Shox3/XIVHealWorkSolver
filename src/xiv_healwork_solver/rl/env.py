"""Real-time event-driven RL wrapper built directly on the shared CombatEngine.

Guarantees 100% physics parity between CP-SAT replay verification and RL rollouts:
- Continuous exact timestamps (GCD 2.50 / 2.40 / 2.31s produce exact 24 / 25 / 26 casts in 60s)
- Separate Action 0 (WAIT) and Action 1 (FILLER_ATTACK_GCD)
- Returns exact elapsed `dt` per step for continuous-time discounting (gamma ** dt)
- Observation exposes special debuffs (Full-Heal Doom, Esuna stacks), active HoT/DoT,
  active AA pressure, server-tick countdown, downtime, and future timeline lookahead
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from xiv_healwork_solver.models import EncounterTimeline, GearSamplingRange, PlayerKit
from xiv_healwork_solver.sim.engine import EPS, ActionEntry, CombatEngine, build_action_entries

LOOKAHEAD_EVENTS = 8
STATE_DIM = 36
SKILL_FEAT_DIM = 20
EVENT_FEAT_DIM = 14


class FF14HealMitEnv:
    """Gym-style RL environment backed directly by CombatEngine (zero duplicated combat rules).

    Supports gear-conditioned training via `gear_sampling`: each training episode samples
    (gcd_s, piety, tenacity, heal_scale, max_hp) and exposes those exact parameters in `obs["state"]`,
    so a single trained checkpoint adapts at inference time to any user gear input.
    """

    def __init__(
        self,
        kit: PlayerKit,
        timeline: EncounterTimeline,
        damage_variance: float = 0.05,
        raise_prob_per_pull: float = 0.35,
        tank_passive_self_heal_per_tick: float = 0.0,
        partner_trust_scale: float = 1.0,
        max_wait_s: float = 1.0,
        gear_sampling: GearSamplingRange | None = None,
        seed: int | None = None,
    ) -> None:
        self.base_kit = kit
        self.kit = kit
        self.timeline = timeline
        self.damage_variance = damage_variance
        self.raise_prob_per_pull = raise_prob_per_pull
        self.tank_passive_self_heal_per_tick = tank_passive_self_heal_per_tick
        self.partner_trust_scale = partner_trust_scale
        self.max_wait_s = max_wait_s
        self.gear_sampling = gear_sampling
        self._gear_rng = np.random.default_rng(seed)
        self.engine = CombatEngine(
            kit=kit,
            timeline=timeline,
            damage_variance=damage_variance,
            raise_prob_per_pull=raise_prob_per_pull,
            partner_trust_scale=partner_trust_scale,
            tank_passive_self_heal_per_tick=tank_passive_self_heal_per_tick,
            seed=seed,
        )
        self.actions: list[ActionEntry] = self.engine.actions
        self.n_actions: int = self.engine.n_actions
        self._static_skill_feats = self._build_static_skill_feats()

    @property
    def t(self) -> float:
        return self.engine.t

    @property
    def hp(self) -> dict[str, float]:
        return self.engine.hp

    @property
    def stats(self) -> dict[str, float]:
        return self.engine.stats

    @property
    def action_log(self) -> list[dict[str, Any]]:
        return self.engine.action_log

    def _build_static_skill_feats(self) -> np.ndarray:
        feats = np.zeros((self.n_actions, SKILL_FEAT_DIM), dtype=np.float32)
        heal_scale = self.kit.gear_stats.heal_scale_multiplier
        # Action 1 = FILLER_ATTACK_GCD marker
        feats[1, 8] = 1.0
        feats[1, 16] = 1.0
        feats[1, 11] = min(1.0, self.kit.gear_stats.filler_gcd_mp_cost / 3000.0)
        for entry in self.actions[2:]:
            sk = entry.skill
            assert sk is not None
            i = entry.index
            feats[i, 0] = min(1.0, sk.cooldown_s / 180.0)
            feats[i, 1] = min(1.0, sk.duration_s / 30.0)
            feats[i, 2] = sk.mitigation_all
            feats[i, 3] = sk.mitigation_physical
            feats[i, 4] = sk.mitigation_magical
            sh_eff = (
                (self.kit.max_hp_tank if entry.target_channel in ("mt", "st") else self.kit.max_hp_party)
                * sk.shield_max_hp_ratio
                if sk.shield_max_hp_ratio > 0
                else sk.shield_value * heal_scale
            )
            feats[i, 5] = min(1.0, sh_eff / 40_000.0)
            feats[i, 6] = min(1.0, (max(sk.heal_instant, sk.delayed_heal_value) * heal_scale) / 50_000.0)
            feats[i, 7] = min(1.0, (sk.heal_hot_total * heal_scale) / 50_000.0)
            feats[i, 8] = 1.0 if sk.is_gcd else 0.0
            feats[i, 9] = min(1.0, sk.cast_time_s / 2.5)
            feats[i, 10] = sk.weave_cost / 2.0
            feats[i, 11] = min(1.0, sk.resource_costs.get("mp", 0.0) / 3000.0)
            feats[i, 12] = min(1.0, (sk.resource_grants.get("mp", 0.0) + sk.mp_regen_per_tick * 7.0) / 4000.0)
            if entry.target_channel == "party":
                feats[i, 13] = 1.0
            elif entry.target_channel == "mt":
                feats[i, 14] = 1.0
            elif entry.target_channel == "st":
                feats[i, 15] = 1.0
            elif entry.target_channel == "boss":
                feats[i, 16] = 1.0
            feats[i, 18] = 1.0 if sk.cleanses_debuff else 0.0
            feats[i, 19] = 1.0 if sk.needs_boss_target() else 0.0
        return feats

    def _sample_gear_kit(self) -> PlayerKit:
        assert self.gear_sampling is not None
        gs = self.gear_sampling
        kit_copy = self.base_kit.model_copy(deep=True)
        kit_copy.gcd_s = round(float(self._gear_rng.uniform(gs.gcd_min, gs.gcd_max)), 2)
        kit_copy.gear_stats.heal_scale_multiplier = round(
            float(self._gear_rng.uniform(gs.heal_scale_min, gs.heal_scale_max)), 3
        )
        if self.base_kit.role == "healer":
            # Healers condition strictly on healer gear parameters (GCD / Piety / Heal Scale / Party HP);
            # tank HP scales naturally with the encounter tier and is never a healer gear input.
            kit_copy.gear_stats.piety = int(self._gear_rng.integers(gs.piety_min, gs.piety_max + 1))
            hp_scale = float(self._gear_rng.uniform(gs.hp_scale_min, gs.hp_scale_max))
            kit_copy.max_hp_party = round(self.base_kit.max_hp_party * hp_scale, 0)
            kit_copy.max_hp_tank = round(self.base_kit.max_hp_tank * hp_scale, 0)
        else:
            kit_copy.gear_stats.tenacity = int(self._gear_rng.integers(gs.tenacity_min, gs.tenacity_max + 1))
            hp_scale_party = float(self._gear_rng.uniform(gs.hp_scale_min, gs.hp_scale_max))
            hp_scale_tank = float(self._gear_rng.uniform(gs.hp_scale_min, gs.hp_scale_max))
            kit_copy.max_hp_party = round(self.base_kit.max_hp_party * hp_scale_party, 0)
            kit_copy.max_hp_tank = round(self.base_kit.max_hp_tank * hp_scale_tank, 0)
        return kit_copy

    def reset(
        self,
        seed: int | None = None,
        deterministic: bool = False,
        fixed_tick_phase: float | None = None,
    ) -> dict[str, np.ndarray]:
        if self.gear_sampling is not None and not deterministic:
            self.kit = self._sample_gear_kit()
            self.engine = CombatEngine(
                kit=self.kit,
                timeline=self.timeline,
                damage_variance=self.damage_variance,
                raise_prob_per_pull=self.raise_prob_per_pull,
                partner_trust_scale=self.partner_trust_scale,
                tank_passive_self_heal_per_tick=self.tank_passive_self_heal_per_tick,
                seed=seed,
            )
            self._static_skill_feats = self._build_static_skill_feats()
        else:
            if self.kit is not self.base_kit:
                self.kit = self.base_kit
                self.engine = CombatEngine(
                    kit=self.kit,
                    timeline=self.timeline,
                    damage_variance=self.damage_variance,
                    raise_prob_per_pull=self.raise_prob_per_pull,
                    partner_trust_scale=self.partner_trust_scale,
                    tank_passive_self_heal_per_tick=self.tank_passive_self_heal_per_tick,
                    seed=seed,
                )
                self._static_skill_feats = self._build_static_skill_feats()
        self.engine.reset(seed=seed, deterministic=deterministic, fixed_tick_phase=fixed_tick_phase)
        return self._get_obs()

    def step(self, action_idx: int, deterministic: bool | None = None) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if deterministic is not None:
            self.engine.deterministic = deterministic
        prev_t = self.engine.t
        imm_reward = self.engine.execute_action(action_idx)

        next_t = self.engine.next_decision_time(max_wait_s=self.max_wait_s)
        if next_t <= self.engine.t + EPS:
            next_t = min(self.timeline.duration_s, self.engine.t + 0.05)
        adv_reward = self.engine.advance_to(next_t)

        dt_elapsed = max(0.0, self.engine.t - prev_t)
        reward = imm_reward + adv_reward
        done = False
        if self.engine.died:
            # Strict survival-first hierarchy: a wipe is catastrophically worse than casting safety heals
            reward -= 200.0
            done = True
        elif self.engine.t >= self.timeline.duration_s - EPS:
            reward += 200.0
            done = True
        else:
            # Danger-zone margin shaping: discourage riding below 15% HP where a +5% damage roll causes a wipe
            min_r = min(
                self.engine.hp["party"] / self.engine.max_hp["party"],
                self.engine.hp["mt"] / self.engine.max_hp["mt"],
                self.engine.hp["st"] / self.engine.max_hp["st"],
            )
            if min_r < 0.15:
                reward -= (0.15 - min_r) * 3.0

        return self._get_obs(), float(reward), done, {
            "stats": dict(self.engine.stats),
            "time_s": self.engine.t,
            "dt": dt_elapsed,
        }

    def _get_obs(self) -> dict[str, np.ndarray]:
        eng = self.engine
        t = eng.t
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = min(1.0, t / max(1.0, self.timeline.duration_s))
        state[1] = max(0.0, eng.hp["party"] / eng.max_hp["party"])
        state[2] = max(0.0, eng.hp["mt"] / eng.max_hp["mt"])
        state[3] = max(0.0, eng.hp["st"] / eng.max_hp["st"])
        state[4] = min(1.0, eng.total_shield("party") / 40_000.0)
        state[5] = min(1.0, eng.total_shield("mt") / 40_000.0)
        state[6] = min(1.0, eng.total_shield("st") / 40_000.0)
        state[7] = eng.resources.get("mp", 10_000.0) / 10_000.0
        # Generic non-MP resource summary (mean normalized fullness)
        other_res = [
            eng.resources.get(n, 0.0) / max(1.0, p.maximum)
            for n, p in eng.resolved_pools.items()
            if n.lower() != "mp"
        ]
        state[8] = float(np.mean(other_res)) if other_res else 0.0
        state[9] = max(0.0, min(1.0, (eng.next_server_tick - t) / 3.0))

        in_move = any(mw.start_s <= t <= mw.end_s for mw in self.timeline.movement_windows)
        next_move = min(
            [0.0 if mw.start_s <= t <= mw.end_s else max(0.0, mw.start_s - t) for mw in self.timeline.movement_windows if mw.end_s >= t],
            default=60.0,
        )
        state[10] = 1.0 if in_move else 0.0
        state[11] = min(1.0, next_move / 60.0)
        state[12] = 1.0 if (
            self.kit.burst_duration_s > 0
            and t >= self.kit.burst_offset_s
            and ((t - self.kit.burst_offset_s) % self.kit.burst_interval_s) < self.kit.burst_duration_s
        ) else 0.0
        state[13] = 0.0 if self.timeline.is_boss_targetable(t) else 1.0
        state[14] = 1.0 if t >= eng.gcd_ready_at - EPS else 0.0
        state[15] = min(1.0, max(0.0, eng.gcd_ready_at - t) / self.kit.gcd_s)
        state[16] = eng.weave_slots / 2.0
        state[17] = (self.kit.gcd_s - 2.0) / 1.0
        state[18] = min(1.0, self.kit.gear_stats.piety / 2000.0)
        state[19] = eng.tenacity_mit * 5.0
        state[20] = 1.0 - eng.compute_mitigation_factor("party", "magical")
        state[21] = 1.0 - eng.compute_mitigation_factor("mt", "physical")
        state[22] = 1.0 - eng.compute_mitigation_factor("st", "physical")
        state[23] = 1.0 if (eng.mp_floor > 0 and eng.resources.get("mp", 10_000.0) >= eng.mp_floor) else 0.0
        # Special mechanics & HoT/DoT/AA awareness
        fh_dooms, esuna_stacks, min_doom_rem = eng.active_doom_counts()
        state[24] = min(1.0, fh_dooms / 3.0)
        state[25] = min(1.0, esuna_stacks / 3.0)
        state[26] = min(1.0, min_doom_rem / 20.0)
        active_hot = sum(b.hot_per_tick for b in eng.active_buffs if b.expires_at >= t - EPS)
        active_dot = sum(d.raw_dmg_per_tick for d in eng.active_dots if d.expires_at >= t - EPS)
        state[27] = min(1.0, active_hot / 20_000.0)
        state[28] = min(1.0, active_dot / 20_000.0)
        aa_active = any(seg.start_s <= t <= seg.end_s for seg in self.timeline.auto_attacks) and self.timeline.is_boss_targetable(t)
        state[29] = 1.0 if aa_active else 0.0
        # Explicit gear parameter channels (heal scale, normalized party max HP, normalized tank max HP)
        state[30] = min(2.0, self.kit.gear_stats.heal_scale_multiplier)
        state[31] = min(2.0, self.kit.max_hp_party / 180_000.0)
        state[32] = min(2.0, self.kit.max_hp_tank / 280_000.0)
        # Generic HP-threshold-triggered damage mechanic awareness (e.g. full-heal triggers raidwide blast)
        tr_cnt, tr_raw, tr_must = eng.active_hp_trigger_summary()
        state[33] = min(1.0, tr_cnt / 3.0)
        state[34] = min(1.0, tr_raw / 250_000.0)
        state[35] = tr_must

        skill_tokens = self._static_skill_feats.copy()
        for entry in self.actions[2:]:
            sk = entry.skill
            assert sk is not None
            charges = eng.skill_charges[sk.id]
            if charges > 0:
                skill_tokens[entry.index, 17] = charges / max(1.0, float(sk.max_charges))
            else:
                rem = max(0.0, eng.skill_next_charge_at[sk.id] - t) if math.isfinite(eng.skill_next_charge_at[sk.id]) else 0.0
                skill_tokens[entry.index, 17] = -min(1.0, rem / max(1.0, sk.cooldown_s))

        future = np.zeros((LOOKAHEAD_EVENTS, EVENT_FEAT_DIM), dtype=np.float32)
        upcoming = eng.events[eng._next_event_idx : eng._next_event_idx + LOOKAHEAD_EVENTS]
        for k, ev in enumerate(upcoming):
            dt_ev = max(0.0, ev.time_s - t)
            future[k, 0] = min(1.0, dt_ev / 120.0)
            future[k, 1] = min(1.0, ev.raw_damage / 250_000.0)
            future[k, 2] = min(1.0, ev.dot_total_damage / 60_000.0)
            future[k, 3] = 1.0 if ev.target_scope == "party" else 0.0
            future[k, 4] = 1.0 if ev.target_scope in ("mt", "both_tanks") else 0.0
            future[k, 5] = 1.0 if ev.target_scope in ("st", "both_tanks") else 0.0
            future[k, 6] = 1.0 if ev.damage_type == "physical" else 0.0
            future[k, 7] = 1.0 if ev.damage_type == "magical" else 0.0
            future[k, 8] = 1.0 if ev.requires_movement else 0.0
            future[k, 9] = min(1.0, ev.hit_count / 6.0)
            future[k, 10] = 1.0 if ev.mechanic_effect == "set_hp_to_1" else 0.0
            future[k, 11] = 1.0  # valid token mask
            future[k, 12] = 1.0 if ev.doom_full_heal_duration_s is not None else 0.0
            future[k, 13] = min(1.0, ev.esuna_required_count / 2.0)

        return {
            "state": state,
            "skills": skill_tokens,
            "future": future,
            "mask": eng.action_mask(),
        }

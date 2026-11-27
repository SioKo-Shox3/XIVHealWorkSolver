"""Real-time event-driven combat engine shared by CP-SAT verification and RL.

Single source of truth for FF14 combat physics:
- Continuous float timestamps (zero 0.5s quantization -> exact GCD 2.50 / 2.40 / 2.31s timing)
- Exact chronological priority-queue execution of casts, snapshots, impacts, AAs, and 3s server ticks
- Per-channel (party / mt / st) HP, stacked shields with stack-group overwrite rules, HoTs, and DoTs
- DoT mitigation snapshot locked at application time (later mit does NOT weaken existing DoTs)
- Strict downtime semantics (no boss debuffs, no filler MP drain, no attack-triggered heals/gauges,
  0 DPS cost for downtime pre-shields that don't delay return GCD)
- Exact cast resolution + animation lock + movement window overlap (with slidecast allowance)
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xiv_healwork_solver.models import (
    DamageEvent,
    EncounterTimeline,
    HpThresholdTrigger,
    OptimizationPlan,
    PlayerKit,
    ReplayEventSnapshot,
    ReplayVerification,
    SkillDef,
)

EPS = 1e-6

# Event ordering priorities at identical timestamps
PRIO_DOWNTIME = 1
PRIO_CAST_RESOLVE = 2
PRIO_PARTNER = 3
PRIO_SNAPSHOT = 4
PRIO_DAMAGE_IMPACT = 5
PRIO_AUTO_ATTACK = 6
PRIO_SERVER_TICK = 7
PRIO_RAISE = 8
PRIO_DEBUFF_EXPIRE = 9


@dataclass(frozen=True)
class ActionEntry:
    """Expanded discrete action:
    0 = WAIT (advance to next meaningful timestamp without casting filler)
    1 = FILLER_ATTACK_GCD (cast attack filler on targetable boss)
    2..N = owned skills from PlayerKit (single_ally skills expand into MT and ST targets)
    """

    index: int
    kind: str  # "wait" | "filler" | "skill"
    skill: SkillDef | None
    target_channel: str  # "none" | "party" | "mt" | "st" | "boss"
    label: str


@dataclass
class _ActiveBuff:
    skill_id: str
    stack_group: str | None
    target_channel: str
    expires_at: float
    mit_all: float = 0.0
    mit_phys: float = 0.0
    mit_mag: float = 0.0
    hot_per_tick: float = 0.0
    mp_per_tick: float = 0.0
    delayed_heal: float = 0.0
    delayed_trigger_ratio: float = 0.5
    delayed_triggered: bool = False


@dataclass
class _ActiveShield:
    source_id: str
    stack_group: str | None
    remaining: float
    expires_at: float


@dataclass
class _ActiveDot:
    source_event_id: str
    target_channel: str
    applied_at: float
    expires_at: float
    ticks_remaining: int
    raw_dmg_per_tick: float
    damage_type: str
    snapshotted_mit_factor: float | None
    custom_interval_s: float | None = None


@dataclass
class _SpecialDebuff:
    kind: str  # "full_heal_doom" | "esuna_doom"
    target_channel: str
    expires_at: float
    stacks_remaining: int = 1
    cleared: bool = False


@dataclass
class _HpTriggerState:
    spec: HpThresholdTrigger
    triggers_remaining: int
    next_allowed_t: float = 0.0


@dataclass(order=True)
class _ScheduledItem:
    time_s: float
    priority: int
    seq: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False)


def build_action_entries(kit: PlayerKit) -> list[ActionEntry]:
    """Build skill-agnostic action list with separate WAIT and FILLER_GCD actions."""
    entries: list[ActionEntry] = [
        ActionEntry(index=0, kind="wait", skill=None, target_channel="none", label="WAIT"),
        ActionEntry(index=1, kind="filler", skill=None, target_channel="boss", label="FILLER_ATTACK_GCD"),
    ]
    idx = 2
    for sk in kit.skills:
        if sk.target_scope == "single_ally":
            entries.append(ActionEntry(index=idx, kind="skill", skill=sk, target_channel="mt", label=f"{sk.name} -> MT"))
            idx += 1
            entries.append(ActionEntry(index=idx, kind="skill", skill=sk, target_channel="st", label=f"{sk.name} -> ST"))
            idx += 1
        elif sk.target_scope == "self":
            ch = "mt" if (kit.role == "tank" and kit.player_is_mt) else ("st" if kit.role == "tank" else "party")
            entries.append(ActionEntry(index=idx, kind="skill", skill=sk, target_channel=ch, label=sk.name))
            idx += 1
        elif sk.target_scope in ("mt", "st"):
            entries.append(ActionEntry(index=idx, kind="skill", skill=sk, target_channel=sk.target_scope, label=sk.name))
            idx += 1
        elif sk.target_scope == "boss_debuff":
            entries.append(ActionEntry(index=idx, kind="skill", skill=sk, target_channel="boss", label=sk.name))
            idx += 1
        else:
            entries.append(ActionEntry(index=idx, kind="skill", skill=sk, target_channel="party", label=sk.name))
            idx += 1
    return entries


class CombatEngine:
    """Exact real-time event-driven combat simulator shared by CP-SAT verification and RL."""

    def __init__(
        self,
        kit: PlayerKit,
        timeline: EncounterTimeline,
        damage_variance: float = 0.05,
        raise_prob_per_pull: float = 0.0,
        partner_trust_scale: float = 1.0,
        tank_passive_self_heal_per_tick: float = 0.0,
        forced_damage_roll: float | None = None,
        expected_partners: bool = False,
        seed: int | None = None,
    ) -> None:
        self.kit = kit
        self.timeline = timeline
        self.damage_variance = damage_variance
        self.raise_prob_per_pull = raise_prob_per_pull
        self.partner_trust_scale = partner_trust_scale
        self.tank_passive_self_heal_per_tick = tank_passive_self_heal_per_tick
        self.forced_damage_roll = forced_damage_roll
        self.expected_partners = expected_partners
        self.rng = np.random.default_rng(seed)

        self.events = sorted(timeline.events, key=lambda e: e.time_s)
        self.actions = build_action_entries(kit)
        self.n_actions = len(self.actions)
        self.resolved_pools = {p.name: p for p in kit.resolved_resources()}
        self.has_explicit_lucid = any(sk.mp_regen_per_tick > 0 for sk in kit.skills)
        self.heal_scale = kit.gear_stats.heal_scale_multiplier
        self.tenacity_mit = kit.gear_stats.tenacity_passive_mitigation(kit.role)

        # Natural 3s MP tick (Piety + passive MP regen such as Aetherflow/Assize; explicit Lucid buffs add on top)
        self.base_mp_per_3s_tick = (
            kit.gear_stats.combat_mp_per_3s_tick() + kit.gear_stats.lucid_dreaming_mp_per_sec * 3.0
        )
        self.mp_floor = kit.gear_stats.mp_floor()

        self.max_hp: dict[str, float] = {
            "party": kit.max_hp_party,
            "mt": kit.max_hp_tank,
            "st": kit.max_hp_tank,
        }

        # Dynamic state
        self.t: float = 0.0
        self.deterministic: bool = False
        self.server_tick_phase: float = 1.5
        self.next_server_tick: float = 1.5
        self.gcd_ready_at: float = 0.0
        self.anim_lock_until: float = 0.0
        self.casting_until: float = 0.0
        self.weave_slots: int = kit.base_weave_slots_per_gcd
        self.hp: dict[str, float] = {}
        self.shields: dict[str, list[_ActiveShield]] = {}
        self.resources: dict[str, float] = {}
        self.skill_charges: dict[str, int] = {}
        self.skill_next_charge_at: dict[str, float] = {}
        self.active_buffs: list[_ActiveBuff] = []
        self.active_dots: list[_ActiveDot] = []
        self.special_debuffs: list[_SpecialDebuff] = []
        self.hp_triggers: list[_HpTriggerState] = []
        self._active_trigger_ids: set[str] = set()
        self._event_snapshots_mit: dict[tuple[str, str], float] = {}
        self._queue: list[_ScheduledItem] = []
        self._seq: int = 0
        self._next_event_idx: int = 0

        self.died: bool = False
        self.first_death_time_s: float | None = None
        self.first_death_channel: str | None = None
        self.max_deficit_hp: float = 0.0
        self.action_log: list[dict[str, Any]] = []
        self.event_snapshots: list[ReplayEventSnapshot] = []
        self.stats: dict[str, float] = {}
        self._step_reward_accum: float = 0.0

    def _push(self, time_s: float, priority: int, kind: str, payload: Any) -> None:
        self._seq += 1
        heapq.heappush(self._queue, _ScheduledItem(time_s=time_s, priority=priority, seq=self._seq, kind=kind, payload=payload))

    def reset(
        self,
        seed: int | None = None,
        deterministic: bool = False,
        fixed_tick_phase: float | None = None,
        initial_hp_ratios: dict[str, float] | None = None,
        initial_resources: dict[str, float] | None = None,
        initial_cooldowns: dict[str, float] | None = None,
        start_time_s: float = 0.0,
    ) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.deterministic = deterministic
        self.t = start_time_s
        if fixed_tick_phase is not None:
            self.server_tick_phase = float(fixed_tick_phase) % 3.0
        elif deterministic:
            self.server_tick_phase = 1.5
        else:
            self.server_tick_phase = float(self.rng.uniform(0.0, 3.0))

        # First tick strictly after start_time_s
        k_tick = math.floor((start_time_s - self.server_tick_phase) / 3.0) + 1
        self.next_server_tick = self.server_tick_phase + 3.0 * k_tick
        if self.next_server_tick <= start_time_s + EPS:
            self.next_server_tick += 3.0

        self.gcd_ready_at = start_time_s
        self.anim_lock_until = start_time_s
        self.casting_until = start_time_s
        self.weave_slots = self.kit.weave_budget_at(start_time_s)

        ratios = initial_hp_ratios or {}
        self.hp = {
            ch: self.max_hp[ch] * float(ratios.get(ch, 1.0))
            for ch in ("party", "mt", "st")
        }
        self.shields = {"party": [], "mt": [], "st": []}
        self.resources = {name: pool.initial for name, pool in self.resolved_pools.items()}
        if initial_resources:
            for k, v in initial_resources.items():
                self.resources[k] = float(v)

        self.skill_charges = {sk.id: sk.max_charges for sk in self.kit.skills}
        self.skill_next_charge_at = {sk.id: math.inf for sk in self.kit.skills}
        if initial_cooldowns:
            for sid, rem in initial_cooldowns.items():
                if sid in self.skill_charges and rem > EPS:
                    self.skill_charges[sid] = max(0, self.skill_charges[sid] - 1)
                    self.skill_next_charge_at[sid] = start_time_s + rem

        self.active_buffs = []
        self.active_dots = []
        self.special_debuffs = []
        self.hp_triggers = [
            _HpTriggerState(spec=tr, triggers_remaining=tr.max_triggers, next_allowed_t=tr.start_s)
            for tr in self.timeline.hp_trigger_mechanics
            if (tr.end_s >= start_time_s - EPS if start_time_s <= EPS else tr.start_s >= start_time_s - EPS)
        ]
        self._active_trigger_ids = set()
        self._event_snapshots_mit = {}

        # Apply countdown pre-pull actions (t <= 0s: 0 DPS loss, out-of-combat MP stays full)
        if start_time_s <= EPS and self.kit.prepull_actions:
            skills_by_id = {sk.id: sk for sk in self.kit.skills}
            for pp in sorted(self.kit.prepull_actions, key=lambda a: a.offset_s):
                sk = skills_by_id.get(pp.skill_id)
                if sk is None:
                    continue
                if pp.consumes_cooldown and sk.cooldown_s > 0 and self.skill_charges[sk.id] > 0:
                    ready_t = pp.offset_s + sk.cooldown_s
                    if ready_t > EPS:
                        was_full = self.skill_charges[sk.id] == sk.max_charges
                        self.skill_charges[sk.id] -= 1
                        if was_full:
                            self.skill_next_charge_at[sk.id] = ready_t
                if pp.consumes_mp and "mp" in self.resources:
                    mp_cost = sk.resource_costs.get("mp", 0.0)
                    if mp_cost > 0:
                        self.resources["mp"] = max(0.0, self.resources["mp"] - mp_cost)
                if pp.override_shield_value is not None:
                    sh_amt = pp.override_shield_value
                elif sk.shield_max_hp_ratio > 0:
                    sh_amt = self.max_hp[pp.target_channel] * sk.shield_max_hp_ratio
                else:
                    sh_amt = sk.shield_value * self.heal_scale
                sh_exp = pp.offset_s + sk.effective_shield_duration()
                if sh_amt > 0 and sh_exp > EPS:
                    self._add_shield(pp.target_channel, sh_amt, sh_exp, sk.id, sk.shield_stack_group)
                buf_exp = pp.offset_s + sk.duration_s
                hot_ticks = max(1, int(round(sk.duration_s / 3.0))) if sk.duration_s >= 3.0 else 1
                hot_per_tick = (sk.heal_hot_total * self.heal_scale) / hot_ticks if sk.heal_hot_total > 0 else 0.0
                if buf_exp > EPS and (
                    sk.mitigation_all > 0
                    or sk.mitigation_physical > 0
                    or sk.mitigation_magical > 0
                    or hot_per_tick > 0
                ):
                    self.active_buffs.append(
                        _ActiveBuff(
                            skill_id=sk.id,
                            stack_group=sk.stack_group,
                            target_channel=pp.target_channel,
                            expires_at=buf_exp,
                            mit_all=sk.mitigation_all,
                            mit_phys=sk.mitigation_physical,
                            mit_mag=sk.mitigation_magical,
                            hot_per_tick=hot_per_tick,
                        )
                    )
        self._queue = []
        self._seq = 0
        self._next_event_idx = 0
        while self._next_event_idx < len(self.events) and self.events[self._next_event_idx].time_s < start_time_s - EPS:
            self._next_event_idx += 1

        self.died = False
        self.first_death_time_s = None
        self.first_death_channel = None
        self.max_deficit_hp = 0.0
        self.action_log = []
        self.event_snapshots = []
        self._step_reward_accum = 0.0
        self.stats = {
            "filler_gcds": 0.0,
            "gcd_heals": 0.0,
            "downtime_free_gcds": 0.0,
            "ogcds_used": 0.0,
            "cast_interrupts": 0.0,
            "raises_handled": 0.0,
            "mp_reserve_breaches": 0.0,
            "mp_min": self.resources.get("mp", 10_000.0),
            "player_heal_raw": 0.0,
            "player_heal_eff": 0.0,
            "expected_potency_gained": 0.0,
            "expected_potency_lost": 0.0,
            "resource_potency_lost": 0.0,
            "hp_triggers_fired": 0.0,
            "min_party_hp_ratio": self.hp["party"] / self.max_hp["party"],
            "min_mt_hp_ratio": self.hp["mt"] / self.max_hp["mt"],
            "min_st_hp_ratio": self.hp["st"] / self.max_hp["st"],
            "survived": 1.0,
        }

        # Schedule initial 3s server tick
        self._push(self.next_server_tick, PRIO_SERVER_TICK, "server_tick", None)

        # Schedule player weapon auto-attacks for AA-driven job gauges (e.g. Paladin Oath Gauge; pauses in downtime!)
        if any(p.gain_per_player_aa > 0.0 for p in self.resolved_pools.values()):
            p_aa_t = start_time_s + self.kit.player_aa_interval_s
            if p_aa_t <= self.timeline.duration_s + EPS:
                self._push(p_aa_t, PRIO_SERVER_TICK, "player_aa", None)

        # Schedule downtime debuff-clear transitions
        for dw in self.timeline.downtime_windows:
            if dw.start_s >= start_time_s - EPS and dw.clears_boss_debuffs:
                self._push(dw.start_s, PRIO_DOWNTIME, "downtime_start", dw)

        # Schedule HP-threshold-trigger expiry checks (for must_trigger_before_end mechanics)
        for tr_state in self.hp_triggers:
            if tr_state.spec.must_trigger_before_end and tr_state.spec.end_s >= start_time_s - EPS:
                self._push(tr_state.spec.end_s, PRIO_DEBUFF_EXPIRE, "hp_trigger_expire", tr_state)

        # Schedule timeline DamageEvents (separate snapshot and individual multi-hit impacts)
        for ev in self.events:
            if ev.time_s < start_time_s - EPS:
                continue
            snap_t = max(start_time_s, ev.effective_snapshot_time())
            self._push(snap_t, PRIO_SNAPSHOT, "snapshot", ev)
            per_hit_raw = ev.raw_damage / ev.hit_count
            for h_idx in range(ev.hit_count):
                hit_t = ev.time_s + h_idx * ev.hit_interval_s
                self._push(
                    hit_t,
                    PRIO_DAMAGE_IMPACT,
                    "damage_impact",
                    {"event": ev, "hit_index": h_idx, "is_last_hit": (h_idx == ev.hit_count - 1), "raw": per_hit_raw},
                )

        # Schedule boss auto-attacks
        for seg in self.timeline.auto_attacks:
            aa_t = seg.start_s
            while aa_t < start_time_s - EPS:
                aa_t += seg.interval_s
            while aa_t <= seg.end_s + EPS and aa_t <= self.timeline.duration_s + EPS:
                if self.timeline.is_boss_targetable(aa_t):
                    self._push(aa_t, PRIO_AUTO_ATTACK, "auto_attack", seg)
                aa_t += seg.interval_s

        # Schedule partner buffs/shields/heals
        ev_by_id = {e.id: e for e in self.events}
        for pa in self.timeline.partner_profile:
            ev = ev_by_id.get(pa.event_id)
            if ev is None or ev.time_s < start_time_s - EPS:
                continue
            p = min(1.0, pa.probability * self.partner_trust_scale)
            if p <= EPS:
                continue
            if deterministic or self.expected_partners:
                # Expected-value scaling matches stochastic expectation E[X] = p * X
                # without losing cross-comp p=0.33 mitigations/shields to unsolvable Bernoulli cliffs.
                pa_eff = pa if p >= 1.0 - EPS else pa.model_copy(
                    update={
                        "mitigation_all": pa.mitigation_all * p,
                        "mitigation_physical": pa.mitigation_physical * p,
                        "mitigation_magical": pa.mitigation_magical * p,
                        "shield_value": pa.shield_value * p,
                        "heal_value": pa.heal_value * p,
                    }
                )
                jitter = 0.0 if deterministic else float(self.rng.normal(0.0, min(0.5, pa.timing_std_s * 0.25)))
            else:
                if float(self.rng.random()) >= p:
                    continue
                pa_eff = pa
                jitter = float(self.rng.normal(0.0, pa.timing_std_s))
            # Align partner timing offset with event snapshot time (matching extractor.py)
            fire_t = max(start_time_s, ev.effective_snapshot_time() + pa.timing_mean_offset_s + jitter)
            self._push(fire_t, PRIO_PARTNER, "partner", pa_eff)

        # Optional stochastic mid-fight Raise disturbance
        if (
            not deterministic
            and self.kit.role == "healer"
            and self.raise_prob_per_pull > 0.0
            and float(self.rng.random()) < self.raise_prob_per_pull
            and self.timeline.duration_s - start_time_s > 40.0
        ):
            raise_t = float(self.rng.uniform(start_time_s + 20.0, self.timeline.duration_s - 20.0))
            self._push(raise_t, PRIO_RAISE, "raise", None)

    def _update_continuous_state(self, new_t: float) -> None:
        dt = max(0.0, new_t - self.t)
        if dt <= 0.0:
            return
        # Update cooldown charges accurately across continuous time
        for sk in self.kit.skills:
            while self.skill_charges[sk.id] < sk.max_charges and self.skill_next_charge_at[sk.id] <= new_t + EPS:
                charge_t = self.skill_next_charge_at[sk.id]
                self.skill_charges[sk.id] += 1
                if self.skill_charges[sk.id] < sk.max_charges and sk.cooldown_s > 0:
                    self.skill_next_charge_at[sk.id] = charge_t + sk.cooldown_s
                else:
                    self.skill_next_charge_at[sk.id] = math.inf

        # Update continuous time-based resources (Addersgall/Lily, plus Tank job MP regen from combos/CDs)
        for rname, pool in self.resolved_pools.items():
            if rname.lower() == "mp" and self.kit.role != "tank":
                continue
            if pool.regen_per_sec > 0.0:
                self.resources[rname] = min(pool.maximum, self.resources.get(rname, 0.0) + pool.regen_per_sec * dt)

        # Prune expired buffs, shields, and DoTs (fire any untriggered delayed/reactive heals on expiry!)
        self.t = new_t
        kept_buffs: list[_ActiveBuff] = []
        for b in self.active_buffs:
            if b.expires_at >= self.t - EPS:
                kept_buffs.append(b)
            elif b.delayed_heal > 0.0 and not b.delayed_triggered and not self.died:
                b.delayed_triggered = True
                self._apply_heal(
                    b.target_channel,
                    b.delayed_heal,
                    is_player_heal=(not b.skill_id.startswith("partner_")),
                    single_channel_only=(b.target_channel != "party"),
                )
        self.active_buffs = kept_buffs
        self.active_dots = [d for d in self.active_dots if d.ticks_remaining > 0 and d.expires_at >= self.t - EPS]
        for ch in ("party", "mt", "st"):
            self.shields[ch] = [sh for sh in self.shields[ch] if sh.expires_at >= self.t - EPS and sh.remaining > EPS]

    def total_shield(self, channel: str) -> float:
        return sum(sh.remaining for sh in self.shields[channel] if sh.expires_at >= self.t - EPS)

    def _add_shield(
        self,
        channel: str,
        amount: float,
        duration_s: float,
        source_id: str,
        stack_group: str | None,
        single_channel_only: bool = False,
    ) -> None:
        if amount <= 0.0:
            return
        targets = (channel,) if single_channel_only else (("party", "mt", "st") if channel == "party" else (channel,))
        exp = self.t + max(3.0, duration_s)
        for ch in targets:
            if stack_group is not None:
                replaced = False
                for i, existing in enumerate(self.shields[ch]):
                    if existing.stack_group == stack_group and existing.expires_at >= self.t - EPS:
                        if amount >= existing.remaining:
                            self.shields[ch][i] = _ActiveShield(source_id, stack_group, amount, exp)
                        replaced = True
                        break
                if not replaced:
                    self.shields[ch].append(_ActiveShield(source_id, stack_group, amount, exp))
            else:
                self.shields[ch].append(_ActiveShield(source_id, None, amount, exp))

    def _apply_heal(
        self,
        channel: str,
        amount: float,
        is_player_heal: bool = False,
        single_channel_only: bool = False,
    ) -> float:
        if amount <= 0.0:
            return 0.0
        targets = (channel,) if single_channel_only else (("party", "mt", "st") if channel == "party" else (channel,))
        eff = 0.0
        raw_total = amount * len(targets)
        crossed_channels: list[str] = []
        # Pass 1: apply healing to ALL target channels first so a party heal heals tanks BEFORE any blast fires!
        for ch in targets:
            before = self.hp[ch]
            before_ratio = before / self.max_hp[ch]
            self.hp[ch] = min(self.max_hp[ch], before + amount)
            eff += self.hp[ch] - before
            after_ratio = self.hp[ch] / self.max_hp[ch]
            # Reaching 100% HP clears Full-Heal Doom debuffs on this channel!
            if self.hp[ch] >= self.max_hp[ch] - 1.0:
                for db in self.special_debuffs:
                    if not db.cleared and db.kind == "full_heal_doom" and db.target_channel == ch:
                        db.cleared = True
            if after_ratio > before_ratio + 1e-6:
                crossed_channels.append((ch, before_ratio, after_ratio))  # type: ignore[arg-type]

        # Pass 2: evaluate HP-threshold-triggered damage mechanics for channels that genuinely crossed upward
        if not self.died and self.hp_triggers and crossed_channels:
            for ch, before_r, _after_r in crossed_channels:  # type: ignore[misc]
                for tr_state in self.hp_triggers:
                    sp = tr_state.spec
                    if sp.id in self._active_trigger_ids or self.died:
                        continue
                    ratio_now = self.hp[ch] / self.max_hp[ch]
                    if (
                        tr_state.triggers_remaining > 0
                        and sp.watch_channel == ch
                        and sp.start_s - EPS <= self.t <= sp.end_s + EPS
                        and self.t >= tr_state.next_allowed_t - EPS
                        and before_r < sp.trigger_hp_ratio - 1e-4
                        and ratio_now >= sp.trigger_hp_ratio - 1e-4
                    ):
                        tr_state.triggers_remaining -= 1
                        tr_state.next_allowed_t = self.t + sp.trigger_cooldown_s
                        self.stats["hp_triggers_fired"] += 1.0
                        self._active_trigger_ids.add(sp.id)
                        try:
                            blast_channels = (
                                ("party", "mt", "st")
                                if sp.blast_target_scope == "party"
                                else (("mt", "st") if sp.blast_target_scope == "both_tanks" else (sp.blast_target_scope,))
                            )
                            for bch in blast_channels:
                                b_raw = (
                                    sp.blast_raw_damage * self.kit.tank_raidwide_damage_ratio
                                    if (sp.blast_target_scope == "party" and bch in ("mt", "st"))
                                    else sp.blast_raw_damage
                                )
                                self._apply_damage_to_channel(bch, b_raw, sp.blast_damage_type)
                        finally:
                            self._active_trigger_ids.discard(sp.id)
        if is_player_heal:
            self.stats["player_heal_raw"] += raw_total
            self.stats["player_heal_eff"] += eff
        return max(0.0, 1.0 - eff / max(1.0, raw_total))

    def active_hp_trigger_summary(self) -> tuple[int, float, float]:
        """Return (armed_remaining_triggers, max_blast_raw_damage, must_resolve_flag) for RL obs."""
        rem_cnt = 0
        max_raw = 0.0
        must_flag = 0.0
        for st in self.hp_triggers:
            sp = st.spec
            if (
                st.triggers_remaining > 0
                and sp.start_s - EPS <= self.t <= sp.end_s + EPS
                and self.t >= st.next_allowed_t - EPS
            ):
                rem_cnt += st.triggers_remaining
                max_raw = max(max_raw, sp.blast_raw_damage)
                if sp.must_trigger_before_end:
                    must_flag = 1.0
        return rem_cnt, max_raw, must_flag

    def _apply_cleanse(self, channel: str) -> bool:
        """Remove one stack of an Esuna-cleansable debuff (returns True if something was cleansed)."""
        candidates = [
            db for db in self.special_debuffs
            if not db.cleared and db.kind == "esuna_doom" and (channel == "party" or db.target_channel == channel)
        ]
        if not candidates:
            return False
        # Cleanse the most urgent (earliest expiring) debuff first
        candidates.sort(key=lambda d: d.expires_at)
        db = candidates[0]
        db.stacks_remaining -= 1
        if db.stacks_remaining <= 0:
            db.cleared = True
        return True

    def active_doom_counts(self) -> tuple[int, int, float]:
        """Return (active_full_heal_dooms, active_esuna_stacks, min_time_to_expiry)."""
        fh = 0
        es = 0
        min_rem = 60.0
        for db in self.special_debuffs:
            if db.cleared:
                continue
            rem = max(0.0, db.expires_at - self.t)
            min_rem = min(min_rem, rem)
            if db.kind == "full_heal_doom":
                fh += 1
            elif db.kind == "esuna_doom":
                es += db.stacks_remaining
        return fh, es, min_rem

    def compute_mitigation_factor(self, channel: str, damage_type: str) -> float:
        rem = 1.0
        if self.tenacity_mit > 0.0:
            if (channel == "mt" and self.kit.player_is_mt) or (channel == "st" and not self.kit.player_is_mt):
                rem *= 1.0 - self.tenacity_mit

        group_best_factor: dict[str, float] = {}
        for b in self.active_buffs:
            if b.expires_at < self.t - EPS:
                continue
            if not (b.target_channel in ("party", "boss") or b.target_channel == channel):
                continue
            m_all = b.mit_all
            m_phys = b.mit_phys if damage_type == "physical" else 0.0
            m_mag = b.mit_mag if damage_type == "magical" else 0.0
            factor = (1.0 - m_all) * (1.0 - m_phys) * (1.0 - m_mag)
            if b.stack_group:
                # Always keep the strongest active buff within each stack_group (e.g. player's 10% Reprisal > 5% expected partner Reprisal)
                prev = group_best_factor.get(b.stack_group, 1.0)
                if factor < prev:
                    group_best_factor[b.stack_group] = factor
            else:
                rem *= factor
        for f in group_best_factor.values():
            rem *= f
        return rem

    def _apply_damage_to_channel(
        self,
        channel: str,
        raw_dmg: float,
        damage_type: str,
        forced_mit_factor: float | None = None,
    ) -> bool:
        if self.forced_damage_roll is not None:
            roll = self.forced_damage_roll
        elif self.deterministic:
            roll = 1.0
        else:
            roll = float(self.rng.uniform(1.0 - self.damage_variance, 1.0 + self.damage_variance))
        mit_factor = self.compute_mitigation_factor(channel, damage_type) if forced_mit_factor is None else forced_mit_factor
        dmg = raw_dmg * roll * mit_factor

        # Consume shields earliest-expiry first
        active_sh = sorted(
            [sh for sh in self.shields[channel] if sh.expires_at >= self.t - EPS and sh.remaining > EPS],
            key=lambda s: s.expires_at,
        )
        kept: list[_ActiveShield] = []
        for sh in active_sh:
            if dmg > 0.0:
                absorbed = min(sh.remaining, dmg)
                sh.remaining -= absorbed
                dmg -= absorbed
            if sh.remaining > EPS:
                kept.append(sh)
        self.shields[channel] = kept

        self.hp[channel] -= dmg
        ratio = self.hp[channel] / self.max_hp[channel]
        self.stats[f"min_{channel}_hp_ratio"] = min(self.stats[f"min_{channel}_hp_ratio"], max(0.0, ratio))
        if self.hp[channel] <= 0.0:
            deficit = -self.hp[channel]
            if not self.died:
                self.died = True
                self.first_death_time_s = round(self.t, 3)
                self.first_death_channel = channel
                self.stats["survived"] = 0.0
            self.max_deficit_hp = max(self.max_deficit_hp, deficit)
            return True
        # Trigger any active reactive/delayed heal (e.g. Excogitation / Shadowed Vigil) if HP fell below threshold!
        for b in self.active_buffs:
            if (
                b.delayed_heal > 0.0
                and not b.delayed_triggered
                and b.expires_at >= self.t - EPS
                and (b.target_channel == channel or b.target_channel == "party")
                and (self.hp[channel] / self.max_hp[channel]) < b.delayed_trigger_ratio
            ):
                b.delayed_triggered = True
                self._apply_heal(
                    channel,
                    b.delayed_heal,
                    is_player_heal=(not b.skill_id.startswith("partner_")),
                    single_channel_only=True,
                )
        return False

    def _process_item(self, item: _ScheduledItem) -> None:
        if item.kind == "downtime_start":
            # Clear boss debuffs when boss jumps/becomes untargetable
            self.active_buffs = [b for b in self.active_buffs if b.target_channel != "boss"]

        elif item.kind == "cast_resolve":
            sk: SkillDef = item.payload["skill"]
            ch: str = item.payload["channel"]
            self._apply_skill_effects(sk, ch)

        elif item.kind == "partner":
            pa = item.payload
            dur = float(pa.duration_s)
            if dur > 0 and (pa.mitigation_all > 0 or pa.mitigation_physical > 0 or pa.mitigation_magical > 0):
                self.active_buffs.append(
                    _ActiveBuff(
                        skill_id=f"partner_{pa.skill_name}",
                        stack_group=pa.stack_group,
                        target_channel=pa.target_channel,
                        expires_at=self.t + dur,
                        mit_all=pa.mitigation_all,
                        mit_phys=pa.mitigation_physical,
                        mit_mag=pa.mitigation_magical,
                    )
                )
            if pa.shield_value > 0:
                self._add_shield(
                    pa.target_channel,
                    pa.shield_value,
                    max(15.0, dur),
                    f"partner_{pa.skill_name}",
                    pa.shield_stack_group,
                    single_channel_only=pa.non_tank_only,
                )
            if pa.heal_value > 0:
                self._apply_heal(
                    pa.target_channel,
                    pa.heal_value,
                    single_channel_only=pa.non_tank_only,
                )

        elif item.kind == "snapshot":
            ev: DamageEvent = item.payload
            channels = (
                ("party", "mt", "st")
                if ev.target_scope == "party"
                else (("mt", "st") if ev.target_scope == "both_tanks" else (ev.target_scope,))
            )
            for ch in channels:
                self._event_snapshots_mit[(ev.id, ch)] = self.compute_mitigation_factor(ch, ev.damage_type)

        elif item.kind == "damage_impact":
            ev = item.payload["event"]
            raw = float(item.payload["raw"])
            is_last_hit = bool(item.payload["is_last_hit"])
            channels = (
                ("party", "mt", "st")
                if ev.target_scope == "party"
                else (("mt", "st") if ev.target_scope == "both_tanks" else (ev.target_scope,))
            )
            hit_killed = False
            for ch in channels:
                snap_mit = self._event_snapshots_mit.get((ev.id, ch))
                ch_raw = raw * self.kit.tank_raidwide_damage_ratio if (ev.target_scope == "party" and ch in ("mt", "st")) else raw
                if ev.mechanic_effect == "set_hp_to_1":
                    # Unmitigable HP=1 mechanic (bypasses mitigation and shields)
                    self.hp[ch] = min(self.hp[ch], 1.0)
                    ratio = self.hp[ch] / self.max_hp[ch]
                    self.stats[f"min_{ch}_hp_ratio"] = min(self.stats[f"min_{ch}_hp_ratio"], max(0.0, ratio))
                    killed = False
                elif ev.mechanic_effect == "hp_percent_damage":
                    pct_dmg = self.max_hp[ch] * ev.hp_percent_fraction
                    killed = self._apply_damage_to_channel(ch, pct_dmg, ev.damage_type, forced_mit_factor=1.0)
                else:
                    killed = self._apply_damage_to_channel(ch, ch_raw, ev.damage_type, forced_mit_factor=snap_mit)
                if killed:
                    hit_killed = True
                else:
                    ratio = self.hp[ch] / self.max_hp[ch]
                    if ev.mechanic_effect == "normal" and ratio < ev.min_survival_hp_ratio:
                        self._step_reward_accum -= (ev.min_survival_hp_ratio - ratio) * 6.0
                    elif is_last_hit:
                        self._step_reward_accum += 2.5

                if is_last_hit and ev.doom_full_heal_duration_s is not None and not killed:
                    if self.hp[ch] < self.max_hp[ch] - 1.0:
                        db = _SpecialDebuff(
                            kind="full_heal_doom",
                            target_channel=ch,
                            expires_at=self.t + ev.doom_full_heal_duration_s,
                            stacks_remaining=1,
                        )
                        self.special_debuffs.append(db)
                        self._push(db.expires_at, PRIO_DEBUFF_EXPIRE, "debuff_expire", db)

                if is_last_hit and ev.esuna_required_count > 0 and not killed:
                    # Attach esuna stacks once on primary channel for party-wide debuffs (or per tank for tank-targeted)
                    if ev.target_scope != "party" or ch == "party":
                        db = _SpecialDebuff(
                            kind="esuna_doom",
                            target_channel=ch,
                            expires_at=self.t + ev.esuna_deadline_s,
                            stacks_remaining=ev.esuna_required_count,
                        )
                        self.special_debuffs.append(db)
                        self._push(db.expires_at, PRIO_DEBUFF_EXPIRE, "debuff_expire", db)

                if is_last_hit and ev.dot_total_damage > 0.0:
                    n_ticks = max(1, int(round(ev.dot_duration_s / ev.dot_tick_interval_s)))
                    custom_iv = ev.dot_tick_interval_s if abs(ev.dot_tick_interval_s - 3.0) > 0.05 else None
                    dot = _ActiveDot(
                        source_event_id=ev.id,
                        target_channel=ch,
                        applied_at=self.t,
                        expires_at=self.t + ev.dot_duration_s,
                        ticks_remaining=n_ticks,
                        raw_dmg_per_tick=ev.dot_total_damage / n_ticks,
                        damage_type=ev.damage_type,
                        snapshotted_mit_factor=snap_mit if ev.dot_snapshot_mitigation else None,
                        custom_interval_s=custom_iv,
                    )
                    self.active_dots.append(dot)
                    if custom_iv is not None:
                        self._push(self.t + custom_iv, PRIO_SERVER_TICK, "custom_dot_tick", dot)

            if is_last_hit:
                while self._next_event_idx < len(self.events) and self.events[self._next_event_idx].time_s <= self.t + EPS:
                    self._next_event_idx += 1
                self.event_snapshots.append(
                    ReplayEventSnapshot(
                        event_id=ev.id,
                        time_s=round(self.t, 3),
                        name=ev.name,
                        target_scope=ev.target_scope,
                        hp_after={k: round(v, 1) for k, v in self.hp.items()},
                        shield_after={k: round(self.total_shield(k), 1) for k in ("party", "mt", "st")},
                        mp_after=round(self.resources.get("mp", 0.0), 1),
                        died=hit_killed or self.died,
                    )
                )

        elif item.kind == "player_aa":
            # Player weapon auto-attack against targetable boss (generates Paladin Oath Gauge etc.; pauses in downtime!)
            if self.timeline.is_boss_targetable(self.t) and self.timeline.can_party_act(self.t):
                for rname, pool in self.resolved_pools.items():
                    if pool.gain_per_player_aa > 0.0:
                        self.resources[rname] = min(
                            pool.maximum,
                            self.resources.get(rname, 0.0) + pool.gain_per_player_aa,
                        )
            nxt_paa = self.t + self.kit.player_aa_interval_s
            if nxt_paa <= self.timeline.duration_s + EPS:
                self._push(nxt_paa, PRIO_SERVER_TICK, "player_aa", None)

        elif item.kind == "auto_attack":
            seg = item.payload
            if self.timeline.is_boss_targetable(self.t):
                targets = ("mt", "st") if seg.target == "both_tanks" else (seg.target,)
                for ch in targets:
                    self._apply_damage_to_channel(ch, seg.raw_damage_per_hit, seg.damage_type)

        elif item.kind == "server_tick":
            # Natural Piety MP tick + active explicit MP regen buffs (e.g. Lucid Dreaming)
            if "mp" in self.resources:
                mp_max = self.resolved_pools["mp"].maximum
                extra_mp = sum(b.mp_per_tick for b in self.active_buffs if b.expires_at >= self.t - EPS and b.mp_per_tick > 0)
                self.resources["mp"] = min(mp_max, self.resources["mp"] + self.base_mp_per_3s_tick + extra_mp)
            # Active HoT ticks
            for b in self.active_buffs:
                if b.expires_at >= self.t - EPS and b.hot_per_tick > 0:
                    oh = self._apply_heal(b.target_channel, b.hot_per_tick, is_player_heal=(not b.skill_id.startswith("partner_")))
                    if not b.skill_id.startswith("partner_"):
                        self._step_reward_accum -= oh * 0.02
            # Automatic single-target pet/fairy heal (e.g. Eos Embrace) onto ONLY the single lowest-HP-ratio channel
            if self.kit.pet_auto_heal_per_3s_tick > 0:
                lowest_ch = min(("mt", "st", "party"), key=lambda c: self.hp[c] / self.max_hp[c])
                if self.hp[lowest_ch] < self.max_hp[lowest_ch] - 1.0:
                    self._apply_heal(
                        lowest_ch,
                        self.kit.pet_auto_heal_per_3s_tick * self.heal_scale,
                        is_player_heal=True,
                        single_channel_only=True,
                    )
            # Partner baseline healing (co-healer GCD/short-CD healing & tank self-sustain from FFLogs)
            pbh = self.timeline.partner_baseline_healing
            if self.partner_trust_scale > 0:
                if pbh.party_per_3s_tick > 0:
                    self._apply_heal("party", pbh.party_per_3s_tick * self.partner_trust_scale)
                if pbh.mt_per_3s_tick > 0:
                    self._apply_heal("mt", pbh.mt_per_3s_tick * self.partner_trust_scale)
                if pbh.st_per_3s_tick > 0:
                    self._apply_heal("st", pbh.st_per_3s_tick * self.partner_trust_scale)
            # Optional passive baseline tank self-sustain
            if self.tank_passive_self_heal_per_tick > 0:
                self._apply_heal("mt", self.tank_passive_self_heal_per_tick)
                self._apply_heal("st", self.tank_passive_self_heal_per_tick)
            # Active 3s DoT bleed ticks (strictly strictly after application & bounded by ticks_remaining)
            for d in self.active_dots:
                if (
                    d.custom_interval_s is None
                    and d.ticks_remaining > 0
                    and self.t > d.applied_at + EPS
                    and d.expires_at >= self.t - EPS
                ):
                    d.ticks_remaining -= 1
                    self._apply_damage_to_channel(
                        d.target_channel,
                        d.raw_dmg_per_tick,
                        d.damage_type,
                        forced_mit_factor=d.snapshotted_mit_factor,
                    )
            self.next_server_tick = self.t + 3.0
            if self.next_server_tick <= self.timeline.duration_s + EPS:
                self._push(self.next_server_tick, PRIO_SERVER_TICK, "server_tick", None)

        elif item.kind == "custom_dot_tick":
            d: _ActiveDot = item.payload
            if d.ticks_remaining > 0 and d.expires_at >= self.t - EPS:
                d.ticks_remaining -= 1
                self._apply_damage_to_channel(
                    d.target_channel,
                    d.raw_dmg_per_tick,
                    d.damage_type,
                    forced_mit_factor=d.snapshotted_mit_factor,
                )
                if d.ticks_remaining > 0 and d.custom_interval_s is not None:
                    nxt = self.t + d.custom_interval_s
                    if nxt <= min(d.expires_at, self.timeline.duration_s) + EPS:
                        self._push(nxt, PRIO_SERVER_TICK, "custom_dot_tick", d)

        elif item.kind == "debuff_expire":
            db: _SpecialDebuff = item.payload
            if not db.cleared:
                # Uncleansed Esuna debuff or unhealed Full-Heal Doom -> instant lethal wipe!
                self.hp[db.target_channel] = 0.0
                if not self.died:
                    self.died = True
                    self.first_death_time_s = round(self.t, 3)
                    self.first_death_channel = f"{db.target_channel}({db.kind})"
                    self.stats["survived"] = 0.0
                self.max_deficit_hp = max(self.max_deficit_hp, self.max_hp[db.target_channel])

        elif item.kind == "hp_trigger_expire":
            tr_state: _HpTriggerState = item.payload
            if tr_state.spec.must_trigger_before_end and tr_state.triggers_remaining > 0:
                ch = tr_state.spec.watch_channel
                self.hp[ch] = 0.0
                if not self.died:
                    self.died = True
                    self.first_death_time_s = round(self.t, 3)
                    self.first_death_channel = f"{ch}(hp_trigger_expired)"
                    self.stats["survived"] = 0.0
                self.max_deficit_hp = max(self.max_deficit_hp, self.max_hp[ch])

        elif item.kind == "raise":
            raise_cost = self.kit.gear_stats.raise_mp_cost
            if "mp" in self.resources:
                if self.resources["mp"] >= raise_cost:
                    self.resources["mp"] -= raise_cost
                    self.stats["mp_min"] = min(self.stats["mp_min"], self.resources["mp"])
                    self.gcd_ready_at = max(self.gcd_ready_at, self.t + self.kit.gcd_s)
                    self.stats["raises_handled"] += 1.0
                else:
                    self._step_reward_accum -= 6.0

    def advance_to(self, target_t: float) -> float:
        """Advance simulation chronologically to `target_t`, executing every intermediate event in exact order."""
        target_t = min(target_t, self.timeline.duration_s)
        self._step_reward_accum = 0.0
        while self._queue and self._queue[0].time_s <= target_t + EPS:
            item = heapq.heappop(self._queue)
            self._update_continuous_state(item.time_s)
            self._process_item(item)
            if self.died:
                return self._step_reward_accum
        if target_t > self.t:
            self._update_continuous_state(target_t)
        return self._step_reward_accum

    def _cast_overlaps_movement(self, start_t: float, sk: SkillDef) -> float:
        """Return max interrupt probability if non-slidecast portion of cast overlaps a MovementWindow."""
        if sk.cast_time_s <= 0.0:
            return 0.0
        vulnerable_end = start_t + max(0.0, sk.cast_time_s - sk.slidecast_window_s)
        if vulnerable_end <= start_t + EPS:
            return 0.0
        max_p = 0.0
        for mw in self.timeline.movement_windows:
            if start_t < mw.end_s and vulnerable_end > mw.start_s:
                max_p = max(max_p, mw.cast_interrupt_prob)
        return max_p

    def _apply_skill_effects(self, sk: SkillDef, channel: str) -> None:
        hot_ticks = max(1, int(round(sk.duration_s / 3.0))) if sk.duration_s >= 3.0 else 1
        hot_per_tick = (sk.heal_hot_total * self.heal_scale) / hot_ticks if sk.heal_hot_total > 0 else 0.0
        delayed_heal = sk.delayed_heal_value * self.heal_scale if sk.delayed_heal_value > 0 else 0.0
        if sk.duration_s > 0 and (
            sk.mitigation_all > 0
            or sk.mitigation_physical > 0
            or sk.mitigation_magical > 0
            or hot_per_tick > 0
            or sk.mp_regen_per_tick > 0
            or delayed_heal > 0
        ):
            self.active_buffs.append(
                _ActiveBuff(
                    skill_id=sk.id,
                    stack_group=sk.stack_group,
                    target_channel=channel,
                    expires_at=self.t + sk.duration_s,
                    mit_all=sk.mitigation_all,
                    mit_phys=sk.mitigation_physical,
                    mit_mag=sk.mitigation_magical,
                    hot_per_tick=hot_per_tick,
                    mp_per_tick=sk.mp_regen_per_tick,
                    delayed_heal=delayed_heal,
                    delayed_trigger_ratio=sk.delayed_heal_trigger_ratio,
                )
            )
        if sk.shield_max_hp_ratio > 0:
            # % Max HP shields (e.g. TBN = 25% Max HP) scale with target Max HP, NOT heal_scale!
            targets = ("party", "mt", "st") if channel == "party" else (channel,)
            for ch in targets:
                self._add_shield(
                    ch,
                    self.max_hp[ch] * sk.shield_max_hp_ratio,
                    sk.effective_shield_duration(),
                    sk.id,
                    sk.shield_stack_group,
                    single_channel_only=True,
                )
        elif sk.shield_value > 0:
            self._add_shield(
                channel,
                sk.shield_value * self.heal_scale,
                sk.effective_shield_duration(),
                sk.id,
                sk.shield_stack_group,
            )
        if sk.heal_instant > 0:
            overheal = self._apply_heal(channel, sk.heal_instant * self.heal_scale, is_player_heal=True)
            self._step_reward_accum -= overheal * (0.15 if sk.is_gcd else 0.06)
        if sk.cleanses_debuff:
            if self._apply_cleanse(channel):
                self._step_reward_accum += 2.0

    def action_mask(self) -> np.ndarray:
        """Return exact boolean legal-action mask at current timestamp `self.t`."""
        mask = np.zeros(self.n_actions, dtype=np.bool_)
        mask[0] = True  # WAIT always legal

        if not self.timeline.can_party_act(self.t) or self.t < self.casting_until - EPS:
            return mask

        boss_ok = self.timeline.is_boss_targetable(self.t)
        gcd_ok = self.t >= self.gcd_ready_at - EPS
        anim_ok = self.t >= self.anim_lock_until - EPS

        # Action 1: Filler Attack GCD
        if boss_ok and gcd_ok and anim_ok:
            filler_mp = self.kit.gear_stats.filler_gcd_mp_cost
            if filler_mp <= 0 or self.resources.get("mp", 10_000.0) >= filler_mp - EPS:
                mask[1] = True

        # Actions 2..N: Owned skills
        for entry in self.actions[2:]:
            sk = entry.skill
            assert sk is not None
            if not anim_ok:
                continue
            if self.skill_charges[sk.id] <= 0:
                continue
            if sk.needs_boss_target() and not boss_ok:
                continue
            if sk.is_gcd:
                if not gcd_ok:
                    continue
                if self._cast_overlaps_movement(self.t, sk) >= 0.8:
                    continue
            else:
                if self.weave_slots < sk.weave_cost:
                    continue
            if any(self.resources.get(rname, 0.0) < rcost - EPS for rname, rcost in sk.resource_costs.items()):
                continue
            mask[entry.index] = True

        return mask

    def execute_action(self, action_idx: int) -> float:
        """Execute one player decision at `self.t`; returns immediate reward delta."""
        mask = self.action_mask()
        if action_idx < 0 or action_idx >= self.n_actions or not mask[action_idx]:
            action_idx = 0

        entry = self.actions[action_idx]
        reward = 0.0

        if entry.kind == "filler":
            self.gcd_ready_at = self.t + self.kit.gcd_s
            self.anim_lock_until = self.t + 0.65
            self.weave_slots = self.kit.weave_budget_at(self.t)
            filler_mp = self.kit.gear_stats.filler_gcd_mp_cost
            if filler_mp > 0 and "mp" in self.resources:
                self.resources["mp"] = max(0.0, self.resources["mp"] - filler_mp)
                self.stats["mp_min"] = min(self.stats["mp_min"], self.resources["mp"])
            # Attack-triggered heals (e.g. Kardia) & gauges (e.g. Oath) fire ONLY on boss attack!
            if self.kit.on_attack_heal_target != "none" and self.kit.on_attack_heal_value > 0:
                self._apply_heal(self.kit.on_attack_heal_target, self.kit.on_attack_heal_value * self.heal_scale)
            for rname, pool in self.resolved_pools.items():
                if pool.gain_per_filler_gcd > 0:
                    self.resources[rname] = min(pool.maximum, self.resources.get(rname, 0.0) + pool.gain_per_filler_gcd)
            self.stats["filler_gcds"] += 1.0
            burst_mult = self.kit.burst_dps_multiplier if self.kit.is_in_burst(self.t) else 1.0
            self.stats["expected_potency_gained"] += self.kit.filler_gcd_potency * burst_mult
            reward += 0.15 * burst_mult

        elif entry.kind == "skill":
            sk = entry.skill
            assert sk is not None
            # Consume cooldown charge
            was_full = self.skill_charges[sk.id] == sk.max_charges
            self.skill_charges[sk.id] -= 1
            if was_full and sk.cooldown_s > 0.0:
                self.skill_next_charge_at[sk.id] = self.t + sk.cooldown_s

            # Pay/grant resources
            for rname, rcost in sk.resource_costs.items():
                self.resources[rname] = max(0.0, self.resources.get(rname, 0.0) - rcost)
            for rname, rgrant in sk.resource_grants.items():
                rmax = self.resolved_pools[rname].maximum if rname in self.resolved_pools else 10_000.0
                self.resources[rname] = min(rmax, self.resources.get(rname, 0.0) + rgrant)
            if "mp" in self.resources:
                self.stats["mp_min"] = min(self.stats["mp_min"], self.resources["mp"])

            # Check downtime DPS-loss exemption for pre-shields that finish before boss returns
            dw = self.timeline.downtime_at(self.t)
            is_free_downtime_gcd = False
            if sk.is_gcd:
                self.gcd_ready_at = self.t + max(self.kit.gcd_s, sk.cast_time_s)
                self.weave_slots = self.kit.weave_budget_at(self.t)
                if dw is not None and not dw.boss_targetable and self.gcd_ready_at <= dw.end_s + EPS:
                    is_free_downtime_gcd = True
                    self.stats["downtime_free_gcds"] += 1.0
                else:
                    self.stats["gcd_heals"] += 1.0
            else:
                self.weave_slots = max(0, self.weave_slots - sk.weave_cost)
                self.stats["ogcds_used"] += 1.0

            # Cast resolution vs instant resolution
            intr_p = self._cast_overlaps_movement(self.t, sk)
            interrupted = False
            if intr_p > 0.0:
                interrupted = (intr_p >= 0.5) if self.deterministic else (float(self.rng.random()) < intr_p)

            if interrupted:
                self.casting_until = self.t + sk.cast_time_s * 0.5
                self.anim_lock_until = self.casting_until
                self.stats["cast_interrupts"] += 1.0
                reward -= 1.5
            elif sk.cast_time_s > 0.0:
                self.casting_until = self.t + sk.cast_time_s
                self.anim_lock_until = self.casting_until + 0.1
                self._push(self.casting_until, PRIO_CAST_RESOLVE, "cast_resolve", {"skill": sk, "channel": entry.target_channel})
            else:
                self.anim_lock_until = self.t + sk.animation_lock_s
                self._apply_skill_effects(sk, entry.target_channel)

            pot_loss = self.kit.expected_skill_potency_loss(sk, self.t, is_free_downtime=is_free_downtime_gcd)
            self.stats["expected_potency_lost"] += pot_loss
            # Track resource opportunity potency separately (un-multiplied by burst since Aetherflow can be spent at any time)
            res_pot = sum(
                rcost * self.resolved_pools[rname].dps_potency_per_unit
                for rname, rcost in sk.resource_costs.items()
                if rname in self.resolved_pools and self.resolved_pools[rname].dps_potency_per_unit > 0.0
            )
            self.stats["resource_potency_lost"] += res_pot
            if not is_free_downtime_gcd:
                # Displaced filler GCD already forfeits +0.15 filler reward (no double penalty!);
                # only penalize damage-convertible resource opportunity cost (e.g. Aetherflow -> Energy Drain)
                # plus a tiny tie-breaker so 0p oGCDs are strictly preferred over costly heals!
                res_pen = (res_pot / max(1.0, self.kit.filler_gcd_potency)) * 0.15
                burst_extra = (0.04 if (sk.is_gcd and self.kit.is_in_burst(self.t)) else 0.0)
                reward -= (res_pen + burst_extra + sk.dps_cost * 0.03)

            self.action_log.append(
                {
                    "time_s": round(self.t, 3),
                    "skill_id": sk.id,
                    "skill_name": sk.name,
                    "target": entry.target_channel,
                    "is_gcd": sk.is_gcd,
                    "free_downtime_gcd": is_free_downtime_gcd,
                    "mp_after": round(self.resources.get("mp", 0.0), 1),
                }
            )

        # MP reserve floor soft penalty
        if "mp" in self.resources and self.mp_floor > 0 and self.resources["mp"] < self.mp_floor:
            self.stats["mp_reserve_breaches"] += 1.0
            reward -= 0.15 * ((self.mp_floor - self.resources["mp"]) / max(1.0, self.mp_floor))

        return reward

    def next_decision_time(self, max_wait_s: float = 1.0) -> float:
        """Compute exact next timestamp where the player can make a meaningful decision."""
        candidates = [self.timeline.duration_s, self.t + max_wait_s]
        for cand in (self.gcd_ready_at, self.anim_lock_until, self.casting_until):
            if cand > self.t + EPS:
                candidates.append(cand)
        if self._queue and self._queue[0].time_s > self.t + EPS:
            candidates.append(self._queue[0].time_s)
        for dw in self.timeline.downtime_windows:
            if dw.end_s > self.t + EPS:
                candidates.append(dw.end_s)
        return min(candidates)

    def build_verification(self) -> ReplayVerification:
        raw_h = self.stats.get("player_heal_raw", 0.0)
        eff_h = self.stats.get("player_heal_eff", 0.0)
        oh_pct = ((raw_h - eff_h) / raw_h) * 100.0 if raw_h > EPS else 0.0
        pot_gain = self.stats.get("expected_potency_gained", 0.0)
        pot_loss = self.stats.get("expected_potency_lost", 0.0)
        res_pot_loss = self.stats.get("resource_potency_lost", 0.0)
        # Net expected DPS potency: exact filler attack potency gained minus damage-convertible resources spent on heals
        # (zero double-counting of displaced filler GCDs!)
        total_pot = max(0.0, pot_gain - res_pot_loss)
        dps_eff = (total_pot / max(1.0, total_pot + pot_loss)) * 100.0 if (total_pot + pot_loss) > EPS else 100.0
        return ReplayVerification(
            survived=not self.died,
            first_death_time_s=self.first_death_time_s,
            first_death_channel=self.first_death_channel,
            deficit_hp=round(self.max_deficit_hp, 1),
            min_hp_ratio={
                "party": round(self.stats["min_party_hp_ratio"], 4),
                "mt": round(self.stats["min_mt_hp_ratio"], 4),
                "st": round(self.stats["min_st_hp_ratio"], 4),
            },
            filler_gcds=int(self.stats["filler_gcds"]),
            gcd_heals=int(self.stats["gcd_heals"]),
            downtime_free_gcds=int(self.stats["downtime_free_gcds"]),
            overheal_pct=round(oh_pct, 1),
            mp_min=round(self.stats["mp_min"], 1),
            expected_total_potency=round(total_pot, 1),
            expected_potency_lost=round(pot_loss, 1),
            resource_potency_lost=round(res_pot_loss, 1),
            dps_efficiency_pct=round(dps_eff, 2),
            hp_triggers_fired=int(self.stats.get("hp_triggers_fired", 0.0)),
            event_snapshots=list(self.event_snapshots),
        )


def replay_plan(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    plan: OptimizationPlan,
    deterministic: bool = True,
    fixed_tick_phase: float = 1.5,
    start_time_s: float = 0.0,
    initial_hp_ratios: dict[str, float] | None = None,
    initial_resources: dict[str, float] | None = None,
    initial_cooldowns: dict[str, float] | None = None,
    forced_damage_roll: float | None = None,
) -> ReplayVerification:
    """Replay a planned action schedule chronologically through the exact CombatEngine."""
    engine = CombatEngine(
        kit=kit,
        timeline=timeline,
        damage_variance=0.0 if deterministic else 0.05,
        raise_prob_per_pull=0.0,
        partner_trust_scale=plan.partner_trust,
        tank_passive_self_heal_per_tick=0.0,
        forced_damage_roll=forced_damage_roll,
    )
    engine.reset(
        deterministic=deterministic,
        fixed_tick_phase=fixed_tick_phase,
        start_time_s=start_time_s,
        initial_hp_ratios=initial_hp_ratios,
        initial_resources=initial_resources,
        initial_cooldowns=initial_cooldowns,
    )

    # Map (skill_id, target_channel) -> action index
    idx_by_key: dict[tuple[str, str], int] = {}
    idx_by_skill: dict[str, int] = {}
    for entry in engine.actions[2:]:
        assert entry.skill is not None
        idx_by_key[(entry.skill.id, entry.target_channel)] = entry.index
        idx_by_skill.setdefault(entry.skill.id, entry.index)

    scheduled = sorted(plan.actions, key=lambda a: (a.cast_at_s, 0 if a.is_gcd else 1))
    ptr = 0
    executed_count = 0
    dropped_count = 0

    while engine.t < timeline.duration_s - EPS and not engine.died:
        # Execute any planned actions due at or before current t
        acted = False
        while ptr < len(scheduled) and scheduled[ptr].cast_at_s <= engine.t + EPS:
            act = scheduled[ptr]
            aidx = idx_by_key.get((act.skill_id, act.target_channel), idx_by_skill.get(act.skill_id, 0))
            sk = next((s for s in kit.skills if s.id == act.skill_id), None)
            if aidx < 2 or sk is None:
                ptr += 1
                dropped_count += 1
                continue
            # Check strict timing deadline BEFORE attempting execution so an on-cooldown cast never silently slides by 5s+!
            if engine.t > act.cast_at_s + 2.5:
                ptr += 1
                dropped_count += 1
                continue
            mask = engine.action_mask()
            if mask[aidx]:
                engine.execute_action(aidx)
                ptr += 1
                executed_count += 1
                acted = True
            else:
                if sk.needs_boss_target() and not timeline.is_boss_targetable(engine.t):
                    ptr += 1
                    dropped_count += 1
                    continue
                break

        if engine.died:
            break

        # Cast filler attack GCD if ready AND won't collide with an imminent planned GCD action
        mask = engine.action_mask()
        if mask[1]:
            imminent_gcd = (
                ptr < len(scheduled)
                and scheduled[ptr].is_gcd
                and scheduled[ptr].cast_at_s < engine.t + kit.gcd_s - EPS
            )
            if not imminent_gcd:
                engine.execute_action(1)
                acted = True

        # Advance to earliest of: next planned action time or engine's next decision time
        next_t = engine.next_decision_time(max_wait_s=1.0)
        if ptr < len(scheduled) and scheduled[ptr].cast_at_s > engine.t + EPS:
            next_t = min(next_t, scheduled[ptr].cast_at_s)
        if next_t <= engine.t + EPS and not acted:
            next_t = min(timeline.duration_s, engine.t + 0.05)
        engine.advance_to(next_t)

    if not engine.died and ptr < len(scheduled):
        dropped_count += len(scheduled) - ptr

    ver = engine.build_verification()
    ver.executed_actions = executed_count
    ver.dropped_actions = dropped_count
    return ver

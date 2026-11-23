"""Skill-agnostic data models for FF14 mitigation & healing planning.

No job names or expansion-specific rules are hardcoded here: all abilities,
gear/substat formulas, resource pools, GCD/weave budgets, timelines, and
partner probabilities are pure data so the planner survives job reworks
and level-cap increases via YAML updates.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DamageType = Literal["physical", "magical", "special"]
SkillTargetScope = Literal["party", "self", "single_ally", "mt", "st", "boss_debuff"]
EventTargetScope = Literal["party", "mt", "st", "both_tanks"]
RoleType = Literal["healer", "tank", "dps", "custom"]
ObjectivePreset = Literal["balanced", "dps_greedy", "safe", "solo_carry"]


class LevelStatConstants(BaseModel):
    """Level-cap formula constants (defaults: Lv.100; editable for future caps)."""

    level: int = 100
    base_substat: int = 420
    base_piety: int = 440
    level_div: int = 2780
    tenacity_mit_coeff: float = 112.0   # floor(coeff * (TEN - base_sub) / div) / 1000
    piety_mp_coeff: float = 150.0       # +floor(coeff * (PIE - base_pie) / div) MP per 3s tick
    base_combat_mp_per_tick: float = 200.0


class GearStats(BaseModel):
    """Player gear & substat parameters that affect mitigation, healing, and MP economy."""

    tenacity: int = Field(default=420, ge=0, description="Tank Tenacity (TEN) -> passive mitigation")
    piety: int = Field(default=440, ge=0, description="Healer Piety (PIE) -> combat MP regen per 3s tick")
    determination: int = Field(default=2000, ge=0, description="Determination (DET)")
    spell_or_skill_speed: int = Field(default=420, ge=0, description="SS/SKS (informational; gcd_s drives timing)")
    heal_scale_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        description="Direct multiplier on kit heal/shield values (e.g. from higher ilvl / MND / DET)",
    )
    filler_gcd_mp_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Baseline MP spent per filler attack GCD (e.g. 400 MP for healers, 0 for tanks)",
    )
    lucid_dreaming_mp_per_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Average MP/s from regular MP-recovery CDs when Lucid is not modeled as an explicit skill",
    )
    raise_mp_cost: float = Field(default=2400.0, ge=0.0, description="MP cost of one Raise")
    planned_raise_reserve: int = Field(
        default=0,
        ge=0,
        description="Number of emergency Raises worth of MP to keep in reserve at all times",
    )
    constants: LevelStatConstants = Field(default_factory=LevelStatConstants)

    def tenacity_passive_mitigation(self, role: RoleType) -> float:
        """Compute passive damage reduction from Tenacity (applies to tanks)."""
        if role != "tank":
            return 0.0
        delta = max(0, self.tenacity - self.constants.base_substat)
        permille = math.floor(self.constants.tenacity_mit_coeff * delta / self.constants.level_div)
        return max(0.0, min(0.30, permille / 1000.0))

    def combat_mp_per_3s_tick(self) -> float:
        """Exact natural combat MP regained per 3.0s server tick from Piety."""
        delta = max(0, self.piety - self.constants.base_piety)
        bonus_per_tick = math.floor(self.constants.piety_mp_coeff * delta / self.constants.level_div)
        return self.constants.base_combat_mp_per_tick + bonus_per_tick

    def combat_mp_regen_per_sec(self) -> float:
        """Combat MP regen per second from Piety + optional averaged Lucid."""
        return (self.combat_mp_per_3s_tick() / 3.0) + self.lucid_dreaming_mp_per_sec

    def net_baseline_mp_per_sec(self, gcd_s: float) -> float:
        """Net MP/s before mitigation/heal skills = (PIE regen + Lucid) - (filler GCD drain)."""
        filler_drain = self.filler_gcd_mp_cost / gcd_s if gcd_s > 0 else 0.0
        return self.combat_mp_regen_per_sec() - filler_drain

    def mp_floor(self) -> float:
        """Minimum MP floor reserved for emergency raises."""
        return self.raise_mp_cost * self.planned_raise_reserve


class ResourcePool(BaseModel):
    """Generic named resource pool (MP, Addersgall, Aetherflow, Lily, Oath, etc.)."""

    name: str
    initial: float = 0.0
    maximum: float = 100.0
    regen_per_sec: float = 0.0
    gain_per_filler_gcd: float = Field(
        default=0.0,
        ge=0.0,
        description="Resource gained on attacking the boss (pauses automatically during downtime)",
    )
    min_reserve: float = Field(default=0.0, ge=0.0, description="Minimum balance to keep in reserve")


class SkillDef(BaseModel):
    """Generic representation of any defensive, barrier, healing, or MP recovery action."""

    id: str
    name: str
    action_id: int | None = None
    cooldown_s: float = Field(ge=0.0)
    max_charges: int = Field(default=1, ge=1)
    duration_s: float = Field(default=0.0, ge=0.0)
    cast_time_s: float = Field(default=0.0, ge=0.0)
    slidecast_window_s: float = Field(default=0.5, ge=0.0)
    animation_lock_s: float = Field(default=0.65, ge=0.0)
    is_gcd: bool = False
    weave_cost: int = Field(default=1, ge=0, le=2)
    target_scope: SkillTargetScope = "party"
    requires_targetable_boss: bool = Field(
        default=False,
        description="True for boss debuffs (Reprisal/Feint/Addle) or attacks unavailable during downtime",
    )

    # Defensive / recovery effects
    mitigation_all: float = Field(default=0.0, ge=0.0, le=1.0)
    mitigation_physical: float = Field(default=0.0, ge=0.0, le=1.0)
    mitigation_magical: float = Field(default=0.0, ge=0.0, le=1.0)
    stack_group: str | None = None
    shield_value: float = Field(default=0.0, ge=0.0)
    shield_duration_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Override shield duration (defaults to max(duration_s, 30s) when shield_value > 0)",
    )
    shield_stack_group: str | None = Field(
        default=None,
        description="Shields sharing the same non-null group overwrite (e.g. 'gcd_barrier'); null stacks independently",
    )
    heal_instant: float = Field(default=0.0, ge=0.0)
    heal_hot_total: float = Field(default=0.0, ge=0.0)
    cleanses_debuff: bool = Field(
        default=False,
        description="True for cleanse skills (e.g. Esuna) that remove one cleansable debuff stack",
    )
    mp_regen_per_tick: float = Field(
        default=0.0,
        ge=0.0,
        description="MP restored per 3s server tick while buff is active (e.g. Lucid Dreaming = 550)",
    )

    # Generic resource interactions
    resource_costs: dict[str, float] = Field(default_factory=dict)
    resource_grants: dict[str, float] = Field(default_factory=dict)

    # Scheduling / ergonomic metadata
    lead_time_s: float = Field(default=1.5, ge=0.0)
    dps_cost: float = Field(default=0.05, ge=0.0)
    notes: str = ""

    def effective_mitigation(self, damage_type: DamageType) -> float:
        """Return effective fractional damage reduction for a given damage type."""
        factors = [1.0 - self.mitigation_all]
        if damage_type == "physical":
            factors.append(1.0 - self.mitigation_physical)
        elif damage_type == "magical":
            factors.append(1.0 - self.mitigation_magical)
        remaining = 1.0
        for f in factors:
            remaining *= f
        return max(0.0, min(1.0, 1.0 - remaining))

    def needs_boss_target(self) -> bool:
        return self.requires_targetable_boss or self.target_scope == "boss_debuff"

    def effective_shield_duration(self) -> float:
        if self.shield_duration_s is not None:
            return self.shield_duration_s
        return self.duration_s if self.duration_s > 0 else 30.0

    def applies_to_event(self, event_scope: EventTargetScope, player_role: RoleType, player_is_mt: bool = True) -> bool:
        """Check whether this skill can mitigate/heal an event with the given target scope."""
        if self.target_scope in ("party", "boss_debuff"):
            return True
        if event_scope == "party":
            return False
        if self.target_scope == "single_ally":
            return True
        if self.target_scope == "self":
            if player_role != "tank":
                return False
            if event_scope == "both_tanks":
                return True
            return (event_scope == "mt" and player_is_mt) or (event_scope == "st" and not player_is_mt)
        if self.target_scope == event_scope:
            return True
        return False


class WeaveOverrideWindow(BaseModel):
    """Explicit weave availability window (e.g., extracted from own FFLogs or downtime)."""

    start_s: float
    end_s: float
    weave_slots_per_gcd: int = Field(ge=0, le=2)
    is_downtime: bool = False


class PrePullAction(BaseModel):
    """Action prepared during the countdown before pull (t < 0s, 0 DPS loss, MP resets to full)."""

    skill_id: str
    offset_s: float = Field(default=-3.0, le=0.0, description="Seconds before pull (e.g. -3.0s for Pre-pull Deploy)")
    target_channel: Literal["party", "mt", "st"] = "party"
    override_shield_value: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional override for combo prep (e.g. Recitation+Protraction+Adlo+Deploy crit shield)",
    )
    consumes_cooldown: bool = True


class GearSamplingRange(BaseModel):
    """Optional parameter ranges for gear-conditioned RL training (so one model adapts to any gear input)."""

    gcd_min: float = 2.28
    gcd_max: float = 2.50
    piety_min: int = 440
    piety_max: int = 1600
    tenacity_min: int = 420
    tenacity_max: int = 2200
    heal_scale_min: float = 0.90
    heal_scale_max: float = 1.15
    hp_scale_min: float = 0.92
    hp_scale_max: float = 1.12


class PlayerKit(BaseModel):
    """User's editable skill kit, gear stats, and GCD/weave profile (completely job-agnostic)."""

    label: str
    role: RoleType = "healer"
    player_is_mt: bool = True
    gcd_s: float = Field(default=2.50, gt=1.0, le=3.5)
    base_weave_slots_per_gcd: int = Field(default=1, ge=0, le=2)
    burst_interval_s: float = Field(default=120.0, gt=0.0)
    burst_duration_s: float = Field(default=20.0, ge=0.0)
    burst_offset_s: float = Field(default=5.0, ge=0.0)
    burst_weave_slots_per_gcd: int = Field(default=0, ge=0, le=2)
    max_hp_party: float = Field(default=120_000.0, gt=0.0)
    max_hp_tank: float = Field(default=190_000.0, gt=0.0)
    tank_raidwide_damage_ratio: float = Field(
        default=1.0,
        gt=0.1,
        le=1.0,
        description="Tank armor + Tank Mastery ratio on party-wide hits (set ~0.62 for Lv.100 Savage kits)",
    )
    on_attack_heal_target: Literal["none", "mt", "st"] = Field(
        default="none",
        description="Optional passive heal on filler attack GCD (e.g. Kardia -> MT); pauses during downtime",
    )
    on_attack_heal_value: float = Field(default=0.0, ge=0.0)
    gear_stats: GearStats = Field(default_factory=GearStats)
    resources: list[ResourcePool] = Field(default_factory=list)
    skills: list[SkillDef] = Field(default_factory=list)
    prepull_actions: list[PrePullAction] = Field(
        default_factory=list,
        description="Countdown prep actions cast before t=0 (e.g. -3s Pre-pull Deploy / Shield / Star)",
    )
    weave_overrides: list[WeaveOverrideWindow] = Field(default_factory=list)

    def weave_budget_at(self, time_s: float) -> int:
        """Return available oGCD weave slots per GCD around `time_s`."""
        for ov in self.weave_overrides:
            if ov.start_s <= time_s <= ov.end_s:
                return ov.weave_slots_per_gcd
        if self.burst_duration_s > 0 and time_s >= self.burst_offset_s:
            phase = (time_s - self.burst_offset_s) % self.burst_interval_s
            if phase < self.burst_duration_s:
                return self.burst_weave_slots_per_gcd
        return self.base_weave_slots_per_gcd

    def passive_self_mitigation_for(self, event_scope: EventTargetScope) -> float:
        """Passive mitigation from gear stats (e.g. Tank Tenacity on self-targeted hits)."""
        if self.role != "tank":
            return 0.0
        if event_scope == "both_tanks":
            return self.gear_stats.tenacity_passive_mitigation(self.role)
        if (event_scope == "mt" and self.player_is_mt) or (event_scope == "st" and not self.player_is_mt):
            return self.gear_stats.tenacity_passive_mitigation(self.role)
        return 0.0

    def resolved_resources(self) -> list[ResourcePool]:
        """Return resource pools with GearStats MP economy applied automatically to `mp`."""
        pools: list[ResourcePool] = []
        has_mp = False
        has_explicit_lucid = any(sk.mp_regen_per_tick > 0 for sk in self.skills)
        for pool in self.resources:
            if pool.name.lower() == "mp":
                has_mp = True
                if (
                    self.gear_stats.filler_gcd_mp_cost > 0
                    or self.gear_stats.lucid_dreaming_mp_per_sec > 0
                    or self.gear_stats.piety != 440
                ):
                    base_regen = self.gear_stats.net_baseline_mp_per_sec(self.gcd_s)
                else:
                    base_regen = pool.regen_per_sec
                mp_reserve = max(pool.min_reserve, self.gear_stats.mp_floor())
                pools.append(
                    ResourcePool(
                        name=pool.name,
                        initial=pool.initial,
                        maximum=pool.maximum,
                        regen_per_sec=base_regen,
                        gain_per_filler_gcd=pool.gain_per_filler_gcd,
                        min_reserve=mp_reserve,
                    )
                )
            else:
                pools.append(pool)
        if not has_mp and self.role == "healer" and (
            self.gear_stats.filler_gcd_mp_cost > 0 or self.gear_stats.planned_raise_reserve > 0
        ):
            pools.append(
                ResourcePool(
                    name="mp",
                    initial=10_000.0,
                    maximum=10_000.0,
                    regen_per_sec=self.gear_stats.net_baseline_mp_per_sec(self.gcd_s),
                    min_reserve=self.gear_stats.mp_floor(),
                )
            )
        return pools

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlayerKit":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class LiveFightState(BaseModel):
    """Mid-fight state snapshot for fast real-time replanning (e.g. after a Raise or slip)."""

    current_time_s: float = Field(ge=0.0)
    cooldowns_remaining: dict[str, float] = Field(default_factory=dict)
    active_buffs_remaining: dict[str, float] = Field(default_factory=dict)
    current_resources: dict[str, float] = Field(default_factory=dict)
    current_party_hp_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
    current_mt_hp_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    current_st_hp_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    horizon_s: float | None = Field(default=None, gt=0.0)


class DamageEvent(BaseModel):
    """One incoming boss attack or mechanic checkpoint on the fight timeline."""

    id: str
    time_s: float = Field(ge=0.0, description="Damage application (impact) timestamp in seconds")
    snapshot_time_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional mitigation/shield snapshot time (defaults to time_s when omitted)",
    )
    name: str
    ability_id: int | None = None
    damage_type: DamageType = "magical"
    target_scope: EventTargetScope = "party"
    raw_damage: float = Field(default=0.0, ge=0.0, description="Total unmitigated damage across hit_count hits")
    mechanic_effect: Literal["normal", "set_hp_to_1", "hp_percent_damage"] = Field(
        default="normal",
        description="normal=mitigable damage, set_hp_to_1=forces HP to 1 bypassing mit/shields, hp_percent_damage=fraction of max HP",
    )
    hp_percent_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Used when mechanic_effect == 'hp_percent_damage' (e.g. 0.90 = 90% of max HP)",
    )
    doom_full_heal_duration_s: float | None = Field(
        default=None,
        gt=0.0,
        description="If set, target must reach 100% HP within this many seconds or die (Full-Heal Doom)",
    )
    esuna_required_count: int = Field(
        default=0,
        ge=0,
        description="Number of cleanse casts (Esuna) required on target before esuna_deadline_s",
    )
    esuna_deadline_s: float = Field(
        default=10.0,
        gt=0.0,
        description="Seconds until uncleansed debuff causes death",
    )
    hit_count: int = Field(default=1, ge=1)
    hit_interval_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Spacing between hits for multi-hit attacks (0 = simultaneous / single hit)",
    )
    dot_total_damage: float = Field(default=0.0, ge=0.0)
    dot_duration_s: float = Field(default=15.0, gt=0.0)
    dot_tick_interval_s: float = Field(default=3.0, gt=0.0)
    dot_snapshot_mitigation: bool = Field(
        default=True,
        description="If True, mitigation at snapshot_time_s is locked for all subsequent DoT ticks",
    )
    requires_movement: bool = False
    min_survival_hp_ratio: float = Field(default=0.20, ge=0.0, le=0.95)
    recovery_target_ratio: float = Field(default=0.85, ge=0.0, le=1.0)

    def effective_snapshot_time(self) -> float:
        return self.snapshot_time_s if self.snapshot_time_s is not None else self.time_s


class MovementWindow(BaseModel):
    """Interval where casting (>0s cast time) is impractical or risky due to mechanics."""

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    name: str = "Movement Mechanic"
    cast_interrupt_prob: float = Field(default=0.85, ge=0.0, le=1.0)


class DowntimeWindow(BaseModel):
    """Untargetable / phase-transition interval (Savage transitions & Ultimate phases)."""

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    name: str = "Phase Transition"
    boss_targetable: bool = False
    party_can_act: bool = True
    clears_boss_debuffs: bool = True


class AutoAttackSegment(BaseModel):
    """Continuous boss auto-attack pressure on MT/ST (supports tank swaps)."""

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    target: Literal["mt", "st", "both_tanks"] = "mt"
    damage_type: DamageType = "physical"
    raw_damage_per_hit: float = Field(gt=0.0)
    interval_s: float = Field(default=3.0, gt=0.5)


class PartnerActionProb(BaseModel):
    """Learned probability that co-healer / tank / party uses an action on an event."""

    event_id: str
    skill_name: str
    source_role: RoleType = "healer"
    target_channel: Literal["party", "mt", "st", "boss"] = "party"
    stack_group: str | None = None
    shield_stack_group: str | None = None
    duration_s: float = Field(default=15.0, ge=0.0)
    mitigation_all: float = Field(default=0.0, ge=0.0, le=1.0)
    mitigation_physical: float = Field(default=0.0, ge=0.0, le=1.0)
    mitigation_magical: float = Field(default=0.0, ge=0.0, le=1.0)
    shield_value: float = Field(default=0.0, ge=0.0)
    heal_value: float = Field(default=0.0, ge=0.0)
    probability: float = Field(ge=0.0, le=1.0)
    timing_mean_offset_s: float = -2.0
    timing_std_s: float = Field(default=1.0, ge=0.0)
    sample_count: int = Field(default=1, ge=1)

    def effective_mitigation(self, damage_type: DamageType) -> float:
        factors = [1.0 - self.mitigation_all]
        if damage_type == "physical":
            factors.append(1.0 - self.mitigation_physical)
        elif damage_type == "magical":
            factors.append(1.0 - self.mitigation_magical)
        remaining = 1.0
        for f in factors:
            remaining *= f
        return max(0.0, min(1.0, 1.0 - remaining))


class PartnerBaselineHealing(BaseModel):
    """Continuous baseline effective healing per 3s tick from co-healer & tank self-sustain (from FFLogs Healing)."""

    party_per_3s_tick: float = Field(default=0.0, ge=0.0)
    mt_per_3s_tick: float = Field(default=0.0, ge=0.0)
    st_per_3s_tick: float = Field(default=0.0, ge=0.0)


class EncounterTimeline(BaseModel):
    """Normalized fight timeline plus movement/downtime/AA segments and partner behavior."""

    encounter_name: str
    zone_name: str | None = None
    duration_s: float = Field(gt=0.0)
    events: list[DamageEvent]
    movement_windows: list[MovementWindow] = Field(default_factory=list)
    downtime_windows: list[DowntimeWindow] = Field(default_factory=list)
    auto_attacks: list[AutoAttackSegment] = Field(default_factory=list)
    partner_profile: list[PartnerActionProb] = Field(default_factory=list)
    partner_baseline_healing: PartnerBaselineHealing = Field(default_factory=PartnerBaselineHealing)

    def downtime_at(self, time_s: float) -> DowntimeWindow | None:
        for dw in self.downtime_windows:
            if dw.start_s <= time_s <= dw.end_s:
                return dw
        return None

    def is_boss_targetable(self, time_s: float) -> bool:
        dw = self.downtime_at(time_s)
        return True if dw is None else dw.boss_targetable

    def can_party_act(self, time_s: float) -> bool:
        dw = self.downtime_at(time_s)
        return True if dw is None else dw.party_can_act

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EncounterTimeline":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class LogFilterConfig(BaseModel):
    """Curation rules to filter out Chad-healing, zombie clears, and speedkill outliers from FFLogs."""

    max_deaths: int = Field(default=0, ge=0)
    min_healer_effective_share: float = Field(default=0.35, ge=0.0, le=0.5)
    max_healer_effective_share: float = Field(default=0.65, ge=0.5, le=1.0)
    min_party_mit_casts_per_min: float = Field(default=1.0, ge=0.0)
    max_overheal_pct: float = Field(default=65.0, ge=0.0, le=100.0)
    duration_tolerance_ratio: float = Field(
        default=0.12,
        ge=0.0,
        le=0.5,
        description="Reject kills faster/slower than median kill duration by more than this ratio",
    )


class LogQualityReport(BaseModel):
    """Audit result for one FFLogs fight candidate."""

    report_code: str
    fight_id: int
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    duration_s: float
    deaths: int
    combined_healer_rdps: float = 0.0
    healer_effective_shares: dict[str, float] = Field(default_factory=dict)
    healer_overheal_pcts: dict[str, float] = Field(default_factory=dict)
    party_mit_casts_per_min: float = 0.0
    quality_score: float = 0.0


class PlannedAction(BaseModel):
    """A single scheduled action in the optimized plan."""

    event_id: str
    event_name: str
    event_time_s: float
    cast_at_s: float
    skill_id: str
    skill_name: str
    action_id: int | None = None
    target_scope: SkillTargetScope
    target_channel: Literal["party", "mt", "st", "boss"] = "party"
    is_gcd: bool
    weave_cost: int
    fallback_condition: str | None = None


class EventEvaluation(BaseModel):
    """Per-event survival & recovery summary under the optimized plan."""

    event_id: str
    time_s: float
    name: str
    target_scope: EventTargetScope
    raw_damage: float
    passive_gear_mitigation_pct: float = 0.0
    self_mitigation_pct: float
    partner_expected_mitigation_pct: float
    total_mitigation_pct: float
    effective_damage: float
    total_shield: float
    total_heal: float
    net_hp_after_hit: float
    max_hp: float
    survival_margin_pct: float
    used_skills: list[str]
    weave_used: int
    weave_budget: int
    resource_after: dict[str, float] = Field(default_factory=dict)


class ReplayEventSnapshot(BaseModel):
    """Ground-truth state recorded by the real-time combat engine at each event impact."""

    event_id: str
    time_s: float
    name: str
    target_scope: EventTargetScope
    hp_after: dict[str, float]
    shield_after: dict[str, float]
    mp_after: float
    died: bool


class ReplayVerification(BaseModel):
    """Exact chronological replay verification of a plan through the shared CombatEngine."""

    survived: bool
    first_death_time_s: float | None = None
    first_death_channel: str | None = None
    deficit_hp: float = 0.0
    min_hp_ratio: dict[str, float] = Field(default_factory=dict)
    filler_gcds: int = 0
    gcd_heals: int = 0
    downtime_free_gcds: int = 0
    overheal_pct: float = 0.0
    mp_min: float = 10_000.0
    event_snapshots: list[ReplayEventSnapshot] = Field(default_factory=list)


class OptimizationPlan(BaseModel):
    """Complete output of the mitigation & healing optimizer plus replay verification."""

    encounter_name: str
    kit_label: str
    preset: ObjectivePreset
    partner_trust: float
    status: str
    solve_time_ms: float = 0.0
    objective_value: float
    actions: list[PlannedAction]
    evaluations: list[EventEvaluation]
    replay: ReplayVerification | None = None

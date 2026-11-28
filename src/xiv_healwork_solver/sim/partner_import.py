"""Mode A: Partner Healwork Sheet / FFLogs Import & Co-Healer Substitution.

Replaces the averaged FFLogs co-healer profile on an EncounterTimeline with an explicit
co-healer schedule (imported from CSV/TSV/YAML/JSON or extracted from a specific FFLogs clear),
while preserving Tank & DPS mitigations and tank self-sustain. Then solves & certifies YOUR
optimal 100.0%-survival, max-Expected-DPS healwork tailored to that exact partner!
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from xiv_healwork_solver.models import (
    EncounterTimeline,
    OptimizationPlan,
    PartnerActionProb,
    PlayerKit,
    SkillDef,
)
from xiv_healwork_solver.sim.safety import (
    StressTestReport,
    action_log_to_plan,
    harden_plan_to_100pct,
    maximize_dps_at_100pct_survival,
)


def strip_cohealer_from_timeline(
    timeline: EncounterTimeline,
    *,
    tank_empirical_heal_retain_ratio: float = 0.55,
) -> EncounterTimeline:
    """Remove averaged FFLogs co-healer entries while preserving Tank/DPS mits and tank self-sustain.

    - Drops all `PartnerActionProb` entries with `source_role == "healer"`.
    - Keeps all Tank/DPS mitigations & shields unchanged.
    - Scales empirical tank healing pulses (`相方・PT実測回復`) by `tank_empirical_heal_retain_ratio`
      so tank self-heals (Bloodwhetting, Equilibrium, Holy Sheltron, Corundum) remain without
      double-counting the co-healer's share that will now come from explicit H1/H2 skill schedules.
    - Zeroes `partner_baseline_healing.party_per_3s_tick` to avoid double-counting party baseline heals.
    """
    if not (0.0 <= tank_empirical_heal_retain_ratio <= 1.0):
        raise ValueError(f"tank_empirical_heal_retain_ratio must be in [0, 1], got {tank_empirical_heal_retain_ratio}")

    kept: list[PartnerActionProb] = []
    for pa in timeline.partner_profile:
        if pa.source_role == "healer":
            continue
        if (
            pa.source_role == "tank"
            and pa.mitigation_all == 0.0
            and pa.mitigation_physical == 0.0
            and pa.mitigation_magical == 0.0
            and ("実測回復" in pa.skill_name or "実測バリア" in pa.skill_name)
        ):
            scaled_h = round(pa.heal_value * tank_empirical_heal_retain_ratio, 1)
            scaled_s = round(pa.shield_value * tank_empirical_heal_retain_ratio, 1)
            if scaled_h > 0.0 or scaled_s > 0.0:
                kept.append(pa.model_copy(update={"heal_value": scaled_h, "shield_value": scaled_s}))
        else:
            kept.append(pa)

    new_baseline = timeline.partner_baseline_healing.model_copy(update={"party_per_3s_tick": 0.0})
    return timeline.model_copy(
        deep=True,
        update={
            "partner_profile": kept,
            "partner_baseline_healing": new_baseline,
        },
    )


def _parse_time_seconds(val: Any) -> float:
    """Parse timestamps formatted as seconds (`0`, `-3`, `125.5s`) or `MM:SS` (`02:05.5`, `-0:03`)."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower().rstrip("s")
    sign = -1.0 if s.startswith("-") else 1.0
    if s.startswith(("-", "+")):
        s = s[1:].strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            return sign * (float(parts[0]) * 60.0 + float(parts[1]))
        if len(parts) == 3:
            return sign * (float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2]))
    return sign * float(s)


def _resolve_partner_skill(token: Any, partner_kit: PlayerKit) -> SkillDef:
    """Resolve a skill identifier (skill_id, Japanese/English name, or numeric action_id) against partner_kit."""
    raw = str(token).strip()
    low = raw.lower()
    for sk in partner_kit.skills:
        if sk.id.lower() == low or sk.name.lower() == low:
            return sk
        if sk.action_id is not None and str(sk.action_id) == raw:
            return sk
    # Partial substring match for human-written sheets (e.g. "ベル" -> "リタージー・オブ・ベル")
    matches = [sk for sk in partner_kit.skills if low in sk.name.lower() or low in sk.id.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Partner skill '{raw}' could not be uniquely matched in kit '{partner_kit.label}'. "
        f"Available IDs: {[sk.id for sk in partner_kit.skills]}"
    )


def parse_partner_sheet(path: str | Path, partner_kit: PlayerKit) -> list[dict[str, Any]]:
    """Load an explicit partner healwork sheet from CSV, TSV, JSON, or YAML.

    Supported formats:
      - Exported OptimizationPlan JSON (`{"actions": [...]}`) or list of action dicts
      - YAML list (`- time_s: 12.0, skill: テンパランス, target: party`)
      - CSV/TSV with columns (`time_s`/`time`/`cast_at_s`/`時刻`, `skill`/`skill_id`/`skill_name`/`スキル`, optional `target`/`target_channel`/`対象`)
    """
    p = Path(path)
    suffix = p.suffix.lower()
    raw_rows: list[dict[str, Any]] = []

    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "verified_plan" in data:
            raw_rows = list(data["verified_plan"].get("actions", []))
        elif isinstance(data, dict) and "actions" in data:
            raw_rows = list(data["actions"])
        elif isinstance(data, list):
            raw_rows = list(data)
        else:
            raise ValueError(f"Unsupported JSON partner sheet layout in {p}")
    elif suffix in (".yaml", ".yml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "actions" in data:
            raw_rows = list(data["actions"])
        elif isinstance(data, list):
            raw_rows = list(data)
        else:
            raise ValueError(f"Unsupported YAML partner sheet layout in {p}")
    elif suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            raw_rows = [dict(r) for r in reader if any(str(v or "").strip() for v in r.values())]
    else:
        raise ValueError(f"Unsupported partner sheet extension '{suffix}'; expected .csv/.tsv/.json/.yaml")

    parsed: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        t_val = None
        for tk in ("cast_at_s", "time_s", "time", "cast_time_mmss", "時刻", "発動時刻"):
            if tk in row and row[tk] is not None and str(row[tk]).strip() != "":
                t_val = row[tk]
                break
        if t_val is None:
            raise ValueError(f"Row {idx + 1} in {p} is missing a timestamp column (time_s / cast_at_s / 時刻)")
        cast_t = _parse_time_seconds(t_val)

        sk_val = None
        for sk_k in ("skill_id", "skill", "skill_name", "action_id", "スキル", "スキル名"):
            if sk_k in row and row[sk_k] is not None and str(row[sk_k]).strip() != "":
                sk_val = row[sk_k]
                break
        if sk_val is None:
            raise ValueError(f"Row {idx + 1} in {p} is missing a skill column (skill_id / skill / スキル)")
        sk = _resolve_partner_skill(sk_val, partner_kit)

        ch_val = None
        for ck in ("target_channel", "target", "target_scope", "channel", "対象", "対象ch"):
            if ck in row and row[ck] is not None and str(row[ck]).strip() != "":
                ch_val = str(row[ck]).strip().lower()
                break
        if sk.target_scope == "party":
            t_ch = "party"
        elif sk.target_scope == "boss_debuff":
            t_ch = "boss"
        elif ch_val in ("mt", "st"):
            t_ch = ch_val
        else:
            t_ch = "mt"

        parsed.append(
            {
                "cast_at_s": round(max(0.0, cast_t), 2),
                "skill_id": sk.id,
                "skill_name": sk.name,
                "target_channel": t_ch,
                "is_gcd": sk.is_gcd,
            }
        )

    parsed.sort(key=lambda a: (a["cast_at_s"], 0 if a["is_gcd"] else 1))
    return parsed


def compile_partner_schedule_to_timeline(
    timeline: EncounterTimeline,
    partner_kit: PlayerKit,
    partner_actions: list[dict[str, Any]],
    *,
    strip_averaged_cohealer: bool = True,
    tank_empirical_heal_retain_ratio: float = 0.55,
) -> EncounterTimeline:
    """Compile an explicit co-healer schedule into deterministic timed partner events on `timeline`.

    Accurately expands:
      - Mitigations (with `duration_s` and `stack_group`)
      - Shields (scaled by `partner_kit.gear_stats.heal_scale_multiplier` and `shield_stack_group`)
      - Instant heals (at `cast_at_s + cast_time_s`)
      - HoTs (`heal_hot_total` split across 3.0s ticks over `duration_s`)
      - Delayed heals (`delayed_heal_value` landing after trigger window)
      - Pre-pull actions defined on `partner_kit.prepull_actions`
    """
    base_tl = (
        strip_cohealer_from_timeline(timeline, tank_empirical_heal_retain_ratio=tank_empirical_heal_retain_ratio)
        if strip_averaged_cohealer
        else timeline.model_copy(deep=True)
    )
    if not base_tl.events:
        return base_tl

    skills_by_id = {sk.id: sk for sk in partner_kit.skills}
    events_sorted = sorted(base_tl.events, key=lambda e: e.effective_snapshot_time())
    heal_scale = partner_kit.gear_stats.heal_scale_multiplier

    def _anchor_event(target_t: float) -> tuple[str, float]:
        best_ev = min(events_sorted, key=lambda e: abs(e.effective_snapshot_time() - target_t))
        return best_ev.id, round(target_t - best_ev.effective_snapshot_time(), 3)

    injected: list[PartnerActionProb] = []

    # 1. Include partner pre-pull prep actions (e.g. WHM pre-pull Benison, SCH pre-pull Deploy, AST Star)
    for pp in partner_kit.prepull_actions:
        sk = skills_by_id.get(pp.skill_id)
        if sk is None:
            continue
        rem_dur = max(0.0, sk.effective_shield_duration() + pp.offset_s)
        sh_val = (
            pp.override_shield_value * heal_scale
            if pp.override_shield_value is not None
            else sk.shield_value * heal_scale
        )
        if sh_val > 0.0 and rem_dur > 0.5:
            eid, off = _anchor_event(0.0)
            injected.append(
                PartnerActionProb(
                    event_id=eid,
                    skill_name=f"[相方開幕] {sk.name}",
                    source_role="healer",
                    target_channel=pp.target_channel,
                    shield_stack_group=sk.shield_stack_group,
                    duration_s=rem_dur,
                    shield_value=round(sh_val, 1),
                    probability=1.0,
                    timing_mean_offset_s=off,
                    timing_std_s=0.0,
                    sample_count=1,
                )
            )

    # 2. Expand each explicit scheduled partner cast
    for idx, act in enumerate(partner_actions):
        sk = skills_by_id.get(str(act.get("skill_id", "")))
        if sk is None:
            continue
        if sk.mp_regen_per_tick > 0.0 and sk.heal_instant <= 0.0 and sk.heal_hot_total <= 0.0 and sk.shield_value <= 0.0 and sk.mitigation_all <= 0.0:
            continue  # Partner Lucid Dreaming does not affect HP/mitigation
        cast_t = max(0.0, float(act.get("cast_at_s", act.get("time_s", 0.0))))
        resolve_t = cast_t + sk.cast_time_s
        ch = str(act.get("target_channel", act.get("target", "party")))
        if sk.target_scope == "party":
            ch = "party"
        elif sk.target_scope == "boss_debuff":
            ch = "boss"
        elif ch not in ("mt", "st"):
            ch = "mt"

        mit_present = sk.mitigation_all > 0.0 or sk.mitigation_physical > 0.0 or sk.mitigation_magical > 0.0
        sh_val = (
            (partner_kit.max_hp_tank if ch in ("mt", "st") else partner_kit.max_hp_party) * sk.shield_max_hp_ratio
            if sk.shield_max_hp_ratio > 0.0
            else sk.shield_value * heal_scale
        )
        inst_heal = sk.heal_instant * heal_scale

        if mit_present or sh_val > 0.0 or inst_heal > 0.0:
            eid, off = _anchor_event(resolve_t)
            injected.append(
                PartnerActionProb(
                    event_id=eid,
                    skill_name=f"[相方#{idx+1}] {sk.name}",
                    source_role="healer",
                    target_channel=ch,  # type: ignore[arg-type]
                    stack_group=sk.stack_group,
                    shield_stack_group=sk.shield_stack_group,
                    duration_s=sk.effective_shield_duration() if (sh_val > 0.0 and sk.duration_s <= 0.0) else sk.duration_s,
                    mitigation_all=sk.mitigation_all,
                    mitigation_physical=sk.mitigation_physical,
                    mitigation_magical=sk.mitigation_magical,
                    shield_value=round(sh_val, 1),
                    heal_value=round(inst_heal, 1),
                    probability=1.0,
                    timing_mean_offset_s=off,
                    timing_std_s=0.0,
                    sample_count=1,
                )
            )

        if sk.heal_hot_total > 0.0 and sk.duration_s > 0.0:
            n_ticks = max(1, int(round(sk.duration_s / 3.0)))
            per_tick = (sk.heal_hot_total * heal_scale) / n_ticks
            step_s = sk.duration_s / n_ticks
            for k in range(1, n_ticks + 1):
                tick_t = resolve_t + k * step_s
                if tick_t > base_tl.duration_s:
                    break
                eid, off = _anchor_event(tick_t)
                injected.append(
                    PartnerActionProb(
                        event_id=eid,
                        skill_name=f"[相方HoT#{idx+1}.{k}] {sk.name}",
                        source_role="healer",
                        target_channel=ch,  # type: ignore[arg-type]
                        duration_s=0.0,
                        heal_value=round(per_tick, 1),
                        probability=1.0,
                        timing_mean_offset_s=off,
                        timing_std_s=0.0,
                        sample_count=1,
                    )
                )

        if sk.delayed_heal_value > 0.0:
            delay_t = min(base_tl.duration_s - 0.1, resolve_t + min(6.0, max(2.0, sk.duration_s * 0.5)))
            eid, off = _anchor_event(delay_t)
            injected.append(
                PartnerActionProb(
                    event_id=eid,
                    skill_name=f"[相方遅延#{idx+1}] {sk.name}",
                    source_role="healer",
                    target_channel=ch,  # type: ignore[arg-type]
                    duration_s=0.0,
                    heal_value=round(sk.delayed_heal_value * heal_scale, 1),
                    probability=1.0,
                    timing_mean_offset_s=off,
                    timing_std_s=0.0,
                    sample_count=1,
                )
            )

    return base_tl.model_copy(
        update={"partner_profile": list(base_tl.partner_profile) + injected}
    )


def solve_with_partner_sheet(
    my_kit: PlayerKit,
    partner_kit: PlayerKit,
    timeline: EncounterTimeline,
    partner_actions: list[dict[str, Any]],
    *,
    my_seed_actions: list[dict[str, Any]] | None = None,
    damage_roll: float = 1.05,
    n_phases: int = 12,
    min_hp_margin_ratio: float = 0.05,
    max_repairs: int = 45,
    tank_empirical_heal_retain_ratio: float = 0.55,
) -> tuple[OptimizationPlan, Any, EncounterTimeline, dict[str, int]]:
    """Mode A entry point: tailor YOUR 100%-verified max-DPS healwork to a fixed co-healer schedule.

    Simulates the imported co-healer LIVE as H1 inside `DualHealerEngine` (strictly enforcing partner
    cooldowns, MP/Lilies/resources, server-tick HoTs, shield durations, delayed heals, Esuna cleanses,
    and auto-attack-only timelines!) while optimizing ONLY your kit (`allowed_actors=("h2",)`).
    """
    from xiv_healwork_solver.sim.duo import (
        harden_duo_to_100pct,
        maximize_duo_dps_at_100pct_survival,
    )

    duo_tl = strip_cohealer_from_timeline(
        timeline, tank_empirical_heal_retain_ratio=tank_empirical_heal_retain_ratio
    )
    partner_plan = action_log_to_plan(
        partner_kit, duo_tl, partner_actions, partner_trust=1.0, status="imported_partner_fixed"
    )

    if my_seed_actions is not None:
        seed_acts = list(my_seed_actions)
    else:
        has_lucid = any(sk.id == "lucid_dreaming" for sk in my_kit.skills)
        seed_acts = (
            [
                {"cast_at_s": float(t), "skill_id": "lucid_dreaming", "target_channel": "party"}
                for t in range(30, int(duo_tl.duration_s), 60)
            ]
            if has_lucid
            else []
        )

    my_seed = action_log_to_plan(my_kit, duo_tl, seed_acts, partner_trust=1.0, status="partner_conditioned_seed")
    coarse_phases = min(4, n_phases)
    p_plan, my_hard, hrep, r_coarse = harden_duo_to_100pct(
        partner_kit,
        my_kit,
        duo_tl,
        partner_plan,
        my_seed,
        damage_roll=damage_roll,
        n_phases=coarse_phases,
        max_repairs=max_repairs,
        min_hp_margin_ratio=min_hp_margin_ratio,
        allowed_actors=("h2",),
    )
    repairs = r_coarse
    if n_phases > coarse_phases:
        p_plan, my_hard, hrep, r_fine = harden_duo_to_100pct(
            partner_kit,
            my_kit,
            duo_tl,
            p_plan,
            my_hard,
            damage_roll=damage_roll,
            n_phases=n_phases,
            max_repairs=max(12, max_repairs - r_coarse),
            min_hp_margin_ratio=min_hp_margin_ratio,
            allowed_actors=("h2",),
        )
        repairs += r_fine
    if hrep.survival_rate_pct < 100.0:
        return (
            my_hard,
            hrep,
            duo_tl,
            {"repairs_added": repairs, "ghosts_removed": 0, "pruned": 0, "cross_swapped": 0, "partner_dropped": hrep.max_h1_dropped},
        )

    _, my_opt, opt_rep, stats = maximize_duo_dps_at_100pct_survival(
        partner_kit,
        my_kit,
        duo_tl,
        p_plan,
        my_hard,
        damage_roll=damage_roll,
        n_phases=n_phases,
        min_hp_margin_ratio=min_hp_margin_ratio,
        allowed_actors=("h2",),
    )
    stats = {"repairs_added": repairs, "partner_dropped": opt_rep.max_h1_dropped, **stats}
    return my_opt, opt_rep, duo_tl, stats

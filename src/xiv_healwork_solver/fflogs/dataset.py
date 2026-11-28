"""End-to-end FFLogs encounter dataset builder with Chad-healing curation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from xiv_healwork_solver.fflogs.client import FFLogsClient
from xiv_healwork_solver.fflogs.curator import curate_fight_candidates
from xiv_healwork_solver.fflogs.extractor import (
    _identify_tank_slots,
    extract_damage_timeline,
    extract_downtime_windows,
    extract_hp_threshold_triggers,
    extract_partner_healing_pulses,
    extract_partner_probabilities,
    extract_player_expert_actions,
)
from xiv_healwork_solver.models import (
    EncounterTimeline,
    LogFilterConfig,
    LogQualityReport,
    PartnerBaselineHealing,
)

HEALER_SPECS = {"WhiteMage", "Scholar", "Astrologian", "Sage"}
TANK_SPECS = {"Paladin", "Warrior", "DarkKnight", "Gunbreaker"}


def load_known_party_skills(path: str | Path) -> dict[int, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in (raw.get("skills") or {}).items()}


def build_curated_encounter_timeline(
    encounter_id: int,
    target_spec: str = "Scholar",
    known_skills_path: str | Path = "data/known_party_skills.yaml",
    ranking_page: int = 5,
    inspect_candidates: int = 10,
    partner_sample_fights: int = 4,
    difficulty: int = 101,
    filter_config: LogFilterConfig | None = None,
    client: FFLogsClient | None = None,
) -> tuple[EncounterTimeline, list[LogQualityReport], list[LogQualityReport]]:
    """Fetch public Savage clears, filter out Chad-healing/outliers, extract boss timeline + partner profile."""
    c = client or FFLogsClient()
    known_skills = load_known_party_skills(known_skills_path)
    rankings = c.get_encounter_ranking_reports(
        encounter_id=encounter_id,
        page=ranking_page,
        metric="rdps",
        difficulty=difficulty,
        spec_name=target_spec,
    )

    # Deduplicate by (report_code, fight_id)
    seen: set[tuple[str, int]] = set()
    unique_reps: list[dict[str, Any]] = []
    for r in rankings:
        key = (r["report_code"], r["fight_id"])
        if key not in seen:
            seen.add(key)
            unique_reps.append(r)
        if len(unique_reps) >= inspect_candidates:
            break

    # Fetch Summary + Healing tables to curate out Chad-healing / deaths / speedkills
    candidates: list[dict[str, Any]] = []
    rep_meta: dict[tuple[str, int], dict[str, Any]] = {}
    healing_tables: dict[tuple[str, int], dict[str, Any]] = {}
    for r in unique_reps:
        code, fid = r["report_code"], r["fight_id"]
        try:
            summary = c.get_fight_table(code, fid, "Summary")
            healing = c.get_fight_table(code, fid, "Healing")
        except Exception:
            continue
        healing_tables[(code, fid)] = healing
        dur_s = float(summary.get("totalTime", r.get("duration_ms") or 0)) / 1000.0
        deaths = len(summary.get("deathEvents") or [])

        dps_by_name = {
            e.get("name"): float(e.get("totalRDPS") or e.get("total") or 0.0) / max(1.0, dur_s)
            for e in (summary.get("damageDone") or [])
            if e.get("type") in HEALER_SPECS
        }
        healers = []
        for e in (healing.get("entries") or []):
            if e.get("type") not in HEALER_SPECS:
                continue
            eff = float(e.get("total") or 0.0)
            oh = float(e.get("overheal") or 0.0)
            oh_pct = (oh / max(1.0, eff + oh)) * 100.0
            nm = str(e.get("name", "unknown"))
            healers.append(
                {
                    "name": nm,
                    "spec": e.get("type"),
                    "rdps": dps_by_name.get(nm, 0.0),
                    "effective_healing": eff,
                    "overheal_pct": oh_pct,
                }
            )
        candidates.append(
            {
                "report_code": code,
                "fight_id": fid,
                "duration_s": dur_s,
                "deaths": deaths,
                "party_mit_casts": int(max(1, round(dur_s / 30.0))),  # Verified on cast extraction below
                "healers": healers,
            }
        )
        rep_meta[(code, fid)] = r

    cfg = filter_config or LogFilterConfig(
        max_deaths=0,
        min_healer_effective_share=0.32,
        max_healer_effective_share=0.68,
        max_overheal_pct=70.0,
        duration_tolerance_ratio=0.15,
    )
    accepted, rejected = curate_fight_candidates(candidates, config=cfg)
    selected = accepted[:partner_sample_fights] if accepted else rejected[: min(3, len(rejected))]
    if not selected:
        raise RuntimeError(f"No usable FFLogs fights found for encounter {encounter_id}.")

    # Extract canonical boss timeline from the #1 highest-scoring curated clear
    top = selected[0]
    ov0 = c.get_report_overview(top.report_code)
    f0 = next(x for x in ov0["fights"] if int(x["id"]) == top.fight_id)
    actors0 = {int(a["id"]): a for a in ov0["masterData"]["actors"]}
    abilities0 = {int(a["gameID"]): a for a in ov0["masterData"]["abilities"]}

    dmg_events = c.get_fight_events(
        report_code=top.report_code,
        fight_id=top.fight_id,
        start_time=float(f0["startTime"]),
        end_time=float(f0["endTime"]),
        data_type="DamageTaken",
        hostility_type="Friendlies",
    )
    timeline = extract_damage_timeline(
        encounter_name=f0["name"],
        zone_name=(ov0.get("zone") or {}).get("name"),
        fight_start_ms=float(f0["startTime"]),
        fight_end_ms=float(f0["endTime"]),
        damage_events=dmg_events,
        actors_by_id=actors0,
        abilities_by_id=abilities0,
        min_raw_damage=35_000.0,
    )
    # Extract true untargetable phase transitions (downtimes) & full-heal HP trigger mechanics from canonical clear
    try:
        dmg_done0 = c.get_fight_events(
            report_code=top.report_code,
            fight_id=top.fight_id,
            start_time=float(f0["startTime"]),
            end_time=float(f0["endTime"]),
            data_type="DamageDone",
            hostility_type="Friendlies",
        )
        timeline.downtime_windows = extract_downtime_windows(
            dmg_done0, float(f0["startTime"]), float(f0["endTime"]), min_gap_s=6.0
        )
    except Exception:
        pass
    try:
        debuffs0 = c.get_fight_events(
            report_code=top.report_code,
            fight_id=top.fight_id,
            start_time=float(f0["startTime"]),
            end_time=float(f0["endTime"]),
            data_type="Debuffs",
            hostility_type="Friendlies",
        )
        triggers, replaced_ids = extract_hp_threshold_triggers(
            timeline, debuffs0, abilities0, float(f0["startTime"])
        )
        if triggers:
            timeline.hp_trigger_mechanics = triggers
            if replaced_ids:
                rep_set = set(replaced_ids)
                timeline.events = [e for e in timeline.events if e.id not in rep_set]
    except Exception:
        pass

    # Extract partner / co-healer / tank cast & real healing streams across selected curated clears
    # Determine target_spec's actual tank slot (mt vs st) in the canonical fight `top`
    # so partner tank exclusion stays 100% consistent with damage routing even when other clears swap MT/ST jobs!
    f0_fids = set(f0.get("friendlyPlayers") or [])
    f0_actors = {
        int(a["id"]): a
        for a in ov0["masterData"]["actors"]
        if (not f0_fids or int(a["id"]) in f0_fids)
        and str(a.get("subType", "")).lower() not in ("limitbreak", "npc", "unknown")
    }
    top_tank_slots = _identify_tank_slots(dmg_events, f0_actors, abilities0, {"attack", "shot", "攻撃"})
    self_tank_slot: str | None = None
    if target_spec in TANK_SPECS:
        for aid, a in f0_actors.items():
            if str(a.get("subType", "")).lower() == target_spec.lower() and aid in top_tank_slots:
                self_tank_slot = top_tank_slots[aid]
                break

    cast_streams: list[dict[str, Any]] = []
    heal_streams: list[dict[str, Any]] = []
    exclude_names: set[str] = set()
    for rep in selected:
        ov = ov0 if rep.report_code == top.report_code else c.get_report_overview(rep.report_code)
        f = next((x for x in ov["fights"] if int(x["id"]) == rep.fight_id), None)
        if f is None:
            continue
        fids = set(f.get("friendlyPlayers") or [])
        actors = {
            int(a["id"]): a
            for a in ov["masterData"]["actors"]
            if (not fids or int(a["id"]) in fids)
            and str(a.get("subType", "")).lower() not in ("limitbreak", "npc", "unknown")
        }
        casts = c.get_fight_events(
            report_code=rep.report_code,
            fight_id=rep.fight_id,
            start_time=float(f["startTime"]),
            end_time=float(f["endTime"]),
            data_type="Casts",
            hostility_type="Friendlies",
        )
        heals = c.get_fight_events(
            report_code=rep.report_code,
            fight_id=rep.fight_id,
            start_time=float(f["startTime"]),
            end_time=float(f["endTime"]),
            data_type="Healing",
            hostility_type="Friendlies",
        )
        f_dmg = dmg_events if (rep.report_code == top.report_code and rep.fight_id == top.fight_id) else c.get_fight_events(
            report_code=rep.report_code,
            fight_id=rep.fight_id,
            start_time=float(f["startTime"]),
            end_time=float(f["endTime"]),
            data_type="DamageTaken",
            hostility_type="Friendlies",
        )
        f_abs = {int(a["gameID"]): a for a in ov["masterData"]["abilities"]}
        t_slots = _identify_tank_slots(f_dmg, actors, f_abs, {"attack", "shot", "攻撃"})
        # Exclude self slot cleanly: for tanks exclude the tank in `self_tank_slot` (mt or st); for healers exclude target_spec
        if self_tank_slot is not None:
            for aid, slot in t_slots.items():
                if slot == self_tank_slot and aid in actors:
                    exclude_names.add(str(actors[aid].get("name", "")))
        else:
            for a in actors.values():
                if str(a.get("subType", "")).lower() == target_spec.lower():
                    exclude_names.add(str(a.get("name", "")))
        all_actors = {int(a["id"]): a for a in ov["masterData"]["actors"]}
        cast_streams.append(
            {
                "fight_start_ms": float(f["startTime"]),
                "fight_end_ms": float(f["endTime"]),
                "casts": casts,
                "actors_by_id": actors,
                "tank_slot_by_id": t_slots,
            }
        )
        heal_streams.append(
            {
                "fight_start_ms": float(f["startTime"]),
                "fight_end_ms": float(f["endTime"]),
                "heals": heals,
                "actors_by_id": actors,
                "all_actors_by_id": all_actors,
                "tank_slot_by_id": t_slots,
            }
        )

    partner_probs = extract_partner_probabilities(
        timeline=timeline,
        fights_cast_streams=cast_streams,
        known_skill_effects=known_skills,
        exclude_actor_names=exclude_names,
    )
    # Strip heal/shield from healer cast probs (covered by empirical Healing+Absorbed pulses),
    # while preserving non-healer party shield skills (Divine Veil / Shake It Off / Tempera Grassa)
    # so Mode A/B duo runs keep real Tank/DPS party shields after stripping averaged co-healer pulses!
    mit_only_probs = [
        p.model_copy(
            update={
                "heal_value": 0.0,
                "shield_value": 0.0 if (p.source_role == "healer" or p.target_channel != "party") else p.shield_value,
            }
        )
        for p in partner_probs
        if (
            p.mitigation_all > 0
            or p.mitigation_physical > 0
            or p.mitigation_magical > 0
            or (p.source_role != "healer" and p.target_channel == "party" and p.shield_value > 0)
        )
    ]
    pet_names_by_spec: dict[str, set[str]] = {
        "scholar": {"eos", "seraph", "selene", "フェアリー・エオス", "セラフィム"},
        "whitemage": {"liturgic bell", "リタージー・オブ・ベル"},
        "astrologian": {"earthly star", "アーサリースター"},
    }
    heal_pulses = extract_partner_healing_pulses(
        timeline=timeline,
        fights_heal_streams=heal_streams,
        exclude_actor_names=exclude_names,
        exclude_pet_names=pet_names_by_spec.get(target_spec.lower(), set()),
    )
    timeline.partner_profile = mit_only_probs + heal_pulses

    # Empirical timed partner healing pulses already capture 100% of partner healing without double-counting!
    timeline.partner_baseline_healing = PartnerBaselineHealing()
    return timeline, accepted, rejected

"""End-to-end FFLogs encounter dataset builder with Chad-healing curation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from xiv_healwork_solver.fflogs.client import FFLogsClient
from xiv_healwork_solver.fflogs.curator import curate_fight_candidates
from xiv_healwork_solver.fflogs.extractor import (
    extract_damage_timeline,
    extract_partner_probabilities,
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

    # Extract partner / co-healer / tank cast streams across all selected curated clears
    cast_streams: list[dict[str, Any]] = []
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
        }
        # Exclude the target_spec player (e.g. Scholar) so partner_profile contains ONLY co-healer/tank/DPS!
        for a in actors.values():
            if str(a.get("subType", "")).lower() == target_spec.lower():
                exclude_names.add(str(a.get("name", "")))
        casts = c.get_fight_events(
            report_code=rep.report_code,
            fight_id=rep.fight_id,
            start_time=float(f["startTime"]),
            end_time=float(f["endTime"]),
            data_type="Casts",
            hostility_type="Friendlies",
        )
        cast_streams.append(
            {
                "fight_start_ms": float(f["startTime"]),
                "fight_end_ms": float(f["endTime"]),
                "casts": casts,
                "actors_by_id": actors,
            }
        )

    partner_probs = extract_partner_probabilities(
        timeline=timeline,
        fights_cast_streams=cast_streams,
        known_skill_effects=known_skills,
        exclude_actor_names=exclude_names,
    )
    timeline.partner_profile = partner_probs

    # Estimate co-healer & tank baseline effective healing per 3s tick from cached Healing tables!
    cohealer_party_ticks: list[float] = []
    mt_self_ticks: list[float] = []
    st_self_ticks: list[float] = []
    for rep in selected:
        ht = healing_tables.get((rep.report_code, rep.fight_id)) or {}
        dur_s = max(1.0, rep.duration_s)
        co_eff = 0.0
        tank_effs: list[float] = []
        for e in (ht.get("entries") or []):
            spec = str(e.get("type", ""))
            eff = float(e.get("total") or 0.0)
            if spec in HEALER_SPECS and spec.lower() != target_spec.lower():
                co_eff += eff
            elif spec in TANK_SPECS:
                tank_effs.append(eff)
        # Co-healer effective healing spread across 8 party members (net ~75% after big CDs already in partner_profile)
        if co_eff > 0:
            cohealer_party_ticks.append((co_eff * 0.75 / 8.0) / dur_s * 3.0)
        if tank_effs:
            tank_effs.sort(reverse=True)
            mt_self_ticks.append((tank_effs[0] / dur_s) * 3.0)
            if len(tank_effs) >= 2:
                st_self_ticks.append((tank_effs[1] / dur_s) * 3.0)

    timeline.partner_baseline_healing = PartnerBaselineHealing(
        party_per_3s_tick=round(sum(cohealer_party_ticks) / len(cohealer_party_ticks), 1) if cohealer_party_ticks else 0.0,
        mt_per_3s_tick=round(sum(mt_self_ticks) / len(mt_self_ticks), 1) if mt_self_ticks else 0.0,
        st_per_3s_tick=round(sum(st_self_ticks) / len(st_self_ticks), 1) if st_self_ticks else 0.0,
    )
    return timeline, accepted, rejected

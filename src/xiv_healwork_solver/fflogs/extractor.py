"""Extract normalized timelines, AA segments, DoTs, partner probabilities, and weave windows from FFLogs."""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from xiv_healwork_solver.models import (
    AutoAttackSegment,
    DamageEvent,
    DamageType,
    DowntimeWindow,
    EncounterTimeline,
    HpThresholdTrigger,
    PartnerActionProb,
    RoleType,
    WeaveOverrideWindow,
)


def _infer_damage_type(ability_type: int | None) -> DamageType:
    if ability_type is None:
        return "magical"
    if ability_type in (1, 128):
        return "physical"
    if ability_type in (32, 64):
        return "special"
    return "magical"


def _actor_role_from_subtype(sub_type: str | None) -> RoleType:
    s = (sub_type or "").lower()
    if s in ("paladin", "warrior", "darkknight", "gunbreaker"):
        return "tank"
    if s in ("whitemage", "scholar", "astrologian", "sage"):
        return "healer"
    return "dps"


def _identify_tank_slots(
    damage_events: list[dict[str, Any]],
    actors_by_id: dict[int, dict[str, Any]],
    abilities_by_id: dict[int, dict[str, Any]],
    aa_names: set[str],
) -> dict[int, str]:
    """Assign 'mt' from earliest single-target tank auto-attack (falling back to single-target tank hits).

    Never uses multi-target raidwide target-list order so an opening raidwide cannot flip MT/ST.
    """
    # Group hits by (timestamp_bucket_100ms, ability_id) to distinguish single-target hits from raidwides
    targets_per_cast: dict[tuple[int, int], set[int]] = defaultdict(set)
    for ev in damage_events:
        if ev.get("type") not in ("damage", "calculateddamage") or ev.get("tick"):
            continue
        tid = ev.get("targetID")
        if tid is None:
            continue
        bucket = int(float(ev.get("timestamp", 0)) // 100)
        targets_per_cast[(bucket, int(ev.get("abilityGameID", 0)))].add(int(tid))

    tank_ids: list[int] = []
    # Pass 1: single-target auto-attacks on tanks
    for ev in sorted(damage_events, key=lambda x: float(x.get("timestamp", 0))):
        if ev.get("type") not in ("damage", "calculateddamage") or ev.get("tick"):
            continue
        tid = ev.get("targetID")
        if tid is None:
            continue
        tid_i = int(tid)
        if _actor_role_from_subtype(actors_by_id.get(tid_i, {}).get("subType")) != "tank":
            continue
        ab_id = int(ev.get("abilityGameID", 0))
        ab_name = str(abilities_by_id.get(ab_id, {}).get("name", "")).lower()
        bucket = int(float(ev.get("timestamp", 0)) // 100)
        if ab_name in aa_names and len(targets_per_cast[(bucket, ab_id)]) == 1:
            if tid_i not in tank_ids:
                tank_ids.append(tid_i)
            if len(tank_ids) >= 2:
                break

    # Pass 2: fallback to any single-target tank hit, then any tank actor
    if len(tank_ids) < 2:
        for ev in sorted(damage_events, key=lambda x: float(x.get("timestamp", 0))):
            if ev.get("type") not in ("damage", "calculateddamage") or ev.get("tick"):
                continue
            tid = ev.get("targetID")
            if tid is None:
                continue
            tid_i = int(tid)
            if _actor_role_from_subtype(actors_by_id.get(tid_i, {}).get("subType")) != "tank":
                continue
            bucket = int(float(ev.get("timestamp", 0)) // 100)
            if len(targets_per_cast[(bucket, int(ev.get("abilityGameID", 0)))]) == 1:
                if tid_i not in tank_ids:
                    tank_ids.append(tid_i)
                if len(tank_ids) >= 2:
                    break
    if len(tank_ids) < 2:
        for tid_i, actor in sorted(actors_by_id.items()):
            if _actor_role_from_subtype(actor.get("subType")) == "tank" and tid_i not in tank_ids:
                tank_ids.append(tid_i)
            if len(tank_ids) >= 2:
                break

    mapping: dict[int, str] = {}
    if len(tank_ids) >= 1:
        mapping[tank_ids[0]] = "mt"
    if len(tank_ids) >= 2:
        mapping[tank_ids[1]] = "st"
    return mapping


def extract_damage_timeline(
    encounter_name: str,
    zone_name: str | None,
    fight_start_ms: float,
    fight_end_ms: float,
    damage_events: list[dict[str, Any]],
    actors_by_id: dict[int, dict[str, Any]],
    abilities_by_id: dict[int, dict[str, Any]],
    min_raw_damage: float = 25_000.0,
    cluster_window_s: float = 1.5,
    auto_attack_ability_names: set[str] | None = None,
) -> EncounterTimeline:
    """Extract DamageEvents (with snapshot vs impact), AutoAttackSegments (MT/ST), and DoT totals."""
    aa_names = {n.lower() for n in (auto_attack_ability_names or {"attack", "shot", "攻撃"})}
    tank_slot_by_id = _identify_tank_slots(damage_events, actors_by_id, abilities_by_id, aa_names)

    # 1. Correlate `calculateddamage` (snapshot + fallback raw amount) and `damage` (impact) via packetID
    snapshot_by_packet: dict[tuple[int, int, int], float] = {}
    calc_amount_by_packet: dict[tuple[int, int, int], float] = {}
    has_impact_packets: set[tuple[int, int, int]] = set()
    for ev in damage_events:
        pkt = ev.get("packetID")
        if pkt is None:
            continue
        key = (int(pkt), int(ev.get("abilityGameID", 0)), int(ev.get("targetID", -1)))
        rel_s = (float(ev["timestamp"]) - fight_start_ms) / 1000.0
        if ev.get("type") == "calculateddamage":
            snapshot_by_packet[key] = rel_s
            calc_unmit = float(
                ev.get("unmitigatedAmount")
                or (ev.get("amount", 0) + ev.get("absorbed", 0) + ev.get("mitigated", 0))
            )
            calc_amount_by_packet[key] = calc_unmit
        elif ev.get("type") == "damage":
            has_impact_packets.add(key)

    hits: list[dict[str, Any]] = []
    aa_hits: list[dict[str, Any]] = []
    dot_ticks: list[dict[str, Any]] = []

    for ev in damage_events:
        ev_type = ev.get("type")
        if ev_type not in ("damage", "calculateddamage"):
            continue
        # Ignore friendly self-damage effects (e.g., Gunbreaker Superbolide HP-to-1 self-hit!)
        src_id = ev.get("sourceID")
        if src_id is not None and int(src_id) in actors_by_id and actors_by_id[int(src_id)].get("type") == "Player":
            continue
        rel_s = (float(ev["timestamp"]) - fight_start_ms) / 1000.0
        if rel_s < 0:
            continue
        ability_id = int(ev.get("abilityGameID", 0))
        target_id = ev.get("targetID")
        pkt = ev.get("packetID")
        pkt_key = (int(pkt), ability_id, int(target_id or -1)) if pkt is not None else None

        # Skip calculateddamage duplicate when matching `damage` packet exists
        if ev_type == "calculateddamage" and pkt_key is not None and pkt_key in has_impact_packets:
            continue

        unmit = float(
            ev.get("unmitigatedAmount")
            or (ev.get("amount", 0) + ev.get("absorbed", 0) + ev.get("mitigated", 0))
        )
        # Preserve calculateddamage unmitigated amount when paired impact record reports 0 (e.g. fully shielded)
        if unmit <= 0.0 and pkt_key is not None:
            unmit = calc_amount_by_packet.get(pkt_key, 0.0)
        ab = abilities_by_id.get(ability_id, {})
        ab_name = str(ab.get("name", f"Ability_{ability_id}"))
        dmg_type = _infer_damage_type(ab.get("type"))

        if ev.get("tick"):
            # Skip FFLogs's synthetic roll-up ability 500000 ("Combined DoTs") to avoid double-counting real DoTs
            if ability_id == 500000 or ab_name.lower() == "combined dots":
                continue
            if unmit > 0:
                dot_ticks.append(
                    {
                        "time_s": rel_s,
                        "ability_id": ability_id,
                        "target_id": target_id,
                        "raw_damage": unmit,
                    }
                )
            continue

        # Detect tank auto-attacks strictly by AA ability name (never sweep low-damage boss mechanics like Brutal Rain!)
        is_tank_target = target_id is not None and int(target_id) in tank_slot_by_id
        if is_tank_target and ab_name.lower() in aa_names and unmit >= 3_000.0:
            aa_hits.append(
                {
                    "time_s": rel_s,
                    "target_slot": tank_slot_by_id[int(target_id)],
                    "damage_type": dmg_type,
                    "raw_damage": unmit,
                }
            )
            continue

        if unmit < min_raw_damage:
            continue

        snap_s = rel_s
        if pkt is not None:
            snap_s = snapshot_by_packet.get((int(pkt), ability_id, int(target_id or -1)), rel_s)

        hits.append(
            {
                "time_s": rel_s,
                "snapshot_s": min(snap_s, rel_s),
                "ability_id": ability_id,
                "ability_name": ab_name,
                "damage_type": dmg_type,
                "target_id": target_id,
                "raw_damage": unmit,
                "amount": float(ev.get("amount") or 0.0),
                "absorbed": float(ev.get("absorbed") or 0.0),
                "mitigated": float(ev.get("mitigated") or 0.0),
                "overkill": float(ev.get("overkill") or 0.0),
            }
        )

    hits.sort(key=lambda h: h["time_s"])

    # 2. Cluster major hits by ability within cluster_window_s, AND merge disjoint-target split waves
    # (e.g. 4+4 Blast Beat or 2+2+2+2 Aetherletting where each player only gets hit once across waves!)
    clusters: list[list[dict[str, Any]]] = []
    cluster_targets: list[set[int]] = []
    for h in hits:
        placed = False
        tid = int(h["target_id"]) if h["target_id"] is not None else -1
        for ci in range(len(clusters) - 1, -1, -1):
            cl = clusters[ci]
            dt_last = h["time_s"] - cl[-1]["time_s"]
            if dt_last > 8.0:
                break
            if cl[0]["ability_id"] != h["ability_id"]:
                continue
            # Same hit cluster within cluster_window_s, OR disjoint-target wave within 6.0s!
            if dt_last <= cluster_window_s or (dt_last <= 6.0 and tid != -1 and tid not in cluster_targets[ci]):
                cl.append(h)
                if tid != -1:
                    cluster_targets[ci].add(tid)
                placed = True
                break
        if not placed:
            clusters.append([h])
            cluster_targets.append({tid} if tid != -1 else set())

    consumed_dot_indices: set[int] = set()
    events: list[DamageEvent] = []
    event_all_targets: list[set[int]] = []
    event_nontank_targets: list[set[int]] = []
    for idx, cl in enumerate(clusters, start=1):
        unique_targets = {int(x["target_id"]) for x in cl if x["target_id"] is not None}
        tank_slots_hit = {tank_slot_by_id[tid] for tid in unique_targets if tid in tank_slot_by_id}
        nontank_targets = {tid for tid in unique_targets if tid not in tank_slot_by_id}

        if len(unique_targets) >= 4:
            scope = "party"
        elif len(unique_targets) == 2 and tank_slots_hit == {"mt", "st"}:
            scope = "both_tanks"
        elif len(unique_targets) == 1 and len(tank_slots_hit) == 1:
            scope = next(iter(tank_slots_hit))  # Exact "mt" or "st"!
        elif len(unique_targets) >= 2:
            scope = "party"
        else:
            # In a full party log (>=4 actors), a 1-target hit on a non-tank is an avoidable AoE clip
            # (e.g. one player eating Half Moon) or single-player bait, NOT a party-wide raidwide.
            if len(actors_by_id) >= 4 and not tank_slots_hit:
                continue
            scope = "party"

        # Detect FFLogs HP-to-1 / invuln-surviving 9,999,999 mechanics (e.g. Nothingness, Beyond Death Antilight,
        # Living Dead 11M Chaotic Holy where mitigated==0 and target survived in a 0-death clear)
        is_set_hp1 = False
        for x in cl:
            if x["amount"] <= 0:
                continue
            if x["mitigated"] == 0.0 and x["raw_damage"] >= 9_000_000.0:
                is_set_hp1 = True
                break
            is_nt = x["target_id"] is None or int(x["target_id"]) not in tank_slot_by_id
            if is_nt and x["mitigated"] == 0.0 and x["raw_damage"] >= 500_000.0 and x["amount"] >= 150_000.0:
                is_set_hp1 = True
                break

        # Use true non-zero unmitigated hits from non-tank targets for party raidwides (avoids 100%-shielded zeros & tank armor bias)
        if is_set_hp1:
            avg_raw = 0.0
        elif scope == "party":
            non_tank_unmit = [
                x["raw_damage"]
                for x in cl
                if (x["target_id"] is None or int(x["target_id"]) not in tank_slot_by_id) and x["raw_damage"] > 0
            ]
            avg_raw = mean(non_tank_unmit) if non_tank_unmit else mean(x["raw_damage"] for x in cl)
        else:
            avg_raw = mean(x["raw_damage"] for x in cl)
        hit_count = 1 if is_set_hp1 else max(1, round(len(cl) / max(1, len(unique_targets))))
        first_t = cl[0]["time_s"]
        last_t = cl[-1]["time_s"]
        hit_interval = round((last_t - first_t) / (hit_count - 1), 2) if hit_count > 1 and last_t > first_t else 0.0
        snap_t = round(min(x["snapshot_s"] for x in cl), 2)

        # Attach subsequent unconsumed DoT ticks from the same ability (never double-count overlapping apps)
        matching_dots: list[float] = []
        for d_idx, d in enumerate(dot_ticks):
            if d_idx in consumed_dot_indices:
                continue
            if d["ability_id"] == cl[0]["ability_id"] and first_t < d["time_s"] <= first_t + 21.0:
                consumed_dot_indices.add(d_idx)
                matching_dots.append(d["raw_damage"])
        dot_per_target = (sum(matching_dots) / max(1, len(unique_targets))) if matching_dots else 0.0

        cand = DamageEvent(
            id=f"e{idx}_{cl[0]['ability_id']}_{int(first_t)}",
            time_s=round(first_t, 2),
            snapshot_time_s=snap_t if snap_t < round(first_t, 2) else None,
            name=cl[0]["ability_name"],
            ability_id=cl[0]["ability_id"],
            damage_type=cl[0]["damage_type"],
            target_scope=scope,  # type: ignore[arg-type]
            raw_damage=0.0 if is_set_hp1 else round(avg_raw * hit_count, 1),
            hit_count=hit_count,
            hit_interval_s=hit_interval,
            dot_total_damage=round(dot_per_target, 1),
            mechanic_effect="set_hp_to_1" if is_set_hp1 else "normal",
        )

        # Merge simultaneous or alternating disjoint-target split-party mechanics (e.g. Group A hit then
        # Group B hit within 5.5s with ZERO overlapping targets across both tanks and non-tanks, and no DoT loss)
        # so the single `party` channel is never hit twice by mutually exclusive mechanics!
        merged_disjoint = False
        if scope == "party" and nontank_targets and cand.dot_total_damage <= 0.0 and cand.mechanic_effect == "normal":
            for prev_i in range(len(events) - 1, -1, -1):
                prev_ev = events[prev_i]
                if cand.time_s - prev_ev.time_s > 5.5:
                    break
                if (
                    prev_ev.target_scope == "party"
                    and prev_ev.mechanic_effect == "normal"
                    and event_nontank_targets[prev_i]
                    and event_all_targets[prev_i].isdisjoint(unique_targets)
                ):
                    if cand.raw_damage > prev_ev.raw_damage:
                        events[prev_i] = prev_ev.model_copy(
                            update={
                                "raw_damage": cand.raw_damage,
                                "name": f"{prev_ev.name} / {cand.name}" if cand.name != prev_ev.name else prev_ev.name,
                            }
                        )
                    elif cand.name != prev_ev.name and cand.name not in prev_ev.name:
                        events[prev_i] = prev_ev.model_copy(update={"name": f"{prev_ev.name} / {cand.name}"})
                    event_all_targets[prev_i] |= unique_targets
                    event_nontank_targets[prev_i] |= nontank_targets
                    merged_disjoint = True
                    break

        if not merged_disjoint:
            events.append(cand)
            event_all_targets.append(set(unique_targets))
            event_nontank_targets.append(set(nontank_targets))

    # Preserve standalone / differently-IDed DoTs as their own events so bleeds never vanish
    unmatched_dots = [d for i, d in enumerate(dot_ticks) if i not in consumed_dot_indices]
    if unmatched_dots:
        unmatched_dots.sort(key=lambda d: d["time_s"])
        dot_clusters: list[list[dict[str, Any]]] = []
        for d in unmatched_dots:
            placed = False
            for dcl in reversed(dot_clusters):
                if d["ability_id"] == dcl[0]["ability_id"] and d["time_s"] - dcl[-1]["time_s"] <= 4.5:
                    dcl.append(d)
                    placed = True
                    break
            if not placed:
                dot_clusters.append([d])
        for dcl in dot_clusters:
            u_targs = {int(x["target_id"]) for x in dcl if x["target_id"] is not None}
            t_slots = {tank_slot_by_id[tid] for tid in u_targs if tid in tank_slot_by_id}
            nt_targs = {tid for tid in u_targs if tid not in tank_slot_by_id}
            if len(actors_by_id) >= 4:
                # In full-party logs, only broadcast as party-wide DoT if >=3 non-tanks received it;
                # otherwise route to affected tank slot(s) or ignore 1-2 player avoidable puddle bleeds.
                if len(nt_targs) >= 3:
                    d_scope = "party"
                elif t_slots == {"mt", "st"}:
                    d_scope = "both_tanks"
                elif len(t_slots) == 1:
                    d_scope = next(iter(t_slots))
                else:
                    continue
            else:
                d_scope = next(iter(t_slots)) if (len(u_targs) == 1 and len(t_slots) == 1) else ("both_tanks" if (len(u_targs) == 2 and t_slots == {"mt", "st"}) else "party")
            ab = abilities_by_id.get(dcl[0]["ability_id"], {})
            d_total = sum(x["raw_damage"] for x in dcl) / max(1, len(u_targs))
            d_dur = max(3.0, round(dcl[-1]["time_s"] - dcl[0]["time_s"] + 3.0, 1))
            t0_dot = max(0.0, round(dcl[0]["time_s"] - 3.0, 2))
            ab_id_dot = dcl[0]["ability_id"]
            dot_name = ab.get("name", f"DoT_{ab_id_dot}")
            events.append(
                DamageEvent(
                    id=f"dot_{ab_id_dot}_{int(t0_dot)}",
                    time_s=t0_dot,
                    name=f"{dot_name} (DoT)",
                    ability_id=ab_id_dot,
                    damage_type=_infer_damage_type(ab.get("type")),
                    target_scope=d_scope,  # type: ignore[arg-type]
                    raw_damage=0.0,
                    dot_total_damage=round(d_total, 1),
                    dot_duration_s=d_dur,
                )
            )
        events.sort(key=lambda e: e.time_s)

    # 3. Group tank auto-attacks into contiguous MT / ST AutoAttackSegments (preserves 1-hit segments too!)
    auto_attacks: list[AutoAttackSegment] = []
    if aa_hits:
        aa_hits.sort(key=lambda a: a["time_s"])
        seg_hits: list[dict[str, Any]] = [aa_hits[0]]

        def _flush_aa(sh: list[dict[str, Any]]) -> None:
            start_s = round(sh[0]["time_s"], 2)
            end_s = round(sh[-1]["time_s"], 2)
            if len(sh) > 1 and end_s > start_s:
                # Floor-clamp interval to 0.001s so start_s + (n-1)*interval_s <= end_s never drops the last AA hit!
                exact_int = (end_s - start_s) / (len(sh) - 1)
                interval_s = max(1.5, math.floor(exact_int * 1000.0) / 1000.0)
            else:
                interval_s = 3.0
            auto_attacks.append(
                AutoAttackSegment(
                    start_s=start_s,
                    end_s=end_s,
                    target=sh[0]["target_slot"],  # type: ignore[arg-type]
                    damage_type=sh[0]["damage_type"],
                    raw_damage_per_hit=round(mean(x["raw_damage"] for x in sh), 1),
                    interval_s=interval_s,
                )
            )

        for ah in aa_hits[1:]:
            if ah["target_slot"] == seg_hits[-1]["target_slot"] and (ah["time_s"] - seg_hits[-1]["time_s"]) <= 8.0:
                seg_hits.append(ah)
            else:
                _flush_aa(seg_hits)
                seg_hits = [ah]
        _flush_aa(seg_hits)

    duration_s = max(10.0, round((fight_end_ms - fight_start_ms) / 1000.0, 2))
    return EncounterTimeline(
        encounter_name=encounter_name,
        zone_name=zone_name,
        duration_s=duration_s,
        events=events,
        auto_attacks=auto_attacks,
    )


def extract_downtime_windows(
    damage_done_events: list[dict[str, Any]],
    fight_start_ms: float,
    fight_end_ms: float,
    min_gap_s: float = 6.0,
) -> list[DowntimeWindow]:
    """Infer untargetable / phase-transition windows from gaps in non-tick friendly DamageDone."""
    hit_ts = sorted(
        (float(ev["timestamp"]) - fight_start_ms) / 1000.0
        for ev in damage_done_events
        if ev.get("type") == "damage" and not ev.get("tick") and float(ev["timestamp"]) >= fight_start_ms
    )
    windows: list[DowntimeWindow] = []
    dur_s = max(0.0, (fight_end_ms - fight_start_ms) / 1000.0)
    idx = 0
    for a, b in zip(hit_ts, hit_ts[1:]):
        if b - a >= min_gap_s and b <= dur_s:
            idx += 1
            windows.append(
                DowntimeWindow(
                    start_s=round(a + 0.5, 2),
                    end_s=round(b - 0.5, 2),
                    name=f"Phase Transition {idx}",
                    boss_targetable=False,
                    party_can_act=True,
                    clears_boss_debuffs=True,
                )
            )
    return windows


def extract_hp_threshold_triggers(
    timeline: EncounterTimeline,
    debuff_events: list[dict[str, Any]],
    abilities_by_id: dict[int, dict[str, Any]],
    fight_start_ms: float,
) -> tuple[list[HpThresholdTrigger], list[str]]:
    """Detect real full-heal-triggered raidwide debuffs (e.g. Dancing Mad 'Accretion' -> Earthquake).

    Returns (triggers, replaced_event_ids) so static events already modeled dynamically aren't double-counted.
    """
    full_heal_debuff_names = {"accretion"}
    applies: list[float] = []
    removes: list[float] = []
    for ev in debuff_events:
        ab_name = str(abilities_by_id.get(int(ev.get("abilityGameID") or 0), {}).get("name", "")).lower()
        if ab_name not in full_heal_debuff_names:
            continue
        t_s = (float(ev["timestamp"]) - fight_start_ms) / 1000.0
        if ev.get("type") == "applydebuff":
            applies.append(t_s)
        elif ev.get("type") == "removedebuff":
            removes.append(t_s)
    if not applies:
        return [], []

    applies.sort()
    removes.sort()
    # Cluster apply windows separated by > 25s
    clusters: list[list[float]] = []
    for t in applies:
        if not clusters or t - clusters[-1][-1] > 25.0:
            clusters.append([t])
        else:
            clusters[-1].append(t)

    triggers: list[HpThresholdTrigger] = []
    replaced_ids: list[str] = []
    for ci, cl in enumerate(clusters, start=1):
        t_start = cl[0]
        # Find removes belonging to this cluster (within 25s of apply)
        cl_rems = [r for r in removes if t_start <= r <= t_start + 25.0]
        t_end = max(cl_rems[-1] + 5.0, t_start + 18.0) if cl_rems else (t_start + 18.0)
        # Match party raidwide events firing 0.5..3.5s after each remove (e.g. Earthquake after Accretion pops)
        matched_evs = []
        for r_t in cl_rems:
            for e in timeline.events:
                if e.target_scope == "party" and 0.4 <= (e.time_s - r_t) <= 3.5 and e.id not in replaced_ids:
                    matched_evs.append(e)
                    replaced_ids.append(e.id)
                    break
        if not matched_evs:
            continue
        blast_dmg = round(mean(e.raw_damage for e in matched_evs), 1)
        triggers.append(
            HpThresholdTrigger(
                id=f"hp_trig_accretion_{ci}",
                name=f"Accretion Full-Heal Trigger ({matched_evs[0].name})",
                start_s=round(t_start, 2),
                end_s=round(min(timeline.duration_s, t_end), 2),
                watch_channel="party",
                trigger_hp_ratio=0.999,
                blast_target_scope="party",
                blast_raw_damage=max(50_000.0, blast_dmg),
                blast_damage_type=matched_evs[0].damage_type,
                max_triggers=len(cl),
                trigger_cooldown_s=4.5,
                must_trigger_before_end=True,
            )
        )
    return triggers, replaced_ids


def extract_partner_probabilities(
    timeline: EncounterTimeline,
    fights_cast_streams: list[dict[str, Any]],
    known_skill_effects: dict[int, dict[str, Any]],
    exclude_actor_names: set[str] | None = None,
    include_only_actor_names: set[str] | None = None,
    pre_window_s: float = 12.0,
    post_heal_window_s: float = 8.0,
) -> list[PartnerActionProb]:
    """Learn empirical mitigation/heal probabilities per event using phase-reached pull denominators.

    - Mitigation/shield casts are only matched BEFORE the event snapshot (never 4s after hit).
    - Pure heal casts may match up to `post_heal_window_s` after hit.
    - Denominator per event counts ONLY pulls that actually survived/reached `event.time_s`.
    """
    exclude_set = {n.lower() for n in (exclude_actor_names or set())}
    include_set = {n.lower() for n in (include_only_actor_names or set())} if include_only_actor_names else None

    observations: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    meta_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    reached_counts: dict[str, int] = defaultdict(int)

    for fight_bundle in fights_cast_streams:
        fight_start_ms = float(fight_bundle["fight_start_ms"])
        fight_end_ms = float(fight_bundle.get("fight_end_ms", fight_start_ms + timeline.duration_s * 1000.0))
        fight_dur_s = max(0.0, (fight_end_ms - fight_start_ms) / 1000.0)
        casts = fight_bundle["casts"]
        actors_by_id: dict[int, dict[str, Any]] = fight_bundle["actors_by_id"]

        # Identify MT/ST actor IDs in this fight so personal tank mits route to the actual caster's tank slot!
        tank_actors = [int(aid) for aid, a in sorted(actors_by_id.items()) if _actor_role_from_subtype(a.get("subType")) == "tank"]
        tank_slot_map = fight_bundle.get("tank_slot_by_id") or (
            {tank_actors[0]: "mt", **({tank_actors[1]: "st"} if len(tank_actors) > 1 else {})} if tank_actors else {}
        )

        for dmg_ev in timeline.events:
            if fight_dur_s >= dmg_ev.time_s - 0.5:
                reached_counts[dmg_ev.id] += 1

        seen_in_fight: set[tuple[str, str, str]] = set()
        for ev in casts:
            if ev.get("type") != "cast":
                continue
            ab_id = int(ev.get("abilityGameID", 0))
            if ab_id not in known_skill_effects:
                continue
            eff = known_skill_effects[ab_id]
            src_id = int(ev.get("sourceID", -1))
            src = actors_by_id.get(src_id, {})
            src_name = str(src.get("name", ""))
            if src_name.lower() in exclude_set:
                continue
            if include_set is not None and src_name.lower() not in include_set:
                continue

            # Route tank personal mitigations to whichever tank (mt or st) actually cast it
            t_channel = str(eff.get("target_channel", "party"))
            if t_channel in ("mt", "st") and src_id in tank_slot_map:
                t_channel = tank_slot_map[src_id]

            # Aggregate across job-equivalent skills using `role_slot` when provided (e.g., PLD/WAR/DRK/GNB 40% CD)
            slot_key = str(eff.get("role_slot") or ab_id)

            cast_s = (float(ev["timestamp"]) - fight_start_ms) / 1000.0
            is_mit_or_shield = (
                float(eff.get("mitigation_all", 0.0)) > 0
                or float(eff.get("mitigation_physical", 0.0)) > 0
                or float(eff.get("mitigation_magical", 0.0)) > 0
                or float(eff.get("shield_value", 0.0)) > 0
            )
            dur_s = float(eff.get("duration_s", 15.0))
            # Include FFXIV ~0.8s buff application latency so 15s mits cast ~15.5s pre-snapshot still match
            effective_pre = min(max(pre_window_s, 16.0), max(2.0, dur_s + 0.8))

            for dmg_ev in timeline.events:
                if fight_dur_s < dmg_ev.time_s - 0.5:
                    continue
                snap_s = dmg_ev.effective_snapshot_time()
                offset = cast_s - snap_s
                if is_mit_or_shield:
                    matched = -effective_pre <= offset <= 0.2
                else:
                    matched = -2.0 <= offset <= post_heal_window_s
                if matched:
                    key = (dmg_ev.id, slot_key, t_channel)
                    if key not in seen_in_fight:
                        seen_in_fight.add(key)
                        observations[key].append(offset)
                        meta_by_key[key] = {
                            "effect": eff,
                            "role": _actor_role_from_subtype(src.get("subType")),
                        }

    results: list[PartnerActionProb] = []
    for (event_id, _slot_key, t_channel), offsets in observations.items():
        denom = max(1, reached_counts.get(event_id, len(fights_cast_streams)))
        prob = len(offsets) / denom
        if prob < 0.15:
            continue
        eff = meta_by_key[(event_id, _slot_key, t_channel)]["effect"]
        role = meta_by_key[(event_id, _slot_key, t_channel)]["role"]
        results.append(
            PartnerActionProb(
                event_id=event_id,
                skill_name=eff["name"],
                source_role=role,
                target_channel=t_channel,  # type: ignore[arg-type]
                stack_group=eff.get("stack_group"),
                shield_stack_group=eff.get("shield_stack_group"),
                duration_s=float(eff.get("duration_s", 15.0)),
                mitigation_all=float(eff.get("mitigation_all", 0.0)),
                mitigation_physical=float(eff.get("mitigation_physical", 0.0)),
                mitigation_magical=float(eff.get("mitigation_magical", 0.0)),
                shield_value=float(eff.get("shield_value", 0.0)),
                heal_value=float(eff.get("heal_value", 0.0)),
                probability=round(prob, 3),
                timing_mean_offset_s=round(mean(offsets), 2),
                timing_std_s=round(pstdev(offsets), 2) if len(offsets) > 1 else 1.0,
                sample_count=denom,
            )
        )
    results.sort(key=lambda x: (x.event_id, -x.probability))
    return results


def extract_partner_healing_pulses(
    timeline: EncounterTimeline,
    fights_heal_streams: list[dict[str, Any]],
    exclude_actor_names: set[str] | None = None,
    exclude_pet_names: set[str] | None = None,
    min_pulse_hp: float = 500.0,
    max_bin_s: float = 6.0,
) -> list[PartnerActionProb]:
    """Extract empirical timed partner effective healing & absorbed shields from FFLogs Healing streams.

    Buckets each inter-event window into empirical <=6s sub-bins (and per-hit slots during multi-hit
    stacks) placed at the actual amount-weighted mean timestamp. Excludes FFLogs synthetic `Combined HoTs`
    (ID 500001) and the target player's own pets, preserves unpaired positive `calculatedheal` records,
    and marks non-tank `party` pulses `non_tank_only=True` so tanks are never double-healed.
    """
    exclude_set = {n.lower() for n in (exclude_actor_names or set())}
    exclude_pets = {n.lower() for n in (exclude_pet_names or set())}
    events = sorted(timeline.events, key=lambda e: e.time_s)
    if not events or not fights_heal_streams:
        return []

    # Precompute contiguous event intervals and explicit sub-bin boundaries [(b_lo, b_hi), ...].
    # For multi-hit events, explicitly split at last_hit_t so post-stack recovery heals are never pulled before the final hit!
    # (eid, ev_snap_t, lo, hi, bins)
    base_windows: list[tuple[str, float, float, float, list[tuple[float, float]]]] = []
    for idx, ev in enumerate(events):
        lo = 0.0 if idx == 0 else max(0.0, ev.time_s - 1.0)
        hi = (events[idx + 1].time_s - 1.0) if idx + 1 < len(events) else timeline.duration_s
        if hi <= lo:
            continue
        bins: list[tuple[float, float]] = []
        sub_spans: list[tuple[float, float]] = []
        if ev.hit_count > 1 and ev.hit_interval_s > 0.0:
            last_hit_t = ev.time_s + (ev.hit_count - 1) * ev.hit_interval_s + 0.05
            if lo < last_hit_t < hi:
                # Split multi-hit stacks per inter-hit slot so mid-stack heals (e.g. Bell/Ixochole)
                # land between the exact hits they occurred on instead of overhealing in one lump!
                prev_b = lo
                for h_i in range(1, ev.hit_count):
                    h_t = min(last_hit_t, ev.time_s + h_i * ev.hit_interval_s)
                    if h_t > prev_b + 0.05:
                        bins.append((prev_b, h_t))
                        prev_b = h_t
                if last_hit_t > prev_b + 0.05:
                    bins.append((prev_b, last_hit_t))
                sub_spans.append((last_hit_t, hi))
            else:
                sub_spans.append((lo, hi))
        else:
            sub_spans.append((lo, hi))
        for s_lo, s_hi in sub_spans:
            n_b = max(1, math.ceil((s_hi - s_lo) / max(1.0, max_bin_s)))
            w_b = (s_hi - s_lo) / n_b
            for k in range(n_b):
                bins.append((s_lo + k * w_b, s_lo + (k + 1) * w_b))
        base_windows.append((ev.id, ev.effective_snapshot_time(), lo, hi, bins))

    # Accumulate per-(eid, bin_idx, channel): list of per-pull effective healing & empirical partner shields
    per_bin_vals: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    per_bin_wtime: dict[tuple[str, int, str], float] = defaultdict(float)
    per_bin_wsum: dict[tuple[str, int, str], float] = defaultdict(float)

    per_bin_shield_vals: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    per_bin_shield_wtime: dict[tuple[str, int, str], float] = defaultdict(float)
    per_bin_shield_wsum: dict[tuple[str, int, str], float] = defaultdict(float)

    ev_by_id = {e.id: e for e in events}

    for bundle in fights_heal_streams:
        t0 = float(bundle["fight_start_ms"])
        t_end = float(bundle.get("fight_end_ms", t0 + timeline.duration_s * 1000.0))
        dur_s = max(0.0, (t_end - t0) / 1000.0)
        heals = bundle["heals"]
        actors_by_id: dict[int, dict[str, Any]] = bundle["actors_by_id"]

        tank_ids = [int(aid) for aid, a in sorted(actors_by_id.items()) if _actor_role_from_subtype(a.get("subType")) == "tank"]
        tank_slot = bundle.get("tank_slot_by_id") or (
            {tank_ids[0]: "mt", **({tank_ids[1]: "st"} if len(tank_ids) > 1 else {})} if tank_ids else {}
        )
        # Count ONLY real combat non-tank players (FFLogs puts LimitBreak / Multiple Players inside friendlyPlayers!)
        non_tank_count = max(
            1,
            sum(
                1
                for aid, a in actors_by_id.items()
                if aid not in tank_slot
                and str(a.get("type", "Player")).lower() == "player"
                and str(a.get("subType", "")).lower() not in ("limitbreak", "npc", "pet", "unknown")
            ),
        )

        active_windows = [w for w in base_windows if dur_s >= w[2] + 0.5]
        fight_sums: dict[tuple[str, int, str], float] = defaultdict(float)
        fight_shield_sums: dict[tuple[str, int, str], float] = defaultdict(float)
        all_actors_by_id: dict[int, dict[str, Any]] = bundle.get("all_actors_by_id") or actors_by_id

        # Track packetIDs that have a resolved `heal` event so unpaired `calculatedheal` records are preserved without duplication
        seen_heal_packets: set[tuple[int, int]] = set()
        for ev in heals:
            if ev.get("type") == "heal" and ev.get("packetID") is not None:
                seen_heal_packets.add((int(ev["packetID"]), int(ev.get("targetID", -1))))

        for ev in heals:
            ev_type = ev.get("type")
            if ev_type == "calculatedheal":
                pkt = ev.get("packetID")
                tid_chk = int(ev.get("targetID", -1))
                if pkt is not None and (int(pkt), tid_chk) in seen_heal_packets:
                    continue
                if pkt is None:
                    continue
                ev_type = "heal"
            elif ev_type not in ("heal", "absorbed"):
                continue
            # Skip FFLogs synthetic Combined HoTs (500001) to prevent double-counting real HoT ticks!
            ab_id = int(ev.get("abilityGameID", 0))
            if ab_id in (500000, 500001):
                continue
            amt = float(ev.get("amount") or 0.0)
            oh = float(ev.get("overheal") or 0.0) if ev_type == "heal" else 0.0
            # Include a modest share of logged partner overheal (35%) so real partner HoTs/AoEs
            # (e.g. Asylum ticks that hit 100% HP in one pull) still top the party up under +5% rolls
            # without erasing the need for the player's own mitigations/heals.
            eff_amt = amt + 0.35 * oh
            if eff_amt <= 0.0:
                continue
            src_id = int(ev.get("sourceID", -1))
            src = all_actors_by_id.get(src_id) or actors_by_id.get(src_id, {})
            src_name = str(src.get("name", "")).lower()
            if src_name in exclude_set:
                continue
            if str(src.get("type", "")).lower() == "pet" and src_name in exclude_pets:
                continue
            rel_s = (float(ev["timestamp"]) - t0) / 1000.0
            tid = int(ev.get("targetID", -1))
            if tid not in actors_by_id and tid not in tank_slot:
                continue
            ch = tank_slot.get(tid, "party")

            for eid, snap_t, lo, hi, bins in active_windows:
                if lo <= rel_s < hi:
                    b_idx = len(bins) - 1
                    for bi, (b_lo, b_hi) in enumerate(bins):
                        if b_lo <= rel_s < b_hi:
                            b_idx = bi
                            break
                    norm_amt = eff_amt / non_tank_count if ch == "party" else eff_amt
                    key = (eid, b_idx, ch)
                    if ev_type == "heal":
                        fight_sums[key] += norm_amt
                        per_bin_wtime[key] += norm_amt * (rel_s - snap_t)
                        per_bin_wsum[key] += norm_amt
                    else:
                        fight_shield_sums[key] += norm_amt
                        per_bin_shield_wtime[key] += norm_amt * (rel_s - snap_t)
                        per_bin_shield_wsum[key] += norm_amt
                    break

        for eid, _snap_t, _lo, _hi, bins in active_windows:
            for b_idx in range(len(bins)):
                for ch in ("party", "mt", "st"):
                    key = (eid, b_idx, ch)
                    per_bin_vals[key].append(fight_sums.get(key, 0.0))
                    per_bin_shield_vals[key].append(fight_shield_sums.get(key, 0.0))

    pulses: list[PartnerActionProb] = []
    for (eid, b_idx, ch), vals in per_bin_vals.items():
        if not vals:
            continue
        avg_h = mean(vals)
        if avg_h < min_pulse_hp:
            continue
        wsum = per_bin_wsum.get((eid, b_idx, ch), 0.0)
        offset_s = (per_bin_wtime[(eid, b_idx, ch)] / wsum) if wsum > 0.0 else 1.2
        pulses.append(
            PartnerActionProb(
                event_id=eid,
                skill_name=f"相方・PT実測回復#{b_idx + 1} ({ch})",
                source_role="healer" if ch == "party" else "tank",
                target_channel=ch,  # type: ignore[arg-type]
                duration_s=0.0,
                heal_value=round(avg_h, 1),
                probability=1.0,
                timing_mean_offset_s=round(offset_s, 2),
                timing_std_s=0.5,
                sample_count=len(vals),
                non_tank_only=(ch == "party"),
            )
        )

    for (eid, b_idx, ch), vals in per_bin_shield_vals.items():
        if not vals:
            continue
        avg_sh = mean(vals)
        if avg_sh < min_pulse_hp:
            continue
        wsum = per_bin_shield_wsum.get((eid, b_idx, ch), 0.0)
        raw_offset = (per_bin_shield_wtime[(eid, b_idx, ch)] / wsum) if wsum > 0.0 else 0.0
        ev_obj = ev_by_id.get(eid)
        # Apply shield slightly BEFORE the observed absorption timestamp (or before event snapshot for bin 0)
        if b_idx == 0:
            sh_offset = min(-1.5, raw_offset - 1.5)
            span_s = ((ev_obj.hit_count - 1) * ev_obj.hit_interval_s) if ev_obj is not None else 0.0
            sh_dur = max(8.0, span_s + 5.0)
        else:
            sh_offset = raw_offset - 1.0
            sh_dur = 8.0
        pulses.append(
            PartnerActionProb(
                event_id=eid,
                skill_name=f"相方・PT実測バリア#{b_idx + 1} ({ch})",
                source_role="healer" if ch == "party" else "tank",
                target_channel=ch,  # type: ignore[arg-type]
                duration_s=round(sh_dur, 1),
                shield_value=round(avg_sh, 1),
                probability=1.0,
                timing_mean_offset_s=round(sh_offset, 2),
                timing_std_s=0.5,
                sample_count=len(vals),
                non_tank_only=(ch == "party"),
            )
        )
    return pulses


def extract_player_expert_actions(
    kit_skills_by_action_id: dict[int, dict[str, Any]],
    fight_start_ms: float,
    casts: list[dict[str, Any]],
    player_actor_id: int,
    tank_slot_by_id: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Extract a real player's chronological defensive/healing cast schedule from FFLogs for BC / replay.

    Uses `begincast` timestamps when available (or subtracts `cast_time_s`) so cast-time spells
    such as Concitation are labeled at cast-start rather than cast-completion.
    """
    tank_map = tank_slot_by_id or {}
    sorted_casts = sorted(casts, key=lambda x: float(x.get("timestamp", 0)))
    # Track most recent begincast per abilityGameID for cast-time spells
    last_begincast_ms: dict[int, float] = {}
    actions: list[dict[str, Any]] = []
    for ev in sorted_casts:
        if int(ev.get("sourceID", -1)) != player_actor_id:
            continue
        ab_id = int(ev.get("abilityGameID", 0))
        if ev.get("type") == "begincast":
            if ab_id in kit_skills_by_action_id:
                last_begincast_ms[ab_id] = float(ev["timestamp"])
            continue
        if ev.get("type") != "cast":
            continue
        if ab_id not in kit_skills_by_action_id:
            continue
        sk_info = kit_skills_by_action_id[ab_id]
        ev_ms = float(ev["timestamp"])
        cast_time_s = float(sk_info.get("cast_time_s", 0.0))
        if cast_time_s > 0.0:
            bg_ms = last_begincast_ms.pop(ab_id, None)
            if bg_ms is not None and 0.0 <= (ev_ms - bg_ms) <= (cast_time_s + 1.5) * 1000.0:
                start_ms = bg_ms
            else:
                start_ms = ev_ms - cast_time_s * 1000.0
        else:
            start_ms = ev_ms
        rel_s = max(0.0, round((start_ms - fight_start_ms) / 1000.0, 2))
        tgt_id = int(ev.get("targetID", -1))
        t_ch = sk_info.get("default_channel") or sk_info.get("target_scope", "party")
        if sk_info.get("target_scope") == "single_ally":
            t_ch = tank_map.get(tgt_id, "mt")
        actions.append(
            {
                "cast_at_s": rel_s,
                "skill_id": sk_info.get("skill_id") or sk_info["id"],
                "skill_name": sk_info.get("skill_name") or sk_info.get("name", ""),
                "target_channel": t_ch,
                "is_gcd": bool(sk_info.get("is_gcd", False)),
            }
        )
    return actions


def extract_weave_windows(
    fight_start_ms: float,
    fight_end_ms: float,
    player_casts: list[dict[str, Any]],
    gcd_ability_ids: set[int],
    gcd_s: float = 2.50,
    max_weave_slots: int = 2,
    bucket_s: float = 10.0,
) -> list[WeaveOverrideWindow]:
    """Infer actual free oGCD weave slots per bucket from a player's own FFLogs casts."""
    duration_s = max(1.0, (fight_end_ms - fight_start_ms) / 1000.0)
    n_buckets = int(duration_s // bucket_s) + 1
    gcd_counts = [0] * n_buckets
    ogcd_counts = [0] * n_buckets

    for ev in player_casts:
        if ev.get("type") != "cast":
            continue
        rel_s = (float(ev["timestamp"]) - fight_start_ms) / 1000.0
        b = int(rel_s // bucket_s)
        if 0 <= b < n_buckets:
            ab_id = int(ev.get("abilityGameID", 0))
            if ab_id in gcd_ability_ids:
                gcd_counts[b] += 1
            else:
                ogcd_counts[b] += 1

    windows: list[WeaveOverrideWindow] = []
    for b in range(n_buckets):
        start_s = b * bucket_s
        end_s = min(duration_s, (b + 1) * bucket_s)
        gcds = gcd_counts[b]
        if gcds == 0:
            windows.append(
                WeaveOverrideWindow(
                    start_s=start_s,
                    end_s=end_s,
                    weave_slots_per_gcd=max_weave_slots,
                    is_downtime=True,
                )
            )
            continue
        used_per_gcd = ogcd_counts[b] / gcds
        free_slots = max(0, min(max_weave_slots, round(max_weave_slots - used_per_gcd)))
        windows.append(
            WeaveOverrideWindow(
                start_s=start_s,
                end_s=end_s,
                weave_slots_per_gcd=free_slots,
                is_downtime=False,
            )
        )
    return windows

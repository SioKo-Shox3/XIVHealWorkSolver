"""Extract normalized timelines, AA segments, DoTs, partner probabilities, and weave windows from FFLogs."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from xiv_healwork_solver.models import (
    AutoAttackSegment,
    DamageEvent,
    DamageType,
    EncounterTimeline,
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

        # Detect tank auto-attacks separately from major mechanics
        is_tank_target = target_id is not None and int(target_id) in tank_slot_by_id
        if is_tank_target and (ab_name.lower() in aa_names or (unmit < min_raw_damage and unmit >= 3_000.0)):
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
            }
        )

    hits.sort(key=lambda h: h["time_s"])

    # 2. Cluster major hits by ability within cluster_window_s
    clusters: list[list[dict[str, Any]]] = []
    for h in hits:
        placed = False
        for cl in reversed(clusters):
            if h["time_s"] - cl[-1]["time_s"] > cluster_window_s:
                break
            if cl[0]["ability_id"] == h["ability_id"]:
                cl.append(h)
                placed = True
                break
        if not placed:
            clusters.append([h])

    consumed_dot_indices: set[int] = set()
    events: list[DamageEvent] = []
    for idx, cl in enumerate(clusters, start=1):
        unique_targets = {int(x["target_id"]) for x in cl if x["target_id"] is not None}
        tank_slots_hit = {tank_slot_by_id[tid] for tid in unique_targets if tid in tank_slot_by_id}

        if len(unique_targets) >= 4:
            scope = "party"
        elif len(unique_targets) == 2 and tank_slots_hit == {"mt", "st"}:
            scope = "both_tanks"
        elif len(unique_targets) == 1 and len(tank_slots_hit) == 1:
            scope = next(iter(tank_slots_hit))  # Exact "mt" or "st"!
        elif len(unique_targets) >= 2:
            scope = "party"
        else:
            scope = "party"

        avg_raw = mean(x["raw_damage"] for x in cl)
        hit_count = max(1, round(len(cl) / max(1, len(unique_targets))))
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

        events.append(
            DamageEvent(
                id=f"e{idx}_{cl[0]['ability_id']}_{int(first_t)}",
                time_s=round(first_t, 2),
                snapshot_time_s=snap_t if snap_t < round(first_t, 2) else None,
                name=cl[0]["ability_name"],
                ability_id=cl[0]["ability_id"],
                damage_type=cl[0]["damage_type"],
                target_scope=scope,  # type: ignore[arg-type]
                raw_damage=round(avg_raw * hit_count, 1),
                hit_count=hit_count,
                hit_interval_s=hit_interval,
                dot_total_damage=round(dot_per_target, 1),
            )
        )

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
            intervals = [b["time_s"] - a["time_s"] for a, b in zip(sh, sh[1:]) if b["time_s"] > a["time_s"]]
            auto_attacks.append(
                AutoAttackSegment(
                    start_s=round(sh[0]["time_s"], 2),
                    end_s=round(sh[-1]["time_s"], 2),
                    target=sh[0]["target_slot"],  # type: ignore[arg-type]
                    damage_type=sh[0]["damage_type"],
                    raw_damage_per_hit=round(mean(x["raw_damage"] for x in sh), 1),
                    interval_s=round(max(1.5, mean(intervals)), 2) if intervals else 3.0,
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


def extract_partner_probabilities(
    timeline: EncounterTimeline,
    fights_cast_streams: list[dict[str, Any]],
    known_skill_effects: dict[int, dict[str, Any]],
    exclude_actor_names: set[str] | None = None,
    include_only_actor_names: set[str] | None = None,
    pre_window_s: float = 12.0,
    post_heal_window_s: float = 3.0,
) -> list[PartnerActionProb]:
    """Learn empirical mitigation/heal probabilities per event using phase-reached pull denominators.

    - Mitigation/shield casts are only matched BEFORE the event snapshot (never 4s after hit).
    - Pure heal casts may match up to `post_heal_window_s` after hit.
    - Denominator per event counts ONLY pulls that actually survived/reached `event.time_s`.
    """
    exclude_set = {n.lower() for n in (exclude_actor_names or set())}
    include_set = {n.lower() for n in (include_only_actor_names or set())} if include_only_actor_names else None

    observations: dict[tuple[str, int], list[float]] = defaultdict(list)
    meta_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    reached_counts: dict[str, int] = defaultdict(int)

    for fight_bundle in fights_cast_streams:
        fight_start_ms = float(fight_bundle["fight_start_ms"])
        fight_end_ms = float(fight_bundle.get("fight_end_ms", fight_start_ms + timeline.duration_s * 1000.0))
        fight_dur_s = max(0.0, (fight_end_ms - fight_start_ms) / 1000.0)
        casts = fight_bundle["casts"]
        actors_by_id: dict[int, dict[str, Any]] = fight_bundle["actors_by_id"]

        for dmg_ev in timeline.events:
            if fight_dur_s >= dmg_ev.time_s - 0.5:
                reached_counts[dmg_ev.id] += 1

        seen_in_fight: set[tuple[str, int]] = set()
        for ev in casts:
            if ev.get("type") != "cast":
                continue
            ab_id = int(ev.get("abilityGameID", 0))
            if ab_id not in known_skill_effects:
                continue
            eff = known_skill_effects[ab_id]
            src = actors_by_id.get(int(ev.get("sourceID", -1)), {})
            src_name = str(src.get("name", ""))
            if src_name.lower() in exclude_set:
                continue
            if include_set is not None and src_name.lower() not in include_set:
                continue

            cast_s = (float(ev["timestamp"]) - fight_start_ms) / 1000.0
            is_mit_or_shield = (
                float(eff.get("mitigation_all", 0.0)) > 0
                or float(eff.get("mitigation_physical", 0.0)) > 0
                or float(eff.get("mitigation_magical", 0.0)) > 0
                or float(eff.get("shield_value", 0.0)) > 0
            )
            dur_s = float(eff.get("duration_s", 15.0))
            effective_pre = min(pre_window_s, max(2.0, dur_s))

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
                    key = (dmg_ev.id, ab_id)
                    if key not in seen_in_fight:
                        seen_in_fight.add(key)
                        observations[key].append(offset)
                        meta_by_key[key] = {
                            "effect": eff,
                            "role": _actor_role_from_subtype(src.get("subType")),
                        }

    results: list[PartnerActionProb] = []
    for (event_id, _ab_id), offsets in observations.items():
        denom = max(1, reached_counts.get(event_id, len(fights_cast_streams)))
        prob = len(offsets) / denom
        if prob < 0.15:
            continue
        eff = meta_by_key[(event_id, _ab_id)]["effect"]
        role = meta_by_key[(event_id, _ab_id)]["role"]
        results.append(
            PartnerActionProb(
                event_id=event_id,
                skill_name=eff["name"],
                source_role=role,
                target_channel=eff.get("target_channel", "party"),
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

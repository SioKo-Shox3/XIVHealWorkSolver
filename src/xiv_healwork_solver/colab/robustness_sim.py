"""Monte Carlo robustness simulation for an optimized mitigation plan.

Can be imported locally OR executed directly on a remote Google Colab runtime:
    colab run src/xiv_healwork_solver/colab/robustness_sim.py -- --plan plan.json --timeline timeline.yaml --trials 10000

Simulates:
- Bernoulli partner drop-outs (partner skips mitigation/heal according to learned probability)
- Random damage variance (+/-5% FF14 damage roll)
- Reports per-event wipe/death probability, p01/p05 worst-case remaining HP, and overall pull survival rate.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def run_robustness_simulation(
    plan_dict: dict[str, Any],
    timeline_dict: dict[str, Any],
    kit_dict: dict[str, Any],
    trials: int = 5000,
    damage_variance: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Monte Carlo robustness simulation over random partner dropouts and damage rolls."""
    rng = random.Random(seed)
    skills_by_id = {s["id"]: s for s in kit_dict["skills"]}
    events = sorted(timeline_dict["events"], key=lambda e: e["time_s"])
    actions = plan_dict["actions"]
    partner_profile = timeline_dict.get("partner_profile", [])

    # Map event_id -> active self skills (accounting for duration carryover)
    actions_by_event: dict[str, list[dict[str, Any]]] = {}
    for a in actions:
        actions_by_event.setdefault(a["event_id"], []).append(a)

    partner_by_event: dict[str, list[dict[str, Any]]] = {}
    for p in partner_profile:
        partner_by_event.setdefault(p["event_id"], []).append(p)

    # Precompute self mitigation (including gear Tenacity passive) & shields per event
    gs = kit_dict.get("gear_stats") or {}
    heal_scale = float(gs.get("heal_scale_multiplier", 1.0))
    consts = gs.get("constants") or {}
    tenacity_mit = 0.0
    if kit_dict.get("role") == "tank":
        ten_delta = max(0, int(gs.get("tenacity", 420)) - int(consts.get("base_substat", 420)))
        tenacity_mit = min(
            0.30,
            math.floor(float(consts.get("tenacity_mit_coeff", 112.0)) * ten_delta / int(consts.get("level_div", 2780)))
            / 1000.0,
        )
    player_is_mt = bool(kit_dict.get("player_is_mt", True))

    self_eval: dict[str, tuple[float, float]] = {}
    for idx, ev in enumerate(events):
        rem = 1.0
        if tenacity_mit > 0.0 and (
            ev["target_scope"] == "both_tanks"
            or (ev["target_scope"] == "mt" and player_is_mt)
            or (ev["target_scope"] == "st" and not player_is_mt)
        ):
            rem *= 1.0 - tenacity_mit
        shield = 0.0
        for act in actions_by_event.get(ev["id"], []):
            sk = skills_by_id[act["skill_id"]]
            shield += float(sk.get("shield_value", 0.0)) * heal_scale
        # Check active buffs from current and prior events
        active_ids: set[str] = set()
        for prev_idx in range(idx + 1):
            prev_ev = events[prev_idx]
            dt = ev["time_s"] - prev_ev["time_s"]
            for act in actions_by_event.get(prev_ev["id"], []):
                sk = skills_by_id[act["skill_id"]]
                if prev_idx == idx or (0.0 < dt <= float(sk.get("duration_s", 0.0)) - float(sk.get("lead_time_s", 1.5))):
                    if sk["id"] not in active_ids:
                        active_ids.add(sk["id"])
                        m_all = float(sk.get("mitigation_all", 0.0))
                        m_phys = float(sk.get("mitigation_physical", 0.0)) if ev["damage_type"] == "physical" else 0.0
                        m_mag = float(sk.get("mitigation_magical", 0.0)) if ev["damage_type"] == "magical" else 0.0
                        rem *= (1.0 - m_all) * (1.0 - m_phys) * (1.0 - m_mag)
        self_eval[ev["id"]] = (rem, shield)

    event_deaths = {ev["id"]: 0 for ev in events}
    event_min_hp_samples: dict[str, list[float]] = {ev["id"]: [] for ev in events}
    pull_survivals = 0

    for _ in range(trials):
        pull_alive = True
        for ev in events:
            self_rem, self_shield = self_eval[ev["id"]]
            p_rem = 1.0
            p_shield = 0.0
            used_groups: set[str] = set()

            for pa in partner_by_event.get(ev["id"], []):
                if rng.random() <= float(pa["probability"]):
                    grp = pa.get("stack_group")
                    if grp and grp in used_groups:
                        continue
                    if grp:
                        used_groups.add(grp)
                    m_all = float(pa.get("mitigation_all", 0.0))
                    m_phys = float(pa.get("mitigation_physical", 0.0)) if ev["damage_type"] == "physical" else 0.0
                    m_mag = float(pa.get("mitigation_magical", 0.0)) if ev["damage_type"] == "magical" else 0.0
                    p_rem *= (1.0 - m_all) * (1.0 - m_phys) * (1.0 - m_mag)
                    p_shield += float(pa.get("shield_value", 0.0))

            roll = 1.0 + rng.uniform(-damage_variance, damage_variance)
            dmg = float(ev["raw_damage"]) * roll * self_rem * p_rem
            max_hp = float(kit_dict["max_hp_tank"] if ev["target_scope"] in ("mt", "st", "both_tanks") else kit_dict["max_hp_party"])
            rem_hp = max_hp + self_shield + p_shield - dmg
            event_min_hp_samples[ev["id"]].append(rem_hp)
            if rem_hp <= 0.0:
                event_deaths[ev["id"]] += 1
                pull_alive = False

        if pull_alive:
            pull_survivals += 1

    per_event_summary = []
    for ev in events:
        samples = sorted(event_min_hp_samples[ev["id"]])
        p01_idx = max(0, int(0.01 * len(samples)))
        p05_idx = max(0, int(0.05 * len(samples)))
        max_hp = float(kit_dict["max_hp_tank"] if ev["target_scope"] in ("mt", "st", "both_tanks") else kit_dict["max_hp_party"])
        per_event_summary.append(
            {
                "event_id": ev["id"],
                "time_s": ev["time_s"],
                "name": ev["name"],
                "death_rate_pct": round((event_deaths[ev["id"]] / trials) * 100.0, 3),
                "p01_remaining_hp": round(samples[p01_idx], 1),
                "p05_remaining_hp": round(samples[p05_idx], 1),
                "p01_margin_pct": round((samples[p01_idx] / max_hp) * 100.0, 2),
            }
        )

    return {
        "encounter_name": timeline_dict["encounter_name"],
        "trials": trials,
        "damage_variance": damage_variance,
        "pull_survival_rate_pct": round((pull_survivals / trials) * 100.0, 3),
        "events": per_event_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo robustness check for a mitigation plan.")
    parser.add_argument("--plan", required=True, help="Path to optimized plan JSON")
    parser.add_argument("--timeline", required=True, help="Path to encounter timeline JSON/YAML")
    parser.add_argument("--kit", required=True, help="Path to player kit JSON/YAML")
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    def _load(p: str) -> dict[str, Any]:
        text = Path(p).read_text(encoding="utf-8")
        if p.endswith(".json"):
            return json.loads(text)
        import yaml
        return yaml.safe_load(text)

    result = run_robustness_simulation(
        plan_dict=_load(args.plan),
        timeline_dict=_load(args.timeline),
        kit_dict=_load(args.kit),
        trials=args.trials,
    )
    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(out_json, encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()

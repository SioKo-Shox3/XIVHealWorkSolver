"""Unit tests for XIVHealWorkSolver."""
from pathlib import Path

from xiv_healwork_solver.colab.robustness_sim import run_robustness_simulation
from xiv_healwork_solver.exporter.sheets import export_csv, export_json, export_markdown
from xiv_healwork_solver.fflogs.extractor import (
    extract_damage_timeline,
    extract_partner_probabilities,
    extract_weave_windows,
)
from xiv_healwork_solver.models import (
    DamageEvent,
    EncounterTimeline,
    GearStats,
    LiveFightState,
    PartnerActionProb,
    PlayerKit,
    ResourcePool,
    SkillDef,
)
from xiv_healwork_solver.optimizer.solver import replan_from_state, solve_mitigation_plan

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundled_healer_and_tank_demos_solve_optimally():
    timeline = EncounterTimeline.from_yaml(REPO_ROOT / "data" / "timelines" / "example_savage_3min.yaml")
    for kit_name in ("example_barrier_healer.yaml", "example_tank.yaml"):
        kit = PlayerKit.from_yaml(REPO_ROOT / "data" / "kits" / kit_name)
        plan = solve_mitigation_plan(kit, timeline, preset="balanced")
        assert plan.status in ("OPTIMAL", "FEASIBLE")
        assert len(plan.evaluations) >= len(timeline.events)
        assert plan.replay is not None

        # Verify cooldowns are strictly respected
        uses_by_skill: dict[str, list[float]] = {}
        skills_by_id = {s.id: s for s in kit.skills}
        for act in plan.actions:
            uses_by_skill.setdefault(act.skill_id, []).append(act.event_time_s)
        for sid, times in uses_by_skill.items():
            sk = skills_by_id[sid]
            times.sort()
            if sk.max_charges == 1 and sk.cooldown_s > 0:
                for a, b in zip(times, times[1:]):
                    assert b - a >= sk.cooldown_s - 1e-5

        sim = run_robustness_simulation(plan.model_dump(), timeline.model_dump(), kit.model_dump(), trials=500)
        assert 0.0 <= sim["pull_survival_rate_pct"] <= 100.0


def test_duration_carryover_covers_two_close_hits():
    kit = PlayerKit(
        label="Carryover Test Kit",
        role="tank",
        skills=[
            SkillDef(
                id="mit15s",
                name="15s Mit",
                cooldown_s=60.0,
                duration_s=15.0,
                lead_time_s=2.0,
                mitigation_all=0.20,
                dps_cost=0.1,
            )
        ],
    )
    timeline = EncounterTimeline(
        encounter_name="Double Hit",
        duration_s=30.0,
        events=[
            DamageEvent(id="h1", time_s=10.0, name="Hit 1", raw_damage=115_000),
            DamageEvent(id="h2", time_s=18.0, name="Hit 2", raw_damage=115_000),
        ],
    )
    plan = solve_mitigation_plan(kit, timeline, preset="safe")
    assert plan.status == "OPTIMAL"
    # Only cast once at h1, yet both h1 and h2 receive 20% mitigation via carryover!
    assert len(plan.actions) == 1
    assert plan.actions[0].event_id == "h1"
    assert plan.evaluations[0].self_mitigation_pct == 20.0
    assert plan.evaluations[1].self_mitigation_pct == 20.0


def test_resource_pool_prevents_overspending():
    kit = PlayerKit(
        label="Resource Test Kit",
        role="healer",
        resources=[ResourcePool(name="stacks", initial=1, maximum=3, regen_per_sec=0.0)],
        skills=[
            SkillDef(
                id="stack_mit",
                name="Stack Mit",
                cooldown_s=5.0,
                duration_s=5.0,
                mitigation_all=0.20,
                resource_costs={"stacks": 1},
            )
        ],
    )
    timeline = EncounterTimeline(
        encounter_name="Two Hits",
        duration_s=40.0,
        events=[
            DamageEvent(id="e1", time_s=10.0, name="Hit 1", raw_damage=115_000),
            DamageEvent(id="e2", time_s=30.0, name="Hit 2", raw_damage=115_000),
        ],
    )
    plan = solve_mitigation_plan(kit, timeline, preset="safe")
    assert len(plan.actions) <= 1


def test_solo_carry_ignores_partner_mitigation():
    kit = PlayerKit(
        label="Solo Carry Test",
        role="healer",
        skills=[
            SkillDef(
                id="party_mit",
                name="Party Mit",
                cooldown_s=30.0,
                duration_s=15.0,
                mitigation_all=0.10,
                shield_value=20_000,
                dps_cost=0.5,
            )
        ],
    )
    timeline = EncounterTimeline(
        encounter_name="Partner Dropout Test",
        duration_s=30.0,
        events=[DamageEvent(id="e1", time_s=15.0, name="Big Hit", raw_damage=110_000)],
        partner_profile=[
            PartnerActionProb(
                event_id="e1",
                skill_name="Partner Big Shield",
                mitigation_all=0.20,
                shield_value=30_000,
                probability=1.0,
            )
        ],
    )
    plan_solo = solve_mitigation_plan(kit, timeline, preset="solo_carry")
    assert plan_solo.evaluations[0].partner_expected_mitigation_pct == 0.0
    assert len(plan_solo.actions) == 1


def test_fflogs_extractors_and_sheet_exporters():
    actors = {
        1: {"id": 1, "name": "Alice", "subType": "Sage"},
        2: {"id": 2, "name": "Bob", "subType": "WhiteMage"},
        3: {"id": 3, "name": "TankA", "subType": "Paladin"},
        4: {"id": 4, "name": "DpsA", "subType": "Viper"},
    }
    abilities = {9001: {"gameID": 9001, "name": "Boss Raidwide", "type": 1024}}
    dmg_events = [
        {"type": "damage", "timestamp": 115_000, "abilityGameID": 9001, "targetID": tid, "unmitigatedAmount": 110_000}
        for tid in (1, 2, 3, 4)
    ]
    tl = extract_damage_timeline(
        encounter_name="Test Boss",
        zone_name="Test Zone",
        fight_start_ms=100_000,
        fight_end_ms=200_000,
        damage_events=dmg_events,
        actors_by_id=actors,
        abilities_by_id=abilities,
    )
    assert len(tl.events) == 1
    assert tl.events[0].target_scope == "party"
    assert tl.events[0].time_s == 15.0

    # Partner extraction (static mode for Bob)
    casts = [{"type": "cast", "timestamp": 112_500, "abilityGameID": 16536, "sourceID": 2}]
    partner_probs = extract_partner_probabilities(
        timeline=tl,
        fights_cast_streams=[{"fight_start_ms": 100_000, "casts": casts, "actors_by_id": actors}],
        known_skill_effects={16536: {"name": "Temperance", "mitigation_all": 0.10}},
        include_only_actor_names={"Bob"},
    )
    assert len(partner_probs) == 1
    assert partner_probs[0].probability == 1.0
    assert partner_probs[0].timing_mean_offset_s == -2.5

    # Weave window extraction
    windows = extract_weave_windows(
        fight_start_ms=100_000,
        fight_end_ms=120_000,
        player_casts=[
            {"type": "cast", "timestamp": 101_000, "abilityGameID": 100},
            {"type": "cast", "timestamp": 103_500, "abilityGameID": 100},
            {"type": "cast", "timestamp": 104_500, "abilityGameID": 200},
        ],
        gcd_ability_ids={100},
    )
    assert len(windows) >= 2

    # Exporters
    kit = PlayerKit.from_yaml(REPO_ROOT / "data" / "kits" / "example_barrier_healer.yaml")
    plan = solve_mitigation_plan(kit, tl)
    assert "Test Boss" in export_markdown(plan)
    assert "cast_at_s" in export_csv(plan)
    assert '"encounter_name"' in export_json(plan)


def test_tenacity_passive_mitigation_reduces_tankbuster_damage():
    timeline = EncounterTimeline(
        encounter_name="TB Check",
        duration_s=30.0,
        events=[DamageEvent(id="tb1", time_s=15.0, name="Tankbuster", target_scope="mt", raw_damage=150_000)],
    )
    kit_low_ten = PlayerKit(label="Low TEN", role="tank", gear_stats=GearStats(tenacity=420), skills=[])
    kit_high_ten = PlayerKit(label="High TEN", role="tank", gear_stats=GearStats(tenacity=2400), skills=[])

    p_low = solve_mitigation_plan(kit_low_ten, timeline)
    p_high = solve_mitigation_plan(kit_high_ten, timeline)

    assert p_low.evaluations[0].passive_gear_mitigation_pct == 0.0
    assert p_high.evaluations[0].passive_gear_mitigation_pct >= 7.0
    assert p_high.evaluations[0].effective_damage < p_low.evaluations[0].effective_damage


def test_healer_mp_economy_and_raise_reserve_prevents_gcd_spam():
    # Fast GCD (2.32s) with 400 MP filler + low PIE drains MP quickly; solver cannot spend below 0 MP
    # and respects raise reserve when a free oGCD alternative exists
    kit = PlayerKit(
        label="Tight MP Healer",
        role="healer",
        gcd_s=2.32,
        gear_stats=GearStats(
            piety=440,
            filler_gcd_mp_cost=400.0,
            lucid_dreaming_mp_per_sec=64.2,
            raise_mp_cost=2400.0,
            planned_raise_reserve=1,
        ),
        resources=[ResourcePool(name="mp", initial=3200, maximum=10000)],
        skills=[
            SkillDef(
                id="expensive_gcd",
                name="Expensive GCD Shield",
                cooldown_s=2.5,
                is_gcd=True,
                weave_cost=0,
                shield_value=20_000,
                resource_costs={"mp": 1000},
                dps_cost=0.1,  # Artificially cheap DPS cost to prove MP reserve steers choice
            ),
            SkillDef(
                id="free_ogcd",
                name="Free oGCD Shield",
                cooldown_s=60.0,
                is_gcd=False,
                weave_cost=1,
                shield_value=20_000,
                dps_cost=0.2,
            ),
        ],
    )
    timeline = EncounterTimeline(
        encounter_name="MP Check",
        duration_s=20.0,
        events=[DamageEvent(id="e1", time_s=10.0, name="Raidwide", raw_damage=112_000)],
    )
    plan = solve_mitigation_plan(kit, timeline, preset="safe")
    used_ids = {a.skill_id for a in plan.actions}
    assert "free_ogcd" in used_ids
    assert "expensive_gcd" not in used_ids
    assert plan.evaluations[0].resource_after["mp"] >= 2400.0


def test_mid_fight_replan_respects_cooldowns_and_runs_fast():
    kit = PlayerKit.from_yaml(REPO_ROOT / "data" / "kits" / "example_barrier_healer.yaml")
    timeline = EncounterTimeline.from_yaml(REPO_ROOT / "data" / "timelines" / "example_savage_3min.yaml")

    # Simulate t=75s right after a Raise: low MP (3000), mit_10_hot on 25s CD, party at 70% HP
    state = LiveFightState(
        current_time_s=75.0,
        cooldowns_remaining={"mit_10_hot": 25.0},
        current_resources={"mp": 3000.0, "gauge": 1.0},
        current_party_hp_ratio=0.70,
        horizon_s=100.0,
    )
    plan = replan_from_state(kit, timeline, state, preset="safe")
    assert plan.status in ("OPTIMAL", "FEASIBLE")
    assert plan.solve_time_ms < 500.0
    # Event e4 at 86s cannot use mit_10_hot because it is on CD until 75 + 25 = 100s
    e4_skills = [a.skill_id for a in plan.actions if a.event_id == "e4_multi_hit"]
    assert "mit_10_hot" not in e4_skills


def test_rl_env_server_tick_movement_and_mt_st_targeting():
    from xiv_healwork_solver.rl.env import FF14HealMitEnv

    kit = PlayerKit.from_yaml(REPO_ROOT / "data" / "kits" / "example_barrier_healer.yaml")
    timeline = EncounterTimeline.from_yaml(REPO_ROOT / "data" / "timelines" / "example_savage_3min.yaml")
    env = FF14HealMitEnv(kit, timeline, seed=7)

    # single_ally skill (tank_single_mit) should expand into separate MT and ST target actions
    labels = [a.label for a in env.actions]
    assert any("-> MT" in lbl for lbl in labels)
    assert any("-> ST" in lbl for lbl in labels)

    obs = env.reset(fixed_tick_phase=1.2)
    assert obs["state"].shape == (30,)
    assert obs["skills"].shape == (env.n_actions, 20)
    assert obs["future"].shape == (8, 14)
    assert obs["mask"][0]

    # Step through full pull with legal random actions to ensure zero crashes
    done = False
    steps = 0
    while not done:
        legal = [i for i, ok in enumerate(obs["mask"]) if ok]
        obs, _r, done, info = env.step(legal[0])
        steps += 1
        assert steps <= 1500
    assert "min_mt_hp_ratio" in info["stats"]
    assert "min_st_hp_ratio" in info["stats"]



"""P0 Acceptance Tests: exact parity between CP-SAT replay, CombatEngine, RL env, and FFLogs curator."""
from __future__ import annotations

from xiv_healwork_solver.fflogs.curator import curate_fight_candidates
from xiv_healwork_solver.fflogs.extractor import extract_damage_timeline, extract_partner_probabilities
from xiv_healwork_solver.models import (
    AutoAttackSegment,
    DamageEvent,
    DowntimeWindow,
    EncounterTimeline,
    GearStats,
    LogFilterConfig,
    PlayerKit,
    ResourcePool,
    SkillDef,
)
from xiv_healwork_solver.optimizer.solver import solve_mitigation_plan
from xiv_healwork_solver.rl.env import FF14HealMitEnv
from xiv_healwork_solver.sim.engine import CombatEngine


def test_p0_1_back_to_back_raidwides_hp_carryover_parity():
    """Acceptance Case 1: 2s and 4s 60k hits on 100k HP MUST die at 4s in BOTH CP-SAT and RL env."""
    kit = PlayerKit(label="Empty", role="healer", max_hp_party=100_000, max_hp_tank=150_000, skills=[])
    tl = EncounterTimeline(
        encounter_name="BackToBack",
        duration_s=10.0,
        events=[
            DamageEvent(id="h1", time_s=2.0, name="Hit 1", raw_damage=60_000),
            DamageEvent(id="h2", time_s=4.0, name="Hit 2", raw_damage=60_000),
        ],
    )
    plan = solve_mitigation_plan(kit, tl, verify_with_replay=True)
    # CP-SAT must see the carried-over HP deficit (-20,000), NOT 40,000 -> 40,000!
    assert plan.evaluations[0].net_hp_after_hit == 40_000.0
    assert plan.evaluations[1].net_hp_after_hit == -20_000.0
    assert plan.replay is not None
    assert plan.replay.survived is False
    assert plan.replay.first_death_time_s == 4.0
    assert plan.replay.first_death_channel == "party"
    assert plan.replay.deficit_hp == 20_000.0

    # RL env must match exact death time and HP
    env = FF14HealMitEnv(kit, tl, damage_variance=0.0, raise_prob_per_pull=0.0)
    env.reset(deterministic=True)
    done = False
    while not done:
        _, _, done, info = env.step(0, deterministic=True)
    assert info["stats"]["survived"] == 0.0
    assert env.engine.first_death_time_s == 4.0
    assert env.hp["party"] == -20_000.0


def test_p0_2_exact_continuous_gcd_counts():
    """Exact real-time engine distinguishes GCD 2.50 (24), 2.40 (25), and 2.31 (26) over 60s."""
    tl = EncounterTimeline(encounter_name="Dummy 60s", duration_s=60.0, events=[])
    expected = {2.50: 24, 2.40: 25, 2.31: 26}
    for gcd_s, exp_count in expected.items():
        kit = PlayerKit(label=f"GCD {gcd_s}", role="healer", gcd_s=gcd_s, skills=[])
        env = FF14HealMitEnv(kit, tl, damage_variance=0.0, raise_prob_per_pull=0.0)
        obs = env.reset(deterministic=True)
        done = False
        while not done:
            # Always press Action 1 (FILLER_ATTACK_GCD) when legal, else WAIT (0)
            act = 1 if obs["mask"][1] else 0
            obs, _, done, info = env.step(act, deterministic=True)
        assert int(info["stats"]["filler_gcds"]) == exp_count, f"GCD {gcd_s}: got {info['stats']['filler_gcds']}"


def test_p0_3_downtime_preshield_free_and_boss_debuff_blocked():
    """Acceptance Case 2: During untargetable downtime, Reprisal is blocked and pre-shield is 0 DPS cost."""
    kit = PlayerKit(
        label="Downtime Test Kit",
        role="healer",
        gcd_s=2.50,
        max_hp_party=100_000,
        on_attack_heal_target="mt",
        on_attack_heal_value=5_000,
        gear_stats=GearStats(piety=440, filler_gcd_mp_cost=400.0),
        resources=[ResourcePool(name="mp", initial=10_000, maximum=10_000)],
        skills=[
            SkillDef(
                id="reprisal_like",
                name="Boss Debuff Mit",
                cooldown_s=60.0,
                duration_s=15.0,
                target_scope="boss_debuff",
                requires_targetable_boss=True,
                mitigation_all=0.10,
                lead_time_s=2.0,
                dps_cost=0.01,
            ),
            SkillDef(
                id="gcd_barrier",
                name="GCD Party Barrier",
                cooldown_s=2.5,
                duration_s=30.0,
                is_gcd=True,
                weave_cost=0,
                target_scope="party",
                shield_value=25_000,
                shield_stack_group="gcd_barrier",
                resource_costs={"mp": 900},
                lead_time_s=3.0,
                dps_cost=1.0,
            ),
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Transition Test",
        duration_s=30.0,
        downtime_windows=[DowntimeWindow(start_s=5.0, end_s=20.0, boss_targetable=False, party_can_act=True)],
        events=[
            DamageEvent(id="post_dt_hit", time_s=20.0, name="Transition Raidwide", raw_damage=115_000),
        ],
    )
    plan = solve_mitigation_plan(kit, tl, preset="balanced", verify_with_replay=True)
    used_ids = {a.skill_id for a in plan.actions}
    assert "reprisal_like" not in used_ids  # Blocked because cast_t=18s is inside untargetable downtime!
    assert "gcd_barrier" in used_ids
    assert plan.replay is not None
    assert plan.replay.survived is True
    assert plan.replay.downtime_free_gcds >= 1  # Pre-shield cast at 17s finished before 20s -> 0 DPS loss!


def test_p0_4_dot_mitigation_snapshot_before_vs_after():
    """Acceptance Case 3: Mitigation BEFORE DoT snapshot weakens all ticks; AFTER does NOT."""
    kit = PlayerKit(
        label="Mit Kit",
        role="healer",
        max_hp_party=100_000,
        skills=[
            SkillDef(
                id="mit20",
                name="20% Mit",
                cooldown_s=60.0,
                duration_s=15.0,
                mitigation_all=0.20,
                lead_time_s=1.0,
            )
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Bleed Snapshot",
        duration_s=25.0,
        events=[
            DamageEvent(
                id="bleed",
                time_s=5.0,
                snapshot_time_s=4.5,
                name="Raidwide Bleed",
                raw_damage=20_000,
                dot_total_damage=50_000,
                dot_duration_s=15.0,
                dot_snapshot_mitigation=True,
            )
        ],
    )

    # Case A: Mit pressed BEFORE snapshot (at t=3.0s)
    eng_before = CombatEngine(kit, tl, damage_variance=0.0)
    eng_before.reset(deterministic=True, fixed_tick_phase=1.0)
    eng_before.advance_to(3.0)
    eng_before.execute_action(2)  # mit20
    eng_before.advance_to(25.0)

    # Case B: Mit pressed AFTER snapshot/impact (at t=5.5s)
    eng_after = CombatEngine(kit, tl, damage_variance=0.0)
    eng_after.reset(deterministic=True, fixed_tick_phase=1.0)
    eng_after.advance_to(5.5)
    eng_after.execute_action(2)  # mit20
    eng_after.advance_to(25.0)

    # Total damage before mit: 20k + 50k = 70k -> HP=30k; with 20% mit on hit+DoT -> 56k -> HP=44k!
    assert eng_before.hp["party"] == 44_000.0
    assert eng_after.hp["party"] == 30_000.0  # Late mit did NOT retroactively weaken DoT!


def test_p0_5_tank_swap_aa_and_st_single_care():
    """Acceptance Case 4: AA swaps from MT to ST; ST single-target care protects ST specifically."""
    kit = PlayerKit(
        label="Single Care Healer",
        role="healer",
        max_hp_party=100_000,
        max_hp_tank=100_000,
        skills=[
            SkillDef(
                id="single_shield",
                name="ST Shield",
                cooldown_s=30.0,
                duration_s=15.0,
                target_scope="single_ally",
                shield_value=30_000,
            )
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Tank Swap AA",
        duration_s=16.0,
        auto_attacks=[
            AutoAttackSegment(start_s=2.0, end_s=8.0, target="mt", raw_damage_per_hit=20_000, interval_s=3.0),
            AutoAttackSegment(start_s=10.0, end_s=16.0, target="st", raw_damage_per_hit=20_000, interval_s=3.0),
        ],
        events=[],
    )
    eng = CombatEngine(kit, tl, damage_variance=0.0)
    eng.reset(deterministic=True)
    # Find action targeting ST
    st_idx = next(a.index for a in eng.actions if a.target_channel == "st")
    eng.advance_to(9.0)
    eng.execute_action(st_idx)  # Shield ST right before swap!
    eng.advance_to(16.0)

    # MT took 3 AAs (2,5,8s) = 60k -> HP=40k. ST took 3 AAs (10,13,16s) = 60k minus 30k shield -> HP=70k!
    assert eng.hp["mt"] == 40_000.0
    assert eng.hp["st"] == 70_000.0


def test_special_mechanics_set_hp_1_full_heal_doom_and_esuna():
    """Verify HP=1 bypasses shields, Full-Heal Doom kills unless topped, and Esuna clears cleansable doom."""
    kit = PlayerKit(
        label="Special Mech Healer",
        role="healer",
        max_hp_party=100_000,
        skills=[
            SkillDef(
                id="full_heal",
                name="Full Heal",
                cooldown_s=30.0,
                heal_instant=200_000,
                target_scope="party",
            ),
            SkillDef(
                id="esuna",
                name="エスナ",
                cooldown_s=2.5,
                is_gcd=True,
                weave_cost=0,
                target_scope="party",
                cleanses_debuff=True,
            ),
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Doom & Esuna",
        duration_s=25.0,
        events=[
            DamageEvent(
                id="wh",
                time_s=5.0,
                name="White Hole (HP=1 + Full-Heal Doom + Esuna)",
                mechanic_effect="set_hp_to_1",
                doom_full_heal_duration_s=8.0,
                esuna_required_count=1,
                esuna_deadline_s=8.0,
            )
        ],
    )

    # Without heal or esuna -> must die at 5 + 8 = 13s!
    eng_fail = CombatEngine(kit, tl, damage_variance=0.0)
    eng_fail.reset(deterministic=True)
    eng_fail.advance_to(6.0)
    assert eng_fail.hp["party"] == 1.0
    eng_fail.advance_to(20.0)
    assert eng_fail.died is True
    assert eng_fail.first_death_time_s == 13.0

    # With BOTH full_heal and esuna before 13s -> survives!
    eng_ok = CombatEngine(kit, tl, damage_variance=0.0)
    eng_ok.reset(deterministic=True)
    heal_idx = next(a.index for a in eng_ok.actions if a.skill and a.skill.id == "full_heal")
    esuna_idx = next(a.index for a in eng_ok.actions if a.skill and a.skill.id == "esuna")
    eng_ok.advance_to(6.0)
    eng_ok.execute_action(heal_idx)
    eng_ok.advance_to(7.0)
    eng_ok.execute_action(esuna_idx)
    eng_ok.advance_to(25.0)
    assert eng_ok.died is False
    assert eng_ok.hp["party"] == 100_000.0


def test_fflogs_curator_rejects_chad_healing_and_speedkills():
    """Verify Chad-healing (18%/82% split), high deaths, and speedkills are filtered out."""
    candidates = [
        {
            "report_code": "GOOD1",
            "fight_id": 1,
            "duration_s": 540.0,
            "deaths": 0,
            "party_mit_casts": 24,
            "healers": [
                {"name": "H1", "rdps": 14500, "effective_healing": 1_200_000, "overheal_pct": 35.0},
                {"name": "H2", "rdps": 14200, "effective_healing": 1_150_000, "overheal_pct": 38.0},
            ],
        },
        {
            "report_code": "CHAD_PARSE",
            "fight_id": 2,
            "duration_s": 535.0,
            "deaths": 0,
            "party_mit_casts": 18,
            "healers": [
                {"name": "GreedyH1", "rdps": 16500, "effective_healing": 400_000, "overheal_pct": 20.0},
                {"name": "MartyrH2", "rdps": 10000, "effective_healing": 1_900_000, "overheal_pct": 45.0},
            ],
        },
        {
            "report_code": "SPEEDKILL",
            "fight_id": 3,
            "duration_s": 430.0,
            "deaths": 0,
            "party_mit_casts": 20,
            "healers": [
                {"name": "H1", "rdps": 15000, "effective_healing": 1_000_000, "overheal_pct": 30.0},
                {"name": "H2", "rdps": 14800, "effective_healing": 1_050_000, "overheal_pct": 30.0},
            ],
        },
        {
            "report_code": "ZOMBIE",
            "fight_id": 4,
            "duration_s": 545.0,
            "deaths": 3,
            "party_mit_casts": 20,
            "healers": [
                {"name": "H1", "rdps": 12000, "effective_healing": 1_200_000, "overheal_pct": 40.0},
                {"name": "H2", "rdps": 12500, "effective_healing": 1_200_000, "overheal_pct": 40.0},
            ],
        },
    ]
    accepted, rejected = curate_fight_candidates(candidates, LogFilterConfig())
    assert [r.report_code for r in accepted] == ["GOOD1"]
    rej_map = {r.report_code: r.reasons for r in rejected}
    assert any("chad_healing" in msg for msg in rej_map["CHAD_PARSE"])
    assert any("speedkill_outlier" in msg for msg in rej_map["SPEEDKILL"])
    assert any("deaths=3" in msg for msg in rej_map["ZOMBIE"])


def test_fflogs_extractor_dedup_snapshot_and_phase_reached_denominator():
    """Verify calculateddamage+damage deduplication, ST identification, and phase-reached denominator."""
    actors = {
        1: {"id": 1, "name": "MT_Tank", "subType": "Paladin"},
        2: {"id": 2, "name": "ST_Tank", "subType": "Warrior"},
        3: {"id": 3, "name": "CoHealer", "subType": "WhiteMage"},
    }
    abilities = {
        7: {"gameID": 7, "name": "Attack", "type": 128},
        901: {"gameID": 901, "name": "ST Tankbuster", "type": 128},
    }
    dmg = [
        # MT takes first AA -> identified as MT
        {"type": "damage", "timestamp": 103_000, "abilityGameID": 7, "targetID": 1, "unmitigatedAmount": 20_000},
        {"type": "damage", "timestamp": 106_000, "abilityGameID": 7, "targetID": 1, "unmitigatedAmount": 20_000},
        # ST takes second AA -> identified as ST
        {"type": "damage", "timestamp": 115_000, "abilityGameID": 7, "targetID": 2, "unmitigatedAmount": 20_000},
        {"type": "damage", "timestamp": 118_000, "abilityGameID": 7, "targetID": 2, "unmitigatedAmount": 20_000},
        # ST Tankbuster has BOTH calculateddamage (snapshot at 19.0s) and damage (impact at 20.0s) with same packetID!
        {"type": "calculateddamage", "packetID": 42, "timestamp": 119_000, "abilityGameID": 901, "targetID": 2, "unmitigatedAmount": 180_000},
        {"type": "damage", "packetID": 42, "timestamp": 120_000, "abilityGameID": 901, "targetID": 2, "unmitigatedAmount": 180_000},
    ]
    tl = extract_damage_timeline("Test", "Zone", 100_000, 150_000, dmg, actors, abilities)
    assert len(tl.events) == 1
    assert tl.events[0].target_scope == "st"
    assert tl.events[0].snapshot_time_s == 19.0
    assert tl.events[0].time_s == 20.0
    assert len(tl.auto_attacks) == 2
    assert tl.auto_attacks[0].target == "mt"
    assert tl.auto_attacks[1].target == "st"

    # Pull 1 reached 50s and cast mit at 17s; Pull 2 wiped at 10s (never reached 20s) -> prob must be 1/1 = 1.0!
    probs = extract_partner_probabilities(
        timeline=tl,
        fights_cast_streams=[
            {
                "fight_start_ms": 100_000,
                "fight_end_ms": 150_000,
                "casts": [{"type": "cast", "timestamp": 117_000, "abilityGameID": 16536, "sourceID": 3}],
                "actors_by_id": actors,
            },
            {
                "fight_start_ms": 200_000,
                "fight_end_ms": 210_000,  # Wiped at 10s!
                "casts": [],
                "actors_by_id": actors,
            },
        ],
        known_skill_effects={16536: {"name": "Temperance", "mitigation_all": 0.10, "duration_s": 20.0}},
    )
    assert len(probs) == 1
    assert probs[0].sample_count == 1
    assert probs[0].probability == 1.0


def test_codex_regression_raidwide_to_mt_buster_carryover():
    """60k raidwide -> 60k MT buster on 100k HP must show -20k MT HP in BOTH CP-SAT and replay."""
    kit = PlayerKit(label="Empty", role="tank", max_hp_party=100_000, max_hp_tank=100_000, skills=[])
    tl = EncounterTimeline(
        encounter_name="RW then TB",
        duration_s=10.0,
        events=[
            DamageEvent(id="rw", time_s=2.0, name="Raidwide", target_scope="party", raw_damage=60_000),
            DamageEvent(id="tb", time_s=4.0, name="MT Buster", target_scope="mt", raw_damage=60_000),
        ],
    )
    plan = solve_mitigation_plan(kit, tl, verify_with_replay=True)
    assert plan.evaluations[1].net_hp_after_hit == -20_000.0
    assert plan.replay is not None and plan.replay.survived is False


def test_codex_regression_aa_checkpoints_planned_automatically():
    """Two 60k AAs on 100k tank must trigger CP-SAT to assign a 40k shield and survive replay."""
    kit = PlayerKit(
        label="AA Shield Tank",
        role="tank",
        max_hp_tank=100_000,
        skills=[
            SkillDef(
                id="tank_sh",
                name="40k Tank Shield",
                cooldown_s=30.0,
                duration_s=15.0,
                target_scope="self",
                shield_value=40_000,
                lead_time_s=1.0,
            )
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Heavy AA",
        duration_s=10.0,
        auto_attacks=[AutoAttackSegment(start_s=2.0, end_s=5.0, target="mt", raw_damage_per_hit=60_000, interval_s=3.0)],
        events=[],
    )
    plan = solve_mitigation_plan(kit, tl, verify_with_replay=True)
    assert any(a.skill_id == "tank_sh" for a in plan.actions)
    assert plan.replay is not None and plan.replay.survived is True


def test_codex_regression_shield_stack_group_and_cast_deadline():
    """Two shields in same stack_group cannot double-count; 2s cast on 1s event is illegal."""
    kit = PlayerKit(
        label="Shield Group Test",
        role="healer",
        max_hp_party=100_000,
        skills=[
            SkillDef(id="sh1", name="Barrier A", cooldown_s=2.5, is_gcd=True, weave_cost=0, shield_value=30_000, shield_stack_group="gcd_barrier", lead_time_s=1.0),
            SkillDef(id="sh2", name="Barrier B", cooldown_s=2.5, is_gcd=True, weave_cost=0, shield_value=30_000, shield_stack_group="gcd_barrier", lead_time_s=1.0),
            SkillDef(id="slow_sh", name="Slow 2s Cast Shield", cooldown_s=2.5, is_gcd=True, weave_cost=0, cast_time_s=2.0, shield_value=50_000, lead_time_s=1.0),
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Stack & Cast Deadline",
        duration_s=15.0,
        events=[
            DamageEvent(id="e_early", time_s=1.0, name="1s Hit", raw_damage=120_000),
            DamageEvent(id="e_big", time_s=10.0, name="140k Hit", raw_damage=140_000),
        ],
    )
    plan = solve_mitigation_plan(kit, tl, verify_with_replay=True)
    early_ids = {a.skill_id for a in plan.actions if a.event_id == "e_early"}
    big_ids = {a.skill_id for a in plan.actions if a.event_id == "e_big"}
    assert "slow_sh" not in early_ids  # 2.0s cast cannot complete before 1.0s!
    assert not ({"sh1", "sh2"} <= big_ids)  # Same shield_stack_group cannot both be assigned!


def test_codex_regression_dot_exact_tick_count_on_server_tick_boundary():
    """A 50k 15s bleed applied exactly on a server tick must tick 5 times (50k), NOT 6 times (60k)."""
    kit = PlayerKit(label="Empty", role="healer", max_hp_party=100_000, skills=[])
    tl = EncounterTimeline(
        encounter_name="Boundary Bleed",
        duration_s=25.0,
        events=[DamageEvent(id="b1", time_s=6.0, name="Bleed", raw_damage=0.0, dot_total_damage=50_000, dot_duration_s=15.0)],
    )
    eng = CombatEngine(kit, tl, damage_variance=0.0)
    eng.reset(deterministic=True, fixed_tick_phase=0.0)  # Server ticks at 3, 6, 9, 12, 15, 18, 21!
    eng.advance_to(25.0)
    assert eng.hp["party"] == 50_000.0


def test_codex_regression_env_preserves_deterministic_and_extractor_edge_cases():
    """Verify env.step() preserves reset(deterministic=True), packet calc fallback, 1-hit AA, and curator 0-heal."""
    kit = PlayerKit(label="Empty", role="healer", max_hp_party=100_000, skills=[])
    tl = EncounterTimeline(encounter_name="Det", duration_s=10.0, events=[DamageEvent(id="h1", time_s=3.0, name="H", raw_damage=50_000)])
    env = FF14HealMitEnv(kit, tl, damage_variance=0.05, raise_prob_per_pull=0.0)
    env.reset(deterministic=True)
    done = False
    while not done:
        _, _, done, _ = env.step(0)  # Omit deterministic arg -> must stay deterministic!
    assert env.hp["party"] == 50_000.0

    # Extractor: calculateddamage 120k + 0-amount impact + opening raidwide with ST listed first + 1-hit AA
    actors = {
        10: {"id": 10, "name": "ST", "subType": "Warrior"},
        20: {"id": 20, "name": "MT", "subType": "Paladin"},
    }
    abilities = {7: {"gameID": 7, "name": "Attack", "type": 128}, 99: {"gameID": 99, "name": "Big", "type": 1024}}
    dmg = [
        # Opening raidwide lists ST before MT
        {"type": "calculateddamage", "packetID": 1, "timestamp": 102_000, "abilityGameID": 99, "targetID": 10, "unmitigatedAmount": 120_000},
        {"type": "damage", "packetID": 1, "timestamp": 102_500, "abilityGameID": 99, "targetID": 10, "amount": 0, "unmitigatedAmount": 0},
        {"type": "damage", "packetID": 2, "timestamp": 102_500, "abilityGameID": 99, "targetID": 20, "unmitigatedAmount": 120_000},
        # Single MT AA then single ST AA
        {"type": "damage", "timestamp": 105_000, "abilityGameID": 7, "targetID": 20, "unmitigatedAmount": 20_000},
        {"type": "damage", "timestamp": 110_000, "abilityGameID": 7, "targetID": 10, "unmitigatedAmount": 20_000},
    ]
    extracted = extract_damage_timeline("Edge", "Z", 100_000, 120_000, dmg, actors, abilities)
    assert len(extracted.events) == 1
    assert extracted.events[0].raw_damage == 120_000.0
    assert len(extracted.auto_attacks) == 2
    assert extracted.auto_attacks[0].target == "mt"
    assert extracted.auto_attacks[1].target == "st"

    # Curator: 0 effective healing with 100% overheal MUST be rejected
    acc, rej = curate_fight_candidates(
        [
            {
                "report_code": "ZERO_HEAL",
                "fight_id": 1,
                "duration_s": 500.0,
                "deaths": 0,
                "party_mit_casts": 20,
                "healers": [
                    {"name": "H1", "rdps": 14000, "effective_healing": 0, "overheal_pct": 100.0},
                    {"name": "H2", "rdps": 14000, "effective_healing": 0, "overheal_pct": 100.0},
                ],
            }
        ],
        LogFilterConfig(),
    )
    assert len(acc) == 0 and len(rej) == 1


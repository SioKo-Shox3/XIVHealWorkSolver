"""P0 Acceptance Tests: exact parity between CP-SAT replay, CombatEngine, RL env, and FFLogs curator."""
from __future__ import annotations

import pytest

from xiv_healwork_solver.fflogs.curator import curate_fight_candidates
from xiv_healwork_solver.fflogs.extractor import (
    extract_damage_timeline,
    extract_partner_healing_pulses,
    extract_partner_probabilities,
    extract_player_expert_actions,
)
from xiv_healwork_solver.models import (
    AutoAttackSegment,
    DamageEvent,
    DowntimeWindow,
    EncounterTimeline,
    GearSamplingRange,
    GearStats,
    HpThresholdTrigger,
    LogFilterConfig,
    PartnerActionProb,
    PlayerKit,
    PrePullAction,
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


def test_prepull_critlo_deploy_and_gear_conditioned_sampling():
    """Verify pre-pull shield carries into t=0 with 0 DPS loss & CD ticks down, and gear sampling varies obs."""
    kit = PlayerKit(
        label="Prepull SCH",
        role="healer",
        max_hp_party=100_000,
        prepull_actions=[
            PrePullAction(
                skill_id="deploy",
                offset_s=-3.0,
                target_channel="party",
                override_shield_value=60_000,
                consumes_cooldown=True,
            )
        ],
        skills=[
            SkillDef(
                id="deploy",
                name="展開戦術",
                cooldown_s=90.0,
                duration_s=30.0,
                is_gcd=True,
                weave_cost=1,
                shield_value=30_000,
                shield_stack_group="gcd_barrier",
                lead_time_s=3.0,
            )
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Opener Raidwide",
        duration_s=30.0,
        events=[DamageEvent(id="op", time_s=10.0, name="Opener", raw_damage=140_000)],
    )
    plan = solve_mitigation_plan(kit, tl, verify_with_replay=True)
    # Pre-pull 60k shield covers the 140k opener (100k + 60k - 140k = 20k) without re-casting Deploy (on CD until 87s!)
    assert len(plan.actions) == 0
    assert plan.replay is not None and plan.replay.survived is True
    assert plan.replay.event_snapshots[0].hp_after["party"] == 20_000.0

    # Gear-conditioned sampling produces distinct GCDs / Piety values across episodes
    env = FF14HealMitEnv(kit, tl, gear_sampling=GearSamplingRange(), seed=99)
    seen_gcds = set()
    for _ in range(5):
        obs = env.reset(deterministic=False)
        seen_gcds.add(env.kit.gcd_s)
        assert obs["state"].shape == (36,)
    assert len(seen_gcds) >= 3

    # Overheal efficiency scoring: lower overheal ranks strictly higher
    acc, _ = curate_fight_candidates(
        [
            {
                "report_code": "CLEAN_OH",
                "fight_id": 1,
                "duration_s": 500.0,
                "deaths": 0,
                "party_mit_casts": 20,
                "healers": [
                    {"name": "H1", "rdps": 14000, "effective_healing": 1_000_000, "overheal_pct": 20.0},
                    {"name": "H2", "rdps": 14000, "effective_healing": 1_000_000, "overheal_pct": 25.0},
                ],
            },
            {
                "report_code": "WASTEFUL_OH",
                "fight_id": 2,
                "duration_s": 500.0,
                "deaths": 0,
                "party_mit_casts": 20,
                "healers": [
                    {"name": "H1", "rdps": 14000, "effective_healing": 1_000_000, "overheal_pct": 60.0},
                    {"name": "H2", "rdps": 14000, "effective_healing": 1_000_000, "overheal_pct": 60.0},
                ],
            },
        ],
        LogFilterConfig(),
    )
    assert [r.report_code for r in acc] == ["CLEAN_OH", "WASTEFUL_OH"]
    assert acc[0].quality_score > acc[1].quality_score


def test_pet_auto_heal_targets_lowest_hp_channel_on_3s_tick():
    """Fairy/pet passive 3s heal heals the lowest HP-ratio channel (e.g. MT taking AAs)."""
    kit = PlayerKit(
        label="Fairy SCH",
        role="healer",
        max_hp_party=180_000,
        max_hp_tank=280_000,
        pet_auto_heal_per_3s_tick=12_000.0,
        skills=[],
    )
    tl = EncounterTimeline(
        encounter_name="MT AA Pet Heal",
        duration_s=12.0,
        events=[],
        auto_attacks=[
            AutoAttackSegment(start_s=2.0, end_s=2.0, interval_s=3.0, target="mt", raw_damage_per_hit=50_000.0)
        ],
    )
    eng = CombatEngine(kit=kit, timeline=tl, damage_variance=0.0, raise_prob_per_pull=0.0)
    eng.reset(deterministic=True, fixed_tick_phase=0.5)
    eng.advance_to(11.0)
    # After 2.0s AA (-50k -> 230k), 3s ticks at 3.5s, 6.5s, 9.5s heal MT by 3 * 12k = +36k -> 266k
    assert eng.hp["mt"] == pytest.approx(266_000.0, abs=10.0)
    assert eng.hp["party"] == pytest.approx(180_000.0, abs=1.0)


def test_fflogs_calibration_fixes_limitbreak_combined_dots_disjoint_waves_and_absorbed_shields():
    """Regression test for real-log calibration fixes:
    1) Disjoint-target split waves & simultaneous disjoint abilities do not double-hit `party`.
    2) Synthetic `Combined DoTs` (ID 500000) and 1-person avoidable clips are ignored in 8-man logs.
    3) `LimitBreak` pseudo-actors are excluded from `non_tank_count`.
    4) `extract_partner_healing_pulses` extracts both empirical heals (split per multi-hit) and absorbed shields.
    5) `extract_player_expert_actions` maps real player casts cleanly to kit skill IDs.
    """
    actors = {
        1: {"id": 1, "name": "MT", "type": "Player", "subType": "DarkKnight"},
        2: {"id": 2, "name": "ST", "type": "Player", "subType": "Paladin"},
        3: {"id": 3, "name": "H1", "type": "Player", "subType": "WhiteMage"},
        4: {"id": 4, "name": "H2", "type": "Player", "subType": "Scholar"},
        5: {"id": 5, "name": "D1", "type": "Player", "subType": "Dragoon"},
        6: {"id": 6, "name": "D2", "type": "Player", "subType": "Samurai"},
        7: {"id": 7, "name": "D3", "type": "Player", "subType": "Bard"},
        8: {"id": 8, "name": "D4", "type": "Player", "subType": "BlackMage"},
        99: {"id": 99, "name": "Limit Break", "type": "Player", "subType": "LimitBreak"},
    }
    abilities = {
        100: {"gameID": 100, "name": "Blast Beat", "type": 1024},
        101: {"gameID": 101, "name": "Ultrasonic Spread", "type": 1024},
        102: {"gameID": 102, "name": "Blood Lash", "type": 1024},
        103: {"gameID": 103, "name": "Half Moon", "type": 1024},
        500000: {"gameID": 500000, "name": "Combined DoTs", "type": 1024},
        9001: {"gameID": 9001, "name": "Real Bleed", "type": 1024},
    }
    dmg_events = [
        # Disjoint 4+4 wave Blast Beat (2.5s apart, disjoint targets -> 1 single hit, NOT 2x damage!)
        *[{"type": "damage", "timestamp": 110_000, "abilityGameID": 100, "targetID": tid, "unmitigatedAmount": 90_000, "amount": 70_000} for tid in (3, 4, 5, 6)],
        *[{"type": "damage", "timestamp": 112_500, "abilityGameID": 100, "targetID": tid, "unmitigatedAmount": 90_000, "amount": 70_000} for tid in (1, 2, 7, 8)],
        # Simultaneous disjoint split abilities (Spread on 3,4,5 + Lash on 6,7,8 within 0.7s -> merged max raw!)
        *[{"type": "damage", "timestamp": 130_000, "abilityGameID": 101, "targetID": tid, "unmitigatedAmount": 110_000, "amount": 80_000} for tid in (3, 4, 5)],
        *[{"type": "damage", "timestamp": 130_700, "abilityGameID": 102, "targetID": tid, "unmitigatedAmount": 120_000, "amount": 90_000} for tid in (6, 7, 8)],
        # 1-person avoidable Half Moon clip on D1 -> ignored in 8-man log
        {"type": "damage", "timestamp": 140_000, "abilityGameID": 103, "targetID": 5, "unmitigatedAmount": 130_000, "amount": 100_000},
        # Synthetic Combined DoTs (500000) -> ignored; Real Bleed on 4 non-tanks -> kept!
        *[{"type": "damage", "tick": True, "timestamp": 153_000, "abilityGameID": 500000, "targetID": tid, "unmitigatedAmount": 30_000, "amount": 30_000} for tid in (3, 4, 5, 6)],
        *[{"type": "damage", "tick": True, "timestamp": 153_000, "abilityGameID": 9001, "targetID": tid, "unmitigatedAmount": 30_000, "amount": 30_000} for tid in (3, 4, 5, 6)],
    ]
    tl = extract_damage_timeline(
        encounter_name="Calibration Test",
        zone_name="Test",
        fight_start_ms=100_000.0,
        fight_end_ms=170_000.0,
        damage_events=dmg_events,
        actors_by_id=actors,
        abilities_by_id=abilities,
        min_raw_damage=35_000.0,
    )
    names = [e.name for e in tl.events]
    assert len(tl.events) == 3  # Blast Beat, Spread/Lash merged, Real Bleed (no Half Moon, no Combined DoTs!)
    assert tl.events[0].hit_count == 1 and tl.events[0].raw_damage == pytest.approx(90_000.0)
    assert tl.events[1].raw_damage == pytest.approx(120_000.0)
    assert all("Half Moon" not in n and "Combined DoTs" not in n for n in names)

    # Empirical healing + absorbed shields divide by 6 real non-tanks (NOT 7 with LimitBreak!)
    heals = [
        *[{"type": "absorbed", "timestamp": 110_000, "sourceID": 4, "targetID": tid, "amount": 30_000} for tid in (3, 4, 5, 6, 7, 8)],
        *[{"type": "heal", "timestamp": 112_000, "sourceID": 3, "targetID": tid, "amount": 42_000} for tid in (3, 4, 5, 6, 7, 8)],
    ]
    pulses = extract_partner_healing_pulses(
        timeline=tl,
        fights_heal_streams=[
            {
                "fight_start_ms": 100_000.0,
                "fight_end_ms": 170_000.0,
                "heals": heals,
                "actors_by_id": actors,
                "tank_slot_by_id": {1: "mt", 2: "st"},
            }
        ],
        exclude_actor_names={"MT"},
    )
    party_heals = [p for p in pulses if p.target_channel == "party" and p.heal_value > 0]
    party_shields = [p for p in pulses if p.target_channel == "party" and p.shield_value > 0]
    assert len(party_heals) == 1 and party_heals[0].heal_value == pytest.approx(42_000.0)
    assert len(party_shields) == 1 and party_shields[0].shield_value == pytest.approx(30_000.0)
    assert party_shields[0].timing_mean_offset_s < 0.0

    # Expert cast extraction
    casts = [
        {"type": "cast", "timestamp": 108_500, "sourceID": 4, "abilityGameID": 188, "targetID": 4},
        {"type": "cast", "timestamp": 115_000, "sourceID": 3, "abilityGameID": 188, "targetID": 3},  # different player -> ignored
    ]
    extracted = extract_player_expert_actions(
        kit_skills_by_action_id={
            188: {"id": "sacred_soil", "name": "野戦治療の陣", "default_channel": "party", "is_gcd": False}
        },
        fight_start_ms=100_000.0,
        casts=casts,
        player_actor_id=4,
    )
    assert len(extracted) == 1
    assert extracted[0]["skill_id"] == "sacred_soil"
    assert extracted[0]["cast_at_s"] == pytest.approx(8.5)


def test_codex_pretrain_regression_no_combined_hots_no_tank_double_heal_strongest_stack_group_and_reactive_heal():
    """Verify every high-impact bug caught by Codex pre-training review:
    1) Combined HoTs (500001) and Scholar's own Eos pet are excluded from partner healing.
    2) `non_tank_only=True` party pulse does NOT duplicate onto MT/ST.
    3) Player's 10% Reprisal overrides a 5% expected-value partner Reprisal in the same stack_group.
    4) TBN 25% max HP shield scales with tank max_hp (not heal_scale) and Shadowed Vigil delayed heal triggers <50% HP.
    """
    actors = {
        1: {"id": 1, "name": "MT", "type": "Player", "subType": "DarkKnight"},
        2: {"id": 2, "name": "ST", "type": "Player", "subType": "Paladin"},
        3: {"id": 3, "name": "H1", "type": "Player", "subType": "WhiteMage"},
        4: {"id": 4, "name": "H2", "type": "Player", "subType": "Scholar"},
        5: {"id": 5, "name": "D1", "type": "Player", "subType": "Dragoon"},
        6: {"id": 6, "name": "D2", "type": "Player", "subType": "Samurai"},
        7: {"id": 7, "name": "D3", "type": "Player", "subType": "Bard"},
        8: {"id": 8, "name": "D4", "type": "Player", "subType": "BlackMage"},
    }
    all_actors = {**actors, 177: {"id": 177, "name": "Eos", "type": "Pet", "subType": "Pet"}}
    tl = EncounterTimeline(
        encounter_name="Codex Pretrain Audit",
        duration_s=30.0,
        events=[DamageEvent(id="e1", time_s=10.0, name="Raidwide", raw_damage=100_000)],
    )
    heals = [
        {"type": "heal", "timestamp": 111_000, "sourceID": 3, "targetID": 5, "abilityGameID": 3571, "amount": 60_000, "packetID": 1},
        # Synthetic Combined HoTs (500001) -> MUST be skipped
        {"type": "heal", "timestamp": 111_000, "sourceID": 3, "targetID": 5, "abilityGameID": 500001, "amount": 60_000, "packetID": 2},
        # Scholar's own Eos pet -> MUST be excluded when exclude_pet_names={"eos"}
        {"type": "heal", "timestamp": 111_000, "sourceID": 177, "targetID": 1, "abilityGameID": 802, "amount": 15_000, "packetID": 3},
        # Unpaired positive calculatedheal on ST -> MUST be preserved
        {"type": "calculatedheal", "timestamp": 112_000, "sourceID": 2, "targetID": 2, "abilityGameID": 3542, "amount": 25_000, "packetID": 99},
    ]
    pulses = extract_partner_healing_pulses(
        timeline=tl,
        fights_heal_streams=[
            {
                "fight_start_ms": 100_000.0,
                "fight_end_ms": 130_000.0,
                "heals": heals,
                "actors_by_id": actors,
                "all_actors_by_id": all_actors,
                "tank_slot_by_id": {1: "mt", 2: "st"},
            }
        ],
        exclude_actor_names={"H2"},
        exclude_pet_names={"eos"},
    )
    party_p = [p for p in pulses if p.target_channel == "party"]
    mt_p = [p for p in pulses if p.target_channel == "mt"]
    st_p = [p for p in pulses if p.target_channel == "st"]
    assert len(party_p) == 1 and party_p[0].heal_value == pytest.approx(10_000.0) and party_p[0].non_tank_only is True
    assert len(mt_p) == 0  # Eos excluded!
    assert len(st_p) == 1 and st_p[0].heal_value == pytest.approx(25_000.0)  # Unpaired calculatedheal preserved!

    # Engine: strongest stack_group wins (player 10% Reprisal > partner 5% Reprisal) + TBN %MaxHP + Shadowed Vigil reactive heal
    tank_kit = PlayerKit(
        label="DRK Audit",
        role="tank",
        max_hp_party=180_000,
        max_hp_tank=280_000,
        burst_weave_slots_per_gcd=2,
        skills=[
            SkillDef(
                id="reprisal",
                name="リプライザル",
                cooldown_s=60.0,
                duration_s=15.0,
                target_scope="boss_debuff",
                requires_targetable_boss=True,
                mitigation_all=0.10,
                stack_group="reprisal",
            ),
            SkillDef(
                id="tbn",
                name="ブラックナイト",
                cooldown_s=15.0,
                duration_s=7.0,
                target_scope="single_ally",
                shield_max_hp_ratio=0.25,
            ),
            SkillDef(
                id="vigil",
                name="シャドウヴィジル",
                cooldown_s=120.0,
                duration_s=15.0,
                target_scope="self",
                mitigation_all=0.40,
                delayed_heal_value=55_000.0,
                delayed_heal_trigger_ratio=0.50,
            ),
        ],
    )
    tl2 = EncounterTimeline(
        encounter_name="Tank Stack & Vigil Test",
        duration_s=20.0,
        events=[DamageEvent(id="tb", time_s=10.0, name="Huge Buster", target_scope="mt", raw_damage=400_000)],
        partner_profile=[
            PartnerActionProb(
                event_id="tb",
                skill_name="Partner Reprisal",
                target_channel="boss",
                stack_group="reprisal",
                duration_s=15.0,
                mitigation_all=0.10,
                probability=0.5,
                timing_mean_offset_s=-5.0,
            ),
            PartnerActionProb(
                event_id="tb",
                skill_name="NonTank Heal Only",
                target_channel="party",
                duration_s=0.0,
                heal_value=50_000.0,
                probability=1.0,
                timing_mean_offset_s=1.0,
                non_tank_only=True,
            ),
        ],
    )
    eng = CombatEngine(kit=tank_kit, timeline=tl2, damage_variance=0.0, raise_prob_per_pull=0.0)
    eng.reset(deterministic=True, fixed_tick_phase=1.5)
    eng.advance_to(6.0)
    # Partner 5% expected Reprisal landed at t=5.0; now player weaves 10% Reprisal, Vigil, and TBN (MT) across GCDs
    idx_rep = next(a.index for a in eng.actions if a.skill and a.skill.id == "reprisal")
    idx_tbn = next(a.index for a in eng.actions if a.skill and a.skill.id == "tbn" and a.target_channel == "mt")
    idx_vig = next(a.index for a in eng.actions if a.skill and a.skill.id == "vigil")
    eng.execute_action(1)  # Filler GCD refreshes weave slots
    eng.advance_to(6.7)
    eng.execute_action(idx_rep)
    eng.advance_to(7.5)
    eng.execute_action(idx_vig)
    eng.advance_to(8.6)
    eng.execute_action(1)  # Next GCD refreshes weave slots
    eng.advance_to(9.3)
    eng.execute_action(idx_tbn)
    # TBN shield must be exact 25% of 280k = 70,000!
    assert eng.total_shield("mt") == pytest.approx(70_000.0)
    # Mitigation on MT must use player's 10% Reprisal (0.9) * Vigil (0.6) = 0.54 (NOT partner's 0.95 * 0.6 = 0.57!)
    assert eng.compute_mitigation_factor("mt", "magical") == pytest.approx(0.90 * 0.60, rel=1e-3)
    # After 400k buster: dmg = 400k * 0.54 = 216k; minus 70k TBN = 146k -> HP drops to 134k (<50% of 280k = 140k!)
    # Vigil reactive heal (+55k) immediately triggers -> 189k HP!
    # And NonTank Heal Only (+50k at t=11s) must NOT heal MT!
    eng.advance_to(12.0)
    assert eng.hp["mt"] == pytest.approx(189_000.0, abs=100.0)


def test_paladin_oath_gauge_builds_on_player_aa_and_pauses_during_downtime():
    """Verify job-specific Tank AA gauge (Paladin Oath +5/AA) builds during boss uptime and pauses in downtime."""
    pld = PlayerKit.from_yaml("data/kits/paladin_heavyweight.yaml")
    tl = EncounterTimeline(
        encounter_name="Oath AA Downtime Test",
        duration_s=30.0,
        events=[],
        downtime_windows=[DowntimeWindow(start_s=10.0, end_s=20.0, name="Boss Untargetable")],
    )
    eng = CombatEngine(kit=pld, timeline=tl, damage_variance=0.0, raise_prob_per_pull=0.0)
    eng.reset(deterministic=True, fixed_tick_phase=1.5)
    # 0..10s at 2.24s AA interval -> 4 AAs (2.24, 4.48, 6.72, 8.96) = 20 Oath
    eng.advance_to(10.0)
    assert eng.resources["oath"] == pytest.approx(20.0)
    # 10..20s boss untargetable -> 0 Oath gained during downtime!
    eng.advance_to(20.0)
    assert eng.resources["oath"] == pytest.approx(20.0)
    # 20..30s boss returns -> Oath resumes building!
    eng.advance_to(30.0)
    assert eng.resources["oath"] >= 40.0


def test_worst_case_105_stress_verifier_and_100pct_hardener():
    """Verify that a plan failing under +5% worst-case damage roll is automatically hardened to 100.0% survival across 12 tick phases."""
    from xiv_healwork_solver.models import OptimizationPlan
    from xiv_healwork_solver.sim.safety import harden_plan_to_100pct, stress_test_plan

    kit = PlayerKit(
        label="Safety SCH",
        role="healer",
        max_hp_party=100_000,
        max_hp_tank=180_000,
        skills=[
            SkillDef(
                id="soil",
                name="野戦治療の陣",
                cooldown_s=30.0,
                duration_s=15.0,
                is_gcd=False,
                weave_cost=1,
                target_scope="party",
                mitigation_all=0.10,
                dps_cost=0.05,
            ),
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Worst-Case Roll Test",
        duration_s=20.0,
        # 98k survives at 1.00x roll (2k left), but WIPES at 1.05x worst-case roll (102.9k > 100k)!
        events=[DamageEvent(id="e1", time_s=10.0, name="Tight Raidwide", raw_damage=98_000)],
    )
    empty = OptimizationPlan(
        encounter_name=tl.encounter_name,
        kit_label=kit.label,
        preset="balanced",
        status="test",
        objective_value=0.0,
        partner_trust=1.0,
        actions=[],
        evaluations=[],
    )
    before = stress_test_plan(kit, tl, empty, damage_roll=1.05, n_phases=12)
    assert before.survival_rate_pct == 0.0

    hardened, after, repairs = harden_plan_to_100pct(kit, tl, empty, damage_roll=1.05, n_phases=12)
    assert after.survival_rate_pct == 100.0
    assert repairs == 1
    assert hardened.actions[0].skill_id == "soil"


def test_hp_full_heal_trigger_blast_mechanic_and_mitigated_resolution():
    """Verify generic HP-threshold-triggered raidwide blast (e.g. Ultimate full-heal trigger):
    1) Topping off to 100% HP without mitigation triggers an unmitigated blast -> WIPE.
    2) Mitigating first and THEN topping off triggers the blast safely -> SURVIVES & resolves must_trigger_before_end.
    3) Never topping off before end_s when must_trigger_before_end=True -> lethal expiry.
    4) RL observation exposes active trigger count, blast raw damage, and must-resolve flag.
    """
    kit = PlayerKit(
        label="Ultimate SCH",
        role="healer",
        max_hp_party=100_000,
        max_hp_tank=180_000,
        base_weave_slots_per_gcd=2,
        burst_weave_slots_per_gcd=2,
        skills=[
            SkillDef(
                id="mit_15",
                name="全体15%軽減",
                cooldown_s=60.0,
                duration_s=15.0,
                is_gcd=False,
                weave_cost=1,
                target_scope="party",
                mitigation_all=0.15,
            ),
            SkillDef(
                id="big_heal",
                name="全体大戻し",
                cooldown_s=30.0,
                duration_s=0.0,
                is_gcd=False,
                weave_cost=1,
                target_scope="party",
                heal_instant=60_000,
            ),
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Full-Heal Trigger Blast Test",
        duration_s=25.0,
        events=[DamageEvent(id="e1", time_s=5.0, name="Opening Hit", raw_damage=50_000)],
        hp_trigger_mechanics=[
            HpThresholdTrigger(
                id="fh_blast",
                name="Full-Heal Party Blast",
                start_s=5.0,
                end_s=18.0,
                watch_channel="party",
                trigger_hp_ratio=1.0,
                blast_target_scope="party",
                blast_raw_damage=110_000,  # Lethal unmitigated (110k > 100k), survivable with 15% mit (93.5k < 100k)!
                max_triggers=1,
                must_trigger_before_end=True,
            )
        ],
    )

    # Case 1: Heal to 100% WITHOUT mitigation -> triggers 110k blast -> WIPE!
    eng1 = CombatEngine(kit=kit, timeline=tl, damage_variance=0.0, raise_prob_per_pull=0.0)
    eng1.reset(deterministic=True, fixed_tick_phase=1.5)
    eng1.advance_to(7.0)
    idx_heal = next(a.index for a in eng1.actions if a.skill and a.skill.id == "big_heal")
    idx_mit = next(a.index for a in eng1.actions if a.skill and a.skill.id == "mit_15")
    eng1.execute_action(idx_heal)
    assert eng1.died is True
    assert eng1.stats["hp_triggers_fired"] == 1.0

    # Case 2: Mitigate FIRST, THEN heal to 100% -> triggers 93.5k blast -> SURVIVES & resolves!
    eng2 = CombatEngine(kit=kit, timeline=tl, damage_variance=0.0, raise_prob_per_pull=0.0)
    eng2.reset(deterministic=True, fixed_tick_phase=1.5)
    eng2.advance_to(7.0)
    eng2.execute_action(idx_mit)
    eng2.advance_to(8.0)
    eng2.execute_action(idx_heal)
    eng2.advance_to(20.0)
    v2 = eng2.build_verification()
    assert v2.survived is True
    assert v2.hp_triggers_fired == 1
    assert v2.min_hp_ratio["party"] == pytest.approx(0.065, abs=0.01)

    # Case 3: Never top off before end_s (18s) -> lethal expiry!
    eng3 = CombatEngine(kit=kit, timeline=tl, damage_variance=0.0, raise_prob_per_pull=0.0)
    eng3.reset(deterministic=True, fixed_tick_phase=1.5)
    eng3.advance_to(20.0)
    assert eng3.died is True
    assert eng3.first_death_channel == "party(hp_trigger_expired)"

    # Case 4: RL env exposes active HP-trigger features in obs["state"][33:36]
    env = FF14HealMitEnv(kit, tl, damage_variance=0.0, raise_prob_per_pull=0.0, seed=1)
    env.reset(deterministic=True, fixed_tick_phase=1.5)
    env.engine.advance_to(7.0)
    obs = env._get_obs()
    assert obs["state"][33] > 0.0  # active trigger count
    assert obs["state"][34] == pytest.approx(110_000 / 250_000, abs=1e-4)
    assert obs["state"][35] == 1.0  # must_trigger_before_end


def test_expected_dps_maximization_prunes_and_swaps_to_0p_ogcd_at_100pct_survival():
    """Verify maximize_dps_at_100pct_survival removes redundant 310p GCD heals and swaps to 0p oGCDs
    while strictly preserving 100.0% survival at worst-case +5% roll x 12 server-tick phases."""
    from xiv_healwork_solver.models import OptimizationPlan, PlannedAction
    from xiv_healwork_solver.sim.safety import maximize_dps_at_100pct_survival, stress_test_plan

    kit = PlayerKit(
        label="DPS Max SCH",
        role="healer",
        gcd_s=2.40,
        max_hp_party=100_000,
        max_hp_tank=180_000,
        filler_gcd_potency=310.0,
        burst_dps_multiplier=1.25,
        burst_offset_s=5.0,
        burst_duration_s=20.0,
        burst_weave_slots_per_gcd=2,
        skills=[
            SkillDef(
                id="free_ogcd_mit",
                name="無料0p全体軽減",
                cooldown_s=60.0,
                duration_s=15.0,
                is_gcd=False,
                weave_cost=1,
                target_scope="party",
                mitigation_all=0.15,
                dps_cost=0.01,
            ),
            SkillDef(
                id="gcd_shield",
                name="意気軒昂の策",
                cooldown_s=2.4,
                duration_s=30.0,
                cast_time_s=0.0,
                is_gcd=True,
                weave_cost=0,
                target_scope="party",
                shield_value=20_000,
                dps_cost=1.0,
            ),
        ],
    )
    tl = EncounterTimeline(
        encounter_name="DPS Maximization Test",
        duration_s=30.0,
        events=[DamageEvent(id="e1", time_s=15.0, name="Raidwide", raw_damage=98_000)],
    )
    # Wasteful plan: casts TWO 310p GCD shields (one redundant, one replaceable by 0p free_ogcd_mit!)
    wasteful = OptimizationPlan(
        encounter_name=tl.encounter_name,
        kit_label=kit.label,
        preset="balanced",
        status="wasteful",
        objective_value=0.0,
        partner_trust=1.0,
        actions=[
            PlannedAction(
                event_id="e1", event_name="Raidwide", event_time_s=15.0, cast_at_s=8.0,
                skill_id="gcd_shield", skill_name="意気軒昂の策", target_scope="party",
                target_channel="party", is_gcd=True, weave_cost=0,
            ),
            PlannedAction(
                event_id="e1", event_name="Raidwide", event_time_s=15.0, cast_at_s=12.0,
                skill_id="gcd_shield", skill_name="意気軒昂の策", target_scope="party",
                target_channel="party", is_gcd=True, weave_cost=0,
            ),
        ],
        evaluations=[],
    )
    before = stress_test_plan(kit, tl, wasteful, damage_roll=1.05, n_phases=12)
    assert before.survival_rate_pct == 100.0
    assert before.mean_gcd_heals == 2.0

    opt_plan, after, stats = maximize_dps_at_100pct_survival(kit, tl, wasteful, damage_roll=1.05, n_phases=12)
    assert after.survival_rate_pct == 100.0
    assert after.mean_gcd_heals == 0.0  # Both 310p GCD shields eliminated (1 pruned, 1 swapped to 0p oGCD!)
    assert after.mean_potency_lost < before.mean_potency_lost
    assert after.mean_expected_potency > before.mean_expected_potency
    assert after.mean_dps_efficiency_pct > before.mean_dps_efficiency_pct
    assert stats["pruned"] >= 1 and stats["swapped"] >= 1


def test_codex_hp_trigger_and_potency_p1_regressions():
    """Regression test for Codex P1 findings:
    1) stress_test_plan catches full-heal trigger wipes at 1.00x / 0.95x rolls even if 1.05x survives.
    2) Party heal heals MT/ST BEFORE full-heal trigger blast fires (no false tank wipe).
    3) Pure overheal while already at 100% HP does not repeatedly consume multi-trigger blasts.
    4) direct_potency_loss overrides derived GCD potency loss cleanly.
    """
    from xiv_healwork_solver.models import OptimizationPlan, PlannedAction
    from xiv_healwork_solver.sim.safety import stress_test_plan

    kit = PlayerKit(
        label="Trigger Edge SCH",
        role="healer",
        max_hp_party=100_000,
        max_hp_tank=100_000,
        filler_gcd_potency=310.0,
        base_weave_slots_per_gcd=2,
        burst_weave_slots_per_gcd=2,
        skills=[
            SkillDef(
                id="aoe_heal_50k",
                name="50k全体ヒール",
                cooldown_s=5.0,
                duration_s=0.0,
                is_gcd=False,
                weave_cost=1,
                target_scope="party",
                heal_instant=50_000,
            ),
            SkillDef(
                id="override_gcd",
                name="上書き威力GCD",
                cooldown_s=2.5,
                duration_s=0.0,
                is_gcd=True,
                weave_cost=0,
                direct_potency_loss=100.0,
            ),
        ],
    )
    # 4) direct_potency_loss override reduces derived 310p loss to 100p (outside burst)
    assert kit.expected_skill_potency_loss(kit.skills[1], 1.0, False) == pytest.approx(100.0)

    # 1) 50k hit + 50k heal + lethal 110k full-HP blast: survives +5% roll (HP=97.5k), WIPES at 1.00x/0.95x!
    tl_low_roll = EncounterTimeline(
        encounter_name="Low Roll Trigger Trap",
        duration_s=15.0,
        events=[DamageEvent(id="e1", time_s=5.0, name="Hit", raw_damage=50_000)],
        hp_trigger_mechanics=[
            HpThresholdTrigger(
                id="tr1",
                name="Lethal Full-Heal Blast",
                start_s=5.0,
                end_s=12.0,
                watch_channel="party",
                trigger_hp_ratio=1.0,
                blast_target_scope="party",
                blast_raw_damage=110_000,
                max_triggers=1,
                must_trigger_before_end=False,
            )
        ],
    )
    plan_trap = OptimizationPlan(
        encounter_name=tl_low_roll.encounter_name,
        kit_label=kit.label,
        preset="balanced",
        status="test",
        objective_value=0.0,
        partner_trust=1.0,
        actions=[
            PlannedAction(
                event_id="e1", event_name="Hit", event_time_s=5.0, cast_at_s=7.0,
                skill_id="aoe_heal_50k", skill_name="50k全体ヒール", target_scope="party",
                target_channel="party", is_gcd=False, weave_cost=1,
            )
        ],
        evaluations=[],
    )
    rep_trap = stress_test_plan(kit, tl_low_roll, plan_trap, damage_roll=1.05, n_phases=4)
    assert rep_trap.survival_rate_pct < 100.0  # Must catch the 0.95x and 1.00x wipes!

    # 2) Party heal heals ALL channels (party/mt/st) from 50k->100k BEFORE 60k blast fires -> everyone ends at 40k ALIVE!
    tl_tank_order = EncounterTimeline(
        encounter_name="Party Heal Order Test",
        duration_s=15.0,
        events=[DamageEvent(id="e1", time_s=5.0, name="Hit", raw_damage=50_000)],
        hp_trigger_mechanics=[
            HpThresholdTrigger(
                id="tr_ord",
                name="60k Blast",
                start_s=5.0,
                end_s=14.0,
                watch_channel="party",
                trigger_hp_ratio=1.0,
                blast_target_scope="party",
                blast_raw_damage=60_000,
                max_triggers=2,
                trigger_cooldown_s=1.0,
                must_trigger_before_end=False,
            )
        ],
    )
    eng = CombatEngine(kit=kit, timeline=tl_tank_order, damage_variance=0.0, raise_prob_per_pull=0.0)
    eng.reset(deterministic=True, fixed_tick_phase=1.5)
    eng.advance_to(7.0)
    idx_h = next(a.index for a in eng.actions if a.skill and a.skill.id == "aoe_heal_50k")
    eng.execute_action(idx_h)
    assert eng.died is False
    assert eng.hp["party"] == pytest.approx(40_000.0)
    assert eng.hp["mt"] == pytest.approx(40_000.0)
    assert eng.stats["hp_triggers_fired"] == 1.0

    # 3) Pure overheal when already at 100% HP must NOT consume another trigger!
    eng.hp["party"] = 100_000.0
    eng.advance_to(10.0)
    eng._apply_heal("party", 20_000.0)
    assert eng.stats["hp_triggers_fired"] == 1.0  # Still 1, NOT 2!


def test_ilvl_progression_scaling_raises_expected_dps_at_100pct_survival():
    """Higher iLvl raises Max HP and heal_scale, enabling fewer GCD heals and higher Expected DPS at 100% survival."""
    from xiv_healwork_solver.models import OptimizationPlan, PlannedAction
    from xiv_healwork_solver.sim.safety import maximize_dps_at_100pct_survival

    kit_base = PlayerKit(
        label="iLvl Test Healer",
        role="healer",
        reference_ilvl=780,
        gcd_s=2.40,
        max_hp_party=160_000,
        max_hp_tank=250_000,
        filler_gcd_potency=310.0,
        skills=[
            SkillDef(
                id="gcd_shield",
                name="GCD Shield",
                cooldown_s=2.4,
                duration_s=30.0,
                is_gcd=True,
                weave_cost=0,
                target_scope="party",
                shield_value=24_000,
                heal_instant=12_000,
            ),
            SkillDef(
                id="ogcd_hot",
                name="oGCD HoT",
                cooldown_s=60.0,
                duration_s=15.0,
                is_gcd=False,
                weave_cost=1,
                target_scope="party",
                heal_hot_total=20_000,
            ),
        ],
    )
    k_low = kit_base.with_gear_overrides(ilvl=770)
    k_high = kit_base.with_gear_overrides(ilvl=795)
    assert k_low.max_hp_party < kit_base.max_hp_party < k_high.max_hp_party
    assert k_low.gear_stats.heal_scale_multiplier < 1.0 < k_high.gear_stats.heal_scale_multiplier
    # Explicit HP override wins over iLvl-derived default
    k_override = kit_base.with_gear_overrides(ilvl=790, max_hp_party=199_999.0)
    assert k_override.max_hp_party == pytest.approx(199_999.0)

    # 158k raw raidwide: +5% roll = 165.9k -> i770 (HP ~148.8k) NEEDS the GCD shield (171.5k), while i795 (HP ~178.4k) survives WITHOUT it!
    tl = EncounterTimeline(
        encounter_name="iLvl DPS Scaling Check",
        duration_s=25.0,
        events=[DamageEvent(id="e1", time_s=12.0, name="Big Raidwide", raw_damage=158_000)],
    )
    seed_plan = OptimizationPlan(
        encounter_name=tl.encounter_name,
        kit_label=kit_base.label,
        preset="balanced",
        partner_trust=1.0,
        status="seed",
        objective_value=0.0,
        actions=[
            PlannedAction(
                event_id="e1", event_name="Big Raidwide", event_time_s=12.0, cast_at_s=8.0,
                skill_id="gcd_shield", skill_name="GCD Shield", target_scope="party",
                target_channel="party", is_gcd=True, weave_cost=0,
            )
        ],
        evaluations=[],
    )
    _, rep_low, _ = maximize_dps_at_100pct_survival(k_low, tl, seed_plan, damage_roll=1.05, n_phases=6)
    _, rep_high, stats_high = maximize_dps_at_100pct_survival(k_high, tl, seed_plan, damage_roll=1.05, n_phases=6)
    assert rep_low.survival_rate_pct == 100.0 and rep_low.mean_gcd_heals >= 1.0
    assert rep_high.survival_rate_pct == 100.0 and rep_high.mean_gcd_heals == 0.0
    assert stats_high["pruned"] >= 1
    assert rep_high.mean_expected_potency > rep_low.mean_expected_potency


def test_dropped_action_tracking_prevents_on_cooldown_ghost_swaps_and_sheet_audit():
    """Verify replay_plan tracks dropped_actions, DPS maximizer strips ghost/illegal casts, and audit_healwork_sheet flags issues."""
    from xiv_healwork_solver.exporter.audit import audit_healwork_sheet, render_audit_markdown
    from xiv_healwork_solver.models import OptimizationPlan, PlannedAction
    from xiv_healwork_solver.sim.engine import replay_plan
    from xiv_healwork_solver.sim.safety import maximize_dps_at_100pct_survival

    kit = PlayerKit(
        label="Ghost Swap Regression Kit",
        role="healer",
        gcd_s=2.40,
        burst_weave_slots_per_gcd=1,
        max_hp_party=150_000,
        max_hp_tank=240_000,
        skills=[
            SkillDef(
                id="cd60_hot",
                name="60s HoT",
                cooldown_s=60.0,
                duration_s=15.0,
                is_gcd=False,
                weave_cost=1,
                target_scope="party",
                heal_hot_total=30_000,
            ),
            SkillDef(
                id="costly_aoe",
                name="Costly AoE",
                cooldown_s=30.0,
                duration_s=0.0,
                is_gcd=True,
                weave_cost=0,
                target_scope="party",
                heal_instant=25_000,
            ),
        ],
    )
    tl = EncounterTimeline(
        encounter_name="Ghost Audit Arena",
        duration_s=45.0,
        events=[
            DamageEvent(id="e1", time_s=10.0, name="Hit 1", raw_damage=60_000),
            DamageEvent(id="e2", time_s=30.0, name="Hit 2", raw_damage=60_000),
        ],
    )
    # Schedule cd60_hot TWICE within 12s (2nd cast is illegal on-cooldown!)
    ghost_plan = OptimizationPlan(
        encounter_name=tl.encounter_name,
        kit_label=kit.label,
        preset="balanced",
        partner_trust=1.0,
        status="noisy",
        objective_value=0.0,
        actions=[
            PlannedAction(
                event_id="e1", event_name="Hit 1", event_time_s=10.0, cast_at_s=8.0,
                skill_id="cd60_hot", skill_name="60s HoT", target_scope="party",
                target_channel="party", is_gcd=False, weave_cost=1,
            ),
            PlannedAction(
                event_id="e2", event_name="Hit 2", event_time_s=30.0, cast_at_s=20.0,
                skill_id="cd60_hot", skill_name="60s HoT", target_scope="party",
                target_channel="party", is_gcd=False, weave_cost=1,
            ),
        ],
        evaluations=[],
    )
    rep_raw = replay_plan(kit, tl, ghost_plan, deterministic=True, fixed_tick_phase=1.5, forced_damage_roll=1.05)
    assert rep_raw.dropped_actions == 1

    # Audit catches the dropped ghost cast as FAIL!
    audit_bad = audit_healwork_sheet(kit, tl, ghost_plan, reference_clear_casts=[[{"skill_id": "cd60_hot"}]], reference_clear_labels=["Ref#1"])
    assert audit_bad.verdict == "FAIL"
    assert any(f.category == "legality" for f in audit_bad.findings)

    # maximize_dps_at_100pct_survival cleans the ghost cast cleanly -> 0 dropped actions & PASS audit!
    clean_plan, clean_rep, stats = maximize_dps_at_100pct_survival(kit, tl, ghost_plan, damage_roll=1.05, n_phases=6)
    assert clean_rep.max_dropped_actions == 0
    assert stats["ghosts_removed"] >= 1
    audit_ok = audit_healwork_sheet(kit, tl, clean_plan, reference_clear_casts=[[{"skill_id": "cd60_hot"}]], reference_clear_labels=["Ref#1"])
    assert audit_ok.verdict in ("PASS", "WARN")
    assert audit_ok.stress_105.max_dropped_actions == 0
    md = render_audit_markdown(audit_ok)
    assert "シート検品レポート" in md and "Ref#1" in md


def test_multi_encounter_parallel_rl_trainer_smoke(tmp_path):
    """Verify multi-encounter train_masked_ppo works cleanly with parallel CPU rollout workers."""
    pytest.importorskip("torch")
    from xiv_healwork_solver.rl.train_colab import train_masked_ppo

    kit = PlayerKit.from_yaml(ROOT / "data" / "kits" / "example_barrier_healer.yaml")
    tl1 = EncounterTimeline(
        encounter_name="Mini Enc 1",
        duration_s=24.0,
        events=[DamageEvent(id="e1", time_s=12.0, name="Blast", raw_damage=55_000)],
    )
    tl2 = EncounterTimeline(
        encounter_name="Mini Enc 2",
        duration_s=24.0,
        events=[DamageEvent(id="e1", time_s=10.0, name="Wave", raw_damage=60_000)],
    )
    summary = train_masked_ppo(
        kit=kit,
        timeline=tl1,
        extra_timelines=[tl2],
        updates=2,
        episodes_per_update=4,
        hidden_dim=32,
        ppo_epochs=2,
        minibatch_size=32,
        num_workers=2,
        out_dir=tmp_path / "rl_multi",
    )
    assert summary["updates"] == 2
    assert len(summary["trained_encounters"]) == 2
    assert (tmp_path / "rl_multi" / "best_policy.pt").exists()









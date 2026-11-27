"""Command-line interface for XIVHealWorkSolver."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from xiv_healwork_solver.colab.robustness_sim import run_robustness_simulation
from xiv_healwork_solver.exporter.audit import audit_healwork_sheet, render_audit_markdown
from xiv_healwork_solver.exporter.sheets import export_csv, export_json, export_markdown
from xiv_healwork_solver.fflogs.client import FFLogsClient
from xiv_healwork_solver.fflogs.dataset import build_curated_encounter_timeline
from xiv_healwork_solver.fflogs.extractor import extract_damage_timeline
from xiv_healwork_solver.models import EncounterTimeline, LiveFightState, ObjectivePreset, OptimizationPlan, PlayerKit
from xiv_healwork_solver.optimizer.solver import replan_from_state, solve_mitigation_plan

app = typer.Typer(help="Skill-agnostic FF14 mitigation & healing timeline solver.")
console = Console()


def _print_plan_table(plan) -> None:
    table = Table(
        title=f"{plan.encounter_name} | {plan.kit_label} [{plan.preset}] ({plan.status}, {plan.solve_time_ms:.1f}ms)"
    )
    table.add_column("Time", style="cyan")
    table.add_column("Event")
    table.add_column("Scope")
    table.add_column("Raw", justify="right")
    table.add_column("Mit%", justify="right")
    table.add_column("EffDmg", justify="right")
    table.add_column("Margin%", justify="right")
    table.add_column("Res", justify="right")
    table.add_column("Weave", justify="center")
    table.add_column("Skills", style="green")

    for ev in plan.evaluations:
        margin_style = "green" if ev.survival_margin_pct >= 20 else ("yellow" if ev.survival_margin_pct > 0 else "red")
        res_str = " ".join(f"{k}:{v:.0f}" for k, v in ev.resource_after.items()) if ev.resource_after else "-"
        table.add_row(
            f"{ev.time_s:5.1f}s",
            ev.name,
            ev.target_scope,
            f"{ev.raw_damage:,.0f}",
            f"{ev.total_mitigation_pct:.1f}%",
            f"{ev.effective_damage:,.0f}",
            f"[{margin_style}]{ev.survival_margin_pct:.1f}%[/{margin_style}]",
            res_str,
            f"{ev.weave_used}/{ev.weave_budget}",
            ", ".join(ev.used_skills) or "-",
        )
    console.print(table)
    if plan.replay is not None:
        r = plan.replay
        if r.survived:
            console.print(
                f"  [green]Replay VERIFIED: SURVIVED[/green] | "
                f"minHP(party={r.min_hp_ratio.get('party',0)*100:.1f}%, "
                f"mt={r.min_hp_ratio.get('mt',0)*100:.1f}%, st={r.min_hp_ratio.get('st',0)*100:.1f}%) | "
                f"fillerGCD={r.filler_gcds} gcdHeal={r.gcd_heals} freeDT={r.downtime_free_gcds} "
                f"overheal={r.overheal_pct:.1f}% minMP={r.mp_min:.0f}"
            )
        else:
            console.print(
                f"  [red]Replay FAILED: DIED at {r.first_death_time_s}s ({r.first_death_channel}, deficit={r.deficit_hp:,.0f} HP)[/red]"
            )


@app.command()
def optimize(
    kit_path: Path = typer.Option(..., "--kit", "-k", help="Path to player skill kit YAML"),
    timeline_path: Path = typer.Option(..., "--timeline", "-t", help="Path to encounter timeline YAML"),
    preset: ObjectivePreset = typer.Option("balanced", "--preset", "-p", help="balanced|dps_greedy|safe|solo_carry"),
    ilvl: Optional[int] = typer.Option(None, "--ilvl", help="Optional average iLvl override (auto-scales HP & heal power)"),
    partner_trust: Optional[float] = typer.Option(None, "--partner-trust", help="Override partner trust [0.0..1.0]"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o", help="Export Markdown, CSV, and JSON to directory"),
    simulate_trials: int = typer.Option(2000, "--sim-trials", help="Local Monte Carlo trials (0 to skip)"),
) -> None:
    """Solve an optimal mitigation & healing timeline for a skill kit and encounter."""
    kit = PlayerKit.from_yaml(kit_path).with_gear_overrides(ilvl=ilvl)
    timeline = EncounterTimeline.from_yaml(timeline_path)
    plan = solve_mitigation_plan(kit, timeline, preset=preset, partner_trust_override=partner_trust)
    _print_plan_table(plan)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        export_markdown(plan, out_dir / "plan.md")
        export_csv(plan, out_dir / "plan.csv")
        export_json(plan, out_dir / "plan.json")
        console.print(f"[bold]Exported[/bold] plan.md / plan.csv / plan.json to {out_dir}")

    if simulate_trials > 0:
        sim = run_robustness_simulation(
            plan.model_dump(), timeline.model_dump(), kit.model_dump(), trials=simulate_trials
        )
        console.print(
            f"Monte Carlo ({simulate_trials} trials, +/-5% roll + partner dropout): "
            f"[bold cyan]Pull survival = {sim['pull_survival_rate_pct']:.2f}%[/bold cyan]"
        )


@app.command()
def demo(
    preset: ObjectivePreset = typer.Option("balanced", "--preset", "-p"),
) -> None:
    """Run an end-to-end demo using bundled example healer & tank kits."""
    root = Path(__file__).resolve().parents[2]
    timeline_path = root / "data" / "timelines" / "example_savage_3min.yaml"
    for kit_file in ("example_barrier_healer.yaml", "example_tank.yaml"):
        kit_path = root / "data" / "kits" / kit_file
        kit = PlayerKit.from_yaml(kit_path)
        timeline = EncounterTimeline.from_yaml(timeline_path)
        plan = solve_mitigation_plan(kit, timeline, preset=preset)
        _print_plan_table(plan)
        sim = run_robustness_simulation(plan.model_dump(), timeline.model_dump(), kit.model_dump(), trials=3000)
        console.print(
            f"  -> Robustness (3,000 trials): Pull survival = [bold]{sim['pull_survival_rate_pct']:.2f}%[/bold]\n"
        )


@app.command("fetch-report")
def fetch_report(
    report_code: str = typer.Argument(..., help="FFLogs report code (from URL .../reports/<CODE>)"),
    fight_id: Optional[int] = typer.Option(None, "--fight", "-f", help="Fight ID (defaults to last kill or last fight)"),
    out: Path = typer.Option(Path("outputs/extracted_timeline.yaml"), "--out", "-o", help="Output timeline YAML"),
) -> None:
    """Fetch a fight from FFLogs v2 and extract its normalized damage timeline."""
    client = FFLogsClient()
    overview = client.get_report_overview(report_code)
    fights = overview.get("fights") or []
    if not fights:
        raise typer.BadParameter(f"No fights found in report {report_code}.")

    if fight_id is None:
        kills = [f for f in fights if f.get("kill")]
        target = kills[-1] if kills else fights[-1]
    else:
        matches = [f for f in fights if f["id"] == fight_id]
        if not matches:
            raise typer.BadParameter(f"Fight ID {fight_id} not found in report {report_code}.")
        target = matches[0]

    actors_by_id = {int(a["id"]): a for a in overview["masterData"]["actors"]}
    abilities_by_id = {int(a["gameID"]): a for a in overview["masterData"]["abilities"]}

    console.print(f"Fetching DamageTaken for '{target['name']}' (fight #{target['id']})...")
    dmg_events = client.get_fight_events(
        report_code=report_code,
        fight_id=int(target["id"]),
        start_time=float(target["startTime"]),
        end_time=float(target["endTime"]),
        data_type="DamageTaken",
        hostility_type="Friendlies",
    )
    timeline = extract_damage_timeline(
        encounter_name=target["name"],
        zone_name=(overview.get("zone") or {}).get("name"),
        fight_start_ms=float(target["startTime"]),
        fight_end_ms=float(target["endTime"]),
        damage_events=dmg_events,
        actors_by_id=actors_by_id,
        abilities_by_id=abilities_by_id,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(timeline.model_dump(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    console.print(f"[green]Extracted {len(timeline.events)} damage events -> {out}[/green]")


@app.command("replan")
def replan(
    kit_path: Path = typer.Option(..., "--kit", "-k", help="Path to player skill kit YAML"),
    timeline_path: Path = typer.Option(..., "--timeline", "-t", help="Path to encounter timeline YAML"),
    now: float = typer.Option(..., "--now", help="Current fight time in seconds"),
    mp: Optional[float] = typer.Option(None, "--mp", help="Current remaining MP (e.g. after a Raise)"),
    hp_ratio: float = typer.Option(1.0, "--hp-ratio", help="Current party HP ratio (0.1..1.0)"),
    cooldown: list[str] = typer.Option([], "--cd", help="Remaining cooldown as skill_id=seconds (repeatable)"),
    preset: ObjectivePreset = typer.Option("safe", "--preset", "-p"),
    horizon_s: Optional[float] = typer.Option(90.0, "--horizon", help="Lookahead window in seconds"),
) -> None:
    """Fast mid-fight replanning from current cooldowns, MP, and HP state."""
    kit = PlayerKit.from_yaml(kit_path)
    timeline = EncounterTimeline.from_yaml(timeline_path)
    cds: dict[str, float] = {}
    for item in cooldown:
        if "=" in item:
            k, v = item.split("=", 1)
            cds[k.strip()] = float(v.strip())
    res: dict[str, float] = {}
    if mp is not None:
        res["mp"] = mp

    state = LiveFightState(
        current_time_s=now,
        cooldowns_remaining=cds,
        current_resources=res,
        current_party_hp_ratio=hp_ratio,
        horizon_s=horizon_s,
    )
    plan = replan_from_state(kit, timeline, state, preset=preset)
    _print_plan_table(plan)


@app.command("build-encounter")
def build_encounter(
    encounter_id: int = typer.Argument(..., help="FFLogs encounter ID (e.g. 101 = M9S Vamp Fatale)"),
    spec: str = typer.Option("Scholar", "--spec", "-s", help="Your job spec name on FFLogs"),
    page: int = typer.Option(5, "--page", help="Ranking page (higher pages avoid #1 speedkill outliers)"),
    candidates: int = typer.Option(10, "--candidates", help="How many candidate reports to audit"),
    partner_fights: int = typer.Option(4, "--partner-fights", help="Curated clears to aggregate partner profile from"),
    out: Path = typer.Option(Path("data/timelines/encounter.yaml"), "--out", "-o", help="Output timeline YAML"),
) -> None:
    """Fetch, curate (filtering Chad-healing & outliers), and extract an encounter timeline + partner profile."""
    root = Path(__file__).resolve().parents[2]
    timeline, accepted, rejected = build_curated_encounter_timeline(
        encounter_id=encounter_id,
        target_spec=spec,
        known_skills_path=root / "data" / "known_party_skills.yaml",
        ranking_page=page,
        inspect_candidates=candidates,
        partner_sample_fights=partner_fights,
    )
    console.print(f"[bold]Curation Summary:[/bold] accepted={len(accepted)}, rejected={len(rejected)}")
    for r in accepted[:5]:
        console.print(
            f"  [green]OK[/green] {r.report_code}#{r.fight_id} dur={r.duration_s}s "
            f"combRDPS={r.combined_healer_rdps:.0f} shares={r.healer_effective_shares} "
            f"overheal={r.healer_overheal_pcts} score={r.quality_score}"
        )
    for r in rejected[:5]:
        console.print(f"  [yellow]REJ[/yellow] {r.report_code}#{r.fight_id} reasons={r.reasons}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(timeline.model_dump(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    console.print(
        f"[green]Saved '{timeline.encounter_name}' ({len(timeline.events)} events, "
        f"{len(timeline.auto_attacks)} AA segments, {len(timeline.partner_profile)} partner probs) -> {out}[/green]"
    )


@app.command("infer-rl")
def infer_rl(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-m", help="Path to trained best_policy.pt"),
    kit_path: Path = typer.Option(..., "--kit", "-k", help="Path to base PlayerKit YAML"),
    timeline_path: Path = typer.Option(..., "--timeline", "-t", help="Path to EncounterTimeline YAML"),
    ilvl: Optional[int] = typer.Option(None, "--ilvl", help="Average iLvl (e.g. 770 Week-1 -> 790 BiS; auto-scales HP & heal)"),
    gcd_s: Optional[float] = typer.Option(None, "--gcd", help="Override GCD seconds (e.g. 2.42)"),
    piety: Optional[int] = typer.Option(None, "--piety", help="Override Piety stat (e.g. 950)"),
    tenacity: Optional[int] = typer.Option(None, "--tenacity", help="Override Tenacity stat"),
    heal_scale: Optional[float] = typer.Option(None, "--heal-scale", help="Override heal/shield multiplier"),
    max_hp_party: Optional[float] = typer.Option(None, "--hp-party", help="Override party max HP"),
    max_hp_tank: Optional[float] = typer.Option(None, "--hp-tank", help="Override tank max HP"),
    min_hp_margin: float = typer.Option(0.0, "--min-hp-margin", help="Minimum worst-case HP ratio floor (e.g. 0.05 = 5%)"),
    pulls: int = typer.Option(12, "--pulls", help="Evaluation pulls across 3s server-tick phases"),
) -> None:
    """Run a pre-trained gear/iLvl-conditioned RL checkpoint with custom runtime gear parameters."""
    from xiv_healwork_solver.rl.train_colab import infer_with_gear

    kit = PlayerKit.from_yaml(kit_path)
    timeline = EncounterTimeline.from_yaml(timeline_path)
    res = infer_with_gear(
        checkpoint_path=checkpoint,
        kit=kit,
        timeline=timeline,
        ilvl=ilvl,
        gcd_s=gcd_s,
        piety=piety,
        tenacity=tenacity,
        heal_scale=heal_scale,
        max_hp_party=max_hp_party,
        max_hp_tank=max_hp_tank,
        min_hp_margin_ratio=min_hp_margin,
        n_eval_pulls=pulls,
    )
    summary = {k: v for k, v in res.items() if k not in ("deterministic_action_log", "verified_plan")}
    console.print(f"[bold cyan]RL Inference Result:[/bold cyan] {json.dumps(summary, ensure_ascii=False)}")
    console.print(f"Scheduled actions (deterministic rollout): {len(res['deterministic_action_log'])}")


@app.command("audit-sheet")
def audit_sheet(
    plan_path: Path = typer.Option(..., "--plan", "-p", help="Path to exported OptimizationPlan JSON"),
    kit_path: Path = typer.Option(..., "--kit", "-k", help="Path to PlayerKit YAML"),
    timeline_path: Path = typer.Option(..., "--timeline", "-t", help="Path to EncounterTimeline YAML"),
    ilvl: Optional[int] = typer.Option(None, "--ilvl", help="Optional iLvl override matching the plan"),
    ref_casts: list[Path] = typer.Option([], "--ref-casts", "-r", help="Reference real FFLogs clear cast JSON(s)"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Optional output Markdown audit report path"),
) -> None:
    """Audit a generated healwork sheet for 100% survival, legality, and realism vs real FFLogs clears."""
    kit = PlayerKit.from_yaml(kit_path).with_gear_overrides(ilvl=ilvl)
    timeline = EncounterTimeline.from_yaml(timeline_path)
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = OptimizationPlan.model_validate(raw["verified_plan"] if "verified_plan" in raw else raw)

    ref_lists: list[list[dict]] = []
    ref_labels: list[str] = []
    for rp in ref_casts:
        data = json.loads(rp.read_text(encoding="utf-8"))
        casts = data["actions"] if isinstance(data, dict) and "actions" in data else data
        if isinstance(casts, list):
            ref_lists.append(casts)
            ref_labels.append(rp.stem)

    report = audit_healwork_sheet(
        kit, timeline, plan, reference_clear_casts=ref_lists, reference_clear_labels=ref_labels
    )
    md = render_audit_markdown(report, output_path=out)
    console.print(md)
    if out is not None:
        console.print(f"[green]Saved audit report -> {out}[/green]")


@app.command("check-fflogs")
def check_fflogs() -> None:
    """Verify FFLogs API v2 credentials and display current hourly rate limit."""
    client = FFLogsClient()
    rl = client.get_rate_limit()
    console.print(f"[green]FFLogs API v2 OK![/green] Rate limit: {json.dumps(rl)}")


if __name__ == "__main__":
    app()

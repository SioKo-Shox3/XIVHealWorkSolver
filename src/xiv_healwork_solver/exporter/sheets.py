"""Export optimized mitigation & healing timelines to Markdown, CSV, and JSON."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from xiv_healwork_solver.models import OptimizationPlan


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def export_markdown(plan: OptimizationPlan, output_path: str | Path | None = None) -> str:
    """Render a clean human-readable Markdown mitigation & healwork sheet."""
    lines = [
        f"# {plan.encounter_name} — ヒール・軽減タイムライン",
        "",
        f"- **スキル構成**: {plan.kit_label}",
        f"- **プリセット**: `{plan.preset}` (相方信頼度: `{plan.partner_trust:.0%}`)",
        f"- **ソルバー状態**: `{plan.status}`",
        "",
        "## タイムライン別評価",
        "",
        "| 時刻 | ギミック / 攻撃 | 対象 | 生ダメ | 自軽減 | 相方期待 | 合計軽減 | 実ダメ | バリア合計 | 残HP余裕 | 使用スキル | Weave |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :---: |",
    ]
    for ev in plan.evaluations:
        skills_str = ", ".join(ev.used_skills) if ev.used_skills else "-"
        lines.append(
            f"| `{_fmt_time(ev.time_s)}` | {ev.name} | {ev.target_scope} | "
            f"{ev.raw_damage:,.0f} | {ev.self_mitigation_pct:.1f}% | "
            f"{ev.partner_expected_mitigation_pct:.1f}% | **{ev.total_mitigation_pct:.1f}%** | "
            f"{ev.effective_damage:,.0f} | {ev.total_shield:,.0f} | "
            f"{ev.survival_margin_pct:.1f}% | {skills_str} | {ev.weave_used}/{ev.weave_budget} |"
        )

    lines.extend(
        [
            "",
            "## アクション実行スケジュール（発動時刻順）",
            "",
            "| 発動時刻 | 対象攻撃時刻 | スキル名 | ActionID | 対象 | 種別 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for act in sorted(plan.actions, key=lambda a: a.cast_at_s):
        kind = "GCD" if act.is_gcd else f"oGCD (weave {act.weave_cost})"
        aid = str(act.action_id) if act.action_id is not None else "-"
        lines.append(
            f"| `{_fmt_time(act.cast_at_s)}` | `{_fmt_time(act.event_time_s)}` ({act.event_name}) | "
            f"{act.skill_name} | {aid} | {act.target_scope} | {kind} |"
        )

    md = "\n".join(lines) + "\n"
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(md, encoding="utf-8")
    return md


def export_csv(plan: OptimizationPlan, output_path: str | Path | None = None) -> str:
    """Export scheduled actions as a UTF-8 CSV timeline sheet."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["cast_at_s", "cast_time_mmss", "event_time_s", "event_id", "event_name", "skill_id", "skill_name", "action_id", "target_scope", "is_gcd", "weave_cost"]
    )
    for act in sorted(plan.actions, key=lambda a: a.cast_at_s):
        writer.writerow(
            [
                act.cast_at_s,
                _fmt_time(act.cast_at_s),
                act.event_time_s,
                act.event_id,
                act.event_name,
                act.skill_id,
                act.skill_name,
                act.action_id if act.action_id is not None else "",
                act.target_scope,
                act.is_gcd,
                act.weave_cost,
            ]
        )
    text = buf.getvalue()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8-sig")
    return text


def export_json(plan: OptimizationPlan, output_path: str | Path | None = None) -> str:
    """Export the full optimization plan as formatted JSON."""
    text = json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
    return text

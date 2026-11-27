"""Real-log sheet audit and verification module for XIVHealWorkSolver.

Audits any generated healwork sheet against:
  1. Strict 100.0% worst-case (+5% damage roll x 12 server-tick phases) survival
  2. Zero dropped/illegal actions (CD, resource, targetable boss, movement cast-time)
  3. Top tightest mechanics & HP safety floor
  4. Side-by-side per-skill cast count & economy comparison against real FFLogs clears
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xiv_healwork_solver.models import EncounterTimeline, OptimizationPlan, PlayerKit
from xiv_healwork_solver.sim.engine import replay_plan
from xiv_healwork_solver.sim.safety import StressTestReport, canonicalize_plan_actions, stress_test_plan


@dataclass
class AuditFinding:
    severity: str  # "FAIL" | "WARN" | "INFO"
    category: str
    message: str


@dataclass
class TightestEventRecord:
    time_s: float
    name: str
    target_scope: str
    hp_party_pct: float
    hp_mt_pct: float
    hp_st_pct: float
    min_hp_pct: float
    active_player_skills: list[str]


@dataclass
class SkillComparisonRow:
    skill_id: str
    skill_name: str
    is_gcd: bool
    potency_loss_base: float
    our_count: int
    ref_avg: float | None = None
    ref_min: int | None = None
    ref_max: int | None = None
    delta: float | None = None


@dataclass
class SheetAuditReport:
    encounter_name: str
    kit_label: str
    item_level: int | None
    gcd_s: float
    max_hp_party: float
    max_hp_tank: float
    heal_scale: float
    verdict: str  # "PASS" | "WARN" | "FAIL"
    stress_105: StressTestReport
    nominal_min_hp_pct: dict[str, float]
    total_scheduled_actions: int
    findings: list[AuditFinding] = field(default_factory=list)
    tightest_events: list[TightestEventRecord] = field(default_factory=list)
    skill_comparison: list[SkillComparisonRow] = field(default_factory=list)
    reference_clear_labels: list[str] = field(default_factory=list)
    reference_avg_gcd_heals: float | None = None
    reference_avg_total_casts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "encounter_name": self.encounter_name,
            "kit_label": self.kit_label,
            "item_level": self.item_level,
            "gcd_s": self.gcd_s,
            "max_hp_party": self.max_hp_party,
            "max_hp_tank": self.max_hp_tank,
            "heal_scale": self.heal_scale,
            "verdict": self.verdict,
            "stress_105": self.stress_105.to_dict(),
            "nominal_min_hp_pct": self.nominal_min_hp_pct,
            "total_scheduled_actions": self.total_scheduled_actions,
            "findings": [f.__dict__ for f in self.findings],
            "tightest_events": [e.__dict__ for e in self.tightest_events],
            "skill_comparison": [r.__dict__ for r in self.skill_comparison],
            "reference_clear_labels": self.reference_clear_labels,
            "reference_avg_gcd_heals": self.reference_avg_gcd_heals,
            "reference_avg_total_casts": self.reference_avg_total_casts,
        }


def _extract_skill_counts(actions: list[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for a in actions:
        sid = a.skill_id if hasattr(a, "skill_id") else str(a.get("skill_id", ""))
        if sid:
            counts[sid] += 1
    return counts


def audit_healwork_sheet(
    kit: PlayerKit,
    timeline: EncounterTimeline,
    plan: OptimizationPlan,
    reference_clear_casts: list[list[dict[str, Any]]] | None = None,
    reference_clear_labels: list[str] | None = None,
    damage_roll: float = 1.05,
    n_phases: int = 12,
    min_safe_hp_warn_pct: float = 5.0,
) -> SheetAuditReport:
    """Run a comprehensive real-raid audit on a healwork plan and compare against real FFLogs clears."""
    skills_by_id = {sk.id: sk for sk in kit.skills}
    raw_actions = sorted(plan.actions, key=lambda a: (a.cast_at_s, 0 if a.is_gcd else 1))

    # Replay the exact original plan (do NOT silently strip unknown skills or rewrite invalid targets before auditing!)
    stress = stress_test_plan(kit, timeline, plan, damage_roll=damage_roll, n_phases=n_phases)
    nominal = replay_plan(kit, timeline, plan, deterministic=True, fixed_tick_phase=1.5, forced_damage_roll=1.00)
    worst_rep = replay_plan(
        kit, timeline, plan, deterministic=True, fixed_tick_phase=0.0, forced_damage_roll=damage_roll
    )

    findings: list[AuditFinding] = []

    # 1. Strict 100% survival check
    if stress.survival_rate_pct < 100.0:
        findings.append(
            AuditFinding(
                severity="FAIL",
                category="survival",
                message=(
                    f"最悪+{int(round((damage_roll - 1.0) * 100))}%乱数で生存率 {stress.survival_rate_pct:.1f}% "
                    f"(初死亡 {stress.worst_failure_time_s}s / {stress.worst_failure_channel}, 不足 {stress.worst_deficit_hp:,.0f} HP)"
                ),
            )
        )

    # 2. Zero dropped / unexecutable actions check
    if stress.max_dropped_actions > 0:
        findings.append(
            AuditFinding(
                severity="FAIL",
                category="legality",
                message=f"リキャスト・リソース・ターゲット不可・未知スキル等で不発(dropped)になったスキルが {stress.max_dropped_actions} 件あります。",
            )
        )

    # 3. Target channel & cast-time movement window & tight animation-lock checks (on ORIGINAL rows!)
    prev_ogcd_t: float | None = None
    for act in raw_actions:
        sk = skills_by_id.get(act.skill_id)
        if sk is None:
            findings.append(AuditFinding("FAIL", "unknown_skill", f"{act.cast_at_s:.2f}s: 未知スキル {act.skill_id}"))
            continue
        if sk.target_scope == "single_ally" and act.target_channel not in ("mt", "st"):
            findings.append(
                AuditFinding("WARN", "target_channel", f"{act.cast_at_s:.2f}s {sk.name}: 単体スキルですが対象が {act.target_channel}")
            )
        if sk.cast_time_s > 0.0:
            effective_lock_end = act.cast_at_s + max(0.0, sk.cast_time_s - sk.slidecast_window_s)
            for mw in timeline.movement_windows:
                if act.cast_at_s < mw.end_s and effective_lock_end > mw.start_s:
                    findings.append(
                        AuditFinding(
                            "WARN",
                            "movement_cast",
                            f"{act.cast_at_s:.2f}s {sk.name} (詠唱{sk.cast_time_s:.1f}s) が移動ギミック区間 {mw.start_s:.1f}-{mw.end_s:.1f}s ({mw.name}) と重複",
                        )
                    )
        if not sk.is_gcd:
            if prev_ogcd_t is not None and 0.0 <= act.cast_at_s - prev_ogcd_t < sk.animation_lock_s - 0.05:
                findings.append(
                    AuditFinding(
                        "INFO",
                        "ogcd_spacing",
                        f"{prev_ogcd_t:.2f}s -> {act.cast_at_s:.2f}s ({sk.name}): oGCD間隔 {act.cast_at_s - prev_ogcd_t:.2f}s (実行時に硬直分だけ自動シフト)",
                    )
                )
            prev_ogcd_t = act.cast_at_s

    # 4. HP safety margin check
    worst_hp_pct = min(stress.worst_min_hp_ratio.values()) * 100.0
    if stress.survival_rate_pct >= 100.0 and worst_hp_pct < min_safe_hp_warn_pct:
        findings.append(
            AuditFinding(
                severity="WARN",
                category="hp_margin",
                message=(
                    f"最悪+5%乱数時の最低HP残量が {worst_hp_pct:.1f}% と非常にタイトです "
                    f"(PT={stress.worst_min_hp_ratio.get('party', 0)*100:.1f}%, "
                    f"MT={stress.worst_min_hp_ratio.get('mt', 0)*100:.1f}%, "
                    f"ST={stress.worst_min_hp_ratio.get('st', 0)*100:.1f}%)。実戦余裕が欲しい場合は min_hp_margin_ratio=0.05〜0.10 を推奨。"
                ),
            )
        )

    # 5. Overheal sanity check
    if stress.mean_overheal_pct > 40.0:
        findings.append(
            AuditFinding(
                severity="WARN",
                category="overheal",
                message=f"オーバーヒール率が {stress.mean_overheal_pct:.1f}% と高めです。過剰ヒール削減の余地があります。",
            )
        )

    # 6. Top-6 tightest mechanics under worst-case roll (ranking by lowest HP among ALL affected channels!)
    tightest: list[TightestEventRecord] = []
    for snap in worst_rep.event_snapshots:
        hp_p = (snap.hp_after.get("party", kit.max_hp_party) / kit.max_hp_party) * 100.0
        hp_m = (snap.hp_after.get("mt", kit.max_hp_tank) / kit.max_hp_tank) * 100.0
        hp_s = (snap.hp_after.get("st", kit.max_hp_tank) / kit.max_hp_tank) * 100.0
        relevant_min = min(hp_p, hp_m, hp_s) if snap.target_scope == "party" else (
            hp_m if snap.target_scope == "mt" else (hp_s if snap.target_scope == "st" else min(hp_m, hp_s))
        )
        active_names: list[str] = []
        for act in raw_actions:
            sk = skills_by_id.get(act.skill_id)
            if sk is None or sk.mp_regen_per_tick > 0:
                continue
            dur = max(sk.duration_s, sk.effective_shield_duration() if (sk.shield_value > 0 or sk.shield_max_hp_ratio > 0) else 0.0, 2.5)
            if act.cast_at_s <= snap.time_s + 0.2 and act.cast_at_s + dur >= snap.time_s - 0.5:
                active_names.append(sk.name)
        tightest.append(
            TightestEventRecord(
                time_s=round(snap.time_s, 1),
                name=snap.name,
                target_scope=snap.target_scope,
                hp_party_pct=round(hp_p, 1),
                hp_mt_pct=round(hp_m, 1),
                hp_st_pct=round(hp_s, 1),
                min_hp_pct=round(relevant_min, 1),
                active_player_skills=list(dict.fromkeys(active_names)),
            )
        )
    tightest.sort(key=lambda r: r.min_hp_pct)
    tightest = tightest[:6]

    # 7. Side-by-side comparison against real FFLogs clear cast lists (preserving empty clears as 0-count samples!)
    our_counts = _extract_skill_counts(raw_actions)
    ref_counters = [_extract_skill_counts(rc) for rc in (reference_clear_casts or []) if rc is not None]
    rows: list[SkillComparisonRow] = []
    for sk in kit.skills:
        oc = int(our_counts.get(sk.id, 0))
        if ref_counters:
            vals = [int(c.get(sk.id, 0)) for c in ref_counters]
            r_avg = round(sum(vals) / len(vals), 1)
            r_min = min(vals)
            r_max = max(vals)
            delta = round(oc - r_avg, 1)
        else:
            r_avg = r_min = r_max = delta = None
        if oc == 0 and (r_max is None or r_max == 0):
            continue
        rows.append(
            SkillComparisonRow(
                skill_id=sk.id,
                skill_name=sk.name,
                is_gcd=sk.is_gcd,
                potency_loss_base=kit.expected_skill_potency_loss(sk, 0.0, False),
                our_count=oc,
                ref_avg=r_avg,
                ref_min=r_min,
                ref_max=r_max,
                delta=delta,
            )
        )

    ref_avg_gcd: float | None = None
    ref_avg_tot: float | None = None
    if ref_counters:
        # Only count healing/shielding GCDs (exclude Esuna / cleanse GCDs)
        gcd_heal_ids = {
            sk.id for sk in kit.skills
            if sk.is_gcd and not sk.cleanses_debuff and (sk.heal_instant > 0 or sk.heal_hot_total > 0 or sk.shield_value > 0 or sk.shield_max_hp_ratio > 0)
        }
        ref_avg_gcd = round(
            sum(sum(v for k, v in c.items() if k in gcd_heal_ids) for c in ref_counters) / len(ref_counters), 1
        )
        ref_avg_tot = round(sum(sum(c.values()) for c in ref_counters) / len(ref_counters), 1)
        our_gcd_cnt = sum(v for k, v in our_counts.items() if k in gcd_heal_ids)
        if our_gcd_cnt > ref_avg_gcd + 4.0:
            findings.append(
                AuditFinding(
                    "WARN",
                    "gcd_count_vs_ref",
                    f"GCDヒール回数 ({our_gcd_cnt}回) が既存クリアログ平均 ({ref_avg_gcd:.1f}回) より多めです。",
                )
            )

    has_fail = any(f.severity == "FAIL" for f in findings)
    has_warn = any(f.severity == "WARN" for f in findings)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    return SheetAuditReport(
        encounter_name=timeline.encounter_name,
        kit_label=kit.label,
        item_level=kit.gear_stats.item_level,
        gcd_s=kit.gcd_s,
        max_hp_party=kit.max_hp_party,
        max_hp_tank=kit.max_hp_tank,
        heal_scale=kit.gear_stats.heal_scale_multiplier,
        verdict=verdict,
        stress_105=stress,
        nominal_min_hp_pct={k: round(v * 100.0, 1) for k, v in nominal.min_hp_ratio.items()},
        total_scheduled_actions=len(raw_actions),
        findings=findings,
        tightest_events=tightest,
        skill_comparison=rows,
        reference_clear_labels=list(reference_clear_labels or []),
        reference_avg_gcd_heals=ref_avg_gcd,
        reference_avg_total_casts=ref_avg_tot,
    )


def render_audit_markdown(report: SheetAuditReport, output_path: str | Path | None = None) -> str:
    """Render a concise Japanese Markdown audit report."""
    s = report.stress_105
    ilvl_str = f"iLvl {report.item_level}" if report.item_level is not None else "custom"
    lines = [
        f"# シート検品レポート: {report.encounter_name} ({report.verdict})",
        "",
        f"- **ジョブ/構成**: {report.kit_label}",
        f"- **装備パラメーター**: `{ilvl_str}` | `GCD {report.gcd_s:.2f}s` | `PT HP {report.max_hp_party:,.0f}` / `Tank HP {report.max_hp_tank:,.0f}` | `Heal倍率 {report.heal_scale:.3f}x`",
        f"- **最悪+5%乱数 × {s.total_phases}位相 生存率**: **{s.survival_rate_pct:.1f}% ({s.survived_phases}/{s.total_phases})**",
        f"- **不発(dropped)スキル数**: `{s.max_dropped_actions}` 件 / 予定スキル総数 `{report.total_scheduled_actions}` 件",
        f"- **火力期待値**: 正味 `{s.mean_expected_potency:,.1f}p` (ロス `{s.mean_potency_lost:,.1f}p` / **DPS効率 `{s.mean_dps_efficiency_pct:.2f}%`**) | 攻撃GCD `{s.mean_filler_gcds:.1f}` | GCDヒール `{s.mean_gcd_heals:.1f}` | OH `{s.mean_overheal_pct:.1f}%`",
        f"- **最低HP残量 (+5%最悪)**: `PT {s.worst_min_hp_ratio.get('party', 0)*100:.1f}%` / `MT {s.worst_min_hp_ratio.get('mt', 0)*100:.1f}%` / `ST {s.worst_min_hp_ratio.get('st', 0)*100:.1f}%`",
        f"- **最低HP残量 (通常1.00x)**: `PT {report.nominal_min_hp_pct.get('party', 0):.1f}%` / `MT {report.nominal_min_hp_pct.get('mt', 0):.1f}%` / `ST {report.nominal_min_hp_pct.get('st', 0):.1f}%`",
        "",
        "## 検品チェック結果",
        "",
    ]
    if not report.findings:
        lines.append("- **PASS**: 重大な問題・不発・移動詠唱衝突・過剰ヒールは検出されませんでした。")
    else:
        for f in report.findings:
            lines.append(f"- **[{f.severity}]** (`{f.category}`) {f.message}")

    lines.extend(
        [
            "",
            "## 最も厳しい攻撃ワースト6 (+5%乱数時)",
            "",
            "| 時刻 | ギミック / 攻撃 | 対象 | 最低HP% | PT% | MT% | ST% | 有効な自スキル |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for ev in report.tightest_events:
        sk_s = ", ".join(ev.active_player_skills) if ev.active_player_skills else "-"
        lines.append(
            f"| {ev.time_s:5.1f}s | {ev.name} | `{ev.target_scope}` | **{ev.min_hp_pct:.1f}%** | "
            f"{ev.hp_party_pct:.1f}% | {ev.hp_mt_pct:.1f}% | {ev.hp_st_pct:.1f}% | {sk_s} |"
        )

    if report.skill_comparison:
        ref_note = (
            f" (比較対象クリアログ {len(report.reference_clear_labels)}件: {', '.join(report.reference_clear_labels)})"
            if report.reference_clear_labels
            else ""
        )
        lines.extend(
            [
                "",
                f"## スキル使用回数比較（生成シート vs 既存クリアログ）{ref_note}",
                "",
                "| スキル | 種別 | 基本ロス | 生成シート | 既存クリア平均 | 既存min–max | 差分 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report.skill_comparison:
            kind = "GCD" if row.is_gcd else "oGCD"
            r_avg = f"{row.ref_avg:.1f}" if row.ref_avg is not None else "-"
            r_mm = f"{row.ref_min}–{row.ref_max}" if row.ref_min is not None else "-"
            dlt = f"{row.delta:+.1f}" if row.delta is not None else "-"
            lines.append(
                f"| {row.skill_name} | {kind} | {row.potency_loss_base:.0f}p | **{row.our_count}** | {r_avg} | {r_mm} | {dlt} |"
            )

    md = "\n".join(lines) + "\n"
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(md, encoding="utf-8")
    return md

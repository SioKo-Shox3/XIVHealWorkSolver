"""FFLogs training-data curation filter.

Rejects degenerate parse logs before partner modeling or imitation learning:
- Chad-healing / co-healer dumping (extreme effective-healing imbalance between the two healers)
- Low party-mitigation participation (ignoring defensive cooldowns)
- High-death / zombie clears
- Extreme speedkill outliers that skip late-fight mechanics
Ranks accepted clears by Combined Healer rDPS + balanced healing + mitigation coverage.
"""
from __future__ import annotations

from statistics import median
from typing import Any

from xiv_healwork_solver.models import LogFilterConfig, LogQualityReport


def evaluate_fight_quality(
    candidate: dict[str, Any],
    config: LogFilterConfig | None = None,
    median_duration_s: float | None = None,
) -> LogQualityReport:
    """Audit a single summarized FFLogs fight candidate against curation rules.

    Expected `candidate` keys:
      - report_code: str
      - fight_id: int
      - duration_s: float
      - deaths: int
      - healers: list[dict] with {name, rdps, effective_healing, overheal_pct}
      - party_mit_casts: int (total defensive party-mitigation casts by target/healers)
    """
    cfg = config or LogFilterConfig()
    reasons: list[str] = []

    duration_s = float(candidate["duration_s"])
    deaths = int(candidate.get("deaths", 0))
    healers: list[dict[str, Any]] = list(candidate.get("healers") or [])
    mit_casts = int(candidate.get("party_mit_casts", 0))
    mit_per_min = (mit_casts / max(1.0, duration_s / 60.0)) if duration_s > 0 else 0.0

    if deaths > cfg.max_deaths:
        reasons.append(f"deaths={deaths} > max_deaths={cfg.max_deaths}")

    if median_duration_s is not None and median_duration_s > 0:
        lo = median_duration_s * (1.0 - cfg.duration_tolerance_ratio)
        hi = median_duration_s * (1.0 + cfg.duration_tolerance_ratio)
        if duration_s < lo:
            reasons.append(f"speedkill_outlier ({duration_s:.1f}s < {lo:.1f}s)")
        elif duration_s > hi:
            reasons.append(f"slow_clear_outlier ({duration_s:.1f}s > {hi:.1f}s)")

    combined_rdps = sum(float(h.get("rdps", 0.0)) for h in healers)
    total_eff_heal = sum(max(0.0, float(h.get("effective_healing", 0.0))) for h in healers)
    shares: dict[str, float] = {}

    if len(healers) >= 2:
        # Always enforce overheal checks regardless of total_eff_heal
        for h in healers:
            name = str(h.get("name", "unknown"))
            overheal = float(h.get("overheal_pct", 0.0))
            if overheal > cfg.max_overheal_pct:
                reasons.append(f"excessive_overheal ({name} {overheal:.1f}% > {cfg.max_overheal_pct:.1f}%)")
        if total_eff_heal <= 0:
            reasons.append("zero_effective_healing")
        else:
            for h in healers:
                name = str(h.get("name", "unknown"))
                share = max(0.0, float(h.get("effective_healing", 0.0))) / total_eff_heal
                shares[name] = round(share, 3)
                if share < cfg.min_healer_effective_share:
                    reasons.append(f"chad_healing_suspected ({name} share={share:.2%} < {cfg.min_healer_effective_share:.0%})")
                elif share > cfg.max_healer_effective_share:
                    reasons.append(f"overburdened_cohealer ({name} share={share:.2%} > {cfg.max_healer_effective_share:.0%})")
    else:
        reasons.append(f"incomplete_healer_pair (count={len(healers)})")

    if mit_per_min < cfg.min_party_mit_casts_per_min:
        reasons.append(f"low_mit_activity ({mit_per_min:.2f}/min < {cfg.min_party_mit_casts_per_min:.2f}/min)")

    # Higher score = higher combined healer rDPS, closer to 50/50 heal balance, strong mitigation cadence
    balance_bonus = 1.0
    if shares:
        max_dev = max(abs(s - 0.5) for s in shares.values())
        balance_bonus = max(0.5, 1.0 - max_dev)
    quality_score = round((combined_rdps / 1000.0) * balance_bonus + min(4.0, mit_per_min) * 2.0, 2)

    return LogQualityReport(
        report_code=str(candidate["report_code"]),
        fight_id=int(candidate["fight_id"]),
        accepted=(len(reasons) == 0),
        reasons=reasons,
        duration_s=round(duration_s, 1),
        deaths=deaths,
        combined_healer_rdps=round(combined_rdps, 1),
        healer_effective_shares=shares,
        party_mit_casts_per_min=round(mit_per_min, 2),
        quality_score=quality_score,
    )


def curate_fight_candidates(
    candidates: list[dict[str, Any]],
    config: LogFilterConfig | None = None,
) -> tuple[list[LogQualityReport], list[LogQualityReport]]:
    """Filter and rank a batch of FFLogs candidates into (accepted_sorted, rejected)."""
    cfg = config or LogFilterConfig()
    durations = [float(c["duration_s"]) for c in candidates if float(c.get("duration_s", 0)) > 0]
    med_dur = median(durations) if len(durations) >= 3 else None

    accepted: list[LogQualityReport] = []
    rejected: list[LogQualityReport] = []
    for c in candidates:
        rep = evaluate_fight_quality(c, config=cfg, median_duration_s=med_dur)
        if rep.accepted:
            accepted.append(rep)
        else:
            rejected.append(rep)

    accepted.sort(key=lambda r: r.quality_score, reverse=True)
    return accepted, rejected

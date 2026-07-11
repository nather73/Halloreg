"""HalloReg.ipd.metrics — 가설 검증 지표."""

from .exploitability import (
    exploitability, payoff_gap, cc_rate, coop_rate,
    cooperation_restoration, lambda_metrics, defense_metrics,
    defense_specificity, welch_t, aggregate,
    first_defection_round, payoff_growth,
)

__all__ = [
    "exploitability", "payoff_gap", "cc_rate", "coop_rate",
    "cooperation_restoration", "lambda_metrics", "defense_metrics",
    "defense_specificity", "welch_t", "aggregate",
    "first_defection_round", "payoff_growth",
]

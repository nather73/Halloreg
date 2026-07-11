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

from .stats import (  # noqa: F401  (보완안 §1 통계 인프라)
    effect_size, effect_size_paired, perm_test, one_sample_perm,
    holm, bh_fdr, wilson_ci, prop_perm_test, hazard_perm_test,
    boot_ci, boot_mean_ci, ols_boot, slope_boot, fmt_es,
)

"""
AIF_IPD.ipd.metrics
===================

가설검증용 통계 인프라 (순열검정·효과크기·다중비교 보정).
"""

from .stats import (
    hedges_g, effect_size, effect_size_paired,
    perm_test, one_sample_perm,
    holm, bh_fdr,
    boot_ci, boot_mean_ci, wilson_ci, corr_boot, slope_boot,
    fmt_es, fmt_p,
)

__all__ = [
    "hedges_g", "effect_size", "effect_size_paired",
    "perm_test", "one_sample_perm", "holm", "bh_fdr",
    "boot_ci", "boot_mean_ci", "wilson_ci", "corr_boot", "slope_boot",
    "fmt_es", "fmt_p",
]

"""
AIF_IPD.ipd.metrics
===================

Statistical infrastructure for hypothesis testing (permutation
tests, effect sizes, multiple-comparison corrections).
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

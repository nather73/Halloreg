"""
ipd.metrics.stats
=================

Shared statistical infrastructure for hypothesis testing.

Every hypothesis uses this module only, so effect-size reporting,
distribution-free testing and multiple-comparison correction are
enforced structurally.

Design principles
-----------------
* p values are always **permutation/bootstrap** based: normal
  approximations are unreliable for bounded metrics with common
  floor/ceiling effects (cooperation rates, survival rates).
* Every function is pure numpy with its own seeded RNG, hence
  reproducible.
* Returns are dicts — JSON serialisation and log formatting share
  one structure.
* Holm for the confirmatory family, BH-FDR for exploratory tests.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

_DEF_BOOT = 4_000
_DEF_PERM = 10_000


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ================================================================ Effect sizes
def hedges_g(a, b) -> float:
    """Hedges' g of two independent samples (small-sample-corrected
    Cohen's d)."""
    a, b = _arr(a), _arr(b)
    na, nb = len(a), len(b)
    va = a.var(ddof=1) if na > 1 else 0.0
    vb = b.var(ddof=1) if nb > 1 else 0.0
    sp2 = ((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)
    if sp2 <= 0:
        return 0.0
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)      # small-sample factor
    return float(J * d)


def effect_size(a, b, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """Hedges' g with a bootstrap 95% CI (two independent
    samples)."""
    a, b = _arr(a), _arr(b)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    gs = np.array([hedges_g(a[ia[k]], b[ib[k]]) for k in range(n_boot)])
    gs = gs[np.isfinite(gs)]
    lo, hi = (np.percentile(gs, [2.5, 97.5]) if gs.size else (np.nan, np.nan))
    return {"g": hedges_g(a, b), "ci": [float(lo), float(hi)],
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "n_a": int(len(a)), "n_b": int(len(b))}


def effect_size_paired(d, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """Standardised paired effect size dz = mean(d)/sd(d) with a
    bootstrap CI."""
    d = _arr(d)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    dz = float(d.mean() / sd) if sd > 0 else 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx]
    sds = boots.std(ddof=1, axis=1)
    ok = sds > 0
    dzs = np.where(ok, boots.mean(axis=1) / np.where(ok, sds, 1.0), np.nan)
    dzs = dzs[np.isfinite(dzs)]
    # Bootstrap resamples with near-zero SDs make dz diverge (common
    # for small n); clip so CIs stay meaningful — |dz| = 50 already
    # means "complete separation", so nothing interpretive is lost.
    dzs = np.clip(dzs, -50.0, 50.0)
    lo, hi = (np.percentile(dzs, [2.5, 97.5]) if dzs.size else (np.nan, np.nan))
    return {"dz": dz, "ci": [float(lo), float(hi)],
            "mean_diff": float(d.mean()),
            "diff_ci": boot_mean_ci(d, seed=seed + 1)["ci"],
            "n": int(len(d))}


# ================================================================ Permutation tests
def perm_test(a, b, paired: bool = False, n_perm: int = _DEF_PERM,
              seed: int = 0, alternative: str = "two-sided") -> Dict:
    """
    Distribution-free test.

    paired=False : pool both samples and randomly reassign labels
                   (two independent samples).
    paired=True  : randomly **flip the signs** of per-seed
                   differences (exchangeability); far more powerful
                   in paired designs.
    alternative in {"two-sided", "greater", "less"} — a vs b.
    """
    a, b = _arr(a), _arr(b)
    rng = np.random.default_rng(seed)

    if paired:
        d = a - b
        d = d[np.isfinite(d)]
        if len(d) == 0:
            # No valid samples — untestable. Return an explicit
            # "no information" (p = 1) instead of an empty-slice
            # warning.
            return {"observed": float("nan"), "p": 1.0, "n_perm": int(n_perm),
                    "alternative": alternative}
        obs = float(d.mean())
        signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
        null = (signs * d).mean(axis=1)
    else:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            return {"observed": float("nan"), "p": 1.0, "n_perm": int(n_perm),
                    "alternative": alternative}
        obs = float(a.mean() - b.mean())
        pool = np.concatenate([a, b])
        na = len(a)
        null = np.empty(n_perm)
        for k in range(n_perm):
            p = rng.permutation(pool)
            null[k] = p[:na].mean() - p[na:].mean()

    # +1 correction (Phipson & Smyth): include the observation so p
    # is never exactly 0.
    if alternative == "greater":
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"observed": obs, "p": float(p), "n_perm": int(n_perm),
            "alternative": alternative}


def one_sample_perm(x, mu0: float = 0.0, n_perm: int = _DEF_PERM,
                    seed: int = 0, alternative: str = "two-sided") -> Dict:
    """One-sample sign-flip test against the constant mu0."""
    x = _arr(x)
    x = x[np.isfinite(x)]
    return perm_test(x, np.full_like(x, mu0), paired=True,
                     n_perm=n_perm, seed=seed, alternative=alternative)


# ================================================================ Multiple comparisons
def holm(pvals: Sequence[float]) -> List[float]:
    """
    Holm-Bonferroni step-down correction (FWER control) for the
    **confirmatory family**. Returns adjusted p values in input
    order.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(running, 1.0)
    return [float(v) for v in adj]


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR correction for **exploratory tests**.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = m * p[i] / (rank + 1)
        running = min(running, val)          # reverse monotonicity
        adj[i] = min(running, 1.0)
    return [float(v) for v in adj]


# ================================================================ Bootstrap
def boot_ci(x, stat: Callable = np.mean, n_boot: int = _DEF_BOOT,
            seed: int = 0) -> Dict:
    """Bootstrap percentile 95% CI of an arbitrary statistic."""
    x = _arr(x)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"stat": float(stat(x)), "ci": [float(lo), float(hi)],
            "n": int(len(x))}


def boot_mean_ci(x, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """Bootstrap CI of the mean (bar-chart error bars); vectorised
    path."""
    x = _arr(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": np.nan, "ci": [np.nan, np.nan], "n": 0}
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci": [float(lo), float(hi)],
            "n": int(len(x))}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Dict:
    """Wilson 95% CI for a binomial proportion (far more accurate
    than the normal approximation)."""
    if n == 0:
        return {"p": np.nan, "ci": [np.nan, np.nan], "k": 0, "n": 0}
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"p": float(p), "ci": [float(max(0.0, ctr - half)),
                                  float(min(1.0, ctr + half))],
            "k": int(k), "n": int(n)}


def corr_boot(x, y, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """
    Pearson correlation with a bootstrap CI and a permutation p —
    the primary metric of parameter recovery (H6) and lambda recovery
    (H1A).
    """
    x, y = _arr(x), _arr(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return {"r": np.nan, "ci": [np.nan, np.nan], "p": np.nan, "n": n}
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rs = []
    for i in idx:
        xa, ya = x[i], y[i]
        if xa.std() > 0 and ya.std() > 0:
            rs.append(np.corrcoef(xa, ya)[0, 1])
    rs = np.asarray(rs)
    lo, hi = (np.percentile(rs, [2.5, 97.5]) if rs.size else (np.nan, np.nan))
    # Permutation test: shuffle y to build the null
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1]
                     for _ in range(1000)])
    p = (np.sum(np.abs(null) >= abs(r)) + 1) / (1000 + 1)
    return {"r": r, "ci": [float(lo), float(hi)], "p": float(p), "n": n}


def slope_boot(x, y, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """Simple linear-regression slope with a bootstrap CI (dose-
    response analyses)."""
    x, y = _arr(x), _arr(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or x.std() == 0:
        return {"slope": np.nan, "ci": [np.nan, np.nan], "n": n}
    b = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for i in rng.integers(0, n, size=(n_boot, n)):
        if x[i].std() > 0:
            bs.append(np.polyfit(x[i], y[i], 1)[0])
    bs = np.asarray(bs)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs.size else (np.nan, np.nan))
    return {"slope": b, "ci": [float(lo), float(hi)], "n": n}


# ================================================================ Formatting
def fmt_es(es: Dict, key: str = "g") -> str:
    """Effect-size dict -> log string."""
    v = es.get(key, np.nan)
    ci = es.get("ci", [np.nan, np.nan])
    return f"{key}={v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def fmt_p(p: float) -> str:
    """p value -> log string (upper-bound notation when tiny)."""
    if not np.isfinite(p):
        return "p=NA"
    return "p<1e-4" if p < 1e-4 else f"p={p:.4g}"

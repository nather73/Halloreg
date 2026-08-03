"""
ipd.metrics.stats
=================

가설검증 공통 통계 인프라.

모든 가설이 이 모듈만 쓰도록 하여, 효과크기 보고·분포무가정 검정·다중비교 보정이
구조적으로 강제되게 한다.

설계 원칙
---------
* p 값은 항상 **순열/부트스트랩** 기반이다. 협력률·생존율처럼 유계이고 바닥/천장
  효과가 흔한 지표에서 정규 근사는 신뢰할 수 없다.
* 모든 함수는 순수 numpy 이며 자체 RNG(seed 인자)로 재현 가능하다.
* 반환은 dict — JSON 직렬화와 로그 포맷이 같은 구조를 공유한다.
* 확증(confirmatory) 가설군에는 Holm, 탐색(exploratory)에는 BH-FDR 을 적용한다.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

_DEF_BOOT = 4_000
_DEF_PERM = 10_000


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ================================================================ 효과크기
def hedges_g(a, b) -> float:
    """두 독립 표본의 Hedges' g (소표본 편향 보정된 Cohen's d)."""
    a, b = _arr(a), _arr(b)
    na, nb = len(a), len(b)
    va = a.var(ddof=1) if na > 1 else 0.0
    vb = b.var(ddof=1) if nb > 1 else 0.0
    sp2 = ((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)
    if sp2 <= 0:
        return 0.0
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)      # 소표본 보정계수
    return float(J * d)


def effect_size(a, b, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """Hedges' g 와 부트스트랩 95% CI (두 독립 표본)."""
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
    """짝지은 차분 d 의 표준화 효과크기 dz = mean(d)/sd(d) + 부트스트랩 CI."""
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
    # 부트스트랩 재표집에서 표준편차가 우연히 0 에 가까워지면 dz 가 발산한다
    # (표본이 작을수록 흔하다). CI 표기가 무의미해지지 않도록 유계로 자른다 —
    # |dz| = 50 은 실질적으로 '완전 분리' 이므로 해석상 손실이 없다.
    dzs = np.clip(dzs, -50.0, 50.0)
    lo, hi = (np.percentile(dzs, [2.5, 97.5]) if dzs.size else (np.nan, np.nan))
    return {"dz": dz, "ci": [float(lo), float(hi)],
            "mean_diff": float(d.mean()),
            "diff_ci": boot_mean_ci(d, seed=seed + 1)["ci"],
            "n": int(len(d))}


# ================================================================ 순열 검정
def perm_test(a, b, paired: bool = False, n_perm: int = _DEF_PERM,
              seed: int = 0, alternative: str = "two-sided") -> Dict:
    """
    분포무가정 검정.

    paired=False : 두 표본을 합친 뒤 라벨을 무작위 재배정 (독립 두 표본).
    paired=True  : 시드별 차분의 **부호를 무작위로 뒤집는다** (교환가능성 가정).
                   짝지은 설계에서는 이쪽이 검정력이 훨씬 높다.
    alternative ∈ {"two-sided", "greater", "less"}  — a 가 b 보다 크다/작다.
    """
    a, b = _arr(a), _arr(b)
    rng = np.random.default_rng(seed)

    if paired:
        d = a - b
        d = d[np.isfinite(d)]
        if len(d) == 0:
            # 유효 표본이 전혀 없음 — 검정 불가. 빈 슬라이스 평균 경고를 내지
            # 않고 명시적으로 '정보 없음'(p=1)을 반환한다.
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

    # +1 보정(Phipson & Smyth): p 가 정확히 0 이 되지 않도록 관측치를 포함한다.
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
    """단일표본 부호뒤집기 검정 (상수 mu0 대비)."""
    x = _arr(x)
    x = x[np.isfinite(x)]
    return perm_test(x, np.full_like(x, mu0), paired=True,
                     n_perm=n_perm, seed=seed, alternative=alternative)


# ================================================================ 다중비교
def holm(pvals: Sequence[float]) -> List[float]:
    """
    Holm-Bonferroni 단계적 보정 (가족오류율 FWER 통제). **확증 가설군**에 쓴다.
    반환: 입력 순서에 대응하는 보정된 p 값.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)          # 단조성 강제
        adj[i] = min(running, 1.0)
    return [float(v) for v in adj]


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR 보정. **탐색 가설군**에 쓴다.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = m * p[i] / (rank + 1)
        running = min(running, val)          # 역방향 단조성
        adj[i] = min(running, 1.0)
    return [float(v) for v in adj]


# ================================================================ 부트스트랩
def boot_ci(x, stat: Callable = np.mean, n_boot: int = _DEF_BOOT,
            seed: int = 0) -> Dict:
    """임의 통계량의 부트스트랩 백분위 95% CI."""
    x = _arr(x)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"stat": float(stat(x)), "ci": [float(lo), float(hi)],
            "n": int(len(x))}


def boot_mean_ci(x, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """평균의 부트스트랩 CI (막대그림 오차막대용). 벡터화 경로."""
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
    """이항 비율의 Wilson 95% CI (정확도가 정규 근사보다 훨씬 낫다)."""
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
    Pearson 상관계수와 부트스트랩 CI + 순열 p 값.
    파라미터 복원(H6)·λ 복원(H1A)의 주 지표.
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
    # 순열 검정: y 를 무작위 재배열해 영분포 생성
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1]
                     for _ in range(1000)])
    p = (np.sum(np.abs(null) >= abs(r)) + 1) / (1000 + 1)
    return {"r": r, "ci": [float(lo), float(hi)], "p": float(p), "n": n}


def slope_boot(x, y, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """단순 선형회귀 기울기와 부트스트랩 CI (용량-반응 분석용)."""
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


# ================================================================ 포맷
def fmt_es(es: Dict, key: str = "g") -> str:
    """효과크기 dict → 로그 문자열."""
    v = es.get(key, np.nan)
    ci = es.get("ci", [np.nan, np.nan])
    return f"{key}={v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def fmt_p(p: float) -> str:
    """p 값 → 로그 문자열 (매우 작으면 상한 표기)."""
    if not np.isfinite(p):
        return "p=NA"
    return "p<1e-4" if p < 1e-4 else f"p={p:.4g}"

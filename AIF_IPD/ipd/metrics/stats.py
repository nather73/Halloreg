"""
stats.py — 가설검증 공통 통계 인프라 (보완안 §1).

모든 가설이 이 모듈만 쓰도록 하여 효과크기 보고·분포무가정 검정·다중비교
보정이 구조적으로 강제되게 한다.

제공 요소
---------
effect_size(a, b)            : Hedges' g + 부트스트랩 95% CI (두 표본)
effect_size_paired(d)        : 짝지은 차분의 표준화 효과크기 dz + CI
perm_test(a, b, paired=)     : 순열(비짝지음) / 부호뒤집기(짝지음) 검정
one_sample_perm(x, mu0)      : 단일표본 부호뒤집기 검정 (상수 대비)
holm(pvals) / bh_fdr(pvals)  : 가족오류율 / FDR 보정
wilson_ci(k, n)              : 비율의 Wilson 95% CI
prop_perm_test(k1,n1,k2,n2)  : 두 비율 차의 순열 검정 (베르누이 재표집)
hazard_perm_test(t1,c1,t2,c2): 우측절단 사건시각의 로그랭크형 순열 검정
boot_ci(x, stat)             : 임의 통계량의 부트스트랩 백분위 CI
boot_mean_ci(x)              : 평균의 부트스트랩 CI (막대그림 오차용)
ols_boot(X, y, cluster)      : OLS + 군집(셀) 부트스트랩 SE/CI + t·p
slope_boot(x, y)             : 원자료 회귀 기울기 + 부트스트랩 CI

설계 원칙
---------
* p 값은 항상 순열/부트스트랩 기반(분포 무가정) — 유계·바닥/천장 지표에 안전.
* 모든 함수는 순수 numpy, 자체 RNG(seed 인자)로 재현 가능.
* 반환은 dict — JSON 직렬화와 로그 포맷이 동일 구조를 공유.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_DEF_BOOT = 10_000
_DEF_PERM = 10_000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


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
        return 0.0 if np.isclose(a.mean(), b.mean()) else np.inf * np.sign(a.mean() - b.mean())
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)   # 소표본 보정
    return float(J * d)


def effect_size(a, b, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """Hedges' g 와 시드 재표집 부트스트랩 95% CI."""
    a, b = _arr(a), _arr(b)
    g = hedges_g(a, b)
    rng = _rng(seed)
    gs = np.empty(n_boot)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    for k in range(n_boot):
        gs[k] = hedges_g(a[ia[k]], b[ib[k]])
    gs = gs[np.isfinite(gs)]
    lo, hi = (np.percentile(gs, [2.5, 97.5]) if gs.size else (np.nan, np.nan))
    return {"g": g, "ci": [float(lo), float(hi)],
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "sd_a": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "sd_b": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
            "n_a": int(len(a)), "n_b": int(len(b))}


def effect_size_paired(d, n_boot: int = _DEF_BOOT, seed: int = 0) -> Dict:
    """짝지은 차분 d 의 표준화 효과크기 dz = mean(d)/sd(d) + 부트스트랩 CI."""
    d = _arr(d)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    dz = float(d.mean() / sd) if sd > 0 else (0.0 if np.isclose(d.mean(), 0) else np.inf * np.sign(d.mean()))
    rng = _rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx]
    sds = boots.std(ddof=1, axis=1)
    ok = sds > 0
    dzs = np.where(ok, boots.mean(axis=1) / np.where(ok, sds, 1.0), np.nan)
    dzs = dzs[np.isfinite(dzs)]
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
      paired=False : 라벨 순열 (독립 두 표본)
      paired=True  : 시드별 차분의 부호뒤집기 (a, b 길이 동일 필수)
    alternative ∈ {"two-sided", "greater", "less"}  (a 대 b 기준)
    """
    a, b = _arr(a), _arr(b)
    rng = _rng(seed)
    if paired:
        if len(a) != len(b):
            raise ValueError("paired 검정은 길이가 같아야 합니다")
        d = a - b
        obs = d.mean()
        signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
        null = (signs * d).mean(axis=1)
    else:
        pooled = np.concatenate([a, b])
        obs = a.mean() - b.mean()
        na = len(a)
        null = np.empty(n_perm)
        for k in range(n_perm):
            rng.shuffle(pooled)
            null[k] = pooled[:na].mean() - pooled[na:].mean()
    if alternative == "greater":
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"stat": float(obs), "p": float(p), "paired": bool(paired),
            "alternative": alternative, "n_perm": int(n_perm)}


def one_sample_perm(x, mu0: float = 0.0, n_perm: int = _DEF_PERM,
                    seed: int = 0, alternative: str = "two-sided") -> Dict:
    """단일표본 부호뒤집기 검정: E[x] = mu0 대비. (퇴화 두표본 검정의 대체)"""
    d = _arr(x) - float(mu0)
    rng = _rng(seed)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * np.abs(d)).mean(axis=1)   # 대칭 영가설 하 부호뒤집기
    if alternative == "greater":
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"stat": float(obs + mu0), "mu0": float(mu0), "p": float(p),
            "alternative": alternative, "n_perm": int(n_perm)}


# ================================================================ 다중비교
def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni 보정 (가족오류율). 입력 순서 보존해 반환."""
    p = _arr(pvals)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)          # 단조성 강제
        adj[idx] = min(running, 1.0)
    return [float(v) for v in adj]


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    """Benjamini–Hochberg FDR q값. 입력 순서 보존해 반환."""
    p = _arr(pvals)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        prev = min(prev, val)
        q[idx] = prev
    return [float(v) for v in q]


# ================================================================ 비율 / 절단
def wilson_ci(k: int, n: int, z: float = 1.96) -> Dict:
    """비율 k/n 의 Wilson 95% CI."""
    if n == 0:
        return {"p": np.nan, "ci": [np.nan, np.nan], "k": 0, "n": 0}
    ph = k / n
    den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return {"p": float(ph), "ci": [float(max(0, ctr - hw)), float(min(1, ctr + hw))],
            "k": int(k), "n": int(n)}


def prop_perm_test(k1: int, n1: int, k2: int, n2: int,
                   n_perm: int = _DEF_PERM, seed: int = 0,
                   alternative: str = "two-sided") -> Dict:
    """두 비율 차의 순열 검정 (관측 단위 라벨 순열)."""
    x = np.concatenate([np.ones(k1), np.zeros(n1 - k1),
                        np.ones(k2), np.zeros(n2 - k2)])
    obs = k1 / max(n1, 1) - k2 / max(n2, 1)
    rng = _rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(x)
        null[k] = x[:n1].mean() - x[n1:].mean()
    if alternative == "greater":
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"diff": float(obs), "p": float(p),
            "p1": wilson_ci(k1, n1), "p2": wilson_ci(k2, n2)}


def hazard_perm_test(times1, cens1, times2, cens2, horizon: int,
                     n_perm: int = 2000, seed: int = 0) -> Dict:
    """
    우측절단 사건시각(예: 첫 배신 라운드)의 이산시간 로그랭크형 순열 검정.
    times: 사건/절단 시각, cens: True 면 절단(사건 미발생, 시각=horizon).
    통계량 = Σ_t (관측 사건₁ − 기대 사건₁)  (표준 로그랭크 분자).
    """
    t1, t2 = _arr(times1).astype(int), _arr(times2).astype(int)
    c1, c2 = np.asarray(cens1, bool), np.asarray(cens2, bool)

    def logrank_num(tA, cA, tB, cB):
        num = 0.0
        for t in range(horizon):
            atA = np.sum(tA >= t); atB = np.sum(tB >= t)
            dA = np.sum((tA == t) & ~cA); dB = np.sum((tB == t) & ~cB)
            atT, dT = atA + atB, dA + dB
            if atT > 0 and dT > 0:
                num += dA - dT * atA / atT
        return num

    obs = logrank_num(t1, c1, t2, c2)
    times = np.concatenate([t1, t2]); cens = np.concatenate([c1, c2])
    n1 = len(t1)
    rng = _rng(seed)
    null = np.empty(n_perm)
    idx = np.arange(len(times))
    for k in range(n_perm):
        rng.shuffle(idx)
        null[k] = logrank_num(times[idx[:n1]], cens[idx[:n1]],
                              times[idx[n1:]], cens[idx[n1:]])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"stat": float(obs), "p": float(p)}


# ================================================================ 부트스트랩
def boot_ci(x, stat: Callable = np.mean, n_boot: int = _DEF_BOOT,
            seed: int = 0, level: float = 95.0) -> Dict:
    x = _arr(x)
    rng = _rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    a = (100 - level) / 2
    lo, hi = np.percentile(vals, [a, 100 - a])
    return {"stat": float(stat(x)), "ci": [float(lo), float(hi)], "n": int(len(x))}


def boot_mean_ci(x, n_boot: int = 4000, seed: int = 0) -> Dict:
    """평균의 부트스트랩 95% CI (벡터화 — 그림 오차막대용)."""
    x = _arr(x)
    if len(x) < 2:
        return {"stat": float(x.mean()) if len(x) else np.nan,
                "ci": [np.nan, np.nan], "n": int(len(x))}
    rng = _rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"stat": float(x.mean()), "ci": [float(lo), float(hi)], "n": int(len(x))}


# ================================================================ 회귀
def ols_boot(X, y, cluster=None, n_boot: int = 2000, seed: int = 0,
             names: Optional[List[str]] = None) -> Dict:
    """
    OLS + 부트스트랩 SE/CI.
      cluster=None : 관측 단위 재표집
      cluster=배열 : 군집(셀) 단위 재표집 — replicate 시드 공유·설계 격자 상관에 강건
    p 는 부트스트랩 분포 기반 양측 (정규근사 아님).
    """
    X, y = np.asarray(X, float), _arr(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rng = _rng(seed)
    k = X.shape[1]
    boots = np.empty((n_boot, k))
    if cluster is None:
        n = len(y)
        for b in range(n_boot):
            i = rng.integers(0, n, n)
            boots[b], *_ = np.linalg.lstsq(X[i], y[i], rcond=None)
    else:
        cluster = np.asarray(cluster)
        uniq = np.unique(cluster)
        groups = {c: np.flatnonzero(cluster == c) for c in uniq}
        for b in range(n_boot):
            cs = rng.choice(uniq, size=len(uniq), replace=True)
            i = np.concatenate([groups[c] for c in cs])
            boots[b], *_ = np.linalg.lstsq(X[i], y[i], rcond=None)
    se = boots.std(axis=0, ddof=1)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    # 양측 부트스트랩 p: 0 이 부트 분포의 어느 꼬리에 있는지
    p = np.array([2 * min((boots[:, j] <= 0).mean(), (boots[:, j] >= 0).mean())
                  for j in range(k)])
    p = np.clip(p, 1.0 / n_boot, 1.0)
    names = names or [f"b{j}" for j in range(k)]
    return {"names": names, "beta": beta.tolist(), "se": se.tolist(),
            "ci": [[float(l), float(h)] for l, h in zip(lo, hi)],
            "p": p.tolist(), "n": int(len(y)), "n_boot": int(n_boot),
            "boots": boots}


def slope_boot(x, y, cluster=None, n_boot: int = 2000, seed: int = 0) -> Dict:
    """원자료(replicate 수준) 단순회귀 기울기 + 부트스트랩 CI. 4점 평균 polyfit 대체."""
    x, y = _arr(x), _arr(y)
    X = np.column_stack([np.ones_like(x), x])
    out = ols_boot(X, y, cluster=cluster, n_boot=n_boot, seed=seed,
                   names=["intercept", "slope"])
    return {"slope": out["beta"][1], "ci": out["ci"][1], "p": out["p"][1],
            "n": out["n"]}


# ================================================================ 보고 포맷
def fmt_es(es: Dict, key: str = "g") -> str:
    """로그용: 'g=2.10 [1.80, 2.40]' — 판정 문장은 p 가 아니라 효과크기 중심."""
    v, ci = es.get(key), es.get("ci", [np.nan, np.nan])
    return f"{key}={v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"

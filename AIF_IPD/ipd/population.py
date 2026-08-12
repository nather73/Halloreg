"""
ipd.population
==============

**혼합 집단(mixed population) 분석 — H3 / H3A / H4 / H4A / H5 의 공통 기반.**

가설은 다음을 요구한다:

    {TFT=a, GTFT=b, WSLS=c, ALLC=d, ALLD=e, HalloReg=f},  a+…+f = 30
    "최대한 많은 조합을 시뮬레이션한다"

[문제] 30 을 6개 유형에 배분하는 조합의 수는 C(35, 5) = 324,632 이다. 각 조합마다
30명 라운드로빈(435 다이애드 × 120 라운드)을 직접 돌리면 1.4억 다이애드가 되어
어떤 계산자원으로도 불가능하다.

[해법 — 평균장이 아니라 **정확한** 분해]
본 설계에서 라운드로빈의 각 다이애드는 **독립적으로 생성된 개체 쌍**으로 실행된다
(개체는 다이애드마다 새로 만들어지며 파트너 간 정보가 전이되지 않는다). 그러면
집단 수준 지표는 유형쌍 수준 지표의 **조합가중 평균과 정확히 일치한다**:

    유형 i 의 라운드당 평균보수
        μ_i(n) = [ Σ_j n_j·Π[i,j] − Π[i,i] ] / (N − 1)
    집단 상호협력률
        CC(n) = [ nᵀ·CCm·n − Σ_i n_i·CCm[i,i] ] / [ N·(N − 1) ]

여기서 Π[i,j] 는 유형 i 가 유형 j 를 상대할 때의 라운드당 평균보수, CCm[i,j] 는
그 다이애드의 상호협력(CC) 발생률이다. 분자에서 대각항을 한 번 빼는 이유는
자기 자신과는 짝을 이루지 않기 때문이다(자기쌍 제거).

이 항등식은 근사가 아니라 **기댓값 수준의 정확한 항등식**이다. 따라서
  · 유형쌍 21개(6×7/2)만 시뮬레이션하고,
  · 324,632개 조합 **전부**를 해석적으로 평가한다.
"최대한 많은 조합" 을 문자 그대로 달성하면서 계산량은 O(21 × seeds) 로 떨어진다.

[검증 의무]
위 항등식은 '다이애드마다 새 개체' 라는 설계 전제에 의존한다. HalloRegAgent 는
identity 기억을 갖기 때문에, 개체가 파트너를 넘나들며 지속되면 항등식이 깨진다.
따라서 실험 스크립트는 대표 조합 몇 개에 대해 **직접 라운드로빈 시뮬레이션**을
수행해 해석적 예측과 대조한다(H3 의 검증 패널).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np

from AIF_IPD.core.constants import CC
from AIF_IPD.core.logging_utils import get_logger
from .env import ALL_TYPES
from .sim import run_dyad, run_many

LOGGER = get_logger("HalloReg.population")


# ==================================================================== 유형 스펙
def type_spec(name: str, **overrides) -> dict:
    """
    유형 이름 → 에이전트 스펙 dict (seed 는 호출부가 채운다).

    HalloReg 만 능동추론 에이전트이고 나머지 5종은 고정전략이다.
    """
    if name == "halloreg":
        spec = {"type": "halloreg"}
    else:
        spec = {"type": "strategy", "kind": name}
    spec.update(overrides)
    return spec


def default_type_specs(halloreg_kwargs: Optional[dict] = None
                       ) -> Dict[str, dict]:
    """6개 유형의 기본 스펙 사전 (순서는 ALL_TYPES)."""
    hk = halloreg_kwargs or {}
    return {n: type_spec(n, **(hk if n == "halloreg" else {}))
            for n in ALL_TYPES}


# ============================================================ 유형쌍 행렬 추정
def estimate_pair_matrices(type_specs: Dict[str, dict], n_rounds: int,
                           seeds: int, n_jobs: int,
                           eval_from: int = 0,
                           regime: Optional[str] = None,
                           env_error: float = 0.05,
                           seed_offset: int = 0,
                           verbose: bool = True) -> Dict:
    """
    유형쌍별 보수행렬 Π 와 상호협력행렬 CCm 을 다이애드 시뮬레이션으로 추정한다.

    Parameters
    ----------
    type_specs : {유형명: 스펙}
        키 순서가 Π 의 행/열 순서가 된다.
    regime : str | None
        보수 레짐 이름. None 이면 고정 PD.
    env_error : float
        환경 계층 실행오류 — **모든 유형에 대칭** 부과. 잡음이 특정 유형에만
        걸리면 비교가 편향되므로 반드시 대칭이어야 한다.

    Returns
    -------
    dict
      names   : 유형명 목록 (k,)
      Pi      : (k, k) 행 유형의 라운드당 평균보수
      Pi_sd   : (k, k) 시드 간 표준편차
      Pi_raw  : (k, k, seeds) 시드별 원자료 — 시드 수준 통계검정에 쓴다
      CCm     : (k, k) 상호협력률
      CCm_raw : (k, k, seeds)
    """
    names = list(type_specs)
    k = len(names)
    # 대칭 게임이므로 i ≤ j 만 돌리고 한 다이애드에서 양방향 표본을 모두 얻는다
    # (my_payoff → Π[i,j], opp_payoff → Π[j,i]). 다이애드 수가 절반으로 준다.
    pairs = [(i, j) for i in range(k) for j in range(i, k)]

    specs, registry = [], {}
    for (i, j) in pairs:
        for sd in range(seeds):
            a = dict(type_specs[names[i]])
            a["seed"] = 10_000 + seed_offset + sd * 17 + i
            b = dict(type_specs[names[j]])
            b["seed"] = 20_000 + seed_offset + sd * 17 + j
            registry[(i, j, sd)] = len(specs)
            specs.append({
                "agent": a, "opponent": b,
                "regime": regime,
                "env_err_agent": env_error, "env_err_opponent": env_error,
                # 조건 간 공통난수(CRN): 같은 (sd, i, j) 면 잡음 실현이 동일
                "noise_seed": 900_000 + seed_offset * 7 + sd * 101 + i * 7 + j,
            })

    res = run_many(specs, n_rounds=n_rounds, n_jobs=n_jobs, verbose=verbose,
                   desc=f"유형쌍 다이애드({regime or 'stationary'})")

    Pi_raw = np.zeros((k, k, seeds))
    CC_raw = np.zeros((k, k, seeds))
    for (i, j) in pairs:
        # eval_from: **학습 후 평가 창** — 워밍업(초기 학습 구간)을 평균에서
        # 제외한다. 0 이면 전 구간(기존 동작).
        _e0 = max(0, min(int(eval_from), n_rounds - 1))
        my = np.array([float(np.mean(
            res[registry[(i, j, sd)]]["hist"]["my_payoff"][_e0:]))
            for sd in range(seeds)])
        op = np.array([float(np.mean(
            res[registry[(i, j, sd)]]["hist"]["opp_payoff"][_e0:]))
            for sd in range(seeds)])
        cc = np.array([float(np.mean(
            res[registry[(i, j, sd)]]["hist"]["state"][_e0:] == CC))
            for sd in range(seeds)])
        if i == j:
            # 자기쌍: 양측 모두 같은 유형이므로 두 관점을 평균한다
            Pi_raw[i, i] = 0.5 * (my + op)
        else:
            Pi_raw[i, j] = my
            Pi_raw[j, i] = op
        CC_raw[i, j] = cc
        CC_raw[j, i] = cc          # 상호협력은 관점 무관 대칭

    return {
        "names": names,
        "Pi": Pi_raw.mean(axis=2),
        "Pi_sd": Pi_raw.std(axis=2, ddof=1) if seeds > 1 else np.zeros((k, k)),
        "Pi_raw": Pi_raw,
        "CCm": CC_raw.mean(axis=2),
        "CCm_raw": CC_raw,
        "n_rounds": n_rounds, "seeds": seeds, "regime": regime,
        "env_error": env_error, "eval_from": int(eval_from),
    }


# ============================================================ 조합 열거
def enumerate_compositions(total: int = 30, k: int = 6,
                           min_each: int = 0) -> np.ndarray:
    """
    total 을 k 개의 비음 정수로 분할하는 **모든** 조합을 열거한다.

    구현: 'stars and bars'. total + k − 1 개의 자리 중 k − 1 개를 막대 위치로
    고르면 각 조합이 유일하게 대응한다. 조합 수는 C(total + k − 1, k − 1).
    total=30, k=6 이면 C(35, 5) = 324,632.

    min_each : 각 유형의 최소 개체 수. 예컨대 1 로 두면 모든 유형이 최소 1명씩
               존재하는 조합만 남는다(비교가 정의되는 조합).

    반환: (M, k) int 배열.
    """
    if min_each > 0:
        # 각 유형에 min_each 를 먼저 배정한 뒤 나머지를 자유 분배
        rest = total - min_each * k
        if rest < 0:
            return np.zeros((0, k), dtype=int)
        base = enumerate_compositions(rest, k, 0)
        return base + min_each

    n = total + k - 1
    out = []
    for bars in combinations(range(n), k - 1):
        prev = -1
        comp = []
        for b in bars:
            comp.append(b - prev - 1)
            prev = b
        comp.append(n - prev - 1)
        out.append(comp)
    return np.asarray(out, dtype=int)


def subsample_compositions(comps: np.ndarray, n_max: int,
                           seed: int = 0) -> np.ndarray:
    """조합이 너무 많을 때 균등 무작위 부분표집 (재현 가능)."""
    if len(comps) <= n_max:
        return comps
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(comps), size=n_max, replace=False)
    return comps[np.sort(idx)]


# ============================================================ 해석적 평가
def type_payoffs(Pi: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    조합별·유형별 라운드당 평균보수.

        μ_i(n) = [ Σ_j n_j·Π[i,j] − Π[i,i] ] / (N − 1)

    counts : (M, k) 또는 (k,)
    반환    : counts 와 같은 shape. 개체 수가 0인 유형의 값은 정의되지 않으므로
             호출부가 마스킹해야 한다(여기서는 형식적으로 계산만 한다).
    """
    c = np.atleast_2d(np.asarray(counts, dtype=float))
    N = c.sum(axis=1, keepdims=True)
    # c @ Pi.T → [m, i] = Σ_j c[m,j]·Pi[i,j]
    tot = c @ np.asarray(Pi, dtype=float).T
    out = (tot - np.diag(Pi)[None, :]) / np.maximum(N - 1.0, 1e-9)
    return out[0] if np.ndim(counts) == 1 else out


def population_cc(CCm: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    조합별 집단 상호협력률.

        CC(n) = [ nᵀ·CCm·n − Σ_i n_i·CCm[i,i] ] / [ N·(N − 1) ]

    유도: 서로 다른 유형쌍 Σ_{i<j} n_i n_j CC[i,j] 와 동일유형 내부쌍
    Σ_i n_i(n_i−1)/2 · CC[i,i] 를 합치면 위 식이 된다(자기 자신과의 짝 제외).
    """
    c = np.atleast_2d(np.asarray(counts, dtype=float))
    M = np.asarray(CCm, dtype=float)
    N = c.sum(axis=1)
    quad = np.einsum("mi,ij,mj->m", c, M, c)
    diag = c @ np.diag(M)
    out = (quad - diag) / np.maximum(N * (N - 1.0), 1e-9)
    return out[0] if np.ndim(counts) == 1 else out


def population_mean_payoff(Pi: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """조합별 집단 전체 평균보수 (개체 가중)."""
    c = np.atleast_2d(np.asarray(counts, dtype=float))
    mu = type_payoffs(Pi, c)
    N = c.sum(axis=1)
    out = np.sum(c * mu, axis=1) / np.maximum(N, 1e-9)
    return out[0] if np.ndim(counts) == 1 else out


def substitution_delta_cc(CCm: np.ndarray, counts: np.ndarray,
                          src: int, dst: int) -> np.ndarray:
    """
    **치환 대비(substitution contrast)** — H3A / H4A 의 기여도 조작화.

    조합 n 에서 유형 src 개체 하나를 유형 dst 로 바꾸었을 때의 집단 CC율 변화:

        Δ = CC(n) − CC(n − e_src + e_dst)

    Δ > 0 이면 "그 자리에 src 가 있는 편이 dst 가 있는 것보다 집단 협력을 더
    끌어올린다" 는 뜻이다. src=HalloReg 로 두고 dst 를 5개 고정전략에 각각
    대입하면, HalloReg 의 **한계 기여도**를 다른 전략 대비로 직접 측정할 수 있다.

    counts 중 n_src = 0 인 조합은 치환이 정의되지 않으므로 NaN 을 반환한다.
    """
    c = np.atleast_2d(np.asarray(counts, dtype=float)).copy()
    base = population_cc(CCm, c)
    alt = c.copy()
    alt[:, src] -= 1.0
    alt[:, dst] += 1.0
    delta = base - population_cc(CCm, alt)
    delta = np.where(c[:, src] >= 1.0, delta, np.nan)
    return delta[0] if np.ndim(counts) == 1 else delta


# ============================================================ 직접 라운드로빈
def run_round_robin(counts: Sequence[int], type_specs: Dict[str, dict],
                    n_rounds: int, seed: int = 0,
                    regime: Optional[str] = None,
                    env_error: float = 0.05,
                    persistent: bool = False,
                    n_jobs: int = 1, eval_from: int = 0) -> Dict:
    """
    **직접** 라운드로빈 시뮬레이션 — 해석적 분해식의 검증용.

    persistent=False (기본)
        다이애드마다 개체를 새로 만든다. 해석적 항등식의 전제와 일치하므로
        두 값이 (표집오차 범위에서) 일치해야 한다. 이 경로는 다이애드가 서로
        독립이므로 `run_many` 로 병렬화한다.
    persistent=True
        개체를 집단 내내 유지한다. HalloReg 의 identity 기억이 파트너를 넘어
        누적되므로 항등식이 성립하지 않을 수 있다 — 그 편차의 크기를 직접
        측정해 '기억의 집단 수준 효과' 를 정량화한다. 개체가 순차적으로 상태를
        누적하므로 **병렬화할 수 없다**(순차 실행).

    반환: cc_rate, mean_payoff, by_type_payoff
    """
    from .payoff_schedule import get_regime
    from .sim import build_agent

    names = list(type_specs)
    labels: List[str] = []
    for i, nm in enumerate(names):
        labels += [nm] * int(counts[i])
    n = len(labels)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def cfg_for(idx: int, dyad_id: int) -> dict:
        cfg = dict(type_specs[labels[idx]])
        cfg["seed"] = seed * 7919 + dyad_id * 131 + idx
        return cfg

    total_cc = 0
    total_rounds = 0
    pay_sum = {nm: 0.0 for nm in names}
    pay_rounds = {nm: 0 for nm in names}

    if persistent:
        # ---- 순차 경로 (개체 상태가 다이애드를 넘어 누적) ----
        ci_fn = get_regime(regime) if regime else None
        agents = [build_agent(cfg_for(i, 0)) for i in range(n)]
        for d, (i, j) in enumerate(pairs):
            h = run_dyad(agents[i], agents[j], n_rounds, ci_schedule=ci_fn,
                         env_err_a=env_error, env_err_b=env_error,
                         noise_seed=seed * 31 + d,
                         partner_id_a=1000 + j, partner_id_b=1000 + i)
            _e0 = max(0, min(int(eval_from), n_rounds - 1))
            _nr = n_rounds - _e0
            total_cc += int(np.sum(h["state"][_e0:] == CC))
            total_rounds += _nr
            pay_sum[labels[i]] += float(np.sum(h["my_payoff"][_e0:]))
            pay_rounds[labels[i]] += _nr
            pay_sum[labels[j]] += float(np.sum(h["opp_payoff"][_e0:]))
            pay_rounds[labels[j]] += _nr
    else:
        # ---- 병렬 경로 (다이애드 독립) ----
        specs = [{"agent": cfg_for(i, d), "opponent": cfg_for(j, d),
                  "regime": regime,
                  "env_err_agent": env_error, "env_err_opponent": env_error,
                  "noise_seed": seed * 31 + d}
                 for d, (i, j) in enumerate(pairs)]
        res = run_many(specs, n_rounds=n_rounds, n_jobs=n_jobs, verbose=False,
                       desc="라운드로빈 검증")
        for d, (i, j) in enumerate(pairs):
            h = res[d]["hist"]
            _e0 = max(0, min(int(eval_from), n_rounds - 1))
            _nr = n_rounds - _e0
            total_cc += int(np.sum(h["state"][_e0:] == CC))
            total_rounds += _nr
            pay_sum[labels[i]] += float(np.sum(h["my_payoff"][_e0:]))
            pay_rounds[labels[i]] += _nr
            pay_sum[labels[j]] += float(np.sum(h["opp_payoff"][_e0:]))
            pay_rounds[labels[j]] += _nr

    by_type = {nm: (pay_sum[nm] / pay_rounds[nm]) if pay_rounds[nm] else np.nan
               for nm in names}
    all_pay = sum(pay_sum.values()) / max(sum(pay_rounds.values()), 1)
    return {"cc_rate": total_cc / max(total_rounds, 1),
            "mean_payoff": all_pay, "by_type_payoff": by_type,
            "n_agents": n, "n_pairs": len(pairs)}

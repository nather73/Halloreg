"""
evolution.py — 집단 '역학' 의 실제 도입 (보완안 §3).

기존 H8 은 전략 갱신이 없는 정적 매칭이었다. 여기서는 2단 구성으로 역학을
도입한다:

1단 — 평균장 복제자 동역학
    유형 쌍별 평균 보수 행렬 Π 를 다이애드 시뮬레이션으로 추정한 뒤
    이산 복제자 갱신 x_i ← x_i·f_i / f̄ 를 수치적분한다. 침입 성장률,
    고정점, 유역(basin) 분석이 거의 무비용으로 가능하다.

2단 — 에이전트 기반 Moran 과정 (경험 보수 근사)
    N 개체, 세대마다 적합도 비례 출생 + 균등 사망 + 돌연변이 ε. 개체의
    세대 적합도는 무작위 상대 k 명에 대해 **경험적 다이애드 보수 분포
    (Π 의 평균·SD)에서 표집**한 보수의 평균으로 근사한다.

    주의(문서화된 근사): 다이애드를 세대마다 재시뮬레이션하는 완전 ABM 은
    계산상 불가하여, 쌍별 보수의 1·2차 모멘트를 보존하는 표집 근사를 쓴다.
    평균장 예측과의 교차 검증이 목적이므로 이 수준이 적절하다.

가설의 재정식화: "AdaptiveAgent 는 혼합 집단을 침입할 수 있고(침입 성장률>0),
협력적 균형의 유역을 넓힌다."
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.sim import run_many

LOGGER = get_logger("HalloReg.evolution")


# ---------------------------------------------------------------- Π 추정
def estimate_payoff_matrix(type_specs: Dict[str, dict], n_rounds: int,
                           seeds: int, n_jobs: int,
                           env_error: float = 0.10) -> Dict:
    """
    유형 쌍별 (행 유형의) 라운드당 평균 보수 행렬 Π 와 시드 간 SD 를 추정.

    type_specs : {유형이름: _build_one 호환 스펙(dict, seed 제외)}
    env_error  : 환경 계층 실행오류 — 모든 유형에 **대칭** 부과 (§2 잡음 대칭화).
    """
    names = list(type_specs)
    k = len(names)
    specs, registry = [], {}
    for i, ri in enumerate(names):
        for j, cj in enumerate(names):
            for sd in range(seeds):
                a = dict(type_specs[ri]); a["seed"] = 10_000 + sd * 17 + i
                b = dict(type_specs[cj]); b["seed"] = 20_000 + sd * 17 + j
                registry[(i, j, sd)] = len(specs)
                specs.append({"agent": a, "opponent": b,
                              "env_err_agent": env_error,
                              "env_err_opponent": env_error,
                              "noise_seed": 900_000 + sd * 101 + i * 7 + j})
    res = run_many(specs, n_rounds=n_rounds, n_jobs=n_jobs, verbose=False)
    Pi = np.zeros((k, k)); Pi_sd = np.zeros((k, k))
    raw = np.zeros((k, k, seeds))               # 시드 원자료 (침입 부트스트랩용)
    for i in range(k):
        for j in range(k):
            v = np.array([float(np.mean(res[registry[(i, j, sd)]]
                                        ["hist"]["my_payoff"]))
                          for sd in range(seeds)])
            raw[i, j] = v
            Pi[i, j] = v.mean()
            Pi_sd[i, j] = v.std(ddof=1) if seeds > 1 else 0.0
    return {"names": names, "Pi": Pi, "Pi_sd": Pi_sd, "raw": raw,
            "n_rounds": n_rounds, "seeds": seeds, "env_error": env_error}


# ---------------------------------------------------------------- 복제자
def replicator_trajectory(Pi: np.ndarray, x0: Sequence[float],
                          steps: int = 400, noise: float = 0.0,
                          seed: int = 0) -> np.ndarray:
    """이산 복제자 갱신 x_i ← x_i f_i / f̄ 궤적. 반환 shape (steps+1, k)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, float)
    x = np.clip(x, 1e-12, None); x /= x.sum()
    traj = [x.copy()]
    for _ in range(steps):
        f = Pi @ x
        fbar = float(x @ f)
        x = x * f / max(fbar, 1e-12)
        if noise > 0:
            x = np.clip(x + rng.normal(0, noise, len(x)), 1e-12, None)
        x /= x.sum()
        traj.append(x.copy())
    return np.asarray(traj)


def invasion_growth(Pi: np.ndarray, resident: Sequence[float],
                    invader_idx: int) -> float:
    """
    상주 혼합 resident 에 희소 침입자 invader 가 들어올 때의 성장률
    f_inv − f̄_res. 양수면 침입 가능.
    """
    x = np.asarray(resident, float); x = x / x.sum()
    f = Pi @ x
    return float(f[invader_idx] - x @ f)


def basin_analysis(Pi: np.ndarray, names: List[str], n_samples: int = 300,
                   steps: int = 600, seed: int = 0,
                   coop_types: Optional[List[int]] = None) -> Dict:
    """
    Dirichlet(1) 초기점에서 복제자 궤적을 적분해 종착 조성을 분류.
    반환: 각 유형이 종착 지배유형(>0.5)이 되는 초기점 비율, '협력 유역'
    (coop_types 합계 > 0.5 종착) 비율, 종착점 평균 조성.
    """
    rng = np.random.default_rng(seed)
    k = Pi.shape[0]
    ends = np.empty((n_samples, k))
    for s in range(n_samples):
        x0 = rng.dirichlet(np.ones(k))
        ends[s] = replicator_trajectory(Pi, x0, steps=steps)[-1]
    dominant = ends.argmax(axis=1)
    dom_frac = {names[i]: float(np.mean((dominant == i) & (ends.max(axis=1) > 0.5)))
                for i in range(k)}
    out = {"dominant_basin_frac": dom_frac,
           "end_mean": ends.mean(axis=0).tolist()}
    if coop_types is not None:
        coop_share = ends[:, coop_types].sum(axis=1)
        out["coop_basin_frac"] = float(np.mean(coop_share > 0.5))
    return out


# ---------------------------------------------------------------- Moran
def moran_process(Pi: np.ndarray, Pi_sd: np.ndarray, init_counts: Sequence[int],
                  generations: int = 30, k_partners: int = 3,
                  mutation: float = 0.01, selection: float = 1.0,
                  seed: int = 0) -> np.ndarray:
    """
    경험 보수 근사 Moran 과정 (출생–사망, 적합도 비례 출생, 균등 사망).

    유형 i 개체의 이벤트 적합도 = 현재 조성 x 에서 무작위 상대 k_partners 명과의
    보수를 Normal(Π[i,·], Π_sd[i,·]) (0 절단) 에서 표집한 평균 → exp(selection·pay).
    유형 수준 카운트로 갱신(이벤트당 O(k)); 세대당 N 회 출생–사망(= 집단 1회 교체).
    반환 shape (generations+1, k) 의 유형 빈도 궤적.
    """
    rng = np.random.default_rng(seed)
    counts = np.asarray(init_counts, int).copy()
    k = len(counts)
    N = int(counts.sum())
    freq = [counts / N]
    for _ in range(generations):
        for _ in range(N):                       # 1 세대 = N 출생–사망
            x = counts / N
            # 유형별 상대 표집 (k_partners 명, 조성 비례) 후 경험 보수 표집
            opp = rng.choice(k, size=(k, k_partners), p=x)
            pay = np.clip(rng.normal(Pi[np.arange(k)[:, None], opp],
                                     np.maximum(Pi_sd[np.arange(k)[:, None], opp],
                                                1e-6)),
                          0.0, None).mean(axis=1)
            fit = np.exp(selection * (pay - pay.max())) * np.maximum(counts, 0)
            birth = rng.choice(k, p=fit / fit.sum())
            if rng.random() < mutation:
                birth = rng.integers(0, k)
            death = rng.choice(k, p=x)           # 균등 사망 (개체 균등 = 조성 비례)
            counts[birth] += 1
            counts[death] -= 1
        freq.append(counts / N)
    return np.asarray(freq)

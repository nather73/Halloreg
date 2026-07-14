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


# ------------------------------------------------- Optimal Replicator Equation
# Bravetti & Padilla (2018), "An optimal strategy to solve the Prisoner's Dilemma"
# (Sci. Rep. 8:1948). 표준 RE 는 **개체 수준** 선택만 담고, T>R>P>S 인 한 항상
# 배신자가 협력자를 지배한다. ORE 는 진화가 개체 수준(빠른 시간척도)뿐 아니라
# **경쟁하는 집단 수준**(느린 시간척도)에서도 작동한다는 가정 하에, 최종 평균
# 적합도 g(x(τ))=xᵀΠx 를 최대화하는 최적제어 문제로 RE 를 확장한다. 그 결과
# 적합도(=제어변수)를 공상태(co-state) p 로 갖는 확장계가 얻어진다:
#
#     forward   ẋ_a = x_a (p_a − ⟨p⟩)                    x(0)=x0                (개체 선택)
#     backward  ṗ_a = ⟨p⟩ p_a − ½ p_a²                    p(τ)=∇g(x(τ))          (집단 선택)
#     terminal  p_a(τ) = ∂g/∂x_a = ((Π+Πᵀ) x(τ))_a
#
# 협력은 p_C>p_D 일 때 창발하며, 이 공상태가 '집단이 최종적으로 나눌 큰 보상'을
# 시간의 역방향으로 실어나른다 → 이기적 개체도 협력을 택한다. 2-유형 PD 에서
# 조건 2R≥T, 2P≥T 하에 협력 고정점 x_C=1 이 점근 안정.
#
# 본 구현은 최적제어 경계값문제(BVP)를 forward-backward sweep(FBSM)으로 푼다.
# K-유형으로 자연 일반화하여, 동일 Π 위에서 RE 유역과 ORE 유역을 짝지어 비교한다
# (§ORE: 집단 수준 경쟁이 협력 유역을 넓히는가).

def _grad_g(Pi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """최종 평균적합도 g(x)=xᵀΠx 의 기울기 ∇g = (Π+Πᵀ)x (터미널 공상태)."""
    return (Pi + Pi.T) @ x


def ore_trajectory(Pi: np.ndarray, x0: Sequence[float], tau: float = 1.0,
                   steps: int = 400, sweeps: int = 60, relax: float = 0.5,
                   tol: float = 1e-7) -> Dict:
    """
    K-유형 ORE 궤적을 FBSM 으로 계산. 반환:
      x     : (steps+1, k)   개체 조성 궤적
      p     : (steps+1, k)   공상태 궤적
      x_end : (k,)           종착 조성 x(τ)
      converged, sweeps_used, resid
    """
    x0 = np.clip(np.asarray(x0, float), 1e-12, None); x0 = x0 / x0.sum()
    k = len(x0)
    dt = tau / steps
    # 공상태 발산(Riccati형 ṗ=⟨p⟩p−½p² 는 유한시간 폭발 가능) 방지용 클램프 범위:
    # 보수 규모의 배수로 상한. 유역 판정은 종착 조성 부호에만 의존하므로 클램프는
    # 정성적 결론을 바꾸지 않되 수치 안정성을 확보한다.
    p_bound = 20.0 * max(1.0, float(np.max(np.abs(_grad_g(Pi, np.ones(k) / k)))),
                         float(np.max(np.abs(Pi))))
    x = np.tile(x0, (steps + 1, 1))
    # 공상태 초기 추정: 모든 시점에서 x0 기준 터미널값
    p = np.tile(_grad_g(Pi, x0), (steps + 1, 1))
    prev = None
    used = sweeps
    resid = np.inf
    for it in range(sweeps):
        # --- forward: x ---
        x[0] = x0
        for n in range(steps):
            pm = float(x[n] @ p[n])                         # ⟨p⟩
            dx = x[n] * (p[n] - pm)
            xn = np.clip(x[n] + dt * dx, 1e-12, None)
            x[n + 1] = xn / xn.sum()
        # --- backward: p (terminal at n=steps) ---
        p_new = p.copy()
        p_new[steps] = np.clip(_grad_g(Pi, x[steps]), -p_bound, p_bound)
        for n in range(steps, 0, -1):
            pm = float(x[n] @ p_new[n])                     # ⟨p⟩
            dpdt = pm * p_new[n] - 0.5 * p_new[n] ** 2       # ṗ_a
            p_new[n - 1] = np.clip(p_new[n] - dt * dpdt,
                                   -p_bound, p_bound)         # 역방향 오일러 + 클램프
        # --- 완화(relaxation) ---
        p = (1 - relax) * p + relax * p_new
        if prev is not None:
            resid = float(np.max(np.abs(p - prev)))
            if resid < tol:
                used = it + 1
                break
        prev = p.copy()
    return {"x": x, "p": p, "x_end": x[steps].copy(),
            "converged": resid < tol, "sweeps_used": used, "resid": resid}


def ore_ends_batch(Pi: np.ndarray, X0: np.ndarray, tau: float = 1.0,
                   steps: int = 220, sweeps: int = 30, relax: float = 0.5) -> np.ndarray:
    """
    다수 초기점의 ORE 종착 조성만 반환 (N, k) — 유역 부트스트랩용.

    **초기점 전체를 벡터화**한 FBSM: 상태 x·공상태 p 를 (steps+1, N, k) 텐서로 두고
    시간에 대한 순차 적분만 남긴다(초기점 축은 완전 병렬). 초기점 N 개 각각을 별도
    호출하던 방식 대비 ~N× 가속.
    """
    X0 = np.clip(np.asarray(X0, float), 1e-12, None)
    X0 = X0 / X0.sum(axis=1, keepdims=True)
    N, k = X0.shape
    dt = tau / steps
    G = Pi + Pi.T                                            # 대칭 (∇g = G x)
    p_bound = 20.0 * max(1.0, float(np.max(np.abs(G @ (np.ones(k) / k)))),
                         float(np.max(np.abs(Pi))))
    x = np.tile(X0, (steps + 1, 1, 1))                      # (S+1, N, k)
    p = np.tile(X0 @ G.T, (steps + 1, 1, 1))               # 터미널값으로 초기화
    prev = None
    for _ in range(sweeps):
        # forward: x
        x[0] = X0
        for n in range(steps):
            pm = np.einsum("nk,nk->n", x[n], p[n])[:, None]  # ⟨p⟩ (N,1)
            xn = np.clip(x[n] + dt * (x[n] * (p[n] - pm)), 1e-12, None)
            x[n + 1] = xn / xn.sum(axis=1, keepdims=True)
        # backward: p (terminal at steps)
        p_new = p.copy()
        p_new[steps] = np.clip(x[steps] @ G.T, -p_bound, p_bound)
        for n in range(steps, 0, -1):
            pm = np.einsum("nk,nk->n", x[n], p_new[n])[:, None]
            dpdt = pm * p_new[n] - 0.5 * p_new[n] ** 2
            p_new[n - 1] = np.clip(p_new[n] - dt * dpdt, -p_bound, p_bound)
        p = (1 - relax) * p + relax * p_new
        if prev is not None and float(np.max(np.abs(p - prev))) < 1e-6:
            break
        prev = p.copy()
    return x[steps].copy()


def ore_coop_basin_frac(Pi: np.ndarray, coop_idx: Sequence[int],
                        n_samples: int = 200, tau: float = 1.0, steps: int = 220,
                        sweeps: int = 30, seed: int = 0,
                        X0: Optional[np.ndarray] = None) -> float:
    """
    ORE 하 협력 유역 점유율: Dirichlet(1) 초기점(또는 주어진 X0)에서 ORE 종착
    조성의 협력 점유(coop_idx 합) > 0.5 인 비율. 동일 X0 를 RE 유역과 공유하면
    '집단 수준 경쟁 도입이 협력 유역을 얼마나 넓히는가'를 짝지어 비교할 수 있다.
    """
    k = Pi.shape[0]
    if X0 is None:
        rng = np.random.default_rng(seed)
        X0 = rng.dirichlet(np.ones(k), size=n_samples)
    ends = ore_ends_batch(Pi, X0, tau=tau, steps=steps, sweeps=sweeps)
    return float(np.mean(ends[:, list(coop_idx)].sum(axis=1) > 0.5))


def ore_two_type(R: float, T: float, P: float, S: float, xC0: float,
                 tau: float = 1.0, steps: int = 400, sweeps: int = 80) -> Dict:
    """
    2-유형(C,D) PD 의 RE vs ORE 궤적 (Bravetti & Padilla Fig.1/2 재현).
    Π=[[R,S],[T,P]]. 반환: RE·ORE 의 x_C(t)·평균적합도(t).
    """
    Pi = np.array([[R, S], [T, P]], float)
    x0 = np.array([xC0, 1 - xC0])
    # RE
    re = replicator_trajectory(Pi, x0, steps=steps)
    re_fit = np.einsum("ti,ij,tj->t", re, Pi, re)
    # ORE
    o = ore_trajectory(Pi, x0, tau=tau, steps=steps, sweeps=sweeps)
    ore_x = o["x"]
    ore_fit = np.einsum("ti,ij,tj->t", ore_x, Pi, ore_x)
    return {"Pi": Pi, "re_xC": re[:, 0], "ore_xC": ore_x[:, 0],
            "re_fit": re_fit, "ore_fit": ore_fit,
            "re_end": float(re[-1, 0]), "ore_end": float(ore_x[-1, 0])}


# ---------------------------------------------------------------- Π 추정
def estimate_payoff_matrix(type_specs: Dict[str, dict], n_rounds: int,
                           seeds: int, n_jobs: int,
                           env_error: float = 0.10,
                           symmetric: bool = True,
                           seed_offset: int = 0) -> Dict:
    """
    유형 쌍별 (행 유형의) 라운드당 평균 보수 행렬 Π 와 시드 간 SD 를 추정.

    type_specs : {유형이름: _build_one 호환 스펙(dict, seed 제외)}
    env_error  : 환경 계층 실행오류 — 모든 유형에 **대칭** 부과 (§2 잡음 대칭화).
    symmetric  : True 면 순서쌍 (i,j), i≤j 만 시뮬레이션하고, 같은 다이애드의
                 my_payoff/opp_payoff 를 각각 Π[i,j]/Π[j,i] 표본으로 재사용한다
                 (환경잡음이 양측 대칭이므로 두 방향 모두 유효한 iid 추정치;
                 다이애드 수를 절반으로 감축 — v0.3 (err×T) 격자 확장 대응).
                 대각 (i,i) 은 양측 평균을 사용한다.
    seed_offset: (err, T) 조건 간 시드 독립화를 위한 오프셋.
    """
    names = list(type_specs)
    k = len(names)
    specs, registry = [], {}
    pair_iter = ([(i, j) for i in range(k) for j in range(i, k)] if symmetric
                 else [(i, j) for i in range(k) for j in range(k)])
    for (i, j) in pair_iter:
        ri, cj = names[i], names[j]
        for sd in range(seeds):
            a = dict(type_specs[ri]); a["seed"] = 10_000 + seed_offset + sd * 17 + i
            b = dict(type_specs[cj]); b["seed"] = 20_000 + seed_offset + sd * 17 + j
            registry[(i, j, sd)] = len(specs)
            specs.append({"agent": a, "opponent": b,
                          "env_err_agent": env_error,
                          "env_err_opponent": env_error,
                          "noise_seed": 900_000 + seed_offset * 7
                                        + sd * 101 + i * 7 + j})
    res = run_many(specs, n_rounds=n_rounds, n_jobs=n_jobs, verbose=False)
    Pi = np.zeros((k, k)); Pi_sd = np.zeros((k, k))
    raw = np.zeros((k, k, seeds))               # 시드 원자료 (침입 부트스트랩용)
    for (i, j) in pair_iter:
        my = np.array([float(np.mean(res[registry[(i, j, sd)]]
                                     ["hist"]["my_payoff"]))
                       for sd in range(seeds)])
        if symmetric:
            op = np.array([float(np.mean(res[registry[(i, j, sd)]]
                                         ["hist"]["opp_payoff"]))
                           for sd in range(seeds)])
            if i == j:
                raw[i, i] = 0.5 * (my + op)
            else:
                raw[i, j] = my
                raw[j, i] = op
        else:
            raw[i, j] = my
    for i in range(k):
        for j in range(k):
            Pi[i, j] = raw[i, j].mean()
            Pi_sd[i, j] = raw[i, j].std(ddof=1) if seeds > 1 else 0.0
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


def cluster_invasion_growth(Pi: np.ndarray, resident: Sequence[float],
                            cluster: Sequence[float]) -> float:
    """
    (v0.4, H11) 상주 혼합 resident 에 **혼합 조성 cluster** 가 희소 침입할 때의
    성장률: g = (w_cluster · f) − f̄_res.  단일 유형 침입(invasion_growth)의
    자연스러운 일반화 — '협력자 클러스터'(adaptive 포함/제외)가 ALLD-heavy
    상주집단을 침입할 수 있는지를 묻는 H11 진화 프레임에 사용한다.
    """
    x = np.asarray(resident, float); x = x / x.sum()
    w = np.asarray(cluster, float); w = w / w.sum()
    f = Pi @ x
    return float(w @ f - x @ f)


def replicator_ends_batch(Pi: np.ndarray, X0: np.ndarray,
                          steps: int = 600) -> np.ndarray:
    """
    (v0.4) 다수 초기점의 이산 복제자 갱신을 **벡터화**하여 종착 조성만 반환.
    X0 shape (n, k) → 반환 shape (n, k).  H11 의 유역 부트스트랩(Π 시드
    재표집마다 basin 재적분)을 실용적 비용으로 만드는 핵심 최적화 —
    per-초기점 파이썬 루프를 단일 (n,k)@(k,k) 행렬곱 수열로 대체한다.
    수치 의미는 replicator_trajectory 와 동일(잡음 없음).
    """
    X = np.clip(np.asarray(X0, float), 1e-12, None)
    X = X / X.sum(axis=1, keepdims=True)
    for _ in range(steps):
        F = X @ Pi.T                                # F[n, i] = (Pi @ x_n)_i
        fbar = np.einsum("ni,ni->n", X, F)
        X = X * F / np.maximum(fbar[:, None], 1e-12)
        X = X / X.sum(axis=1, keepdims=True)
    return X


def coop_basin_frac(Pi: np.ndarray, coop_idx: Sequence[int],
                    n_samples: int = 300, steps: int = 600,
                    seed: int = 0, X0: Optional[np.ndarray] = None) -> float:
    """
    (v0.4) Dirichlet(1) 초기점(또는 주어진 X0)에서 복제자 종착 조성의
    협력 점유율(coop_idx 합) > 0.5 인 초기점 비율 — 벡터화 경로.
    같은 X0 를 조건 간 공유하면(공통 초기점) 유형 추가/제거의 유역 효과를
    짝지어 비교할 수 있다 (H11 basin-widening 대조).
    """
    k = Pi.shape[0]
    if X0 is None:
        rng = np.random.default_rng(seed)
        X0 = rng.dirichlet(np.ones(k), size=n_samples)
    ends = replicator_ends_batch(Pi, X0, steps=steps)
    return float(np.mean(ends[:, list(coop_idx)].sum(axis=1) > 0.5))


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


# ------------------------------------------------------- 끌개(attractor)
def attractor_analysis(Pi: np.ndarray, names: List[str], n_samples: int = 400,
                       steps: int = 800, seed: int = 0,
                       round_to: float = 0.02) -> Dict:
    """
    Dirichlet(1) 초기점 n_samples 개에서 복제자 궤적을 적분해 종착점을
    `round_to` 격자로 양자화·군집화한다. 반환: 끌개 목록(평균 조성, 유역
    비율)과 원 종착점 배열 — Replicator 상태 공간의 끌개 구조 시각화용.
    """
    rng = np.random.default_rng(seed)
    k = Pi.shape[0]
    ends = np.empty((n_samples, k))
    for s in range(n_samples):
        ends[s] = replicator_trajectory(Pi, rng.dirichlet(np.ones(k)),
                                        steps=steps)[-1]
    keys = np.round(ends / round_to).astype(int)
    uniq, inv, cnt = np.unique(keys, axis=0, return_inverse=True,
                               return_counts=True)
    order = np.argsort(-cnt)
    attractors = []
    for o in order:
        mask = inv == o
        attractors.append({
            "composition": ends[mask].mean(axis=0).tolist(),
            "basin_frac": float(cnt[o] / n_samples),
        })
    return {"names": names, "attractors": attractors, "ends": ends}


def restrict_matrix(Pi: np.ndarray, names: List[str],
                    subset: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """유형 부분집합으로 제한한 부분 보수 행렬 (3-유형 위상 초상용)."""
    idx = [names.index(s) for s in subset]
    return Pi[np.ix_(idx, idx)], list(subset)


# ------------------------------------------------- 3-유형 심플렉스(ternary)
def ternary_xy(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    바리센트릭 조성 x(..., 3) → 2D 데카르트 좌표.
    꼭짓점: 유형0=(0,0), 유형1=(1,0), 유형2=(0.5, √3/2).
    """
    x = np.asarray(x, float)
    px = x[..., 1] + 0.5 * x[..., 2]
    py = (np.sqrt(3) / 2.0) * x[..., 2]
    return px, py


def ternary_grid(n: int = 25, margin: float = 0.01) -> np.ndarray:
    """심플렉스 내부의 규칙 바리센트릭 격자 (초기점/유역 지도용)."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            a = i / n; b = j / n; c = 1.0 - a - b
            x = np.array([a, b, c])
            x = np.clip(x, margin, None); x = x / x.sum()
            pts.append(x)
    return np.asarray(pts)


def basin_map_3(Pi3: np.ndarray, coop_idx: Sequence[int], n: int = 25,
                steps: int = 800) -> Dict:
    """
    3-유형 부분계의 협력 유역 지도: 격자 초기점별 (종착 협력 점유율,
    종착 지배유형)을 계산. Replicator 상태 공간·attractor·cooperation
    basin 시각화의 데이터 소스.
    """
    grid = ternary_grid(n)
    coop_share = np.empty(len(grid))
    dominant = np.empty(len(grid), dtype=int)
    ends = np.empty((len(grid), 3))
    for s, x0 in enumerate(grid):
        e = replicator_trajectory(Pi3, x0, steps=steps)[-1]
        ends[s] = e
        coop_share[s] = float(e[list(coop_idx)].sum())
        dominant[s] = int(np.argmax(e))
    return {"grid": grid, "ends": ends, "coop_share": coop_share,
            "dominant": dominant}


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

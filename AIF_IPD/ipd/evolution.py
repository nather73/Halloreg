"""
ipd.evolution
=============

**진화 동역학 — H5 의 기반.**

두 가지 집단 수준 동역학을 같은 보수행렬 Π 위에서 짝지어 비교한다.

────────────────────────────────────────────────────────────────────────
1. 복제자 방정식 (Replicator Equation, RE)
────────────────────────────────────────────────────────────────────────
표준 개체 수준 선택. 유형 i 의 빈도 x_i 는 그 유형의 적합도가 집단 평균보다
높을 때 증가한다.

    ẋ_i = x_i · ( (Πx)_i − xᵀΠx )

이산 갱신형:  x_i ← x_i · f_i / f̄,   f = Πx,  f̄ = xᵀΠx.

RE 의 알려진 한계: T > R > P > S 인 한 배신자가 협력자를 항상 지배한다.
즉 순수 개체 수준 선택만으로는 PD 에서 협력이 진화하지 않는다.

────────────────────────────────────────────────────────────────────────
2. 최적 복제자 방정식 (Optimal Replicator Equation, ORE)
────────────────────────────────────────────────────────────────────────
Bravetti & Padilla (2018), *An optimal strategy to solve the Prisoner's Dilemma*,
Sci. Rep. 8:1948.

진화가 개체 수준(빠른 시간척도)뿐 아니라 **경쟁하는 집단 수준**(느린 시간척도)
에서도 작동한다고 보고, 최종 평균적합도 g(x(τ)) = x(τ)ᵀ Π x(τ) 를 최대화하는
최적제어 문제로 RE 를 확장한다. 적합도를 제어변수로, 공상태(co-state) p 를 갖는
확장계가 얻어진다:

    forward   ẋ_a = x_a · ( p_a − ⟨p⟩ )              x(0) = x₀      (개체 선택)
    backward  ṗ_a = ⟨p⟩·p_a − ½·p_a²                 p(τ) = ∇g(x(τ))  (집단 선택)
    terminal  p_a(τ) = ∂g/∂x_a = ( (Π + Πᵀ) x(τ) )_a

공상태는 '집단이 최종적으로 나눌 큰 보상' 을 **시간의 역방향**으로 실어 나른다.
그 결과 p_C > p_D 인 영역에서는 이기적 개체도 협력을 택한다.

본 구현은 이 경계값문제를 forward-backward sweep(FBSM)으로 푼다. K-유형으로
자연 일반화하여, 동일 Π 위에서 RE 궤적과 ORE 궤적을 짝지어 비교한다.

────────────────────────────────────────────────────────────────────────
H5 의 조작화: "HalloReg 는 세대에 걸쳐 생존하는가"
────────────────────────────────────────────────────────────────────────
주어진 초기 조성 x₀ (= 조합/30) 에서 출발해 RE·ORE 를 각각 적분한 뒤,
종착 조성 x(τ) 에서 HalloReg 빈도가 생존 임계를 넘는지 본다.

    survival(x₀) = 1[ x_HalloReg(τ) > threshold ]

임계는 소멸(extinction)과 잔존을 가르는 값으로, 기본 1/30 (= 개체 1명분)을 쓴다.
'생존 유역(basin of survival)' 의 점유율을 유형 간에 비교하면 어떤 유형이 더 넓은
초기조건 집합에서 살아남는지 알 수 있다.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# ==================================================================== RE
def replicator_trajectory(Pi: np.ndarray, x0: Sequence[float],
                          steps: int = 600) -> np.ndarray:
    """
    이산 복제자 갱신의 궤적. 반환 shape (steps + 1, k).

    보수가 음수일 수 있으므로(교착 레짐에서 R < 0 은 아니지만 일반성을 위해)
    적합도를 최소값 기준으로 이동시켜 양수화한다. 복제자 동역학은 보수의
    **아핀 변환에 불변**이므로 이 이동은 궤적의 정성적 성질을 바꾸지 않는다.
    """
    P = np.asarray(Pi, dtype=float)
    shift = max(0.0, -P.min()) + 1e-6
    x = np.clip(np.asarray(x0, dtype=float), 1e-12, None)
    x = x / x.sum()
    traj = np.empty((steps + 1, len(x)))
    traj[0] = x
    for t in range(steps):
        f = P @ x + shift
        fbar = float(x @ f)
        x = x * f / max(fbar, 1e-12)
        x = np.clip(x, 1e-15, None)
        x = x / x.sum()
        traj[t + 1] = x
    return traj


def replicator_ends_batch(Pi: np.ndarray, X0: np.ndarray,
                          steps: int = 600) -> np.ndarray:
    """
    다수 초기점의 RE 종착 조성만 반환 (N, k) — 유역 분석용.
    초기점 축을 완전 벡터화하여 시간에 대한 순차 적분만 남긴다.
    """
    P = np.asarray(Pi, dtype=float)
    shift = max(0.0, -P.min()) + 1e-6
    X = np.clip(np.asarray(X0, dtype=float), 1e-12, None)
    X = X / X.sum(axis=1, keepdims=True)
    for _ in range(steps):
        F = X @ P.T + shift                       # (N, k)
        fbar = np.sum(X * F, axis=1, keepdims=True)
        X = X * F / np.maximum(fbar, 1e-12)
        X = np.clip(X, 1e-15, None)
        X = X / X.sum(axis=1, keepdims=True)
    return X


# ==================================================================== ORE
def _grad_g(Pi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """최종 평균적합도 g(x) = xᵀΠx 의 기울기 ∇g = (Π + Πᵀ)x — 터미널 공상태."""
    return (Pi + Pi.T) @ x


def ore_trajectory(Pi: np.ndarray, x0: Sequence[float], tau: float = 1.0,
                   steps: int = 400, sweeps: int = 60, relax: float = 0.5,
                   tol: float = 1e-7) -> Dict:
    """
    K-유형 ORE 궤적을 forward-backward sweep 으로 계산한다.

    반환: x (steps+1, k), p (steps+1, k), x_end (k,), converged, resid
    """
    P = np.asarray(Pi, dtype=float)
    x0 = np.clip(np.asarray(x0, dtype=float), 1e-12, None)
    x0 = x0 / x0.sum()
    k = len(x0)
    dt = tau / steps

    # 공상태 방정식 ṗ = ⟨p⟩p − ½p² 는 Riccati 형이라 유한시간 폭발이 가능하다.
    # 보수 규모의 배수로 클램프해 수치 안정성을 확보한다. 유역 판정은 종착
    # 조성의 부호에만 의존하므로 클램프가 정성적 결론을 바꾸지 않는다.
    p_bound = 20.0 * max(1.0,
                         float(np.max(np.abs(_grad_g(P, np.ones(k) / k)))),
                         float(np.max(np.abs(P))))

    x = np.tile(x0, (steps + 1, 1))
    p = np.tile(_grad_g(P, x0), (steps + 1, 1))
    prev = None
    resid = np.inf

    for _ in range(sweeps):
        # --- forward: 상태 x ---
        x[0] = x0
        for n in range(steps):
            pm = float(x[n] @ p[n])
            xn = np.clip(x[n] + dt * (x[n] * (p[n] - pm)), 1e-12, None)
            x[n + 1] = xn / xn.sum()
        # --- backward: 공상태 p (터미널 조건에서 역방향) ---
        p_new = p.copy()
        p_new[steps] = np.clip(_grad_g(P, x[steps]), -p_bound, p_bound)
        for n in range(steps, 0, -1):
            pm = float(x[n] @ p_new[n])
            dpdt = pm * p_new[n] - 0.5 * p_new[n] ** 2
            p_new[n - 1] = np.clip(p_new[n] - dt * dpdt, -p_bound, p_bound)
        # --- 완화(relaxation): 진동 억제 ---
        p = (1.0 - relax) * p + relax * p_new
        if prev is not None:
            resid = float(np.max(np.abs(p - prev)))
            if resid < tol:
                break
        prev = p.copy()

    return {"x": x, "p": p, "x_end": x[steps].copy(),
            "converged": bool(resid < tol), "resid": float(resid)}


def ore_ends_batch(Pi: np.ndarray, X0: np.ndarray, tau: float = 1.0,
                   steps: int = 220, sweeps: int = 30,
                   relax: float = 0.5) -> np.ndarray:
    """
    다수 초기점의 ORE 종착 조성 (N, k) — 초기점 축 완전 벡터화 FBSM.
    상태·공상태를 (steps+1, N, k) 텐서로 두고 시간 축만 순차 적분한다.
    """
    P = np.asarray(Pi, dtype=float)
    X0 = np.clip(np.asarray(X0, dtype=float), 1e-12, None)
    X0 = X0 / X0.sum(axis=1, keepdims=True)
    N, k = X0.shape
    dt = tau / steps
    G = P + P.T
    p_bound = 20.0 * max(1.0, float(np.max(np.abs(G @ (np.ones(k) / k)))),
                         float(np.max(np.abs(P))))

    x = np.tile(X0, (steps + 1, 1, 1))                       # (S+1, N, k)
    p = np.tile((X0 @ G.T)[None, :, :], (steps + 1, 1, 1))   # (S+1, N, k)

    for _ in range(sweeps):
        x[0] = X0
        for n in range(steps):
            pm = np.sum(x[n] * p[n], axis=1, keepdims=True)
            xn = np.clip(x[n] + dt * (x[n] * (p[n] - pm)), 1e-12, None)
            x[n + 1] = xn / xn.sum(axis=1, keepdims=True)
        p_new = p.copy()
        p_new[steps] = np.clip(x[steps] @ G.T, -p_bound, p_bound)
        for n in range(steps, 0, -1):
            pm = np.sum(x[n] * p_new[n], axis=1, keepdims=True)
            dpdt = pm * p_new[n] - 0.5 * p_new[n] ** 2
            p_new[n - 1] = np.clip(p_new[n] - dt * dpdt, -p_bound, p_bound)
        p = (1.0 - relax) * p + relax * p_new
    return x[steps].copy()


# ==================================================================== 생존지표
def survival_fraction(ends: np.ndarray, idx: int,
                      threshold: float = 1.0 / 30.0) -> float:
    """
    종착 조성 집합에서 유형 idx 의 **생존 유역 점유율**.
    threshold 기본값 1/30 은 30명 집단의 개체 1명분에 해당한다.
    """
    return float(np.mean(np.asarray(ends)[:, idx] > threshold))


def mean_terminal_frequency(ends: np.ndarray, idx: int) -> float:
    """종착 조성에서 유형 idx 의 평균 빈도."""
    return float(np.mean(np.asarray(ends)[:, idx]))


def dominance_fraction(ends: np.ndarray, idx: int) -> float:
    """유형 idx 가 종착 조성에서 최대 빈도를 갖는 초기점의 비율."""
    E = np.asarray(ends)
    return float(np.mean(np.argmax(E, axis=1) == idx))


def terminal_cc(CCm: np.ndarray, ends: np.ndarray) -> float:
    """
    종착 조성들의 평균 **행동적** 상호협력률 CC(x) = xᵀ·CCm·x.

    유형 '라벨' 이 아니라 관측된 CC 행동에서 직접 유도되는 연속 지표라,
    임의 임계에 의한 이분화나 라벨-행동 괴리(교착 레짐에서 협력 라벨 유형이
    실제로는 배신하는 경우) 문제를 피한다.
    """
    E = np.asarray(ends)
    return float(np.mean(np.einsum("ni,ij,nj->n", E, np.asarray(CCm), E)))

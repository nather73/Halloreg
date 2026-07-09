"""
core.generative
===============

4-상태 joint-outcome POMDP 의 생성 모형(A, B, C, D)과 그에 대한 **해석적 EFE**.

[배경]
Albarracin et al. (2026) 은 pymdp 로 변분 상태추론과 EFE 를 계산한다. 그러나 본 과제의
생성 모형은 매우 특수하다:

    A = I₄  (joint outcome 직접 관측; 완전 관측)
    policy_len = 1  (myopic; sophisticated planning 은 별도 모듈이 담당)

이 조건에서 pymdp 의 상태추론 posterior 는 관측 상태의 one-hot 에 수렴하고,
EFE 는 아래의 닫힌 형태와 **해석적으로 정확히 동일**하다:

    neg_EFE(a) = E_{s'~B(·|a,pc)}[ C[s'] ]  +  H[ B(·|a,pc) ]
                 └── 실용가치(pragmatic) ──┘    └ 상태 정보이득(epistemic; A=I) ┘

    G(a) = - neg_EFE(a)     (작을수록 선호)

여기서 pc 는 ToM 이 예측한 상대 협력확률이고, B(·|a,pc) 는:

    a=C  ->  {CC: pc, CD: 1-pc}
    a=D  ->  {DC: pc, DD: 1-pc}

[검증] pymdp 1.0.3(JAX) 의 `infer_policies` 가 반환한 neg_efe 와 본 공식이
소수점 이하까지 일치함을 확인했다 (pc=0.5, C=PAYOFF_SELF → [2.19314718, 3.69314718]).
따라서 본 numpy 경로는 pymdp 경로의 정확한 대체이며, 훨씬 빠르고(수천 배) 재귀적
확장(상대의 epistemic value, depth-2 ToM)을 명료하게 표현할 수 있다.

pymdp 1.0.x(JAX) 백엔드를 직접 쓰고 싶다면 `core.pymdp_backend` 를 사용하라
(동일한 EFE 를 pymdp Agent + eqx.tree_at 조망수용으로 계산; 등가성 테스트 포함).
"""

from __future__ import annotations

import numpy as np

from .constants import (
    CC, CD, DC, DD, COOP, DEFECT,
    N_STATES, N_ACTIONS, PAYOFF_SELF, PAYOFF_OTHER,
)

_EPS = 1e-12


# ------------------------------------------------------------------ 생성 모형
def build_A() -> np.ndarray:
    """관측 우도 A = I₄ (joint outcome 직접 관측)."""
    return np.eye(N_STATES)


def build_B(prior_opp_coop: float = 0.5) -> np.ndarray:
    """
    행동조건부 전이 B[s', s, a] : 내 행동 a 가 고정되면 다음 결과는 상대 협력확률
    pc 에 따라 두 결과로 분산 (현재 상태 s 와 무관 — Albarracin et al. 단순화).

    반환 shape: (N_STATES, N_STATES, N_ACTIONS).
    """
    pc = float(np.clip(prior_opp_coop, _EPS, 1 - _EPS))
    pd = 1.0 - pc
    B = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        B[CC, s, COOP] = pc
        B[CD, s, COOP] = pd
        B[DC, s, DEFECT] = pc
        B[DD, s, DEFECT] = pd
    return B


def build_C(perspective: str = "self") -> np.ndarray:
    """로그-선호 C (표준 PD 보수). perspective ∈ {'self','other'}."""
    return PAYOFF_SELF.copy() if perspective == "self" else PAYOFF_OTHER.copy()


def build_D() -> np.ndarray:
    """균등 초기 상태 사전."""
    return np.ones(N_STATES) / N_STATES


# ------------------------------------------------------------------ 해석적 EFE
def predicted_next_state(pc: float) -> np.ndarray:
    """
    각 행동에 대한 예측 다음-상태 분포를 (N_ACTIONS, N_STATES) 로 반환.
    행 0 = COOP 시 분포, 행 1 = DEFECT 시 분포.
    """
    pc = float(np.clip(pc, _EPS, 1 - _EPS))
    pd = 1.0 - pc
    dist = np.zeros((N_ACTIONS, N_STATES))
    dist[COOP, CC] = pc
    dist[COOP, CD] = pd
    dist[DEFECT, DC] = pc
    dist[DEFECT, DD] = pd
    return dist


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def efe_terms(pc: float, C: np.ndarray) -> dict:
    """
    선호 C 하에서, 상대 협력확률 pc 를 가정한 각 행동의 EFE 분해.

    반환:
      pragmatic : (N_ACTIONS,)  E_{s'}[C[s']]
      epistemic : (N_ACTIONS,)  H[B(·|a,pc)]   (상태 정보이득; A=I)
      neg_efe   : (N_ACTIONS,)  pragmatic + epistemic   (클수록 선호)
      G         : (N_ACTIONS,)  -neg_efe                (작을수록 선호)
    """
    dist = predicted_next_state(pc)                 # (2, 4)
    pragmatic = dist @ np.asarray(C, dtype=float)   # (2,)
    epistemic = np.array([_entropy(dist[a]) for a in range(N_ACTIONS)])
    neg_efe = pragmatic + epistemic
    return {
        "pragmatic": pragmatic,
        "epistemic": epistemic,
        "neg_efe": neg_efe,
        "G": -neg_efe,
    }


def G_self(pc: float) -> np.ndarray:
    """내 관점 EFE G_self(a)  (작을수록 선호). shape (N_ACTIONS,)."""
    return efe_terms(pc, PAYOFF_SELF)["G"]


def G_other_perspective(pc: float) -> np.ndarray:
    """
    조망수용(perspective-taking) EFE : 선호만 상대 보수로 교체한 '상대의 EFE'를
    내 모형 안에서 시뮬레이션. shape (N_ACTIONS,), 인덱스는 *내 행동* 기준.

    주의: 이는 "내가 a 를 두었을 때 상대 관점의 EFE" 이며, Albarracin eq.(4) 의
    첫 항(내 정책이 상대 후생에 주는 영향)에 대응한다. 상대의 *행동선택* EFE
    (eq.5, best-response)는 tom_core 에서 별도로 다룬다.
    """
    return efe_terms(pc, PAYOFF_OTHER)["G"]


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """온도 파라미터가 있는 softmax (수치 안정)."""
    z = np.asarray(x, dtype=float) / max(temperature, _EPS)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

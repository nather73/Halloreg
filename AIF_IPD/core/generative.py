"""
core.generative
===============

4-상태 joint-outcome POMDP 의 생성모형(A, B, C, D)과 그에 대한 **해석적 EFE**.

[왜 해석적으로 풀 수 있는가]
Albarracin et al. (2026) 은 pymdp 로 변분 상태추론과 기대자유에너지(EFE)를 계산한다.
그러나 본 과제의 생성모형은 두 가지 이유로 매우 특수하다.

    (1) A = I₄  — joint outcome 을 직접 관측한다(완전 관측). 상태추론 사후는
                  관측 상태의 one-hot 으로 수렴한다.
    (2) policy_len = 1 — 한 스텝 EFE. 다단계 계획은 별도 플래너 모듈이 담당한다.

이 조건에서 pymdp 의 EFE 는 아래 닫힌 형태와 **해석적으로 정확히 동일**하다.

    neg_EFE(a) = E_{s'~B(·|a, pc)}[ C[s'] ]  +  H[ B(·|a, pc) ]
                 └─── 실용가치(pragmatic) ───┘   └ 상태 정보이득(A=I 이므로 엔트로피) ┘

    G(a) = − neg_EFE(a)                      (작을수록 선호)

여기서 pc 는 ToM 이 예측한 상대 협력확률이고, 전이 B 는:

    a = C  →  {CC: pc, CD: 1 − pc}
    a = D  →  {DC: pc, DD: 1 − pc}

따라서 numpy 해석해는 pymdp 경로의 정확한 대체이며, 수천 배 빠르고 재귀적 확장
(상대의 EFE, 자기-사영)을 명료하게 표현할 수 있다.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    CC, CD, DC, DD, COOP, DEFECT,
    N_STATES, N_ACTIONS, PAYOFF_SELF, PAYOFF_OTHER,
)

_EPS = 1e-12


# ------------------------------------------------------------------ 생성모형
def build_A() -> np.ndarray:
    """관측 우도 A = I₄ (joint outcome 직접 관측)."""
    return np.eye(N_STATES)


def build_B(prior_opp_coop: float = 0.5) -> np.ndarray:
    """
    행동조건부 전이 B[s', s, a].

    내 행동 a 가 정해지면 다음 결과는 상대 협력확률 pc 로만 결정되고 현재 상태 s
    와는 무관하다(Albarracin et al. 의 단순화). 반환 shape (S', S, A).
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
    """로그-선호 C. perspective ∈ {'self', 'other'} — 조망수용은 C 만 교체한다."""
    return PAYOFF_SELF.copy() if perspective == "self" else PAYOFF_OTHER.copy()


def build_D() -> np.ndarray:
    """균등 초기 상태 사전 D."""
    return np.ones(N_STATES) / N_STATES


# ------------------------------------------------------------------ 해석적 EFE
def predicted_next_state(pc: float) -> np.ndarray:
    """
    각 행동에 대한 예측 다음-상태 분포. 반환 shape (N_ACTIONS, N_STATES).
    행 0 = 내가 C 를 둘 때의 분포, 행 1 = 내가 D 를 둘 때의 분포.
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
    """Shannon 엔트로피 (자연로그)."""
    p = np.clip(p, _EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def efe_terms(pc: float, C: np.ndarray) -> dict:
    """
    선호 C 하에서 상대 협력확률 pc 를 가정한 각 행동의 EFE 분해.

    반환 dict:
      pragmatic : (A,)  E_{s'}[C[s']]        — 기대효용
      epistemic : (A,)  H[B(·|a, pc)]        — 상태 불확실성 해소분
      neg_efe   : (A,)  pragmatic + epistemic (클수록 선호)
      G         : (A,)  −neg_efe             (작을수록 선호)
    """
    dist = predicted_next_state(pc)                  # (2, 4)
    pragmatic = dist @ np.asarray(C, dtype=float)    # (2,)
    epistemic = np.array([_entropy(dist[a]) for a in range(N_ACTIONS)])
    neg_efe = pragmatic + epistemic
    return {"pragmatic": pragmatic, "epistemic": epistemic,
            "neg_efe": neg_efe, "G": -neg_efe}


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """온도 파라미터가 있는 수치안정 softmax."""
    z = np.asarray(x, dtype=float) / max(temperature, _EPS)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

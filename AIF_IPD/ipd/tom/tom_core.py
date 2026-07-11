"""
ipd.tom.tom_core
================

Theory-of-Mind 핵심: 상대 행동 예측 q(a_j|h_t) 과 **재귀적 social EFE**.

Albarracin et al. (2026) eq.(4,7):
    G_social(a_i) = (1-λ)·G_self(a_i) + λ·E_{q(a_j|h_t)}[G_other(a_j)] + G_epistemic

[HalloReg 확장 — 두 가지 재귀성]
  (R1) 재귀적 ToM: 원 논문은 상대 EFE 계산 시 static ToM 만 가정했으나, 본 모형은
       상대 또한 static ⊕ learned ToM 을 모두 사용한다고 가정한다. 즉 상대의 행동선택
       q(a_j|h_t) 를 구할 때, 상대가 '나'를 예측하는 믿음 π_i 자체를 best-response 한
       단계(depth-2)로 정련한다.
  (R2) 재귀적 epistemic: 원 논문은 social EFE 에 상대의 expected *pragmatic* value 만
       포함했으나, 본 모형은 상대의 expected *epistemic* value(상대가 나를 학습하며
       얻는 정보이득)도 λ 가중으로 추가한다.

이 모든 EFE 는 `core.generative` 의 해석적 numpy EFE(=pymdp EFE 와 등가)로 계산된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np

from AIF_IPD.core.constants import (
    COOP, DEFECT, PD_PAYOFFS, PAYOFF_SELF, PAYOFF_OTHER,
)
from AIF_IPD.core.generative import efe_terms, softmax
from .inversion import OpponentInversion, ObservationContext


# --------------------------------------------------------------- static ToM
class TheoryOfMind:
    """
    상대를 '나와 구조적으로 동형인 합리적 행위자'로 보는 static ToM.

    상대의 행동선택:  q(a_j|h_t) ∝ exp( β_j · negEFE_j(a_j) )
    상대의 negEFE_j(a_j) = 상대가 자기 보수 하에서 얻는 기대가치. 상대는 내 정책 π_i
    (내 협력율에 대한 상대의 믿음)에 대해 기대한다.
    """

    def __init__(self, beta_other: float = 4.0,
                 use_pragmatic: bool = True, use_epistemic: bool = True):
        self.beta_other = beta_other
        self.use_pragmatic = use_pragmatic
        self.use_epistemic = use_epistemic
        self._believed_my_policy = np.array([0.5, 0.5])  # 상대가 믿는 내 (C,D) 확률

    def update_my_policy_belief(self, my_coop_rate: float):
        self._believed_my_policy = np.array([my_coop_rate, 1.0 - my_coop_rate])

    def opponent_efe(self, believed_my_policy: Optional[np.ndarray] = None) -> np.ndarray:
        """상대의 각 행동에 대한 negEFE (상대 보수 관점). shape (2,) = [C,D]."""
        pi = self._believed_my_policy if believed_my_policy is None else believed_my_policy
        negG = np.zeros(2)
        for a_j in (COOP, DEFECT):
            val = 0.0
            for a_i in (COOP, DEFECT):
                _, other_payoff = PD_PAYOFFS[(a_i, a_j)]
                val += pi[a_i] * other_payoff
            negG[a_j] = val
        return negG

    def predict_opponent_action(self,
                                believed_my_policy: Optional[np.ndarray] = None
                                ) -> np.ndarray:
        """static ToM 상대 행동 분포 q(a_j|h_t) = softmax(β_j·negEFE_j)."""
        negG = self.opponent_efe(believed_my_policy)
        return softmax(negG, temperature=1.0 / self.beta_other)


# --------------------------------------------------------------- gated ToM
class GatedToM:
    """
    신뢰도 게이팅된 ToM: static prior 와 learned posterior(입자필터)를
    입자필터 신뢰도 r 로 부드럽게 보간.

        q_gated = r · q_learned + (1 - r) · q_static
    """

    def __init__(self, tom: TheoryOfMind, inversion: OpponentInversion):
        self.tom = tom
        self.inversion = inversion

    def predict_opponent_action(self, ctx: Optional[ObservationContext]) -> np.ndarray:
        r = self.inversion.reliability()
        q_static = self.tom.predict_opponent_action()
        q_learned = self.inversion.predict_action(ctx)
        q = r * q_learned + (1 - r) * q_static
        return q / q.sum()


# --------------------------------------------------------------- social EFE
@dataclass
class SocialEFEResult:
    G_social: np.ndarray            # (2,) 각 내 행동
    G_self: np.ndarray
    G_other_pragmatic: np.ndarray
    G_epistemic_self: np.ndarray
    G_epistemic_other: np.ndarray
    q_response: np.ndarray          # 상대 행동 예측
    info: dict


class RecursiveSocialEFE:
    """
    재귀적 social EFE 계산기.

    G_social(a_i) = (1-λ)·G_self(a_i | q(a_j))
                    + λ·E_{q(a_j)}[ G_other(a_j) ]                       (R: pragmatic)
                    − w_epi_self  · IG_self(a_i)                          (내 epistemic)
                    − λ·w_epi_other · IG_other(a_i)                       (R2: 상대 epistemic)

    q(a_j) 는 GatedToM 예측이며, recursive_depth=2 면 상대가 나를 best-response 하는
    한 단계를 추가로 정련한다(R1).
    """

    def __init__(self, gated_tom: GatedToM, inversion: OpponentInversion,
                 empathy_factor: float = 0.4, beta_self: float = 4.0,
                 w_epi_self: float = 0.6, w_epi_other: float = 0.3,
                 recursive_depth: int = 2):
        self.gated = gated_tom
        self.inversion = inversion
        self.lam = empathy_factor
        self.beta_self = beta_self
        self.w_epi_self = w_epi_self
        self.w_epi_other = w_epi_other
        self.recursive_depth = recursive_depth
        self.my_coop_rate = 0.5

    # ---- 재귀적 상대 예측 (R1) ----
    def _recursive_opponent_prediction(self, ctx: Optional[ObservationContext],
                                       my_last_action: int) -> np.ndarray:
        q = self.gated.predict_opponent_action(ctx)   # depth-1 (gated)
        if self.recursive_depth >= 2:
            # depth-2: 상대가 '나'를 best-response 한다고 가정하고 상대 예측을 정련.
            # 상대가 믿는 내 정책 π_i 를, 내 협력율(level-0) 로부터 한 단계 best-response
            # (상대는 내가 상대에게 최적반응한다고 가정) 하여 갱신.
            tom = self.gated.tom
            tom.update_my_policy_belief(self.my_coop_rate)
            # 상대가 예상하는 나의 best-response: 내 G_self 최소화 행동
            pc_learned = float(q[COOP])  # 상대 협력확률 추정
            g_self = efe_terms(pc_learned, PAYOFF_SELF)["G"]
            my_br = softmax(-g_self, temperature=1.0 / self.beta_self)
            # 상대는 이 π_i 로 자기 EFE 를 다시 계산 → 정련된 q(a_j)
            q2 = tom.predict_opponent_action(believed_my_policy=my_br)
            r = self.inversion.reliability()
            # 신뢰도만큼 재귀 정련을 반영
            q = (1 - 0.5 * r) * q + 0.5 * r * q2
            q = q / q.sum()
        return q

    def compute(self, ctx: Optional[ObservationContext],
                my_last_action: int, lam: Optional[float] = None) -> SocialEFEResult:
        lam = self.lam if lam is None else lam
        q = self._recursive_opponent_prediction(ctx, my_last_action)
        pc = float(q[COOP])

        # 내 EFE (pc 하에서)
        G_self = efe_terms(pc, PAYOFF_SELF)["G"]

        # 상대 pragmatic EFE: 내가 a_i 를 두고 상대가 q 로 반응할 때 상대 보수
        G_other = np.zeros(2)
        for a_i in (COOP, DEFECT):
            val = 0.0
            for a_j in (COOP, DEFECT):
                _, other_payoff = PD_PAYOFFS[(a_i, a_j)]
                val += q[a_j] * other_payoff
            G_other[a_i] = -val   # EFE(작을수록 선호) = -기대보수
        G_other_expected = G_other  # 이미 내 행동별로 정리됨

        # 내 epistemic: 내가 a_i 를 둘 때 상대 다음행동으로부터 θ 정보이득
        f_next = np.array([+1.0, -1.0])  # COOP→+1, DEFECT→-1
        IG_self = np.array([
            self.inversion.expected_infogain(COOP, f_next[COOP]),
            self.inversion.expected_infogain(DEFECT, f_next[DEFECT]),
        ])

        # 상대 epistemic (R2): 상대가 '나'를 학습하며 얻는 정보이득.
        # 대칭 근사 — 상대 관점에서 내 행동이 상대에게 주는 정보이득은, 내가 협력/배신을
        # 얼마나 예측가능하게(=상대의 나에 대한 불확실성 감소) 하는가로 근사한다.
        # 협력(예측된 관계)일수록 상대의 나에 대한 posterior 를 더 잘 정련한다고 본다.
        r = self.inversion.reliability()
        IG_other = np.array([r * 0.5, r * 0.2])  # C 가 D 보다 상대에게 더 정보적

        G_epi_self = -self.w_epi_self * IG_self
        G_epi_other = -lam * self.w_epi_other * IG_other

        G_social = ((1 - lam) * G_self
                    + lam * G_other_expected
                    + G_epi_self
                    + G_epi_other)

        return SocialEFEResult(
            G_social=G_social,
            G_self=G_self,
            G_other_pragmatic=G_other_expected,
            G_epistemic_self=G_epi_self,
            G_epistemic_other=G_epi_other,
            q_response=q,
            info={"pc": pc, "lam": lam, "reliability": r},
        )

    def select_action(self, ctx: Optional[ObservationContext],
                      my_last_action: int, lam: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[int, SocialEFEResult]:
        res = self.compute(ctx, my_last_action, lam)
        q_pi = softmax(-res.G_social, temperature=1.0 / self.beta_self)
        rng = rng or np.random.default_rng()
        action = COOP if rng.random() < q_pi[COOP] else DEFECT
        res.info["q_pi"] = q_pi
        res.info["action"] = action
        return action, res

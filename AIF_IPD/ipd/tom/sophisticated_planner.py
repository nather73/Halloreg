"""
ipd.tom.sophisticated_planner
=============================

Albarracin et al. (2026) 의 **planning horizon** 을 그대로 이식한 다단계 계획.

1. 2^H 개 후보 정책(C/D 시퀀스)을 열거
2. 각 정책을 H 스텝 forward rollout 하며 social EFE 누적
3. softmax 정책분포를 첫 행동으로 marginalize 하여 행동 선택

원 논문의 핵심 발견(planning depth 증가가 중간 λ 에서 협력 임계를 우측 이동)을
재현할 수 있도록, 각 스텝의 social EFE 는 (1-λ)G_self + λ·G_other 로 구성한다.

동시행동 게임: 상대 예측 q(a_j|h_t) 는 내 현재행동에 조건화되지 않는다.
"""

from __future__ import annotations

from itertools import product
from typing import Tuple, Dict

import numpy as np

from AIF_IPD.core.constants import COOP, DEFECT, PD_PAYOFFS
from AIF_IPD.core.generative import softmax
from .opponent_simulator import OpponentSimulator
from .inversion import ObservationContext


class SophisticatedPlanner:
    """
    social EFE 기반 다단계 rollout 계획기.

    [v0.9.0 §5.3] planner 의 각 rollout 스텝에 **전축 histogram IG** 를 이식한다.
    현행(v0.8.2)은 pragmatic 만 써서(S7) h1↔h2 대조가 "깊이+인식항" 혼합이었다.
    social_efe(RecursiveSocialEFE)의 step_terms 를 재사용해 depth-EFE 경로와 항
    구성을 일치시키면 h2 에서 순수 깊이 대조에 가까워진다. social_efe 가 없으면
    pragmatic-만(구 동작) 으로 후퇴한다(하위호환).
    """

    def __init__(self, opponent_sim: OpponentSimulator,
                 empathy_factor: float = 0.4, horizon: int = 3,
                 beta_self: float = 4.0, social_efe=None,
                 base_ctx: ObservationContext = None):
        self.opponent_sim = opponent_sim
        self.lam = empathy_factor
        self.horizon = horizon
        self.beta_self = beta_self
        self.social_efe = social_efe            # RecursiveSocialEFE (§5.3 항 재사용)
        self.base_ctx = base_ctx
        self.policies = list(product([COOP, DEFECT], repeat=horizon))

    def evaluate_policy(self, policy: Tuple[int, ...]) -> float:
        """
        정책의 평균 social EFE. 낮을수록 선호.

        [v0.7.1 §3] rollout_reciprocity=True 면 가상 궤적을 따라 f_virtual 을
        갱신하며 rollout 한다. step t 의 상대는 내 **직전** 가상행동 policy[t−1] 에
        반응하므로 호혜의 도구적 가치가 G_self 에 자연 발생한다(λ 무관).
        [v0.9.0 §5.3] 각 스텝의 항을 social_efe.step_terms 로 계산 —
        pragmatic 키 + 전축 IG_self/IG_other, 가중은 (1−λ),λ 뿐.
        """
        total_G = 0.0
        lam = self.lam
        my_prev = None
        opp_prev = None
        for t, a_i in enumerate(policy):
            q = self.opponent_sim.predict_response(
                step=t, my_virtual_last=my_prev, opp_virtual_last=opp_prev)
            if self.social_efe is not None:
                # rollout 스텝의 맥락: 내/상대의 가상 직전행동(step 0 은 실제 ctx).
                if t == 0 and self.base_ctx is not None:
                    ctx_t = self.base_ctx
                else:
                    ctx_t = ObservationContext(
                        my_last_action=my_prev, their_last_action=opp_prev,
                        round_number=t)
                terms = self.social_efe.step_terms(ctx_t, q, lam)
                self_branch = terms["prag_self"][a_i] - terms["IG_self"][a_i]
                other_branch = terms["prag_other"][a_i] - terms["IG_other"][a_i]
                total_G += (1 - lam) * self_branch + lam * other_branch
            else:
                G_self = 0.0
                G_other = 0.0
                for a_j in (COOP, DEFECT):
                    my_p, other_p = PD_PAYOFFS[(a_i, a_j)]
                    G_self += q[a_j] * (-my_p)
                    G_other += q[a_j] * (-other_p)
                total_G += (1 - lam) * G_self + lam * G_other
            my_prev = a_i
            opp_prev = COOP if q[COOP] >= 0.5 else DEFECT
        return total_G / self.horizon

    def plan(self, lam: float = None) -> Tuple[np.ndarray, Tuple[int, ...], Dict]:
        if lam is not None:
            self.lam = lam
        G = np.array([self.evaluate_policy(p) for p in self.policies])
        q_pi = softmax(-G, temperature=1.0 / self.beta_self)
        q_action = np.zeros(2)
        for i, p in enumerate(self.policies):
            q_action[p[0]] += q_pi[i]
        q_action /= q_action.sum()
        best = self.policies[int(np.argmin(G))]
        return q_action, best, {"G_policies": G, "q_pi": q_pi}

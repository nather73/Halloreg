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


class SophisticatedPlanner:
    """social EFE 기반 다단계 rollout 계획기."""

    def __init__(self, opponent_sim: OpponentSimulator,
                 empathy_factor: float = 0.4, horizon: int = 3,
                 beta_self: float = 4.0):
        self.opponent_sim = opponent_sim
        self.lam = empathy_factor
        self.horizon = horizon
        self.beta_self = beta_self
        self.policies = list(product([COOP, DEFECT], repeat=horizon))

    def evaluate_policy(self, policy: Tuple[int, ...]) -> float:
        """
        정책의 평균 social EFE. 낮을수록 선호.

        [v0.7.1 §3] rollout_reciprocity=True 면 가상 궤적을 따라 f_virtual 을
        갱신하며 rollout 한다. step t 의 상대는 내 **직전** 가상행동 policy[t−1] 에
        반응하므로, "협력 → 상대 협력 유도 → 내 미래 보수↑" 의 도구적 경로가
        G_self 에 자연 발생한다. λ 는 이 경로에 전혀 관여하지 않는다(구성개념 보존).
        """
        total_G = 0.0
        lam = self.lam
        # 가상 궤적 상태: 내 직전 행동 / 상대 직전 행동(기대 최빈)
        my_prev = None
        opp_prev = None
        for t, a_i in enumerate(policy):
            q = self.opponent_sim.predict_response(
                step=t, my_virtual_last=my_prev, opp_virtual_last=opp_prev)
            G_self = 0.0
            G_other = 0.0
            for a_j in (COOP, DEFECT):
                my_p, other_p = PD_PAYOFFS[(a_i, a_j)]
                G_self += q[a_j] * (-my_p)
                G_other += q[a_j] * (-other_p)
            total_G += (1 - lam) * G_self + lam * G_other
            # 다음 step 의 조건화: 내 이번 행동이 다음 라운드 상대의 f 가 된다.
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

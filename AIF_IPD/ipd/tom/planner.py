"""
ipd.tom.planner
===============

**다단계 계획(sophisticated planning).**

1. 2^H 개 후보 정책(C/D 시퀀스)을 열거한다.
2. 각 정책을 H 스텝 forward rollout 하며 social EFE 를 누적한다.
3. 정책 softmax 분포를 첫 행동으로 주변화하여 행동을 선택한다.

[rollout 안의 상대 예측 — 호혜의 도구적 가치]
동시행동 게임이므로 **같은 라운드**의 상대 예측은 내 현재 행동에 조건화되지
않는다. 그러나 **다음** 라운드의 상대는 내 이번 행동에 반응한다. 따라서 rollout
step > 0 에서는 추론된 ρ̂ (호혜성)로 내 가상 직전행동에 조건화한 예측을 낸다:

    P(a_j = C | f_virt, g_virt) = σ( β̂ (α̂ + ρ̂·f_virt + ω̂·g_virt + η̂·f_virt·g_virt + s) )

이렇게 해야 "내가 지금 협력하면 다음 라운드에 되돌아온다"는 계산이 **λ 가 아니라
EFE 안에서** 발생한다. 이 구분은 구성개념상 결정적이다: 호혜의 도구적 가치를 λ 로
처리하면 λ 는 공감이 아니라 전략 파라미터가 되어버린다.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np

from AIF_IPD.core.constants import COOP, DEFECT
from AIF_IPD.core.generative import softmax
from .inversion import ObservationContext, OpponentInversion
from .tom_core import GatedToM, RecursiveSocialEFE, TheoryOfMind


class OpponentSimulator:
    """
    rollout 각 스텝에서 상대 행동 분포를 공급한다.

      step = 0 : GatedToM (학습된 사후 ⊕ 정적 사전, 신뢰도 게이팅)
      step > 0 : ρ̂ 전파 — 내 가상 직전행동에 조건화한 예측을 신뢰도 게이팅
    """

    def __init__(self, tom: TheoryOfMind, gated_tom: GatedToM,
                 context: Optional[ObservationContext],
                 inversion: OpponentInversion):
        self.tom = tom
        self.gated_tom = gated_tom
        self.context = context
        self.inversion = inversion

    def predict_response(self, step: int = 0,
                         my_virtual_last: Optional[int] = None,
                         opp_virtual_last: Optional[int] = None) -> np.ndarray:
        """rollout step 에서 q(a_j) = [P(C), P(D)]."""
        if step == 0:
            return self.gated_tom.predict_opponent_action(self.context)

        f_virt = (0.0 if my_virtual_last is None
                  else 1.0 - 2.0 * float(my_virtual_last))
        g_virt = (0.0 if opp_virtual_last is None
                  else 1.0 - 2.0 * float(opp_virtual_last))
        pc = self.inversion.predict_coop(f_virt, g_virt)
        r = self.inversion.reliability()
        q = r * np.array([pc, 1.0 - pc]) \
            + (1.0 - r) * self.tom.predict_opponent_action()
        return q / q.sum()


class SophisticatedPlanner:
    """
    social EFE 기반 다단계 rollout 계획기.

    각 rollout 스텝의 항 구성은 RecursiveSocialEFE.step_terms 를 재사용한다.
    이렇게 해야 horizon=1 과 horizon>1 의 대조가 '깊이'의 순수 대조가 되고,
    '깊이 + 인식항 유무'의 혼합 대조가 되지 않는다.
    """

    def __init__(self, opponent_sim: OpponentSimulator,
                 social_efe: RecursiveSocialEFE,
                 base_ctx: Optional[ObservationContext],
                 empathy_factor: float = 0.4, horizon: int = 2,
                 beta_self: float = 4.0):
        self.sim = opponent_sim
        self.social_efe = social_efe
        self.base_ctx = base_ctx
        self.lam = float(empathy_factor)
        self.horizon = int(horizon)
        self.beta_self = float(beta_self)
        self.policies = list(product([COOP, DEFECT], repeat=self.horizon))
        # 스텝 항 메모이제이션 캐시.
        # 서로 다른 정책들이 같은 (step, my_prev, opp_prev) 조합을 반복해서 거친다.
        # 예컨대 horizon=2 에서 step 0 은 4개 정책 모두 동일하다. step_terms 는
        # 전축 정보이득 계산이 들어 있어 가장 비싼 연산이므로, 이 캐시가 라운드당
        # 비용을 8회 → 3회로 줄인다(수치는 완전히 동일; 순수 속도 개선).
        self._cache: Dict[tuple, dict] = {}

    def _terms(self, step: int, my_prev, opp_prev) -> Tuple[dict, np.ndarray]:
        """(step, my_prev, opp_prev) 조건의 상대 예측 q 와 항 분해를 캐시와 함께 반환."""
        key = (step if step == 0 else 1, my_prev, opp_prev)
        if key in self._cache:
            return self._cache[key]
        q = self.sim.predict_response(step=step, my_virtual_last=my_prev,
                                      opp_virtual_last=opp_prev)
        ctx_t = (self.base_ctx if (step == 0 and self.base_ctx is not None)
                 else ObservationContext(my_last_action=my_prev,
                                         their_last_action=opp_prev,
                                         round_number=step))
        val = (self.social_efe.step_terms(ctx_t, q), q)
        self._cache[key] = val
        return val

    def evaluate_policy(self, policy: Tuple[int, ...]) -> float:
        """정책의 스텝 평균 social EFE (낮을수록 선호)."""
        total = 0.0
        my_prev = None
        opp_prev = None
        for t, a_i in enumerate(policy):
            terms, q = self._terms(t, my_prev, opp_prev)
            self_branch = terms["prag_self"][a_i] - terms["IG_self"][a_i]
            other_branch = terms["prag_other"][a_i] - terms["IG_other"][a_i]
            total += (1.0 - self.lam) * self_branch + self.lam * other_branch
            my_prev = a_i
            # 가상 궤적의 상대 행동은 예측분포의 MAP 로 확정(결정론적 rollout)
            opp_prev = COOP if q[COOP] >= 0.5 else DEFECT
        return total / self.horizon

    def plan(self, lam: Optional[float] = None
             ) -> Tuple[np.ndarray, Tuple[int, ...], Dict]:
        """
        반환: (첫 행동의 주변 분포 (2,), 최적 정책, 진단 dict).
        """
        if lam is not None:
            self.lam = float(lam)
        self._cache.clear()          # 라운드마다 신념이 바뀌므로 캐시를 비운다
        G = np.array([self.evaluate_policy(p) for p in self.policies])
        q_pi = softmax(-G, temperature=1.0 / self.beta_self)
        q_action = np.zeros(2)
        for i, p in enumerate(self.policies):
            q_action[p[0]] += q_pi[i]
        q_action /= q_action.sum()
        return q_action, self.policies[int(np.argmin(G))], {"G": G, "q_pi": q_pi}

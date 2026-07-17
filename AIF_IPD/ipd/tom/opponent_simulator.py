"""
ipd.tom.opponent_simulator
==========================

다단계 rollout 중 상대 행동을 예측하는 시뮬레이터.

  - step 0 : GatedToM (learned ⊕ static, 신뢰도 게이팅)
  - step>0 : (legacy)              static ToM 으로 후퇴
             (rollout_reciprocity) 추론된 ρ̂ 로 **내 가상 행동에 조건화된** 예측

동시행동 게임이므로 같은 라운드 내 예측은 내 현재 라운드 행동에 조건화되지 않는
q(a_j|h_t) 이다. 그러나 **다음** 라운드의 상대는 내 이번 행동에 반응한다.

[v0.7.1 §3] ρ 의 도구적 가치 경로
---------------------------------
현행(legacy)에는 "내 협력이 다음 라운드에 되돌아온다"는 계산이 어디에도 없다:
  · 동시행동·horizon=1 에서 `predict_action(ctx)` 의 pc 는 내 후보 행동에 의존하지
    않는다 → G_self 실용항은 항상 D 선호(T>R, P>S), 두 행동의 상태 엔트로피가 같아
    epistemic 도 상쇄. 협력은 오직 λ·G_other 와 infogain 에서만 발생한다.
  · `predict_response` 가 step>0 에서 static ToM 으로 후퇴 → ρ 를 rollout 에
    전파하지 않아 planning horizon 을 늘려도 호혜의 도구적 가치가 반영되지 않는다.

**경계 (명세서 §3.1, 중요).** 이 문제를 λ 로 해결하면 안 된다. "호혜적이니 협력이
이득 → λ↑" 는 λ 를 공감이 아니라 전략 파라미터로 바꿔 구성개념을 붕괴시킨다.
ρ 의 도구적 가치는 **λ 가 아니라 EFE/rollout** 이 다뤄야 한다. 따라서 본 확장은
LambdaRegulator 를 전혀 건드리지 않고 planner 의 rollout 예측만 바꾼다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .tom_core import TheoryOfMind, GatedToM
from .inversion import ObservationContext


class OpponentSimulator:
    """SophisticatedPlanner 가 각 rollout step 에서 호출하는 상대 예측 인터페이스.

    Parameters
    ----------
    rollout_reciprocity : bool
        False(기본) 면 step>0 에서 static ToM 으로 후퇴(v0.6.6 동작 보존).
        True 면 추론된 ρ̂ 로 내 가상 직전행동 f_virtual 에 조건화해 예측한다:
            P(a_j=C | f_virt) = σ(β̂(α̂ + ρ̂·f_virt + ω̂·g + η̂·f_virt·g + s))
        horizon=1 에서는 rollout 이 없으므로 무영향.
    inversion : OpponentInversion | None
        ρ̂ 전파에 쓰는 입자필터. rollout_reciprocity=True 면 필수.
    """

    def __init__(self, tom: TheoryOfMind, gated_tom: Optional[GatedToM] = None,
                 context: Optional[ObservationContext] = None,
                 rollout_reciprocity: bool = False,
                 inversion=None):
        self.tom = tom
        self.gated_tom = gated_tom
        self.context = context
        self.rollout_reciprocity = bool(rollout_reciprocity)
        # gated_tom 이 있으면 그 입자필터를 재사용(별도 주입도 허용)
        self.inversion = inversion if inversion is not None else (
            getattr(gated_tom, "inversion", None))

    def predict_response(self, step: int = 0,
                         my_virtual_last: Optional[int] = None,
                         opp_virtual_last: Optional[int] = None) -> np.ndarray:
        """
        rollout step 에서 q(a_j|h_t) = [P(C), P(D)] 반환.

        my_virtual_last : 가상 궤적에서 **내 직전 행동**(step>0 에서만 유효).
            rollout_reciprocity=True 일 때 f_virt = +1(C) / −1(D) 로 변환되어
            상대의 호혜 반응을 유도한다.
        opp_virtual_last : 가상 궤적에서 **상대의 직전 행동** (fg 기저의 g).
        """
        if step == 0 and self.gated_tom is not None:
            return self.gated_tom.predict_opponent_action(self.context)

        # ---- [v0.7.1 §3] ρ̂ 전파 경로 ----
        if (self.rollout_reciprocity and self.inversion is not None
                and my_virtual_last is not None):
            f_virt = 1.0 - 2.0 * float(my_virtual_last)   # C=0→+1, D=1→−1
            g_virt = (None if opp_virtual_last is None
                      else 1.0 - 2.0 * float(opp_virtual_last))
            pc = self.inversion.predict_coop(f_virt, g=g_virt)
            # 신뢰도 게이팅: 입자필터를 아직 못 믿으면 static 쪽으로 수축.
            # (step 0 의 GatedToM 과 동일한 논리를 rollout 에 일관 적용)
            r = self.inversion.reliability()
            q_learned = np.array([pc, 1.0 - pc])
            q_static = self.tom.predict_opponent_action()
            q = r * q_learned + (1.0 - r) * q_static
            return q / q.sum()

        return self.tom.predict_opponent_action()

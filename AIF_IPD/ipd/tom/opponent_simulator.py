"""
ipd.tom.opponent_simulator
==========================

다단계 rollout 중 상대 행동을 예측하는 시뮬레이터.

  - step 0 : GatedToM (learned ⊕ static, 신뢰도 게이팅)
  - step>0 : static ToM (rollout 중에는 새 관측이 없으므로 정적 예측으로 후퇴)

동시행동 게임이므로 예측은 내 현재 라운드 행동에 조건화되지 않는 q(a_j|h_t) 이다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .tom_core import TheoryOfMind, GatedToM
from .inversion import ObservationContext


class OpponentSimulator:
    """SophisticatedPlanner 가 각 rollout step 에서 호출하는 상대 예측 인터페이스."""

    def __init__(self, tom: TheoryOfMind, gated_tom: Optional[GatedToM] = None,
                 context: Optional[ObservationContext] = None):
        self.tom = tom
        self.gated_tom = gated_tom
        self.context = context

    def predict_response(self, step: int = 0) -> np.ndarray:
        """rollout step 에서 q(a_j|h_t) = [P(C), P(D)] 반환."""
        if step == 0 and self.gated_tom is not None:
            return self.gated_tom.predict_opponent_action(self.context)
        return self.tom.predict_opponent_action()

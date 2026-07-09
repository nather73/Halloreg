"""
core.constants
==============

이산 죄수의 딜레마(Prisoner's Dilemma, PD)의 전역 상수와 인덱스 규약.

Albarracin et al. (2026) 및 첨부 프로토타입 `empathic_ipd_pymdp.py` 와 동일한 규약을
사용한다. 4-상태 joint-outcome POMDP:

    state = (내 행동, 상대 행동)
      0 = CC  (나 C, 상대 C)
      1 = CD  (나 C, 상대 D)   -- 내가 협력했는데 상대가 배신 = sucker (예측못한 손실)
      2 = DC  (나 D, 상대 C)
      3 = DD  (나 D, 상대 D)

행동 인덱스:  C(협력)=0, D(배신)=1.

표준 PD 보수 (T > R > P > S) :
    R=3 (상호협력), T=5 (배신 유혹), S=0 (호구), P=1 (상호배신).
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ payoffs
R, T, S, P = 3.0, 5.0, 0.0, 1.0

# ------------------------------------------------------------------ indices
CC, CD, DC, DD = 0, 1, 2, 3
COOP, DEFECT = 0, 1

N_STATES = 4
N_ACTIONS = 2

ACTION_NAMES = {COOP: "C", DEFECT: "D"}
STATE_NAMES = {CC: "CC", CD: "CD", DC: "DC", DD: "DD"}

# 내 관점 / 상대 관점 보수 (대칭). 인덱스: [CC, CD, DC, DD]
PAYOFF_SELF = np.array([R, S, T, P], dtype=float)   # 내 보수
PAYOFF_OTHER = np.array([R, T, S, P], dtype=float)   # 상대 보수

# (my_action, opp_action) -> (my_payoff, opp_payoff)
PD_PAYOFFS = {
    (COOP, COOP): (R, R),
    (COOP, DEFECT): (S, T),
    (DEFECT, COOP): (T, S),
    (DEFECT, DEFECT): (P, P),
}


def joint_index(my_act: int, opp_act: int) -> int:
    """(내 행동, 상대 행동) -> joint-outcome 상태 인덱스."""
    return int(my_act) * 2 + int(opp_act)


def split_joint(state: int) -> tuple[int, int]:
    """joint-outcome 상태 -> (내 행동, 상대 행동)."""
    my_a, opp_a = divmod(int(state), 2)
    return my_a, opp_a


def opponent_action_from_state(state: int) -> int:
    """joint-outcome 상태에서 상대 행동만 추출."""
    return int(state) % 2


def my_action_from_state(state: int) -> int:
    """joint-outcome 상태에서 내 행동만 추출."""
    return int(state) // 2


def mirror_state(state: int | None) -> int | None:
    """focal 의 joint state 를 상대 관점 joint state 로 미러링 ((a_i,a_j)->(a_j,a_i))."""
    if state is None:
        return None
    my_a, opp_a = split_joint(state)
    return joint_index(opp_a, my_a)

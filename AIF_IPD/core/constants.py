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


DEFAULT_COOP_INDEX = round((R - P) / (T - S), 6)   # 현행 게임의 협력지수 = 0.4


def set_coop_index(ci: float | None):
    """
    보수행렬을 협력지수 CI=(R−P)/(T−S) 로 매개화해 재설정한다 (게임구조 일반화).

    T=5, S=0, P=1 고정, R = P + CI·(T−S). 유효 IPD 조건(T>R>P>S, 2R>T+S)을
    만족하려면 CI ∈ (0.2, 0.8] 이며 CI>0.5 에서 2R>T+S 가 성립, CI=0.4(현행)는
    2R=6>5 로 성립. numpy 배열/딕셔너리를 **제자리(in-place)** 갱신하므로
    이미 import 된 참조들(에이전트 생성 이전이라면)에도 반영된다.
    spawn 워커는 태스크마다 이 함수를 호출해 게임을 재설정한다(멱등).
    """
    global R, P
    ci = DEFAULT_COOP_INDEX if ci is None else float(ci)
    R_new = 1.0 + ci * (T - S)
    if not (T > R_new > P > S):
        raise ValueError(f"유효하지 않은 협력지수 {ci}")
    R = R_new
    PAYOFF_SELF[:] = [R, S, T, P]
    PAYOFF_OTHER[:] = [R, T, S, P]
    PD_PAYOFFS.update({(COOP, COOP): (R, R), (COOP, DEFECT): (S, T),
                       (DEFECT, COOP): (T, S), (DEFECT, DEFECT): (P, P)})


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

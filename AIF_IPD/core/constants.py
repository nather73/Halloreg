"""
core.constants
==============

이산 죄수의 딜레마(Prisoner's Dilemma)의 전역 상수·인덱스 규약·보수(payoff) 조작 API.

[상태 표현]
동시행동 IPD 의 한 라운드 결과를 **joint outcome** 4-상태로 인코딩한다.
focal(=나) 관점에서:

    state = 2 * (내 행동) + (상대 행동)
      0 = CC   나 C, 상대 C   → 상호협력
      1 = CD   나 C, 상대 D   → 내가 호구(sucker)
      2 = DC   나 D, 상대 C   → 내가 착취(temptation)
      3 = DD   나 D, 상대 D   → 상호배신

[보수]
표준 PD 조건 T > R > P > S 및 2R > T + S.
기본값 R=3(상호협력), T=5(배신 유혹), S=0(호구), P=1(상호배신).

[가변 보수(non-stationary payoff) 지원 — H4/H4A/H5]
협력지수 CI = (R − P) / (T − S) 를 라운드마다 바꾸면 게임 구조 자체가 변한다.
`set_payoffs(ci)` 는 T=5, S=0, P=1 을 고정한 채 R = P + CI·(T − S) 로 R 만 변조하며,
모듈 전역 배열을 **제자리(in-place)** 갱신한다. 따라서 이미 import 된 참조
(PAYOFF_SELF 등)도 즉시 새 보수를 본다 — 라운드 단위 보수 변동의 구현 근거.

    CI  < 0    : 교착(deadlock). 상호협력이 상호배신보다 나쁨 → 협력이 손해.
    CI = 0.4   : 기본 PD (R=3).
    CI  ≥ 1    : 조화(harmony). R > T → 협력이 우월전략.
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ 기본 보수
R, T, S, P = 3.0, 5.0, 0.0, 1.0

# ------------------------------------------------------------------ 인덱스 규약
CC, CD, DC, DD = 0, 1, 2, 3
COOP, DEFECT = 0, 1

N_STATES = 4
N_ACTIONS = 2

ACTION_NAMES = {COOP: "C", DEFECT: "D"}
STATE_NAMES = {CC: "CC", CD: "CD", DC: "DC", DD: "DD"}

# joint outcome → 보수. 인덱스 순서는 [CC, CD, DC, DD].
#   PAYOFF_SELF  : 내 보수  (CC=R, CD=S, DC=T, DD=P)
#   PAYOFF_OTHER : 상대 보수 (CC=R, CD=T, DC=S, DD=P)  — 대칭게임이므로 CD/DC 만 뒤바뀐다.
PAYOFF_SELF = np.array([R, S, T, P], dtype=float)
PAYOFF_OTHER = np.array([R, T, S, P], dtype=float)

# (내 행동, 상대 행동) → (내 보수, 상대 보수). EFE 계산에서 조망수용에 사용.
PD_PAYOFFS = {
    (COOP, COOP): (R, R),
    (COOP, DEFECT): (S, T),
    (DEFECT, COOP): (T, S),
    (DEFECT, DEFECT): (P, P),
}

# 기본 PD 의 협력지수 = (3 − 1) / (5 − 0) = 0.4
DEFAULT_COOP_INDEX = round((R - P) / (T - S), 6)


# ------------------------------------------------------------------ 보수 setter
def set_payoffs(ci: float, strict: bool = False) -> tuple:
    """
    협력지수 CI 로 보수를 재설정한다 (가변 보수 환경의 핵심 훅).

    T=5, S=0, P=1 을 고정하고 R = P + CI·(T − S) = 1 + 5·CI 로 둔다.
    strict=True 면 유효 PD 조건 T > R > P > S 를 강제(위반 시 예외).
    기본은 strict=False — 교착(CI<0)·조화(CI≥1) 같은 **PD 밖 게임구조**도
    비정상(non-stationary) 스케줄의 구성원으로 허용하기 위함이다.

    전역 배열을 in-place 로 갱신하므로 반드시 슬라이스 대입([:])을 쓴다.
    반환: 갱신된 (R, T, S, P).
    """
    global R, P
    ci = float(ci)
    R_new = P + ci * (T - S)
    if strict and not (T > R_new > P > S):
        raise ValueError(f"유효하지 않은 협력지수: {ci}")
    R = R_new
    PAYOFF_SELF[:] = [R, S, T, P]
    PAYOFF_OTHER[:] = [R, T, S, P]
    PD_PAYOFFS.update({(COOP, COOP): (R, R), (COOP, DEFECT): (S, T),
                       (DEFECT, COOP): (T, S), (DEFECT, DEFECT): (P, P)})
    return (R, T, S, P)


def set_payoff_matrix(R_new: float, T_new: float,
                      S_new: float, P_new: float) -> tuple:
    """
    (R, T, S, P) 를 **직접** 지정하는 일반 setter — 파라미터 복원(H6) 설계 전용.

    왜 필요한가: 상대 공감 λ_j 의 회귀자(regressor)는 아래 `empathy_shift` 에서
    보듯 계수 (T − S) 이다. `set_payoffs` 는 T·S 를 고정하고 R 만 바꾸므로
    λ_j 의 회귀자가 상수 5 로 고정되어 **절편 α 와 완전 공선(collinear)** 이 된다.
    → 설계행렬의 rank 결손 → α 와 λ_j 를 개별 식별할 수 없다.
    H6 복원 배터리는 블록마다 (T, S) 를 변조해 λ_j 회귀자를 중심화해야 하므로
    유효 PD 조건을 강제하지 않는 이 일반 setter 를 쓴다.

    주의: 사용 후에는 반드시 `reset_payoffs()` 로 기본 PD 로 복귀할 것.
    """
    global R, T, S, P
    R, T, S, P = float(R_new), float(T_new), float(S_new), float(P_new)
    PAYOFF_SELF[:] = [R, S, T, P]
    PAYOFF_OTHER[:] = [R, T, S, P]
    PD_PAYOFFS.update({(COOP, COOP): (R, R), (COOP, DEFECT): (S, T),
                       (DEFECT, COOP): (T, S), (DEFECT, DEFECT): (P, P)})
    return (R, T, S, P)


def reset_payoffs() -> tuple:
    """기본 PD 보수 (R=3, T=5, S=0, P=1) 로 복귀."""
    return set_payoff_matrix(3.0, 5.0, 0.0, 1.0)


def current_payoffs() -> tuple:
    """현재 (R, T, S, P) 스냅샷 — 워커·로깅에서 보수 상태 확인용."""
    return (R, T, S, P)


# ------------------------------------------------------------------ 공감 유도항
def empathy_shift(lam, my_coop_rate):
    """
    상대(공감 가중치 λ_j)의 협력–배신 효용 격차 ΔU = U(C) − U(D) 중
    **보수로부터 유도되는 부분**을 로짓 스케일로 반환한다.

    유도. 내(focal) 협력확률을 p 라 하면 상대 입장에서:
        자기 이득부  Δself(p) = p·(R − T) + (1 − p)·(S − P)
        공감 이득부  λ_j·(T − S)
            └ 내가 협력할 때 상대가 나에게 전달할 수 있는 보수 범위가 (T − S) 이고,
              공감 가중치 λ_j 만큼 그 이득을 자기 효용에 반영한다.
    합쳐서 p 에 대해 정리하면:
        empathy_shift(λ, p) = (T − S)·λ + (R − T + P − S)·p + (S − P)

    기본 PD 에서 계수는 (5, −1, −1) → 5λ − p − 1.
    가변 보수 환경에서는 호출 시점의 전역 R 을 읽으므로 ToM 우도가
    **현재 맥락의 효용**을 자동 반영한다 (맥락-의존 조망수용).

    lam, my_coop_rate 는 스칼라 또는 ndarray (입자필터 벡터화 지원).
    """
    return (T - S) * lam + (R - T + P - S) * my_coop_rate + (S - P)


# ------------------------------------------------------------------ 상태 변환
def joint_index(my_act: int, opp_act: int) -> int:
    """(내 행동, 상대 행동) → joint outcome 인덱스."""
    return int(my_act) * 2 + int(opp_act)


def split_joint(state: int) -> tuple[int, int]:
    """joint outcome → (내 행동, 상대 행동)."""
    my_a, opp_a = divmod(int(state), 2)
    return my_a, opp_a


def opponent_action_from_state(state: int) -> int:
    """joint outcome 에서 상대 행동만 추출."""
    return int(state) % 2


def my_action_from_state(state: int) -> int:
    """joint outcome 에서 내 행동만 추출."""
    return int(state) // 2


def mirror_state(state):
    """
    focal 의 joint state 를 **상대 관점**으로 미러링: (a_i, a_j) → (a_j, a_i).

    AIF vs AIF 다이애드에서 두 에이전트가 각자 자기 관점의 관측을 받도록 하는 데 쓴다.
    """
    if state is None:
        return None
    my_a, opp_a = split_joint(state)
    return joint_index(opp_a, my_a)

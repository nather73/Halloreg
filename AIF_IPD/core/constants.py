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


def set_payoff_matrix(R_new: float, T_new: float,
                      S_new: float, P_new: float) -> tuple:
    """
    (R,T,S,P) 를 **직접** 지정하는 일반 보수 setter — 파라미터 복원 과제 설계 전용
    (v0.6.4).

    `set_coop_index`/`set_payoffs` 는 T=5, S=0, P=1 을 고정하고 R 만 변조한다.
    그런데 `empathy_shift` 의 λ 계수는 (T−S) 이므로, R 만 바꾸는 조작에서는 λ 의
    회귀자가 상수 5 로 고정된다 → 로짓에서 λ 열이 절편(α)과 공선이 되어 **α·λ 를
    개별 식별할 수 없다**(설계행렬 rank 결손). 복원 과제에서 λ 회귀자를 블록마다
    변조·**중심화**(평균 0)하려면 T·S 를 포함한 일반 조작이 필요하다.

    유효 PD 조건(T>R>P>S)을 강제하지 않는다 — 복원 배터리는 식별을 위해 PD 가
    아닌 2×2 게임(예: T<S)을 의도적으로 포함한다. 본 실험(run_ipd_experiment)은
    이 함수를 사용하지 않으므로 H1–H12·GS·VP·ABA·ORE 결과에 영향이 없다.

    주의: 호출 후에는 `reset_payoffs()` 로 기본 PD 로 복귀해야 `set_coop_index`
    (T−S=5 를 가정)가 정상 동작한다.
    반환: (R, T, S, P).
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


def set_payoffs(ci: float, strict: bool = False) -> tuple:
    """
    협력지수 CI=(R−P)/(T−S) 로 보수를 재설정하되, **극단 영역(CI<0, CI≥1)까지
    허용**한다 (가변 페이오프 환경용; §VP).

    T=5, S=0, P=1 고정, R = P + CI·(T−S) = 1 + 5·CI.
      · CI<0    : R<P — 상호협력이 상호배신보다 나쁨(교착/deadlock 류 게임).
      · CI=0.4  : 현행 PD (2R=6>T+S=5).
      · CI=1    : R=6>T=5 — 협력이 우월(조화/harmony 게임).
      · CI>1    : 더 강한 협력 우월.
    strict=True 이면 유효 PD 조건(T>R>P>S)을 강제(set_coop_index 와 동일).

    **in-place** 로 PAYOFF_SELF/PAYOFF_OTHER/PD_PAYOFFS 를 갱신하므로, 라운드마다
    호출하면 에이전트의 EFE 가 **현재 맥락의 효용**을 반영한다(맥락-의존 효용).
    반환: (R, T, S, P).
    """
    global R, P
    ci = float(ci)
    R_new = 1.0 + ci * (T - S)
    if strict and not (T > R_new > P > S):
        raise ValueError(f"유효하지 않은 협력지수 {ci}")
    R = R_new
    PAYOFF_SELF[:] = [R, S, T, P]
    PAYOFF_OTHER[:] = [R, T, S, P]
    PD_PAYOFFS.update({(COOP, COOP): (R, R), (COOP, DEFECT): (S, T),
                       (DEFECT, COOP): (T, S), (DEFECT, DEFECT): (P, P)})
    return (R, T, S, P)


def joint_index(my_act: int, opp_act: int) -> int:
    """(내 행동, 상대 행동) -> joint-outcome 상태 인덱스."""
    return int(my_act) * 2 + int(opp_act)


def empathy_shift(lam, my_coop_rate):
    """
    상대(공감 가중치 λ)의 협력–배신 효용 격차 ΔU(λ)=U(C,λ)−U(D,λ) 중 **보수 유도부**
    를 현재 보수 구성 (R,T,S,P) 로부터 계산해 로짓 스케일에 더하는 항 (v0.6.3).

    유도: 내(focal) 협력확률을 p 라 할 때, 상대의
      자기 이득부  Δself(p)  = p·(R−T) + (1−p)·(S−P)
      공감 이득부  λ·(T−S)   — 내 협력이 상대에게 전달하는 보수 범위(T−S)에 λ 가중
    합계를 p 에 대해 정리하면:
        empathy_shift(λ, p) = (T−S)·λ + (R−T+P−S)·p + (S−P)

    기본 PD 보수 (R=3, T=5, S=0, P=1) 에서 계수가 (5, −1, −1) 이 되어 레거시 상수식
        5·λ − p − 1
    과 **비트 단위로 동일**하다(H1–H12 재현성 보존). `set_payoffs`/`set_coop_index`
    로 보수가 라운드마다 바뀌는 가변 페이오프 환경(§VP)에서는 모듈 전역 R 을 호출
    시점에 읽어 ToM 공감항이 **현재 맥락의 효용**을 반영한다(예: 조화 R=7 이면
    p 계수가 +1 로 반전 — 상대가 협력할수록 협력이 더 유리).

    lam, my_coop_rate 는 스칼라 또는 ndarray (입자필터 벡터화 지원).
    """
    return (T - S) * lam + (R - T + P - S) * my_coop_rate + (S - P)


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

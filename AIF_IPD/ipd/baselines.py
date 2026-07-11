"""
baselines.py — 비-ToM 학습 베이스라인 (보완안 §3).

"잠재 형질 추론(ToM)의 이득" 을 "아무 학습이나 하면 얻는 이득" 과 분리 귀속하기
위한 대조군. 세 계열 모두 상대를 **memory-1 프로세스**로만 모형화하며(잠재 형질
없음), StrategyAgent 와 동일한 act()/observe() 인터페이스를 따른다.

QLearnerAgent        : 표 형태 Q-학습 (상태 = 직전 joint outcome, ε-greedy)
BayesBestResponse    : 상대의 memory-1 조건부 협력확률 4개에 Beta 사전 →
                       Thompson 표집 후 4-상태 MDP 가치반복으로 최적반응
FictitiousPlayAgent  : 위와 동일하되 사후평균(점추정)에 대한 결정적 최적반응

보수는 core.constants.PD_PAYOFFS 를 **런타임에** 조회하므로 게임구조 일반화
(set_coop_index)와 자동으로 정합한다.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from AIF_IPD.core import constants as C

_INIT = 4          # 초기(직전 결과 없음) 상태 인덱스
_N_STATES = 5      # CC, CD, DC, DD, INIT


def _payoff(my_a: int, opp_a: int) -> float:
    return float(C.PD_PAYOFFS[(my_a, opp_a)][0])


class _Memory1Base:
    """act/observe 인터페이스 + memory-1 상태 추적 공통부."""

    def __init__(self, seed: int = 0, name: str = "baseline"):
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.last_my: Optional[int] = None
        self.last_opp: Optional[int] = None
        self._pending: Optional[int] = None   # 이번 라운드 내 행동 (observe 에서 확정)

    # StrategyAgent 프로토콜
    def act(self) -> int:
        a = self._decide()
        self._pending = a
        return a

    def observe(self, opp_action: int):
        self.last_opp = int(opp_action)
        my_a = self._pending if self._pending is not None else C.COOP
        self._learn(my_a, int(opp_action))
        self.last_my = my_a
        self._pending = None

    # 하위 클래스 구현
    def _decide(self) -> int:
        raise NotImplementedError

    def _learn(self, my_a: int, opp_a: int):
        pass

    def _state(self) -> int:
        if self.last_my is None:
            return _INIT
        return C.joint_index(self.last_my, self.last_opp)


class QLearnerAgent(_Memory1Base):
    """
    표 형태 Q-학습. 상태 = 직전 joint outcome(+초기), 행동 = {C, D}.
    Q(s,a) ← Q + α·(r + γ·max_a' Q(s',a') − Q). ε-greedy 탐색 (ε 선형 감쇠).
    """

    def __init__(self, seed: int = 0, alpha: float = 0.15, gamma: float = 0.95,
                 eps0: float = 0.20, eps_decay: float = 0.985,
                 optimistic: float = 3.0, name: str = "qlearner"):
        super().__init__(seed, name)
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.eps = float(eps0)
        self.eps_decay = float(eps_decay)
        self.Q = np.full((_N_STATES, 2), float(optimistic))   # 낙관적 초기화

    def _decide(self) -> int:
        s = self._state()
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, 2))
        q = self.Q[s]
        return int(np.argmax(q + 1e-9 * self.rng.random(2)))   # 동률 무작위

    def _learn(self, my_a: int, opp_a: int):
        s = self._state()
        r = _payoff(my_a, opp_a)
        s2 = C.joint_index(my_a, opp_a)
        self.Q[s, my_a] += self.alpha * (
            r + self.gamma * self.Q[s2].max() - self.Q[s, my_a])
        self.eps *= self.eps_decay


class BayesBestResponse(_Memory1Base):
    """
    베이지안 최적반응 (Thompson 표집).

    상대를 memory-1 로 모형화: p_k = P(상대가 C | 직전 결과 k), k ∈ {CC,CD,DC,DD,INIT}.
    각 p_k 에 Beta(1,1) 사전. 매 라운드 사후에서 p 를 표집(Thompson)한 뒤,
    표집된 memory-1 상대에 대한 4-상태 MDP 를 가치반복으로 풀어 현 상태의
    최적행동을 취한다. 잠재 형질(의도/맥락)의 개념이 없다는 점이 ToM 과의
    결정적 차이다.
    """

    def __init__(self, seed: int = 0, gamma: float = 0.95, vi_iters: int = 60,
                 thompson: bool = True, name: str = "bayes_br"):
        super().__init__(seed, name)
        self.gamma = float(gamma)
        self.vi_iters = int(vi_iters)
        self.thompson = bool(thompson)
        # 상대 관점 상태로 카운트: 상대가 본 직전 결과 k 에서 상대가 C 를 낸 횟수
        self.a_cnt = np.ones(_N_STATES)    # Beta α (협력)
        self.b_cnt = np.ones(_N_STATES)    # Beta β (배신)

    def _sample_p(self) -> np.ndarray:
        if self.thompson:
            return self.rng.beta(self.a_cnt, self.b_cnt)
        return self.a_cnt / (self.a_cnt + self.b_cnt)   # 사후평균 (fictitious)

    def _best_response(self, p: np.ndarray, s0: int) -> int:
        """표집된 memory-1 상대에 대한 가치반복 → s0 에서의 최적행동."""
        V = np.zeros(_N_STATES)
        g = self.gamma
        Qs = np.zeros((_N_STATES, 2))
        for _ in range(self.vi_iters):
            for s in range(_N_STATES):
                pc = p[s]
                for a in (C.COOP, C.DEFECT):
                    # 상대 C 확률 pc → 다음 상태와 즉시 보수 기대
                    sc = C.joint_index(a, C.COOP)
                    sd = C.joint_index(a, C.DEFECT)
                    Qs[s, a] = (pc * (_payoff(a, C.COOP) + g * V[sc])
                                + (1 - pc) * (_payoff(a, C.DEFECT) + g * V[sd]))
            V = Qs.max(axis=1)
        return int(np.argmax(Qs[s0]))

    def _decide(self) -> int:
        return self._best_response(self._sample_p(), self._state())

    def _learn(self, my_a: int, opp_a: int):
        # 상대가 관측한 직전 상태 = 내 상태의 미러
        if self.last_my is None:
            k = _INIT
        else:
            k = C.joint_index(self.last_opp, self.last_my)   # 상대 관점
        if opp_a == C.COOP:
            self.a_cnt[k] += 1
        else:
            self.b_cnt[k] += 1


class FictitiousPlayAgent(BayesBestResponse):
    """조건부 경험빈도(사후평균)에 대한 결정적 최적반응 — Thompson 없는 변형."""

    def __init__(self, seed: int = 0, gamma: float = 0.95, vi_iters: int = 60,
                 name: str = "fictitious"):
        super().__init__(seed=seed, gamma=gamma, vi_iters=vi_iters,
                         thompson=False, name=name)


def make_baseline(kind: str, seed: int = 0, **kw):
    if kind == "qlearner":
        return QLearnerAgent(seed=seed, **kw)
    if kind == "bayes_br":
        return BayesBestResponse(seed=seed, **kw)
    if kind == "fictitious":
        return FictitiousPlayAgent(seed=seed, **kw)
    raise ValueError(f"알 수 없는 베이스라인: {kind}")

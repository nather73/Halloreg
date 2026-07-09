"""
ipd.env
=======

IPD 환경과 고정전략 상대(생성 프로세스)들.

Environment : (a_i, a_j) -> joint outcome 관측 매핑.
StrategyAgent : TFT, ALLC, ALLD, WSLS, Generous-TFT, Random 등 이산 전략을
                하나의 클래스로 내포 (프로토타입 `FixedOpponent` 확장).
                파트너/상대 어느 쪽으로도 다이애드/집단에 투입 가능.

전략은 실행 잡음(execution noise, 낮은 β 조작화)을 선택적으로 가질 수 있어,
'의도적 착취자(ALLD, 잡음 0)'와 'noisy 상대(TFT+20% 오류 / 완전 무작위)'를 구분한다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from HalloReg.core.constants import COOP, DEFECT, joint_index


class Environment:
    """동시행동 IPD 환경. 두 행동을 joint outcome 으로 사상."""

    def __init__(self, n_agents: int = 2):
        self.K = n_agents

    @staticmethod
    def step(my_action: int, opp_action: int) -> int:
        """(내 행동, 상대 행동) -> joint outcome 상태 인덱스."""
        return joint_index(my_action, opp_action)


class StrategyAgent:
    """
    이산 전략 에이전트. `act(my_last_seen)` / `observe(focal_action)` 인터페이스.

    kind ∈ {
        'tit_for_tat', 'allc', 'alld', 'wsls', 'generous_tft',
        'random', 'exploiter', 'noisy_tft', 'noisy'
    }

    error : 실행 잡음률(의도와 반대로 행동할 확률; 낮은 β 조작화).
    generosity : Generous-TFT 가 상대 배신을 용서(협력)할 확률.
    schedule : [(round, kind), ...] 지정 라운드에 **형질(의도)이 전환**되는 변덕 상대.
               실행잡음(β)이 아니라 기질(α/ρ/λ_j)이 바뀌는 조작 — H7/H9 용.
    """

    def __init__(self, kind: str = "tit_for_tat", error: float = 0.0,
                 generosity: float = 0.3, seed: int = 0,
                 schedule: Optional[list] = None):
        self.kind = kind
        self.schedule = sorted(schedule or [], key=lambda x: x[0])
        self.error = float(error)
        self.generosity = float(generosity)
        self.rng = np.random.default_rng(seed)
        self.my_last = COOP          # 상대(=이 전략) 자신의 직전 행동
        self.other_last = COOP       # 관측한 focal 의 직전 행동
        self.round = 0

    # ------------------------------------------------------------ 의도
    def _intended(self) -> int:
        k = self.kind
        if k == "allc":
            return COOP
        if k == "alld" or k == "exploiter":
            return DEFECT
        if k in ("tit_for_tat", "noisy_tft"):
            return self.other_last
        if k == "generous_tft":
            if self.other_last == DEFECT and self.rng.random() < self.generosity:
                return COOP           # 관대하게 용서
            return self.other_last
        if k == "wsls":               # win-stay, lose-shift
            # 직전 결과가 '승리'(상호협력 CC 또는 유혹 DC)면 유지, 아니면 전환
            win = (self.my_last == COOP and self.other_last == COOP) or \
                  (self.my_last == DEFECT and self.other_last == COOP)
            return self.my_last if win else (1 - self.my_last)
        if k == "random":
            return COOP if self.rng.random() < 0.5 else DEFECT
        if k == "noisy":              # 완전 무작위(내 행동 무관)
            return COOP if self.rng.random() < 0.5 else DEFECT
        return COOP

    def act(self) -> int:
        # 형질 전환 스케줄 적용 (변덕스러운 상대)
        for r, k in self.schedule:
            if self.round >= r:
                self.kind = k
        self.round += 1
        intend = self._intended()
        # 실행 잡음 (noisy_tft 는 error>0, exploiter/alld 는 error≈0)
        err = self.error
        if self.kind == "noisy_tft" and err == 0.0:
            err = 0.20
        if err > 0.0 and self.rng.random() < err:
            intend = 1 - intend
        self.my_last = intend
        return intend

    def observe(self, focal_action: int):
        self.other_last = int(focal_action)


def make_opponent(kind: str, seed: int = 0, **kwargs) -> StrategyAgent:
    """상대 유형 dispatch (편의 함수)."""
    presets = {
        "exploiter": dict(kind="alld", error=0.0),
        "intentional_exploiter": dict(kind="alld", error=0.0),
        "noisy_tft": dict(kind="noisy_tft", error=0.20),
        "noisy": dict(kind="noisy", error=0.0),
        "tit_for_tat": dict(kind="tit_for_tat", error=0.0),
        "tft": dict(kind="tit_for_tat", error=0.0),
        "allc": dict(kind="allc", error=0.0),
        "alld": dict(kind="alld", error=0.0),
        "wsls": dict(kind="wsls", error=0.0),
        "generous_tft": dict(kind="generous_tft"),
        "gtft": dict(kind="generous_tft"),
        "random": dict(kind="random"),
        # 정적 형질 + 모호한 배신(약한 실행잡음): H10 의 '정적 상대'
        "static_noisy_tft": dict(kind="tit_for_tat", error=0.10),
    }
    cfg = presets.get(kind, dict(kind=kind))
    cfg.update(kwargs)
    return StrategyAgent(seed=seed, **cfg)


def make_capricious(seed: int = 0, n_rounds: int = 120,
                    phases=("tit_for_tat", "alld", "generous_tft")) -> StrategyAgent:
    """
    **변덕스러운(의도 변동) 상대.** 실행잡음(β)이 아니라 기질(α/ρ/λ_j)이 구간마다 바뀐다.

    기본: 협력적 상호성(TFT) → 착취(ALLD) → 관대한 상호성(GTFT, 화해 국면).
    H7(고정전략 대비 의도추론의 이득), H9(α 전용 귀인의 한계) 조건에 사용.
    """
    k = len(phases)
    cuts = [int(round(i * n_rounds / k)) for i in range(k)]
    sched = [(c, ph) for c, ph in zip(cuts, phases)]
    return StrategyAgent(kind=phases[0], seed=seed, schedule=sched)

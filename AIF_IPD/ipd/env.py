"""
ipd.env
=======

IPD 환경과 고정전략 상대(생성 프로세스).

[본 판에서 다루는 전략 집합]
가설 H1·H3·H4·H5 가 지정한 6개 유형 중 5개가 고정전략이다.

  TFT   (tit_for_tat)   : 상대의 직전 행동을 그대로 되갚는다.
  GTFT  (generous_tft)  : TFT 이되, 상대가 배신해도 확률 `generosity` 로 용서한다
                          (Nowak & Sigmund 1992 의 고전적 관대한 TFT).
  WSLS  (wsls)          : Win-Stay Lose-Shift. 직전 결과가 '승리'(상호협력 CC 또는
                          일방 착취 DC)면 자기 행동을 유지하고, 아니면 전환한다.
  ALLC  (allc)          : 항상 협력.
  ALLD  (alld)          : 항상 배신 — 의도적 착취자.

여섯 번째 유형 HalloReg 는 ipd.agent.HalloRegAgent 이다.

[실행잡음(execution noise) 과 의도의 분리 — H2A 의 설계적 근거]
`error` 는 **의도와 반대로 행동이 방출될 확률**이다. 낮은 행동정밀도 β 의
조작화이며, 형질(α/ρ/λ_j) 변화와는 다른 층위다. 이 구분이 있어야
  · 의도적 착취자(ALLD, error = 0)  와
  · 잡음 있는 협력자(noisy TFT, error = 0.2)
가 **같은 배신 관측**을 만들면서도 서로 다른 잠재 원인을 갖는 조건이 성립한다.
H2A 는 정확히 이 구분을 검증한다.

[형질 전환 스케줄 — H1A 의 설계적 근거]
`schedule = [(round, kind), ...]` 를 주면 지정 라운드에 **동일 개체의 기질이
바뀐다**(잡음이 아니라 의도의 변화). memory-1 내부 상태는 국면을 넘어 연속되므로
'다른 개체로 교체' 가 아니라 '같은 사람이 변했다' 는 의미론을 갖는다.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from AIF_IPD.core.constants import COOP, DEFECT, joint_index

# 집단 실험(H3~H5)에서 쓰는 고정전략 5종의 표준 이름 및 표기 순서.
FIXED_STRATEGIES = ("tft", "gtft", "wsls", "allc", "alld")
# HalloReg 를 포함한 전체 유형 순서 — 보수행렬 Π 의 행/열 순서와 일치시킨다.
ALL_TYPES = FIXED_STRATEGIES + ("halloreg",)

# 그림·표에 쓸 한글 표기
TYPE_LABEL_KO = {
    "tft": "TFT", "gtft": "GTFT", "wsls": "WSLS",
    "allc": "ALLC", "alld": "ALLD", "halloreg": "HalloReg",
}


class Environment:
    """동시행동 IPD 환경. 두 행동을 joint outcome 으로 사상한다."""

    @staticmethod
    def step(my_action: int, opp_action: int) -> int:
        return joint_index(my_action, opp_action)


class StrategyAgent:
    """
    고정전략 에이전트. `act()` / `observe(focal_action)` 인터페이스.

    Parameters
    ----------
    kind : str
        {'tft', 'gtft', 'wsls', 'allc', 'alld', 'random'} 중 하나.
    error : float
        실행잡음률 — 의도와 반대 행동이 방출될 확률.
    generosity : float
        GTFT 의 용서 확률 (상대 배신 시 그럼에도 협력할 확률).
    schedule : [(round, kind), ...] | None
        형질 전환 스케줄. 해당 라운드부터 kind 가 바뀐다.
    """

    def __init__(self, kind: str = "tft", error: float = 0.0,
                 generosity: float = 0.3, seed: int = 0,
                 schedule: Optional[List[Tuple[int, str]]] = None):
        self.kind = kind
        self.base_kind = kind
        self.schedule = sorted(schedule or [], key=lambda x: x[0])
        self.error = float(error)
        self.generosity = float(generosity)
        self.rng = np.random.default_rng(seed)

        # memory-1 상태: 자기 직전 행동과 관측한 focal 의 직전 행동.
        # 초기 관례는 둘 다 협력 — TFT 가 첫 라운드에 협력하도록 한다.
        self.my_last = COOP
        self.other_last = COOP
        self.round = 0

    # ------------------------------------------------------------ 의도
    def _intended(self) -> int:
        """실행잡음을 적용하기 **전**의 의도된 행동."""
        k = self.kind
        if k == "allc":
            return COOP
        if k == "alld":
            return DEFECT
        if k == "tft":
            return self.other_last
        if k == "gtft":
            # 상대가 배신했더라도 확률 generosity 로 용서(협력)
            if self.other_last == DEFECT and self.rng.random() < self.generosity:
                return COOP
            return self.other_last
        if k == "wsls":
            # '승리' = 상대가 협력했을 때 (내가 C 면 CC, 내가 D 면 DC 로 착취 성공).
            # 승리면 유지, 패배면 전환.
            win = (self.other_last == COOP)
            return self.my_last if win else (1 - self.my_last)
        if k == "random":
            return COOP if self.rng.random() < 0.5 else DEFECT
        return COOP

    # ------------------------------------------------------------ 방출
    def act(self) -> int:
        """이번 라운드에 실제로 방출할 행동."""
        # 형질 전환 스케줄 적용 (누적 적용 — 마지막으로 도달한 국면이 유효)
        for r, k in self.schedule:
            if self.round >= r:
                self.kind = k
        self.round += 1

        a = self._intended()
        if self.error > 0.0 and self.rng.random() < self.error:
            a = 1 - a               # 실행잡음: 의도와 반대로 방출
        self.my_last = a
        return a

    def observe(self, focal_action: int) -> None:
        """focal 이 실제로 방출한 행동을 관측한다."""
        self.other_last = int(focal_action)

    def begin_partner(self, identity: int) -> None:
        """인터페이스 호환용(고정전략은 identity 기억이 없다)."""
        return None


# ------------------------------------------------------------------ 팩토리
_PRESETS = {
    "tft": dict(kind="tft", error=0.0),
    "gtft": dict(kind="gtft", error=0.0, generosity=0.3),
    "wsls": dict(kind="wsls", error=0.0),
    "allc": dict(kind="allc", error=0.0),
    "alld": dict(kind="alld", error=0.0),
    "random": dict(kind="random", error=0.0),
    # H2/H2A 전용 조건
    "exploiter": dict(kind="alld", error=0.0),        # 의도적 착취자
    "noisy_tft": dict(kind="tft", error=0.20),        # 잡음 있는 협력자
}


def make_opponent(kind: str, seed: int = 0, **kwargs) -> StrategyAgent:
    """전략 이름 → StrategyAgent. 프리셋을 kwargs 로 덮어쓸 수 있다."""
    cfg = dict(_PRESETS.get(kind, dict(kind=kind)))
    cfg.update(kwargs)
    return StrategyAgent(seed=seed, **cfg)


# ------------------------------------------------------ 형질 전환(H1A) 스케줄
#: H1A 에서 쓰는 '변덕스러운 상대' 시나리오. 값은 국면 순환(cycle)이다.
#: 120 라운드를 4 등분(30 라운드 주기)해 순환시킨다.
SWITCH_SCENARIOS = {
    # 상호성 → 착취 → 화해 → 상호성
    "recip_expl_recon": ("tft", "alld", "gtft", "tft"),
    # 순진한 협력 → 착취 (신뢰를 유인한 뒤 배신하는 함정형)
    "coop_trap": ("allc", "alld", "allc", "alld"),
    # 결과-조건형 ↔ 착취 교대 (WSLS 를 포함한 이질 순환)
    "wsls_flip": ("wsls", "alld", "wsls", "gtft"),
}

#: 형질 전환 주기 (라운드). 120 라운드 / 4 국면 = 30.
SWITCH_PERIOD = 30


def make_switching_opponent(scenario: str, n_rounds: int = 120,
                            seed: int = 0, error: float = 0.05
                            ) -> StrategyAgent:
    """
    형질 전환 상대를 생성한다.

    실행잡음 error 를 함께 주어, '잡음' 과 '의도 변화' 가 동시에 존재하는
    조건을 만든다 — 잠재 형질 추론이 실제로 이득이 되는 상황.
    """
    cycle = SWITCH_SCENARIOS[scenario]
    sched = [(r, cycle[(r // SWITCH_PERIOD) % len(cycle)])
             for r in range(0, n_rounds, SWITCH_PERIOD)]
    return StrategyAgent(kind=cycle[0], seed=seed, error=error, schedule=sched)


def switch_rounds(n_rounds: int = 120) -> List[int]:
    """형질이 실제로 전환되는 라운드 인덱스 목록(0 제외) — 정렬 분석용."""
    return list(range(SWITCH_PERIOD, n_rounds, SWITCH_PERIOD))


# ------------------------------------------------ 파라미터 복원용 생성 상대(H6)
class LikelihoodAgent:
    """
    **OpponentInversion 의 생성 우도로부터 직접 행동을 생성하는 상대.**

    파라미터 복원(H6) 전용이다. 진짜 θ = (α, ρ, ω, η, β, λ_j) 를 알고 있는
    상태에서 행동을 표집하므로, 추정된 θ̂ 와 참값을 직접 비교할 수 있다.
    이는 '모형이 자기가 가정한 데이터 생성과정을 되돌릴 수 있는가' 라는
    복원 타당성(recovery validity)의 표준 검사다.

        P(a_j = C) = σ( β·( α + ρ·f + ω·g + η·f·g + s(λ_j, p) ) )

    p (focal 의 협력률에 대한 믿음)는 관측한 focal 행동의 이동평균으로 갱신한다 —
    추론기 쪽 `my_cooperation_rate` 와 같은 방식이라 우도가 정확히 정합한다.
    """

    def __init__(self, alpha: float = 0.0, rho: float = 0.0,
                 omega: float = 0.0, eta: float = 0.0,
                 beta: float = 3.0, lambda_j: float = 0.5,
                 seed: int = 0):
        self.theta = dict(alpha=float(alpha), rho=float(rho),
                          omega=float(omega), eta=float(eta),
                          beta=float(beta), lambda_j=float(lambda_j))
        self.rng = np.random.default_rng(seed)
        self.my_last = COOP
        self.other_last = COOP
        self.other_actions: List[int] = []
        self.round = 0

    def _coop_prob(self) -> float:
        from AIF_IPD.core.constants import empathy_shift
        th = self.theta
        f = 1.0 - 2.0 * float(self.other_last)     # focal 직전 행동
        g = 1.0 - 2.0 * float(self.my_last)        # 자기 직전 행동
        p = (float(np.mean([a == COOP for a in self.other_actions]))
             if self.other_actions else 0.5)
        z = th["beta"] * (th["alpha"] + th["rho"] * f + th["omega"] * g
                          + th["eta"] * f * g
                          + empathy_shift(th["lambda_j"], p))
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))

    def act(self) -> int:
        self.round += 1
        a = COOP if self.rng.random() < self._coop_prob() else DEFECT
        self.my_last = a
        return a

    def observe(self, focal_action: int) -> None:
        self.other_last = int(focal_action)
        self.other_actions.append(int(focal_action))

    def begin_partner(self, identity: int) -> None:
        return None


class ProbeAgent:
    """
    **식별성을 위한 능동 자극 정책** (H6 의 focal).

    파라미터 복원의 성패는 설계행렬 (1, f, g, f·g) 가 실제로 스팬되는지에 달려
    있다. focal 이 늘 협력만 하면 f 가 상수가 되어 α 와 ρ 가 공선이 되고,
    ρ·ω·η 는 식별 불가능해진다.

    ProbeAgent 는 매 라운드 독립적으로 확률 p_coop 로 협력한다. 이 무작위화가
    f 를 두 수준에서 균형 있게 점유하게 하고, 상대의 자기 이력 g 와의 조합도
    네 칸을 모두 채우게 한다.
    """

    def __init__(self, p_coop: float = 0.5, seed: int = 0):
        self.p_coop = float(p_coop)
        self.rng = np.random.default_rng(seed)

    def act(self) -> int:
        return COOP if self.rng.random() < self.p_coop else DEFECT

    def observe(self, focal_action: int) -> None:
        return None

    def begin_partner(self, identity: int) -> None:
        return None

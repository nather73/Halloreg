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

from AIF_IPD.core.constants import COOP, DEFECT, joint_index


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


# ------------------------------------------------------------------ H7 형질전환 case 카탈로그
# v0.3 전면 확장: case = 전환 주기(period) × 순환 구성(cycle family) 격자.
#
#   주기(period) ∈ {10, 30, 60, 120} — 모든 주기를 모두 시뮬레이션한다.
#     주의(설계상 명시): 지평 T=60 에서 p60/p120 case 는 **지평 내 전환이 0회**
#     (capricious_switch_rounds → []) 인 '무전환 대조'로 퇴화한다. 전환 정렬
#     분석에서는 자동 제외되며, 주기 의존성 해석에서 이 점을 명시해야 한다.
#
#   순환(cycle) — 고정전략 간 전환에 국한하지 않고 **정교형 AdaptiveAgent 국면**
#     ("adaptive")을 포함한 이질 전환을 지원한다. "adaptive" 국면이 포함된 case 는
#     capricious_case_spec 이 SwitchingAgent 스펙(type="switching")을 반환한다.
#
# 이름 규약: p{period}_{cycle family}.
_CAPRICIOUS_CYCLES = {
    # 상호성 → 착취 → 화해 (원 구현의 표준 3상 순환)
    "recip_expl_recon": ("tit_for_tat", "alld", "generous_tft"),
    # 착취-선행 3상 (첫인상이 나쁜 상대)
    "expl_first":       ("alld", "tit_for_tat", "generous_tft"),
    # 상호성 ↔ 착취 2상 교대
    "flip":             ("tit_for_tat", "alld"),
    # 관대함-선행 순환 (화해 후 착취)
    "gen_first":        ("generous_tft", "alld", "tit_for_tat"),
    # WSLS 를 포함한 이질 순환
    "wsls_mix":         ("wsls", "alld", "generous_tft"),
    # 순진한 협력 → 착취 (신뢰 유인 후 배신하는 함정형)
    "coop_trap":        ("allc", "alld"),
    # 정교형 AIF ↔ 의도적 착취자 (잠재 성향이 지각을 유지하는 온난 전환)
    "adaptive_expl":    ("adaptive", "alld"),
    # 정교형 AIF → 착취 → 관대한 상호성 (이질 3상)
    "adaptive_mix":     ("adaptive", "alld", "generous_tft"),
}
CAPRICIOUS_PERIODS = (10, 30, 60, 120)
CAPRICIOUS_CASES = {
    f"p{p}_{fam}": dict(period=p, cycle=cyc)
    for fam, cyc in _CAPRICIOUS_CYCLES.items()
    for p in CAPRICIOUS_PERIODS
}

# H8/H8E 등 집단 구성원용 기본 case (고정전략 순환만; AIF 국면 없음)
DEFAULT_CAPRICIOUS_CASE = "p30_recip_expl_recon"


def fixed_capricious_cases() -> list:
    """AIF("adaptive") 국면이 없는 — 순수 고정전략 순환 — case 이름 목록."""
    return [c for c, cfg in CAPRICIOUS_CASES.items()
            if "adaptive" not in cfg["cycle"]]


def capricious_cases_by_period(period: int) -> list:
    """주어진 전환 주기의 case 이름 목록 (주기 의존성 분석용)."""
    return [c for c, cfg in CAPRICIOUS_CASES.items() if cfg["period"] == period]


def capricious_switch_rounds(case: str, n_rounds: int) -> list:
    """해당 case 에서 형질(의도)이 실제로 전환되는 라운드 인덱스 목록(0 제외)."""
    cfg = CAPRICIOUS_CASES[case]
    return [r for r in range(cfg["period"], n_rounds, cfg["period"])]


def _case_schedule(case: str, n_rounds: int) -> list:
    cfg = CAPRICIOUS_CASES[case]
    period, cycle = cfg["period"], cfg["cycle"]
    return [(r, cycle[(r // period) % len(cycle)])
            for r in range(0, n_rounds, period)]


def make_capricious_case(case: str, seed: int = 0, n_rounds: int = 60,
                         error: float = 0.10):
    """
    CAPRICIOUS_CASES 카탈로그의 case 이름으로 형질전환 상대를 생성한다.

    실행잡음 `error`(β 조작화)와 형질 전환(α/ρ/λ_j 조작화)이 동시에 존재하여,
    '잡음'과 '의도 변화'를 구분하는 잠재 형질 추론이 이득이 되는 조건을 만든다.

    cycle 에 "adaptive" 국면이 포함되면 SwitchingAgent(고정전략×AIF 혼합 전환)
    를, 아니면 기존과 동일한 schedule 기반 StrategyAgent 를 반환한다.
    """
    sched = _case_schedule(case, n_rounds)
    cycle = CAPRICIOUS_CASES[case]["cycle"]
    if "adaptive" in cycle:
        return SwitchingAgent(schedule=sched, seed=seed, error=error)
    return StrategyAgent(kind=cycle[0], seed=seed, error=error, schedule=sched)


def capricious_case_spec(case: str, seed: int = 0, n_rounds: int = 60,
                         error: float = 0.10) -> dict:
    """make_capricious_case 와 동일하되 build_from_spec 용 스펙 dict 를 반환."""
    sched = _case_schedule(case, n_rounds)
    cycle = CAPRICIOUS_CASES[case]["cycle"]
    if "adaptive" in cycle:
        return dict(type="switching", seed=seed, error=error, schedule=sched)
    return dict(type="strategy", kind=cycle[0], seed=seed, error=error,
                schedule=sched)


class SwitchingAgent:
    """
    이질 형질전환 상대 — 고정전략 국면과 **정교형 AdaptiveAgent(AIF) 국면**의
    혼합 전환을 지원한다 (v0.3, H7 확장).

    인터페이스는 StrategyAgent 와 동일한 act()/observe() 로, run_dyad 의
    비-AIF 경로에 그대로 투입된다.

    설계 (문서화된 의미론)
    ----------------------
    * 전략 국면 : 내부 StrategyAgent 하나의 kind 를 스케줄에 따라 교체한다.
      memory-1 상태(my_last/other_last)가 국면 간 **연속** — 기존 schedule
      기반 StrategyAgent 와 동일한 '동일 개체의 기질 전환' 의미론.
    * adaptive 국면 : 지속(persistent) AdaptiveAgent 서브에이전트가 **매 라운드**
      step() 으로 지각·신념 갱신을 계속한다 (온난 전환, warm switching —
      잠재 성향은 통제권이 없어도 상대를 계속 관측한다는 이론적 선택).
      자신이 통제권을 가진 국면에서만 행동이 방출된다.
    * 방출 행동이 서브에이전트의 선택과 다른 경우(비활성 국면 또는 실행잡음),
      다음 step 의 관측 상태에는 **실제 방출 행동**이 인코딩된다 — run_dyad 의
      환경 계층 실행오류(env_err)와 동일한 방식으로 self-model 이 정합된다.
    * 실행잡음 error 는 국면 종류와 무관하게 방출 계층에서 **대칭** 부과된다
      (잡음-의도 구분 문제의 대칭성 유지).
    """

    def __init__(self, schedule: list, seed: int = 0, error: float = 0.0,
                 adaptive_kwargs: Optional[dict] = None):
        self.schedule = sorted(schedule or [], key=lambda x: x[0])
        if not self.schedule:
            raise ValueError("SwitchingAgent 는 비어있지 않은 schedule 이 필요")
        self.error = float(error)
        self.rng = np.random.default_rng(seed)
        self.round = 0
        self.kind = self.schedule[0][1]          # 현재 활성 국면 (분석용)

        first_strat = next((k for _, k in self.schedule if k != "adaptive"),
                           "tit_for_tat")
        self.strat = StrategyAgent(kind=first_strat, seed=seed + 1, error=0.0)

        self.aif = None
        if any(k == "adaptive" for _, k in self.schedule):
            # 지연 import (env → agent 는 이 지점에서만; 순환 의존 방지)
            from AIF_IPD.ipd.agent import AdaptiveAgent
            kw = dict(kappa=0.9, sophisticated=True,
                      lam_base=0.4, lam_max=0.8)
            kw.update(adaptive_kwargs or {})
            self.aif = AdaptiveAgent(seed=seed + 2, **kw)

        self.my_last_emitted = COOP
        self._prev_state = None                  # AIF 관점 joint state

    # ------------------------------------------------------------ 내부
    def _active(self) -> str:
        k = self.schedule[0][1]
        for r, kk in self.schedule:
            if self.round >= r:
                k = kk
        return k

    # ------------------------------------------------------- 인터페이스
    def act(self) -> int:
        self.kind = self._active()
        self.round += 1

        # AIF 서브에이전트는 국면과 무관하게 매 라운드 지각·신념 갱신 (온난 전환)
        aif_act = None
        if self.aif is not None:
            aif_act = self.aif.step(self._prev_state)

        if self.kind == "adaptive":
            intend = int(aif_act)
        else:
            self.strat.kind = self.kind
            self.strat.round += 1
            intend = self.strat._intended()

        emitted = intend
        if self.error > 0.0 and self.rng.random() < self.error:
            emitted = 1 - emitted

        # memory-1 상태를 방출 행동으로 일관 유지 (모든 국면 공통)
        self.strat.my_last = emitted
        self.my_last_emitted = emitted
        return emitted

    def observe(self, focal_action: int):
        fa = int(focal_action)
        self.strat.other_last = fa
        # AIF 관점의 직전 joint state: (내 방출 행동, 상대=focal 행동)
        self._prev_state = joint_index(self.my_last_emitted, fa)


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

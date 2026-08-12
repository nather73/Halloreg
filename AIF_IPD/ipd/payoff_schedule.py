"""
ipd.payoff_schedule
===================

**비정상(non-stationary) 보수 구조 스케줄** — H4 / H4A / H5 의 환경.

[동기]
고정전략(TFT/GTFT/WSLS/ALLC/ALLD)은 보수행렬을 **읽지 않는다**. 오직 행동 이력에만
반응한다. 반면 능동추론 에이전트는 EFE 의 선호 C 를 현재 보수로 계산하므로,
게임 구조가 변하는 환경에서 맥락-의존적으로 자·타 효용을 재계산할 수 있다.
초록의 주장 — *Under non-stationary payoffs, it achieves higher final payoffs than
static strategies* — 은 이 비대칭에 근거한다.

[조작화]
이산 IPD 를 유지하되 **협력지수** CI = (R − P) / (T − S) 를 라운드마다 변동시킨다.
T=5, S=0, P=1 을 고정하고 R = P + CI·(T − S) = 1 + 5·CI.

    CI  < 0     교착(deadlock)  : R < P — 상호협력이 상호배신보다 나쁘다.
    CI = 0.2    가혹한 PD       : 2R < T + S — 협력이 매우 취약.
    CI = 0.4    기본 PD
    CI = 0.8    관대한 PD       : 여전히 T > R 이지만 협력 유인이 크다.
    CI  ≥ 1     조화(harmony)   : R > T — 협력이 우월전략.

[레짐 카탈로그]
"최대한 다양한 가변적 payoff 구조의 조합" 요구에 따라, 변동의 **성격**이 서로
다른 6개 레짐을 둔다. 각 레짐은 (t, T) → CI 함수다.

  stationary   : 대조군. 고정 PD (CI = 0.4). H3 와 동일한 구조.
  blocks       : 계단형 블록 전환. 조화 → PD → 교착 → PD → 조화.
                 급격한 구조 전환에 대한 적응을 본다.
  oscillate    : 정현파 진동. 연속적·주기적 변동에 대한 추적을 본다.
  aba          : A → B → A 복귀. 맥락 복귀 시 회복(reversal learning)을 본다.
  drift        : 결정론적 선형 표류. 느린 단조 변화에 대한 적응을 본다.
  shock        : 대부분 기본 PD 이나 특정 구간에서만 급격한 교착 충격.
                 희소·급성 위기에 대한 자기보호를 본다.

레짐 함수는 **결정론적**이다(라운드 인덱스만의 함수). 따라서 같은 레짐 안에서는
모든 유형·모든 시드가 동일한 보수 궤적을 겪으며, 유형 간 비교가 짝지어진다.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

CIFn = Callable[[int, int], float]


# ------------------------------------------------------------------ 기본형
def ci_constant(ci: float) -> CIFn:
    """모든 라운드에서 상수 CI."""
    return lambda t, T: float(ci)


def ci_blocks(values) -> CIFn:
    """지평을 등분해 블록마다 다른 CI (계단형 구조 전환)."""
    vals = [float(v) for v in values]

    def f(t: int, T: int) -> float:
        k = len(vals)
        b = min(k - 1, int(t * k / max(T, 1)))
        return vals[b]
    return f


def ci_oscillate(lo: float, hi: float, period: int) -> CIFn:
    """lo ↔ hi 사이를 period 라운드로 진동하는 CI (연속 변동)."""
    def f(t: int, T: int) -> float:
        phase = 0.5 * (1.0 + np.sin(2.0 * np.pi * t / max(period, 1)))
        return float(lo + (hi - lo) * phase)
    return f


def ci_aba(a: float, b: float) -> CIFn:
    """A → B → A 세 구간 (맥락 전환 후 복귀)."""
    def f(t: int, T: int) -> float:
        third = max(T // 3, 1)
        return float(a if (t < third or t >= 2 * third) else b)
    return f


def ci_drift(start: float, end: float) -> CIFn:
    """start → end 로의 선형 표류 (느린 단조 변화)."""
    def f(t: int, T: int) -> float:
        frac = t / max(T - 1, 1)
        return float(start + (end - start) * frac)
    return f


def ci_shock(base: float, shock: float,
             onset_frac: float = 0.4, width_frac: float = 0.15) -> CIFn:
    """기본 CI 를 유지하다 특정 구간에서만 급격히 shock 으로 떨어지는 스케줄."""
    def f(t: int, T: int) -> float:
        lo = int(onset_frac * T)
        hi = int((onset_frac + width_frac) * T)
        return float(shock if lo <= t < hi else base)
    return f


# ------------------------------------------------------------------ 카탈로그
#: 이름 → CI 함수. 실험 CLI 와 워커가 이 이름으로만 레짐을 참조한다
#: (함수 객체는 spawn 에서 pickle 하기 곤란하므로 **문자열로 전달**한다).
#: 레짐 패턴의 반복 주기 (라운드). 지평 T 가 아니라 **고정 주기**에 패턴을
#: 정의하고 t mod PERIOD 로 반복한다 (v3.7). 이유: 지평 전체에 패턴을 펼치면
#: '600R 학습 후 601~800R 평가' 프로토콜에서 평가창이 패턴의 꼬리 일부만
#: 보게 된다. 주기 반복이면 평가창(200R)이 정확히 한 주기의 전 국면을 담고,
#: 학습 구간(600R)이 같은 패턴을 3회 경험해 '학습된 비정상 적응'을 잰다.
REGIME_PERIOD: int = 200


def ci_periodic(fn: CIFn, period: int = REGIME_PERIOD) -> CIFn:
    """(t, T) → (t mod period, period) 로 감싸 패턴을 주기 반복시킨다."""
    def f(t: int, T: int) -> float:
        return fn(int(t) % int(period), int(period))
    return f


CI_REGIMES: Dict[str, CIFn] = {
    "stationary": ci_constant(0.4),
    "blocks": ci_periodic(ci_blocks([1.2, 0.4, -0.3, 0.4, 1.2])),
    "oscillate": ci_oscillate(-0.3, 1.2, period=30),   # 이미 30R 주기
    "aba": ci_periodic(ci_aba(1.0, -0.3)),
    "drift": ci_periodic(ci_drift(1.2, -0.2)),
    "shock": ci_periodic(ci_shock(0.4, -0.5, onset_frac=0.40,
                                  width_frac=0.15)),
}

#: H4/H4A/H5 에서 사용하는 비정상 레짐 목록 (stationary 제외).
NONSTATIONARY_REGIMES: List[str] = ["blocks", "oscillate", "aba",
                                    "drift", "shock"]

#: 한글 표기
REGIME_LABEL_KO = {
    "stationary": "정상(고정 PD)",
    "blocks": "블록 전환",
    "oscillate": "정현파 진동",
    "aba": "A-B-A 복귀",
    "drift": "선형 표류",
    "shock": "급성 충격",
}


def get_regime(name: str) -> CIFn:
    """레짐 이름 → CI 함수. 워커가 문자열로부터 함수를 복원할 때 쓴다."""
    if name not in CI_REGIMES:
        raise KeyError(f"알 수 없는 보수 레짐: {name}")
    return CI_REGIMES[name]


def regime_trace(name: str, n_rounds: int = 120) -> np.ndarray:
    """레짐의 CI 궤적 (시각화용)."""
    f = get_regime(name)
    return np.array([f(t, n_rounds) for t in range(n_rounds)])

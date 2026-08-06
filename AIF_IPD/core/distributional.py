"""
core.distributional
===================

**분위수 격자 유틸리티.**

v1.5.0 에서 `QuantileCode`(보상의 주변분포)는 폐기되었다. 정서의 두 축이 모두
QRTD 의 상황가치 Z(s,a) 에 근거하게 되면서(§core.core_affect), 주변분포가 맡던
역할이 사라졌기 때문이다.

  · valence : 현재 상대의 기대 보상 분포(중앙값)가, SelfModel 이 보유한 **다른
              타인들의 기대 보상 분포**(거리반비례 가중 평균)에서 차지하는 순위.
              — 시간에 대한 자기비교가 아니라 **사회적 모집단에 대한 횡단비교**.
  · arousal : Z(s,a) 믿음의 갱신량 (W₁ 이동).

남은 것은 격자 정의뿐이며, `core.qrtd` 와 `core.self_model` 이 공유한다.
"""

from __future__ import annotations

import numpy as np


def midpoint_taus(n: int) -> tuple:
    """
    **중점 분위수 격자** τ_i = (2i + 1) / (2n),  i = 0 … n−1.

    (1) 기대값이 단순 평균이 된다 (중점법 적분).
    (2) n 이 홀수면 정확히 τ = 0.5 매듭이 존재한다 — valence 부호 항등성의 근거.
        → 채널 수는 홀수여야 한다.
    """
    if n % 2 == 0:
        raise ValueError("채널 수는 홀수여야 합니다 (τ=0.5 매듭 보장)")
    return tuple((2 * i + 1) / (2.0 * n) for i in range(n))


#: 기본 격자 — 21채널. 11번째(i=10)가 정확히 0.5.
DEFAULT_TAUS = midpoint_taus(21)


def vector_cdf(values: np.ndarray, taus: np.ndarray, x: float) -> float:
    """
    분위수 벡터가 표상하는 분포에서 x 의 누적확률 F̂(x) ∈ (0, 1).

    격자 내부는 선형보간, 격자 밖은 지수꼬리(감쇠길이 = 분포 폭 — "놀람의
    단위는 경험의 폭"). 열린구간을 유지해 valence 가 경계에 붙지 않게 한다.
    """
    v = np.asarray(values, dtype=float)
    t = np.asarray(taus, dtype=float)
    s = float(v[-1] - v[0])
    if s < 1e-9:
        return 0.5
    x = float(x)
    eps = 1e-6
    if x <= v[0]:
        return float(np.clip(t[0] * np.exp(-(v[0] - x) / s), eps, 1 - eps))
    if x >= v[-1]:
        return float(np.clip(1.0 - (1.0 - t[-1]) * np.exp(-(x - v[-1]) / s),
                             eps, 1 - eps))
    k = int(np.searchsorted(v, x) - 1)
    k = max(0, min(k, len(v) - 2))
    lo, hi = v[k], v[k + 1]
    frac = 0.0 if hi - lo < 1e-12 else (x - lo) / (hi - lo)
    return float(t[k] + frac * (t[k + 1] - t[k]))

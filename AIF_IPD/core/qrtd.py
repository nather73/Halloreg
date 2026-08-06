"""
core.qrtd
=========

**QRTD — 분위수 회귀 시간차 학습 (Dabney et al. 2017, QR-DQN).**

`core.distributional.QuantileCode` 와의 관계를 먼저 분명히 해 둔다.

  · `QuantileCode` : **받아온 보상의 주변분포**. 행위로 색인되지 않고 부트스트랩
                     하지 않는다. 역할은 CoreAffect 의 valence 영점(할로스타틱
                     참조점)과 arousal 이다. 이것은 가치함수가 아니다.
  · `QuantileTD`   : **Z(s, a) — 행위로 색인된 수익(return) 분포.** TD 부트스트랩
                     을 하는 진짜 분포적 가치함수다. 역할은 행동 대안 간
                     기대효용 차이의 산출이다.

────────────────────────────────────────────────────────────────────────
왜 필요한가 — 보수행렬을 모를 때
────────────────────────────────────────────────────────────────────────
`empathy_shift(λ, p)` 는 행동 대안 간 기대효용 차이를

    Δ = (1−λ)·(E[u_self|C] − E[u_self|D]) + λ·(E[u_other|C] − E[u_other|D])

로 **보수행렬로부터 해석적으로** 계산한다. 이는 에이전트가 (R, T, S, P) 를 알 때만
쓸 수 있다. 그런데 이전 판은 비정상 보수 레짐에서도 HalloReg 에게만 매 라운드
현재 보수행렬을 특권적으로 넘겨주고 있었다(`set_payoffs(PAYOFF_SELF)`). 고정전략은
그 정보를 받지 못하므로, H4(비정상 보수에서의 우위)의 상당 부분이 모형의 우수성이
아니라 **정보 접근 특권**일 수 있었다. 이는 이론적 근거가 아니라 교란이다.

QRTD 는 그 교란을 제거한다. 관측된 보상만으로 Z(s, a) 를 학습하면

    Δ̂_λ(s) = (1−λ)·[V̂_self(s,C) − V̂_self(s,D)]
            +   λ ·[V̂_other(s,C) − V̂_other(s,D)]

를 **회고적으로** 얻을 수 있고, 보수행렬을 몰라도 행동 대안을 비교할 수 있다.
그러면 "보수 구조를 관측으로부터 학습해야 하는 조건에서도 비정상 환경에서
우위를 보인다" 는 더 강한 주장이 가능해진다.

`empathy_shift` 는 폐기되지 않는다 — **아는 경우의 닫힌 해(오라클)** 로 남으며,
ARCH 검사가 학습된 Δ̂ 가 그 해석해로 수렴하는지 확인한다.

────────────────────────────────────────────────────────────────────────
역할 분담 — 이중계산 방지
────────────────────────────────────────────────────────────────────────
Z 는 수익(할인 누적)이므로 그 차이에는 **미래 가치가 이미 들어 있다.** 형질
rollout 이 다시 미래를 누적하면 이중계산이 된다. 따라서 다음과 같이 자른다.

  · `RewardModel` R̂(joint) : **1-step 보상**의 분포. 우도 절편 ŝ(λ,p) 와 rollout
                              각 스텝의 효용 u_t 를 공급한다.
  · `QuantileTD`  Z(s, a)  : **종단 가치**로만 쓴다.
                              G(θ_i) = −E[ Σ_{t<H} u_t + γ^H · Z(s_H) ]

즉 유한 지평 안은 R̂ 로 명시적으로 굴리고, 지평 너머는 Z 가 요약한다. 표준적인
모형기반 부트스트랩 계획의 형태다.

────────────────────────────────────────────────────────────────────────
역할 분담 — 이중계산 방지
────────────────────────────────────────────────────────────────────────
Z 는 수익(할인 누적)이므로 그 차이에는 **미래가 이미 들어 있다.** 형질 rollout
이 다시 미래를 누적하면 이중계산이 된다. 따라서 다음과 같이 자른다.

  · `RewardModel` R̂(joint) : **1-step 보상**의 분포. 우도 절편 ŝ(λ,p) 와 rollout
                              각 스텝의 효용 u_t 를 공급한다.
  · `QuantileTD`  Z(s, a)  : **종단 가치**로만 쓴다.
                              G(θ_i) = −E[ Σ_{t<H} u_t + γ^H · Z(s_H) ]

즉 유한 지평 안은 R̂ 로 명시적으로 굴리고, 지평 너머는 Z 가 요약한다.

가치 조회는 **기대값**이다. 이전 판의 위험민감 축약(CVaR_τ)은 폐기했다
(mean_value 문서 참조).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .distributional import DEFAULT_TAUS

_EPS = 1e-12

#: 상태 = (내 직전 행동, 상대 직전 행동) → 4가지. 행위 = {C, D} → 2가지.
N_STATES = 4
N_ACTIONS = 2


def state_index(my_last: int, opp_last: int) -> int:
    """s = 2·my_last + opp_last  (C=0, D=1)."""
    return 2 * int(my_last) + int(opp_last)


def mean_value(quantiles: np.ndarray) -> float:
    """
    분포의 기대값 — 중점 분위수 격자에서는 채널 단순평균이 곧 중점법 적분이다.

    [tau_risk 폐기 — v1.5.3]
    이전 판은 하위 τ 꼬리 평균(CVaR_τ, 기본 0.3)을 써서 '위험민감 평가' 를
    표방했다. 폐기하는 이유는 두 가지다.

     (1) **자기보호는 이미 λ 가 담당한다.** 위험회피를 τ 로 한 번 더 부과하면
         같은 구성개념이 두 경로로 들어가 λ 의 해석이 흐려진다.
     (2) **추정 편향과 구별되지 않는다.** 방문이 적은 (상태,행위)는 분포가
         초깃값 근처에 머무는데, 하위 꼬리만 보면 그 미학습 상태가 '위험' 으로
         오독된다. 실제로 naive 모드에서 미방문 칸의 저평가를 τ 가 증폭했다.

    분포 표상의 가치는 valence(모집단 순위)와 arousal(W₁ 이동량)에서 이미
    쓰이고 있으므로, 가치 조회는 기대값으로 단순화한다.
    """
    return float(np.mean(quantiles))


class _QuantileArray:
    """분위수 벡터 묶음에 대한 공통 QR-Huber 갱신."""

    def __init__(self, shape: Tuple[int, ...], taus, init: float = 0.0,
                 spread: float = 1.0, lr: float = 0.05, kappa: float = 1.0):
        self.taus = np.asarray(taus, dtype=float)
        self.n = len(self.taus)
        self.lr = float(lr)
        self.kappa = max(float(kappa), 1e-6)
        # (..., n) 형태. τ 에 선형으로 벌려 초기 순위를 정의한다.
        base = float(init) + float(spread) * (self.taus - 0.5)
        self.values = np.broadcast_to(base, shape + (self.n,)).copy()

    def set_scale(self, span: float) -> None:
        """
        Huber 임계 κ 를 대상의 지지범위에 맞춘다.

        κ 는 '이 크기를 넘는 오차는 이상치로 보고 스텝을 포화시킨다' 는 경계다.
        비정규화 기울기를 쓰므로 κ 가 곧 한 갱신의 최대 이동량(× lr·w)이 된다.
        """
        self.kappa = max(0.25 * float(span), 1e-6)

    def _apply(self, idx: tuple, target: np.ndarray,
               lr: Optional[float] = None) -> None:
        """
        하나의 (상태, 행위) 칸에 분위수 Huber 갱신을 적용한다.

        target 은 **분포**(길이 n 의 표본)다. 분포 회귀이므로 각 목표 원소가
        모든 분위수 채널에 대해 손실을 만든다 — QR-DQN 의 교차 손실이다.
        """
        a = self.lr if lr is None else float(lr)
        cur = self.values[idx]                      # (n,)
        # delta[i, j] = target_j − θ_i
        delta = target[None, :] - cur[:, None]      # (n, n)
        w = np.where(delta > 0.0, self.taus[:, None], 1.0 - self.taus[:, None])
        # **비정규화 Huber 기울기** clip(δ, −κ, κ) 를 쓴다 (κ 로 나누지 않는다).
        # κ 로 나누면 한 갱신의 최대 이동량이 lr 로 상한되는데, 수익 Z 의 척도는
        # 보상의 1/(1−γ) 배(기본 10배)라 참값에 닿는 데 수천 갱신이 걸린다
        # (실측: 600R 후 V=7.4, 이론값 30). 비정규화하면 이동량이 오차 크기에
        # 비례하되 κ 에서 포화하므로, 척도가 다른 대상에도 같은 lr 로 수렴한다.
        step = np.clip(delta, -self.kappa, self.kappa)
        upd = a * np.mean(w * step, axis=1)         # 목표 표본에 대해 평균
        self.values[idx] = np.maximum.accumulate(cur + upd)


class QuantileTD:
    """
    **Z(s, a) — 분포적 수익 가치함수.** 자기·타자 두 벌을 함께 학습한다.

    IPD 는 양쪽 보수가 모두 관측되므로 타자 가치도 학습할 수 있다. 이것이
    Δ̂_λ 의 λ 가중 항을 공급한다.

    Parameters
    ----------
    gamma : float
        수익 할인. **스모크로 0.5 로 정했다.** γ=0.9 면 가치 범위가 보상 범위의
        10배(기본 PD 에서 [0,50])가 되어 세션 120라운드·8개 (s,a) 칸으로는
        수렴하지 못하고, TD 목표가 참조 부근에 뭉쳐 valence 분해능이 0.145 에
        그친다. γ=0.5 면 범위가 [0,10] 이라 학습 가능해져 분해능 0.806,
        충격 각성 배율 11.6배가 된다. 대가로 종단가치가 요약하는 미래가 짧아지나,
        형질 rollout 이 H=6 까지 명시적으로 굴리므로 그 너머의 기여는 작다.
    lr : float
        분위수 학습률.
    init, spread : float
        분위수 벡터의 초깃값과 초기 폭. SelfModel 의 참조분포가 같은 출발점을
        쓰도록 `init_value` / `init_spread` 로 노출된다 (척도 정합).
    """

    def __init__(self, taus=DEFAULT_TAUS, gamma: float = 0.5,
                 lr: float = 0.20,
                 init: float = 2.0, spread: float = 2.0,
                 seed: Optional[int] = None):
        self.taus = np.asarray(taus, dtype=float)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        shape = (N_STATES, N_ACTIONS)
        # 참조분포가 같은 출발점을 쓰도록 초깃값을 노출한다 (척도 정합).
        self.init_value = float(init)
        self.init_spread = float(spread)
        self.z_self = _QuantileArray(shape, taus, init=init, spread=spread,
                                     lr=lr)
        self.z_other = _QuantileArray(shape, taus, init=init, spread=spread,
                                      lr=lr)
        self.n_obs = 0

    def set_scale(self, span: float) -> None:
        """
        **가치 척도로** Huber 임계를 잡는다 (보상 척도가 아니다).

        Z 는 수익이므로 값의 범위가 보상 범위의 1/(1−γ) 배다(기본 PD·γ=0.9 에서
        [0,5] → [0,50]). 보상 범위로 κ 를 잡으면 스텝이 강하게 포화해 세션 안에
        수렴하지 못하고, TD 목표가 참조 분포보다 한참 아래에 머문다.
        """
        vspan = float(span) / max(1.0 - self.gamma, 1e-6)
        self.z_self.set_scale(vspan)
        self.z_other.set_scale(vspan)

    def init_values(self, r_min: float, r_max: float) -> None:
        """
        Z 를 **가치 지지범위의 중앙**에서 출발시킨다.

        기본 init=2.0 은 보상 척도의 값이라 가치 척도([0,50])에서는 바닥에 가깝다.
        그러면 학습 구간 내내 TD 목표가 참조보다 낮아 valence 가 포화하고, 동시에
        모든 관측이 '예상보다 좋음' 이 되어 부호가 왜곡된다. 중앙에서 출발하면
        실제 경험이 Z 를 위아래 어느 쪽으로든 움직일 수 있다.
        """
        v_lo, v_hi = self.value_support(r_min, r_max)
        mid, half = 0.5 * (v_lo + v_hi), 0.5 * (v_hi - v_lo)
        for arr in (self.z_self, self.z_other):
            arr.values = np.broadcast_to(
                mid + half * (arr.taus - 0.5),
                (N_STATES, N_ACTIONS, arr.n)).copy()

    # ============================================================ 갱신
    def update(self, s: int, a_i: int, r_self: float, r_other: float,
               s_next: int, p_coop_next: float) -> Tuple[float, float]:
        """
        한 전이 (s, a_i) → (r_self, r_other, s') 로 QRTD 갱신.

        다음 행위 a' 는 **현재 정책에서 표집**한다(on-policy, SARSA 형). 최대값을
        취하지 않는 이유는 여기서 필요한 것이 최적가치가 아니라 **현재 정책의
        가치평가**이기 때문이다 — 형질 rollout 이 정책을 바꾸어 가며 비교하므로,
        가치함수는 지금 정책의 결과를 정직하게 반영해야 한다.

        반환: (td_target, shift) — CoreAffect 의 valence·arousal 인자.
        """
        a_next = 0 if self.rng.random() < float(p_coop_next) else 1
        tgt_s = float(r_self) + self.gamma * self.z_self.values[s_next, a_next]
        tgt_o = float(r_other) + self.gamma * self.z_other.values[s_next,
                                                                 a_next]
        prev = self.z_self.values[s, int(a_i)].copy()
        self.z_self._apply((s, int(a_i)), tgt_s)
        self.z_other._apply((s, int(a_i)), tgt_o)
        self.n_obs += 1

        # CoreAffect 가 쓰는 두 양을 함께 돌려준다.
        #   td_target : **현재 보상 + 다음 상태의 가치** — "지금 일어난 일이
        #               그 함의까지 포함해 얼마나 좋은가" 의 스칼라 요약.
        #               valence 는 이 값을 SelfModel 의 전역 참조에 견준다.
        #   shift     : W₁(Z_before, Z_after) — 이 (상태,행위)의 **장기 가치에
        #               대한 믿음이 얼마나 흔들렸는가**. arousal 의 인자.
        #               즉각 보상이 아니라 전망이 놀람의 대상이라는 점에서
        #               보상 분포의 이동량보다 구성개념에 맞다.
        td_target = float(np.mean(tgt_s))
        shift = float(np.mean(np.abs(self.z_self.values[s, int(a_i)] - prev)))
        return td_target, shift

    # ============================================================ 조회
    def value(self, s: int, a: int, which: str = "self",
              tau: Optional[float] = None) -> float:
        """기대 가치 V̂(s, a). tau 인자는 서명 호환용이며 무시된다."""
        arr = self.z_self if which == "self" else self.z_other
        return mean_value(arr.values[int(s), int(a)])

    def terminal_value(self, s: int, p_coop: float, lam: float) -> float:
        """
        종단 상태의 λ-가중 가치. 행위는 정책 협력확률로 주변화한다.
        형질 rollout 의 지평 너머를 요약하는 값이다.
        """
        v = 0.0
        for a, pa in ((0, float(p_coop)), (1, 1.0 - float(p_coop))):
            if pa <= _EPS:
                continue
            v += pa * ((1.0 - lam) * self.value(s, a, "self")
                       + lam * self.value(s, a, "other"))
        return float(v)

    def value_support(self, r_min: float, r_max: float) -> Tuple[float, float]:
        """
        가치의 **구조적 지지범위** [r_min/(1−γ), r_max/(1−γ)].

        학습된 참조와 달리 보수 구조에서 직접 나오므로 경험을 따라 표류하지
        않는다. SelfModel 의 전역 가치참조를 초기화할 때 쓴다.
        """
        d = max(1.0 - self.gamma, 1e-6)
        return float(r_min / d), float(r_max / d)

    def shift(self, s: int, lam: float) -> float:
        """
        **Δ̂_λ(s) = V̂(s,C) − V̂(s,D) — 수익 차이. 진단·검증 전용.**

        [왜 이것을 우도 절편으로 쓰면 안 되는가 — 이중계산]
        정책 평가식은 다음과 같다.

            G(θ_i) = −E[ Σ_{t<H} u_t(R̂) + γ^H·Z(s_H) ] + w_cplx·KL

        여기서 **미래는 이미 두 번 다 계산되어 있다** — 지평 안(t<H)은 rollout 이
        명시적으로 굴리고, 지평 밖은 종단 Z(s_H) 가 요약한다. 그런데 각 스텝의
        행동확률을 만드는 우도

            P(a_i=C) = σ( β·( ρf + ωg + ηfg + s ) )

        의 절편 s 에 다시 수익 차이를 넣으면, rollout 의 **매 스텝마다** 그 시점
        이후의 미래가 한 번 더 들어간다. H 스텝이면 미래가 H+1 번 세어진다.
        절편이 맡아야 하는 것은 "지금 이 한 수의 즉각적 손익" 이고, 그 이후는
        rollout 과 종단항의 몫이다.

        [규모로도 확인된다]
        표준 PD·γ=0.9 에서 상호협력 상태의 수렴값을 실측하면
            Z(s,C) − Z(s,D) ≈ 20.0     (수익 차이)
            R̂ 기반 ŝ(λ=0.5, p=1) = 0.50 = empathy_shift 해석해
        로 **40배** 차이가 난다. β=4 이므로 σ(4·20) 은 즉시 포화해 ρ·ω·η 가
        로짓에 기여할 여지가 사라지고, 형질공간 정책 자체가 무력해진다.

        따라서 절편은 **1-step 보상모형** `RewardModel.shift` 가 맡는다. 같은
        QRTD 계열이면서 미래를 포함하지 않는 유일한 항이기 때문이다.
        """
        ds = self.value(s, 0, "self") - self.value(s, 1, "self")
        do = self.value(s, 0, "other") - self.value(s, 1, "other")
        return float((1.0 - lam) * ds + lam * do)


class RewardModel:
    """
    **R̂(joint) — 1-step 보상의 분포 모형.**

    4개 joint outcome (CC, CD, DC, DD) 각각에 대해 자기·타자 보상의 분위수를
    학습한다. 보수가 결정론적이면 점질량으로 수렴하지만, 비정상 레짐에서는
    분포가 실제로 퍼지며 그 폭 자체가 정보가 된다.

    이것이 `PAYOFF_SELF` / `PAYOFF_OTHER` 를 대체한다 — 즉 **에이전트가 보수행렬을
    특권적으로 받지 않아도 되게 만드는 부품**이다.
    """

    def __init__(self, taus=DEFAULT_TAUS, lr: float = 0.10,
                 init: float = 2.0,
                 spread: float = 2.0):
        self.taus = np.asarray(taus, dtype=float)
        self.r_self = _QuantileArray((4,), taus, init=init, spread=spread,
                                     lr=lr)
        self.r_other = _QuantileArray((4,), taus, init=init, spread=spread,
                                      lr=lr)
        self.n_obs = 0

    def set_scale(self, span: float) -> None:
        self.r_self.set_scale(span)
        self.r_other.set_scale(span)

    def update(self, joint: int, r_self: float, r_other: float) -> None:
        """관측된 joint outcome 의 보상으로 갱신. 목표는 점(길이 1 표본)."""
        self.r_self._apply((int(joint),), np.array([float(r_self)]))
        self.r_other._apply((int(joint),), np.array([float(r_other)]))
        self.n_obs += 1

    def payoff_vector(self, which: str = "self",
                      tau: Optional[float] = None) -> np.ndarray:
        """
        학습된 (4,) 보수 벡터 — 위험민감 통계량으로 축약.
        `PAYOFF_SELF` 자리에 그대로 꽂아 쓸 수 있다.
        """
        arr = self.r_self if which == "self" else self.r_other
        return np.array([mean_value(arr.values[j]) for j in range(4)],
                        dtype=float)

    def shift(self, lam: float, p_coop: float) -> float:
        """
        **ŝ(λ, p) — 학습된 1-step 기대효용 차이.**

        `empathy_shift(λ, p)` 와 동일한 식을 참 보수 대신 학습된 보수로 계산한다.
        정상 보수에서 관측이 쌓이면 해석해로 수렴해야 하며, ARCH 가 검사한다.
        """
        us = self.payoff_vector("self")
        uo = self.payoff_vector("other")
        p = float(np.clip(p_coop, 0.0, 1.0))
        # joint 인덱스 규약: CC=0, CD=1, DC=2, DD=3
        d_self = (p * us[0] + (1 - p) * us[1]) - (p * us[2] + (1 - p) * us[3])
        d_other = (p * uo[0] + (1 - p) * uo[1]) - (p * uo[2] + (1 - p) * uo[3])
        return float((1.0 - lam) * d_self + lam * d_other)

    def uncertainty(self) -> np.ndarray:
        """
        **결합결과별 인식적 불확실성 (4,)** — R̂ 의 각 칸이 얼마나 모르는가.

        분위수 벡터의 폭(최상위 − 최하위)을 쓴다. 관측이 없으면 초기 폭이
        그대로 남고, 관측이 쌓이면 참값 주위로 수축한다. 방문 횟수를 직접
        세는 대신 **표상 자체의 폭**을 쓰는 이유는, 비정상 보수에서는 많이
        방문해도 분포가 넓게 유지되는 것이 정상이기 때문이다 — 그때는 실제로
        불확실한 것이 맞다.

        환경 구조에 대한 epistemic affordance 의 원자료다.
        """
        return np.array([float(self.r_self.values[j][-1]
                               - self.r_self.values[j][0])
                         for j in range(4)], dtype=float)

    def snapshot(self) -> dict:
        return {"payoff_self": self.payoff_vector("self").tolist(),
                "payoff_other": self.payoff_vector("other").tolist(),
                "n_obs": int(self.n_obs)}

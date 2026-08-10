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
from .constants import joint_index

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
        정규화 기울기를 쓰므로 κ 는 **손실의 곡률**만 정하고, 한 갱신의 최대
        이동량은 lr·w 로 따로 정해진다 — 두 파라미터의 역할이 분리된다.
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
        # **표준 QR-Huber 기울기** (v2.7.0) — κ 로 정규화하므로 [−1, 1] 유계다.
        #     ∂/∂δ [ L_κ(δ)/κ ] = clip(δ, −κ, κ) / κ
        # 이전에는 비정규화 clip 을 썼다. 수익 Z 의 척도가 보상의 1/(1−γ) 배라
        # 정규화형에서는 한 갱신 이동량이 lr 로 상한되어 수렴이 느렸기 때문이다
        # (실측: 600R 후 이론값의 25%). 그러나 그 대가로 α·κ = 0.20 × 12.5 =
        # 2.5 > 1 이 되어 Dabney et al. 의 수축 사상 보장을 벗어나 있었다.
        # v2.7.0 은 **Z 자체를 정규화**(Z̃ = (1−γ)Z, 라운드당 평균 수익)해
        # 이동 거리를 1/10 로 줄였으므로, 표준 정규화 Huber 로 같은 속도를 낸다
        # (수치 대조: 600 갱신 후 도달률 94.5% vs 92.6%).
        step = np.clip(delta, -self.kappa, self.kappa) / self.kappa
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

    def __init__(self, taus=DEFAULT_TAUS, gamma: float = 0.9,
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
        #: (상태, 행위)별 방문 횟수 — 표본평균 학습률의 분모.
        self.n_sa = np.zeros((N_STATES, N_ACTIONS), dtype=float)
        self.lr_base = float(lr)
        self.bootstrap = "sarsa"   # 'sarsa' | 'greedy'
        self.z_self = _QuantileArray(shape, taus, init=init, spread=spread,
                                     lr=lr)
        self.z_other = _QuantileArray(shape, taus, init=init, spread=spread,
                                      lr=lr)
        self.n_obs = 0

    def set_scale(self, span: float) -> None:
        """
        **보상 척도로** Huber 임계를 잡는다 (v2.7.0).

        Z̃ = (1−γ)Z 는 '라운드당 평균 수익' 이라 보상과 같은 척도다. 따라서
        κ 를 1/(1−γ) 로 부풀릴 필요가 없고, 표준 정규화 Huber 가 그대로 쓰인다.
        """
        vspan = float(span)
        self.z_self.set_scale(vspan)
        self.z_other.set_scale(vspan)

    def reinit(self, center: float, spread: float) -> None:
        """
        **Z 를 '전형적 관계의 가치' 에서 출발시킨다** (SelfModel.value_prior).

        낙관적 초기화가 아니다 — 최대 보상이 아니라 자기 경험의 중앙값이며,
        사회사가 협력적이면 높고 착취적이면 낮다. "새 관계는 내가 겪어온
        관계들과 비슷할 것이다" 라는, 낙관도 비관도 아닌 사전이다.

        이로써 참조(기억 속 관계들)와 Z(현재 관계)가 **구성상 같은 척도**에서
        출발하므로 첫 라운드 valence 가 중립이 된다. init=2.0 은 γ=0.9 의 참
        가치 범위 [0,50] 에서 거의 최하단이라 사실상 비관적 초기화였다.
        """
        self.init_value = float(center)
        self.init_spread = float(spread)
        self.n_sa[:] = 0.0
        for arr in (self.z_self, self.z_other):
            base = float(center) + float(spread) * (arr.taus - 0.5)
            arr.values = np.broadcast_to(
                base, arr.values.shape[:-1] + (arr.n,)).copy()

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
               s_next: int, a_next: int) -> Tuple[float, float]:
        """
        한 전이 (s, a_i) → (r_self, r_other, s', a') 로 QRTD 갱신.

        **참 SARSA (v2.6.0)** — a' 는 표집하지 않고, s' 에서 **실제로 선택된**
        행위를 받는다. 그래서 호출자는 전이를 한 라운드 보류했다가 a' 가
        관측된 뒤 갱신한다.

        [왜 바뀌었는가]
        v2.5 까지는 a' ~ Bern(q_c) 로 표집했는데, 넘겨받는 q_c 가 `_last_qc`,
        즉 **s' 이 아니라 그 이전 상태 s 에서 산출된 정책**이었다. 부트스트랩은
        s' 에서의 행위를 요구하므로 한 상태 뒤처진 정책을 쓴 셈이다. 실측
        불일치(400R, 4시드): 평균 |Δq_c| 이 TFT 0.203 / WSLS 0.119 / ALLD
        0.087 이고 상관은 TFT 0.419 로 낮았다 — 상태 의존성이 강할수록 오차가
        컸다. 평균은 일치해 장기 편향은 작지만 상태 조건부로 계통 오차가 있었다.

        원인은 순환이었다: q_c(s') 를 알려면 λ_t 가 필요하고 λ_t 는 갱신된 Z 를
        필요로 하는데 그 갱신이 다시 q_c(s') 를 요구한다. 한 라운드 보류하면
        실제 a' 를 관측할 수 있으므로 순환이 **구조적으로** 풀린다.

        부수 효과: a' 표집이 사라져 목표의 표집 분산이 소멸한다. Z_self 와
        Z_other 가 같은 a' 를 쓰던 공통난수 장치도 불필요해진다 — 실제 행위
        하나뿐이므로 자동으로 일치한다.

        반환: (td_target, shift) — CoreAffect 의 valence·arousal 인자.
        """
        # **부트스트랩 모드** (v3.0)
        #   'sarsa'  : a' = 실제 관측된 다음 행위 (on-policy 평가)
        #   'greedy' : a' = argmax_a' Z̃_self(s', a') (최적가치 근사)
        # greedy 는 '실행하지 않아도 최선의 대안을 안다' 는 모형이고, sarsa 는
        # '하고 있는 것의 값만 안다' 는 모형이다. IPD 에서 이 차이는 착취의
        # 표상 여부로 나타난다 — 자세한 대조는 커밋 메시지 참조.
        if self.bootstrap == "greedy":
            a_next = int(np.argmax([
                float(np.mean(self.z_self.values[s_next, 0])),
                float(np.mean(self.z_self.values[s_next, 1]))]))
        else:
            a_next = int(a_next)
        # **정규화 수익** Z̃ = (1−γ)Z 이므로 보상도 (1−γ) 배로 들어간다:
        #     Z̃(s,a) = (1−γ)r + γ Z̃(s',a')
        # 고정점은 Z̃ → r̄ (라운드당 평균 보상)이라 보상과 같은 [0, 5] 척도다.
        _w_r = 1.0 - self.gamma
        tgt_s = _w_r * float(r_self) + self.gamma * self.z_self.values[s_next,
                                                                      a_next]
        tgt_o = _w_r * float(r_other) + self.gamma * self.z_other.values[
            s_next, a_next]
        prev = self.z_self.values[s, int(a_i)].copy()

        # **방문 횟수 적응 학습률** (v1.9.0)
        #   lr_eff(s,a) = max( 1/(1+n(s,a)), lr_base )
        # 첫 방문은 lr=1 로 목표에 곧바로 붙고, 이후 표본평균 감쇠(1/2, 1/3 …)
        # 를 거쳐 기저 학습률로 내려간다(Robbins–Monro).
        #
        # [왜 필요한가 — 초깃값 오염]
        # 고정 lr 에서는 드물게 선택되는 행위의 칸이 초깃값에 머문다. Z 를 사회사
        # 사전(≈17.6)에서 출발시키므로 그 잔류가 **낙관적 초기화를 뒷문으로**
        # 들여왔다: ALLD 상대에서 협력 분기 방문이 400R 중 16회뿐이라
        # A_self = Z(s,C) − Z(s,D) 가 −0.12 (이론값 −1.0 의 1/8)로 무력했다.
        # 한 번 겪어 보상이 낮으면 그 즉시 그 행위의 기대 보상이 내려가야 한다.
        n_sa = self.n_sa[s, int(a_i)]
        lr_eff = max(1.0 / (1.0 + n_sa), self.lr_base)
        self.z_self._apply((s, int(a_i)), tgt_s, lr=lr_eff)
        self.z_other._apply((s, int(a_i)), tgt_o, lr=lr_eff)
        self.n_sa[s, int(a_i)] += 1.0
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

    # ============================================ 모형 기반 계획 스윕 (v3.0)
    def _mix_quantiles(self, va, vb, w):
        """
        두 분위수 벡터의 **참 혼합분포 분위수** (무게중심이 아니다).

        w·F_a + (1−w)·F_b 의 분위수를 구한다. 분위수 벡터를 각각 w/N,
        (1−w)/N 가중 원자로 보고 정렬 → 누적가중 → τ 격자에 보간한다.
        분위수 평균 w·va + (1−w)·vb 는 Wasserstein 무게중심이라 이봉성을
        뭉개므로 (예: 30·50 혼합이 전 채널 40) 쓰지 않는다.
        """
        n = va.shape[0]
        x = np.concatenate([va, vb])
        ww = np.concatenate([np.full(n, w / n), np.full(n, (1.0 - w) / n)])
        o = np.argsort(x)
        x = x[o]; ww = ww[o]
        cw = np.cumsum(ww) - 0.5 * ww
        return np.interp(self.taus, cw, x)

    def plan_sweep(self, rmodel, p_coop_j: float, p_coop_self: float,
                   n_sweeps: int = 1) -> None:
        """
        **학습된 생성모형 위에서의 분포적 가치 반복** (Dyna 형 계획 스윕).

            Z̃(s,a) ← mix_{a_j∼Bern(p_j)}[ (1−γ)R̂(a,a_j)
                                          + γ·mix_{a'∼π}Z̃(s'(a,a_j), a') ]

        8 개 (s, a) 칸 **전부**를 매 라운드 갱신하므로, 수렴이 표본 방문이
        아니라 **γ-수축**에 지배된다. 오차가 스윕당 0.9 배로 줄어 ~40R 이면
        1% 이내다.

        [무엇을 고치는가 — 진단]
        ALLD 상대 400R 후 실측 Z̃(DD,C)=1.684 / Z̃(DD,D)=1.718 로, 참
        on-policy 값(0.609 / 0.709)보다 두 칸 모두 +1.0 떠 있었다. 원인은
        (i) 사회사 사전 초기화 2.11 이 이 관계 참값의 3 배, (ii) 부트스트랩
        목표 자체가 오염되어 축차 수축만 진행, (iii) 방문 불균형(C 77 회 vs
        D 175 회)이 적게 방문한 칸에 편향을 더 남김. 그 결과 이점이
        −0.100(참값) 대신 −0.034 로 압착되어, λ=0.017 을 달성하고도 ALLD
        상대 협력률이 0.32 로 남았다.
        전 칸 스윕은 셋을 동시에 없앤다 — 미방문 반사실 분기도 모형 안에서
        올바른 값을 받고(p_j → 0 이 그 분기를 정확히 평가), 초기화는 수축에
        씻겨 나간다.

        [학술적 위치] Sutton (1990) Dyna, Moore & Atkeson (1993) prioritized
        sweeping, van Seijen et al. (2009) Expected SARSA 의 분포적 확장.
        a_j 주변화는 무게중심이 아니라 **참 혼합 분위수**로 하므로 편향이 없다.

        [대가] 모형을 신뢰하는 만큼 R̂·p_j 의 오차가 가치로 전이된다. 보수
        레짐 전환 직후에는 잘못된 값으로 스윕하지만, R̂ 도 방문 적응 학습률로
        빠르게 재학습되므로 전이 창은 짧다.
        """
        # p_coop_j 는 **상태별 벡터 (4,)** 다. 스칼라 하나로 두면 TFT 처럼
        # 내 직전 행위에 조건부인 상대의 조건부성이 통째로 사라져, 모형
        # 안에서 '무작위 협력자' 로 오표상된다 (실측: TFT CC 0.850 → 0.674).
        pj = np.clip(np.asarray(p_coop_j, dtype=float).reshape(-1), 0.0, 1.0)
        if pj.size == 1:
            pj = np.full(4, float(pj[0]))
        pc = float(np.clip(p_coop_self, 0.0, 1.0))
        w_r = 1.0 - self.gamma
        for _ in range(int(n_sweeps)):
            for arr, rvec in ((self.z_self, rmodel.r_self),
                              (self.z_other, rmodel.r_other)):
                # 다음 상태에서 정책 π 로 주변화한 가치 (4,) × N
                nxt = np.stack([
                    self._mix_quantiles(arr.values[sp, 0],
                                        arr.values[sp, 1], pc)
                    for sp in range(4)])
                new = np.empty_like(arr.values)
                for s in range(4):
                    for a in range(2):
                        # 상대 행위 a_j 두 분기를 각각 만든 뒤 참 혼합
                        br = []
                        for aj in (0, 1):
                            j = joint_index(a, aj)
                            br.append(w_r * float(mean_value(rvec.values[j]))
                                      + self.gamma * nxt[j])
                        new[s, a] = self._mix_quantiles(br[0], br[1],
                                                       float(pj[s]))
                arr.values[:] = np.maximum.accumulate(new, axis=-1)


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
        #: 결합결과별 관측 횟수 — 인식적 불확실성의 감쇠 인자.
        self.n_visit = np.zeros(4, dtype=float)

    def set_scale(self, span: float) -> None:
        self.r_self.set_scale(span)
        self.r_other.set_scale(span)

    def update(self, joint: int, r_self: float, r_other: float) -> None:
        """관측된 joint outcome 의 보상으로 갱신. 목표는 점(길이 1 표본)."""
        self.r_self._apply((int(joint),), np.array([float(r_self)]))
        self.r_other._apply((int(joint),), np.array([float(r_other)]))
        self.n_visit[int(joint)] += 1.0
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
        **결합결과별 인식적 불확실성 (4,)** — 환경 구조에 대한 epistemic
        affordance 의 원자료.

            U_j = spread_j / sqrt(1 + n_j)

        [왜 방문 횟수로 감쇠시키는가 — v1.7.0]
        이전 판은 분위수 폭(spread) 만 썼다. 그런데 폭에는 **줄어들 수 있는
        불확실성(epistemic)** 과 **줄어들 수 없는 변동성(aleatoric)** 이 섞여
        있다. 정보이득은 전자만을 겨냥해야 하는데, 폭만 보면 후자까지 탐색
        유인으로 오독한다. 그 결과 학습이 끝난 뒤에도 IG_R 이 소멸하지 않고
        (0.38 → 0.31 정체) 계속 배신을 유인했다.

        절제 실측이 그 대가를 보여준다 (301~600R 정상상태, CC율):
            TFT  0.356 → 0.587   WSLS 0.541 → 0.644   (IG_R 을 끄면)
            ALLC 0.904 → 0.903   ALLD 0.006 → 0.006   (영향 없음)
        엄격 호혜 상대에게만 해로운데, 탐침적 배신 한 번이 메아리 사슬을 만들기
        때문이다. 협력 진화에서 가장 중요한 상대들이다.

        1/sqrt(1 + n_j) 는 표본평균 추정의 표준오차 감쇠율이다. 관측이 쌓이면
        0 으로 가므로 **탐색이 저절로 꺼진다.** spread 를 곱해 남겨두는 이유는,
        같은 방문 횟수라면 변동이 큰 칸이 실제로 더 불확실하기 때문이다 —
        비정상 보수 레짐에서 이 성질이 필요하다.
        """
        sp = np.array([float(self.r_self.values[j][-1]
                             - self.r_self.values[j][0]) for j in range(4)],
                      dtype=float)
        return sp / np.sqrt(1.0 + self.n_visit)

    def snapshot(self) -> dict:
        return {"payoff_self": self.payoff_vector("self").tolist(),
                "payoff_other": self.payoff_vector("other").tolist(),
                "n_obs": int(self.n_obs)}

"""
ipd.tom.inversion
=================

**OpponentInversion — 상대 특성 θ_j 에 대한 입자필터(particle filter) 베이지안 추론.**

초록의 (i) *perspective-taking, approximated via particle filter Bayesian inference
over the opponent's model parameters* 에 해당하는 모듈이다.

[생성 우도 — (1, f, g, f·g) 기저]
각 입자는 하나의 행동 프로파일 가설 θ_k 를 담는다.

    P(a_j = C | h_t, θ_k)
        = σ( β_k · ( α_k + ρ_k·f + ω_k·g + η_k·f·g + s(λ_k, p) ) )

    f      : 내(focal) 직전 행동의 호혜신호.  협력 → +1, 배신 → −1, 이력없음 → 0
    g      : 상대 **자신의** 직전 행동 신호.  협력 → +1, 배신 → −1
    α      : 협력 편향 (높으면 ALLC 성향, 낮으면 ALLD 성향)
    ρ      : 호혜성 (양수면 TFT 성향 — 내 직전 행동에 반응)
    ω      : 관성/자기일관성 (자기 직전 행동을 반복하는 경향)
    η      : 결과-조건성 (f·g 상호작용 — WSLS 의 시그니처)
    β      : 행동정밀도 (높으면 의도적/결정론적, 낮으면 잡음 = 맥락적)
    λ_j    : 상대의 공감 가중 (재귀적 잠재변수)
    s(λ,p) : 상대의 공감이 협력에 주는 이득. core.constants.empathy_shift 참조.

[왜 (1, f, g, f·g) 기저가 필수인가 — H1 의 식별가능성]
기억-1(memory-one) 전략의 행동은 (자기 직전행동, 상대 직전행동) 4개 조합 위의
협력확률로 완전히 결정된다. 이 4차원 공간을 스팬하려면 기저가 4개 자유도를
가져야 하고, (1, f, g, f·g) 가 정확히 그것이다. 특히:

    WSLS : 상대(j)는 "이겼으면 유지, 졌으면 전환" 한다. j 의 승리는 focal 이
           협력했을 때(f = +1)이므로,
               a_j = (자기 직전행동)      if f = +1
                     (자기 직전행동의 반전) if f = −1
           부호로 쓰면 next_sign = f · g. → **순수 η 축**으로 표현된다.

즉 (1, f) 기저만 쓰면 WSLS 는 α·ρ 로 표현할 수 없어 필연적으로 오분류된다.
H1 이 WSLS 를 포함하므로 본 판의 기본 기저는 "fg" 이다.

[시제(tense) 규약 — off-by-one 재발 방지]
동시행동 게임에서 라운드 k 에 관측되는 상대 행동은 opp_{k−1} 이며, 이는 상대가
**직전에 본 것**에 반응한 결과다. 따라서:

    경로     | 대상            | f (내 직전행동)  | g (상대 직전행동)
    ---------|-----------------|------------------|-------------------
    갱신용   | 관측 opp_{k−1}  | my_{k−2}         | opp_{k−2}
    결정용   | 예측 opp_k      | my_{k−1} (최신)  | opp_{k−1} (최신)

갱신 경로의 f·g 는 **둘 다 한 시점 더 과거**를 가리킨다. 호출부(agent.py)가
ObservationContext 에 올바른 시제를 담아 넘길 책임을 진다.

[SelfModel 연동]
`set_prior(prior)` 로 SelfModel 이 보관한 (theta, dist) 를 주입받아 입자를 재표집한다.
신규 상대면 무정보 사전, 재조우면 과거 관계 지점에서 출발한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from AIF_IPD.core.constants import COOP, empathy_shift

_EPS = 1e-10

# θ 축 순서 — core.self_model.THETA_AXES 와 반드시 일치해야 한다.
THETA_AXES = ("alpha", "rho", "omega", "eta", "beta", "lambda_j")

# 축별 물리적 허용 범위 (입자 클리핑).
#   β > 0  : 정밀도는 양수.
#   λ_j ∈ [0,1] : 볼록결합 가중.
#   ρ, ω, η 는 로짓 계수라 유계일 필요는 없으나, 수치 폭주 방지를 위해 넉넉히 자른다.
_BOUNDS = {
    "alpha": (-6.0, 6.0),
    "rho": (-4.0, 4.0),
    "omega": (-4.0, 4.0),
    "eta": (-4.0, 4.0),
    "beta": (0.05, 12.0),
    "lambda_j": (0.0, 1.0),
}


def _logistic(x: np.ndarray) -> np.ndarray:
    """수치안정 로지스틱 σ(x)."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


@dataclass
class ObservationContext:
    """우도 계산에 필요한 라운드 맥락 (시제는 호출부 책임)."""
    my_last_action: Optional[int] = None      # f 의 원천
    their_last_action: Optional[int] = None   # g 의 원천
    joint_outcome: Optional[int] = None
    round_number: int = 0


class OpponentInversion:
    """
    상대 θ_j = (α, ρ, ω, η, β, λ_j) 에 대한 축차 중요도 표집(SIR) 입자필터.

    Parameters
    ----------
    n_particles : int
        입자 수 N. 6차원 사후를 다루므로 400 이상 권장.
    resample_frac : float
        유효표본수 ESS 가 resample_frac·N 밑으로 내려가면 리샘플링.
    jitter_scale : float
        리샘플 후 추가하는 확산(roughening) 배율. 입자 고갈(degeneracy)을 막고
        **비정상 상대의 형질 전환을 추적**할 수 있게 하는 인공 확산항이다.
        0 이면 입자가 한 점으로 붕괴해 H1A(형질 추적)가 불가능해진다.
    """

    # 축별 무정보 사전 (평균, 표준편차)
    _PRIOR = {
        "alpha": (0.0, 2.0),
        "rho": (0.5, 1.2),
        "omega": (0.0, 1.0),
        "eta": (0.0, 1.0),
        "beta": (3.0, 1.8),
        # λ_j 사전 평균은 **0.5(중립)** 이다. Empathy 의 λ_ctx 는 (2λ̂_j − 1) 을
        # 쓰므로, 평균이 0.5 가 아니면 관측이 전혀 없는 상태에서도 λ_ctx 가
        # 0 이 아닌 값을 갖게 되어 사전만으로 λ 가 표류한다(사전 인공물).
        "lambda_j": (0.5, 0.3),
    }
    # 축별 기본 리샘플 jitter 표준편차
    _JITTER = {"alpha": 0.10, "rho": 0.10, "omega": 0.10,
               "eta": 0.10, "beta": 0.15, "lambda_j": 0.05}

    def __init__(self, n_particles: int = 400, resample_frac: float = 0.5,
                 jitter_scale: float = 1.0, seed: int = 0):
        self.N = int(n_particles)
        self.resample_frac = float(resample_frac)
        self.jitter_scale = float(jitter_scale)
        self.rng = np.random.default_rng(seed)

        # 내 협력률에 대한 상대의 믿음 p — empathy_shift 의 인자.
        self.my_cooperation_rate = 0.5

        # 현재 사전 (SelfModel 주입 시 교체)
        self._prior = {ax: self._PRIOR[ax] for ax in THETA_AXES}
        self._init_particles()

    # ============================================================ 초기화
    def _init_particles(self) -> None:
        """현재 사전에서 입자를 표집하고 가중치를 균등화한다."""
        self.theta = {}
        for ax in THETA_AXES:
            mu, sd = self._prior[ax]
            lo, hi = _BOUNDS[ax]
            self.theta[ax] = np.clip(
                self.rng.normal(mu, max(sd, 1e-6), self.N), lo, hi)
        self.weights = np.ones(self.N) / self.N
        self._h0_cache = None

    def set_prior(self, prior: Dict[str, tuple], reinit: bool = True) -> None:
        """
        SelfModel 이 공급한 θ 사전 {축: (평균, 표준편차)} 을 주입한다.
        reinit=True 면 즉시 입자를 재표집(새 상대와의 관계 시작).
        """
        for ax in THETA_AXES:
            if ax in prior:
                mu, sd = prior[ax]
                self._prior[ax] = (float(mu), float(sd))
        if reinit:
            self._init_particles()

    # ============================================================ 특징 추출
    @staticmethod
    def _feature_f(ctx: Optional[ObservationContext]) -> float:
        """f = 내 직전 행동의 호혜신호 (+1 협력 / −1 배신 / 0 이력없음)."""
        if ctx is None or ctx.my_last_action is None:
            return 0.0
        return 1.0 - 2.0 * float(ctx.my_last_action)

    @staticmethod
    def _feature_g(ctx: Optional[ObservationContext]) -> float:
        """g = 상대 자신의 직전 행동 신호 (+1 협력 / −1 배신 / 0 이력없음)."""
        if ctx is None or ctx.their_last_action is None:
            return 0.0
        return 1.0 - 2.0 * float(ctx.their_last_action)

    # ============================================================ 우도
    def _pC(self, f: float, g: float) -> np.ndarray:
        """각 입자가 예측하는 상대 협력확률 (N,)."""
        th = self.theta
        shift = empathy_shift(th["lambda_j"], self.my_cooperation_rate)
        logit = th["beta"] * (th["alpha"] + th["rho"] * f
                              + th["omega"] * g + th["eta"] * f * g
                              + shift)
        return _logistic(logit)

    def update(self, opp_action: int, ctx: ObservationContext) -> None:
        """
        관측된 상대 행동으로 입자 가중치를 갱신한다(중요도 갱신 + 조건부 리샘플).

        ctx 의 시제는 **갱신용**(f = my_{k−2}, g = opp_{k−2})이어야 한다.
        """
        f = self._feature_f(ctx)
        g = self._feature_g(ctx)
        pC = self._pC(f, g)
        lik = pC if int(opp_action) == COOP else (1.0 - pC)

        self.weights = self.weights * np.clip(lik, _EPS, 1.0)
        s = float(self.weights.sum())
        if s <= _EPS:
            # 모든 입자가 관측을 설명 못 함 → 사전으로 재출발(수치 안전장치)
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights = self.weights / s

        ess = 1.0 / float(np.sum(self.weights ** 2))
        if ess < self.resample_frac * self.N:
            self._resample()
        self._h0_cache = None      # 가중치가 바뀌었으므로 엔트로피 캐시 무효화

    def _resample(self) -> None:
        """
        체계적 중요도 리샘플 + roughening.

        리샘플만 하면 중복 입자가 생겨 사후가 점질량으로 붕괴한다(sample
        impoverishment). 축별 jitter 를 더해 입자를 다시 퍼뜨리는데, 이 인공
        확산은 동시에 **θ 가 시간에 따라 천천히 변할 수 있다**는 상태공간 가정을
        구현한다 — H1A(형질 전환 추적)가 성립하는 이유.
        """
        idx = self.rng.choice(self.N, size=self.N, p=self.weights)
        for ax in THETA_AXES:
            lo, hi = _BOUNDS[ax]
            sd = self._JITTER[ax] * self.jitter_scale
            self.theta[ax] = np.clip(
                self.theta[ax][idx] + self.rng.normal(0.0, sd, self.N), lo, hi)
        self.weights = np.ones(self.N) / self.N

    # ============================================================ 예측
    def predict_coop(self, f: float, g: float = 0.0) -> float:
        """사후 가중 상대 협력확률 P(a_j = C)."""
        return float(np.clip(np.sum(self.weights * self._pC(f, g)),
                             _EPS, 1.0 - _EPS))

    def predict_action(self, ctx: Optional[ObservationContext]) -> np.ndarray:
        """ctx(결정용 시제) 하의 상대 행동 분포 [P(C), P(D)]."""
        pc = self.predict_coop(self._feature_f(ctx), self._feature_g(ctx))
        return np.array([pc, 1.0 - pc])

    # ============================================================ 요약통계
    def posterior_means(self) -> Dict[str, float]:
        """축별 사후 평균 (SelfModel 의 theta 에 해당)."""
        return {ax: float(np.sum(self.weights * self.theta[ax]))
                for ax in THETA_AXES}

    def posterior_stds(self) -> Dict[str, float]:
        """축별 사후 표준편차 (SelfModel 의 dist 에 해당)."""
        m = self.posterior_means()
        out = {}
        for ax in THETA_AXES:
            v = float(np.sum(self.weights * (self.theta[ax] - m[ax]) ** 2))
            out[ax] = float(np.sqrt(max(v, 0.0)))
        return out

    def reliability(self) -> float:
        """
        입자 가중치의 집중도 기반 신뢰도 r ∈ [0, 1] — GatedToM 의 게이팅 신호.
        ESS/N 의 제곱근을 쓴다(가중치가 균등이면 1, 한 입자에 몰리면 0).
        """
        ess = 1.0 / float(np.sum(self.weights ** 2))
        return float(np.clip(ess / self.N, 0.0, 1.0)) ** 0.5

    def belief_update_magnitude(self, prev_means: Dict[str, float]) -> float:
        """
        직전 사후 평균 대비 현재 평균의 **표준화 변화량 L2 노름**.
        모형기반 fMRI 의 belief-update 회귀자에 대응하는 스칼라 진단값.
        """
        cur = self.posterior_means()
        d = [(cur[ax] - prev_means.get(ax, cur[ax])) / self._PRIOR[ax][1]
             for ax in THETA_AXES]
        return float(np.sqrt(np.sum(np.square(d))))

    # ============================================================ 정보이득
    # EFE 의 인식적(epistemic) 항. 축별 주변 히스토그램 엔트로피의 합으로
    # 이산 상호정보 I(a_j ; θ) 를 근사한다. 이산 MI 이므로 비음이 자동 보장된다.
    _HIST_NB = 11          # 히스토그램 빈 수
    _HIST_PAD = 0.5        # 적응적 구간의 여유 = 0.5 · SD

    def _hist_entropy(self, arr: np.ndarray, w: np.ndarray, axis: str) -> float:
        """가중 히스토그램의 Shannon 엔트로피."""
        m = float(np.sum(w * arr))
        sd = float(np.sqrt(max(float(np.sum(w * (arr - m) ** 2)), 1e-12)))
        if axis == "lambda_j":
            lo, hi = 0.0, 1.0                    # 자연 구간이 있는 축
        else:
            lo, hi = m - 3.0 * sd - self._HIST_PAD, m + 3.0 * sd + self._HIST_PAD
        if hi - lo < 1e-9:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, self._HIST_NB + 1)
        idx = np.clip(np.digitize(arr, edges) - 1, 0, self._HIST_NB - 1)
        b = np.zeros(self._HIST_NB)
        np.add.at(b, idx, w)
        s = b.sum()
        if s <= _EPS:
            return 0.0
        b = b / s
        nz = b[b > _EPS]
        return float(-np.sum(nz * np.log(nz)))

    def _total_entropy(self, w: np.ndarray) -> float:
        """전 축 주변 엔트로피의 합."""
        return sum(self._hist_entropy(self.theta[ax], w, ax)
                   for ax in THETA_AXES)

    def _current_entropy(self) -> float:
        """
        현재 사후의 전축 엔트로피 H0. 한 라운드 안에서 여러 후보 행동·정책에
        대해 반복 호출되지만 신념이 불변이므로 값이 같다 → 메모이제이션.
        가중치 배열 **객체 자체**를 캐시 키로 보유해 id 재활용 오류를 막는다.
        """
        if self._h0_cache is not None and self._h0_cache[0] is self.weights:
            return self._h0_cache[1]
        h0 = self._total_entropy(self.weights)
        self._h0_cache = (self.weights, h0)
        return h0

    def expected_infogain(self, f_next: float, g_next: float = 0.0) -> float:
        """
        내가 특정 행동을 두어 다음 호혜신호가 f_next 가 될 때, 상대의 다음 행동
        관측이 θ̂ 사후를 얼마나 좁힐지의 **기대** 엔트로피 감소.

            IG = H[θ] − E_{a_j}[ H[θ | a_j] ]
        """
        pC = self._pC(f_next, g_next)
        p_obsC = float(np.sum(self.weights * pC))
        H0 = self._current_entropy()

        def post_H(lik):
            w2 = self.weights * lik
            s = w2.sum()
            return H0 if s <= _EPS else self._total_entropy(w2 / s)

        H_exp = p_obsC * post_H(pC) + (1.0 - p_obsC) * post_H(1.0 - pC)
        return max(0.0, H0 - H_exp)

    def observed_infogain(self, obs_action: int, f: float,
                          g: float = 0.0) -> float:
        """
        **실현** 정보이득 — 특정 관측 행동이 이 필터의 사후를 얼마나 좁히는가.

        자기-사영(self-projection) 필터에 쓴다: 내가 obs_action 을 두면(내 행동은
        내가 확정적으로 안다) 상대가 나에 대해 갖는 믿음 θ̂_self 가 얼마나
        좁아지는가. "기대"가 아니라 "실현"인 점이 expected_infogain 과 다르다.
        """
        pC = self._pC(f, g)
        lik = pC if int(obs_action) == COOP else (1.0 - pC)
        H0 = self._current_entropy()
        w2 = self.weights * lik
        s = w2.sum()
        if s <= _EPS:
            return 0.0
        return max(0.0, H0 - self._total_entropy(w2 / s))

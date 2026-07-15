"""
ipd.tom.inversion
=================

상대 특성 파라미터 θ_j 에 대한 **입자필터(particle filter) 기반 베이지안 추론**.

각 입자는 parametric behavioral profile 을 담는다:

    P(a_j = C | h_t, θ_k) = σ( β_k · ( α_k + ρ_k·f(h_t) + s(λ_k, p) ) )

  α : 협력편향(cooperation bias)   — 높으면 ALLC 성향, 낮으면 ALLD 성향
  ρ : 호혜성(reciprocity)          — 양수면 TFT 성향(내 직전 행동에 반응)
  β : 행동정밀도(precision)        — 높으면 '의도적/결정론적', 낮으면 '잡음(맥락 불확실)'
  λ_j : 상대 공감(opponent empathy) — 상대가 내 후생을 얼마나 중시하는가(재귀적 잠재변수)
  f(h_t) : 직전 내 행동에 대한 호혜신호 (+1 협력, -1 배신, 0 이력없음)
  s(λ_k, p) : 상대의 공감이 협력에 주는 social-EFE 이득
              (PD 보수에서 유도: 5·λ_j − p − 1, p=내 협력율에 대한 상대의 믿음)

Albarracin et al. (2026) 의 particle-based inversion 및 참조 저장소
`mahault/empathy-prisonner-dilemma` 의 inversion.py 를 pymdp 비의존 numpy 로 재구성하고,
**HalloReg 확장**을 추가했다:

  * 귀인 개인차의 우도-수준 반영: core allostatic belief 가 산출한 축별 신뢰도
    가중치 w_θ 로 (i) 초기 입자 사전 분산과 (ii) 리샘플링 jitter 를 변조한다.
    → 특정 파라미터(예: β)에 높은 신뢰도를 둔 개인은 그 축의 변동을 암시하는
      예측못한 관측에 더 빠르고 즉각적으로 반응한다(사양서의 '논리적 방식').
  * λ_j 를 잠재변수로 명시하여 재귀적 ToM(상대도 내 특성을 추론)에 사용.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np

from AIF_IPD.core.constants import COOP, empathy_shift as C_empathy_shift, DEFECT

_EPS = 1e-10

# θ 축 순서 (core.allostasis 의 마스크와 일치)
THETA_AXES = ("alpha", "rho", "beta", "lambda_j")


def _logistic(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + np.exp(-(x - center) / scale))


@dataclass
class ObservationContext:
    """우도 계산에 필요한 라운드 맥락."""
    my_last_action: Optional[int] = None
    their_last_action: Optional[int] = None
    joint_outcome: Optional[int] = None
    round_number: int = 0


@dataclass
class InversionState:
    """입자필터 상태 스냅샷."""
    reliability: float
    entropy: float
    ess: float
    means: Dict[str, float]


class OpponentInversion:
    """
    상대 θ_j=(α,ρ,β,λ_j) 에 대한 입자필터.

    HalloReg 확장: `reliability_weights`(core allostatic belief 유도)로 축별
    사전 분산과 jitter 를 변조하여, 귀인 개인차가 베이지안 갱신에 자연스럽게
    스며들도록 한다.
    """

    # 축별 기본 사전(prior)  (mean, std)
    _PRIOR = {
        "alpha": (0.0, 2.0),
        "rho": (0.5, 1.2),
        "beta": (3.0, 1.8),      # 감마 대신 절단 정규로 근사
        "lambda_j": (0.4, 0.3),
    }
    # 축별 기본 리샘플 jitter
    _JITTER = {"alpha": 0.10, "rho": 0.10, "beta": 0.15, "lambda_j": 0.05}

    def __init__(self, n_particles: int = 400,
                 reliability_weights: Optional[dict] = None,
                 resample_frac: float = 0.5, seed: int = 0):
        self.N = int(n_particles)
        self.rng = np.random.default_rng(seed)
        self.resample_frac = resample_frac
        # 축별 신뢰도 가중치(없으면 균등). β 신뢰도가 높으면 β 축을 더 넓게/빠르게 탐색.
        self.w_theta = {k: 1.0 for k in THETA_AXES}
        if reliability_weights:
            self.w_theta.update(reliability_weights)

        self.my_cooperation_rate = 0.5
        # H2 기제 절제용: β 축 동결 (사전 평균에 클램프 → β 로 설명 불가)
        self._beta_clamp_val = None
        self._init_particles()

    def clamp_beta(self, value=None):
        """입자필터의 β 축을 상수에 동결한다 (H2 절제 대조: β-귀인 경로 차단)."""
        self._beta_clamp_val = float(value) if value is not None \
            else float(self._PRIOR["beta"][0])
        self.beta[:] = self._beta_clamp_val

    # ------------------------------------------------------------ 초기화
    def set_reliability(self, weights: dict, reinit: bool = False):
        """core allostatic belief 로부터 축별 신뢰도 갱신. reinit 이면 사전 재표집."""
        self.w_theta.update(weights)
        if reinit:
            self._init_particles()

    def _spread(self, axis: str) -> float:
        """축별 신뢰도 → 사전/jitter 확산 배율. w=0 이면 거의 동결."""
        return 0.08 + 1.20 * float(self.w_theta.get(axis, 1.0))

    def _init_particles(self):
        def draw(axis):
            mu, sd = self._PRIOR[axis]
            # 신뢰도 높은 축 → 사전 분산 확대(더 많은 가설) → 그 축 변동에 민감.
            # 신뢰도 0(귀인 대상에서 제외된 축) → 사전이 거의 점질량으로 붕괴 →
            # 그 축의 변동을 '설명 자원'으로 쓸 수 없다(과대/과소귀인의 원천).
            sd_eff = sd * self._spread(axis)
            return self.rng.normal(mu, sd_eff, self.N)

        self.alpha = draw("alpha")
        self.rho = draw("rho")
        self.beta = np.clip(draw("beta"), 0.05, 12.0)
        self.lambda_j = np.clip(draw("lambda_j"), 0.0, 1.0)
        self.weights = np.ones(self.N) / self.N
        if getattr(self, "_beta_clamp_val", None) is not None:
            self.beta[:] = self._beta_clamp_val

    # ------------------------------------------------------------ 우도
    def _empathy_shift(self) -> np.ndarray:
        """상대 공감 λ_j 가 협력에 주는 이득 — 현재 보수 (R,T,S,P) 유도 (v0.6.3).
        기본 PD 보수에서 레거시 5λ_j − p − 1 과 비트 단위 동일; 가변 페이오프
        환경(§VP)에서는 호출 시점의 보수를 반영(맥락-의존 ToM 공감항)."""
        return C_empathy_shift(self.lambda_j, self.my_cooperation_rate)

    def _pC(self, f: float) -> np.ndarray:
        """각 입자의 상대 협력확률."""
        logit = self.beta * (self.alpha + self.rho * f + self._empathy_shift())
        return _logistic(logit)

    @staticmethod
    def _feature(ctx: Optional[ObservationContext]) -> float:
        if ctx is None or ctx.my_last_action is None:
            return 0.0
        return 1.0 - 2.0 * ctx.my_last_action   # C=0 → +1, D=1 → -1

    def update(self, opp_action: int, ctx: ObservationContext) -> InversionState:
        """관측된 상대 행동으로 가중치를 갱신."""
        f = self._feature(ctx)
        pC = self._pC(f)
        lik = pC if opp_action == COOP else (1.0 - pC)
        self.weights = self.weights * np.clip(lik, _EPS, 1.0)
        s = self.weights.sum()
        if s <= _EPS:
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights /= s

        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.resample_frac * self.N:
            self._resample()

        return self.snapshot()

    def _resample(self):
        idx = self.rng.choice(self.N, size=self.N, p=self.weights)

        def jit(x, axis, lo=None, hi=None):
            # 신뢰도 높은 축 → jitter 확대 → 빠른 추적
            sd = self._JITTER[axis] * self._spread(axis)
            y = x[idx] + self.rng.normal(0.0, sd, self.N)
            if lo is not None:
                y = np.clip(y, lo, hi)
            return y

        self.alpha = jit(self.alpha, "alpha")
        self.rho = jit(self.rho, "rho")
        if self._beta_clamp_val is not None:
            self.beta[:] = self._beta_clamp_val
        else:
            self.beta = jit(self.beta, "beta", 0.05, 12.0)
        self.lambda_j = jit(self.lambda_j, "lambda_j", 0.0, 1.0)
        self.weights = np.ones(self.N) / self.N

    # ------------------------------------------------------------ 예측/요약
    def predict_coop(self, f: float) -> float:
        """posterior-weighted 상대 협력확률."""
        return float(np.clip(np.sum(self.weights * self._pC(f)), _EPS, 1 - _EPS))

    def predict_action(self, ctx: Optional[ObservationContext]) -> np.ndarray:
        f = self._feature(ctx)
        pc = self.predict_coop(f)
        return np.array([pc, 1.0 - pc])

    def posterior_means(self) -> Dict[str, float]:
        return {
            "alpha": float(np.sum(self.weights * self.alpha)),
            "rho": float(np.sum(self.weights * self.rho)),
            "beta": float(np.sum(self.weights * self.beta)),
            "lambda_j": float(np.sum(self.weights * self.lambda_j)),
        }

    def posterior_stds(self) -> Dict[str, float]:
        m = self.posterior_means()
        out = {}
        for ax, arr in (("alpha", self.alpha), ("rho", self.rho),
                        ("beta", self.beta), ("lambda_j", self.lambda_j)):
            v = np.sum(self.weights * (arr - m[ax]) ** 2)
            out[ax] = float(np.sqrt(max(v, 0.0)))
        return out

    def _param_entropy(self) -> float:
        """(α,β) 가중분산 로그합 — 미분엔트로피 대용치."""
        m = self.posterior_means()
        v_a = np.sum(self.weights * (self.alpha - m["alpha"]) ** 2) + 1e-6
        v_b = np.sum(self.weights * (self.beta - m["beta"]) ** 2) + 1e-6
        return 0.5 * (np.log(v_a) + np.log(v_b))

    def reliability(self) -> float:
        """가중치 집중도 기반 신뢰도 r∈[0,1] (게이팅용)."""
        ess = 1.0 / np.sum(self.weights ** 2)
        return float(np.clip(ess / self.N, 0.0, 1.0)) ** 0.5

    def belief_update_magnitude(self, prev_means: Dict[str, float]) -> float:
        """
        Buergi et al.(2026) 의 belief-update(BU) 신호에 대응하는 스칼라.
        직전 posterior mean 대비 현재 mean 의 표준화 변화량 L2 노름.
        model-based fMRI regressor 로 사용 가능.
        """
        cur = self.posterior_means()
        scale = {"alpha": 2.0, "rho": 1.2, "beta": 1.8, "lambda_j": 0.3}
        d = [(cur[a] - prev_means.get(a, cur[a])) / scale[a] for a in THETA_AXES]
        return float(np.sqrt(np.sum(np.square(d))))

    def snapshot(self) -> InversionState:
        return InversionState(
            reliability=self.reliability(),
            entropy=self._param_entropy(),
            ess=float(1.0 / np.sum(self.weights ** 2)),
            means=self.posterior_means(),
        )

    # ------------------------------------------------------------ epistemic
    def expected_infogain(self, my_action: int, f_next: float) -> float:
        """
        내가 my_action 을 둘 때(→ 다음 호혜신호 f_next) 상대 다음 행동 관측으로부터
        기대되는 θ 정보이득(엔트로피 감소). social EFE 의 epistemic 항.
        """
        pC = self._pC(f_next)
        p_obsC = float(np.sum(self.weights * pC))
        H0 = self._param_entropy()

        def post_entropy(obs_lik):
            w2 = self.weights * obs_lik
            ssum = w2.sum()
            if ssum <= _EPS:
                return H0
            w2 = w2 / ssum
            m_a = np.sum(w2 * self.alpha)
            m_b = np.sum(w2 * self.beta)
            v_a = np.sum(w2 * (self.alpha - m_a) ** 2) + 1e-6
            v_b = np.sum(w2 * (self.beta - m_b) ** 2) + 1e-6
            return 0.5 * (np.log(v_a) + np.log(v_b))

        H_ifC = post_entropy(pC)
        H_ifD = post_entropy(1.0 - pC)
        H_exp = p_obsC * H_ifC + (1.0 - p_obsC) * H_ifD
        return max(0.0, H0 - H_exp)

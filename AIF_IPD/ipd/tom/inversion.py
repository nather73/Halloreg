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
from typing import Optional, List, Dict, Tuple

import numpy as np

from AIF_IPD.core.constants import COOP, empathy_shift as C_empathy_shift, DEFECT

_EPS = 1e-10

# θ 축 순서 (core.allostasis 의 마스크와 일치)
THETA_AXES = ("alpha", "rho", "beta", "lambda_j")

# [v0.8.0 §7] fg 기저 확장 축. 기저 (1, f, g, fg) 는 기억-1 시그니처 전 공간
# (4자유도)을 스팬한다 → 기억-1 상대(WSLS 포함)는 모두 표현 가능해진다.
THETA_AXES_FG = ("alpha", "rho", "omega", "eta", "beta", "lambda_j")


def theta_axes(basis: str = "f"):
    """likelihood_basis 에 따른 θ 축 튜플."""
    return THETA_AXES_FG if basis == "fg" else THETA_AXES


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
        # [v0.8.0 §7.2] ω(관성/자기일관성), η(결과-조건성/WSLS성).
        # 사전 범위 ω, η ∈ [−2,+2] 근방 = ρ 와 동급 스케일. 평균 0(무정보).
        "omega": (0.0, 1.0),
        "eta": (0.0, 1.0),
    }
    # 축별 기본 리샘플 jitter
    _JITTER = {"alpha": 0.10, "rho": 0.10, "beta": 0.15, "lambda_j": 0.05,
               "omega": 0.10, "eta": 0.10}

    def __init__(self, n_particles: int = 400,
                 reliability_weights: Optional[dict] = None,
                 resample_frac: float = 0.5, seed: int = 0,
                 likelihood_basis: str = "f"):
        """
        likelihood_basis : {"f", "fg"}
            "f"  (기본) : P(a_j=C|f,θ) = σ(β(α + ρf + s(λ_j,p)))     — v0.6.6 보존
            "fg" (§7)   : P(a_j=C|f,g,θ) = σ(β(α + ρf + ωg + η·fg + s(λ_j,p)))
                기저 (1,f,g,fg) 가 기억-1 시그니처 전 공간을 스팬 → WSLS 표현 가능.
        """
        if likelihood_basis not in ("f", "fg"):
            raise ValueError(f"알 수 없는 likelihood_basis: {likelihood_basis}")
        self.likelihood_basis = likelihood_basis
        self.axes = theta_axes(likelihood_basis)
        self.N = int(n_particles)
        self.rng = np.random.default_rng(seed)
        self.resample_frac = resample_frac
        # 축별 신뢰도 가중치(없으면 균등). β 신뢰도가 높으면 β 축을 더 넓게/빠르게 탐색.
        self.w_theta = {k: 1.0 for k in self.axes}
        if reliability_weights:
            self.w_theta.update(reliability_weights)

        self.my_cooperation_rate = 0.5
        # H2 기제 절제용: β 축 동결 (사전 평균에 클램프 → β 로 설명 불가)
        self._beta_clamp_val = None
        self._init_particles()

    @property
    def _fg(self) -> bool:
        return self.likelihood_basis == "fg"

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
        # [v0.8.0 §7] fg 기저에서만 활성. f 기저에서는 0 상수 → 항이 소멸하여
        # 우도가 v0.6.6 과 비트 단위 동일(하위호환 보장).
        if self._fg:
            self.omega = np.clip(draw("omega"), -3.0, 3.0)
            self.eta = np.clip(draw("eta"), -3.0, 3.0)
        else:
            self.omega = np.zeros(self.N)
            self.eta = np.zeros(self.N)
        self.weights = np.ones(self.N) / self.N
        if getattr(self, "_beta_clamp_val", None) is not None:
            self.beta[:] = self._beta_clamp_val

    # ------------------------------------------------------------ 우도
    def _empathy_shift(self) -> np.ndarray:
        """상대 공감 λ_j 가 협력에 주는 이득 — 현재 보수 (R,T,S,P) 유도 (v0.6.3).
        기본 PD 보수에서 레거시 5λ_j − p − 1 과 비트 단위 동일; 가변 페이오프
        환경(§VP)에서는 호출 시점의 보수를 반영(맥락-의존 ToM 공감항)."""
        return C_empathy_shift(self.lambda_j, self.my_cooperation_rate)

    def _pC(self, f: float, g: float = 0.0) -> np.ndarray:
        """각 입자의 상대 협력확률.

        f 기저 : σ(β(α + ρf + s))                       — ω=η=0 이므로 g 무시
        fg 기저: σ(β(α + ρf + ωg + η·fg + s))           — §7.2
        """
        logit = self.beta * (self.alpha + self.rho * f
                             + self.omega * g + self.eta * f * g
                             + self._empathy_shift())
        return _logistic(logit)

    # ================================================================ 시제 표
    # [v0.8.0 §7.2] f·g 의 시제(tense) 규약 — **오프바이원 재발 방지**.
    #
    # 동시행동 게임에서 라운드 k 에 관측되는 상대 행동은 opp_{k−1} 이며, 이는
    # 상대가 **직전에 본 것**에 반응한 결과다. 따라서:
    #
    #   경로     | 대상            | f (내 직전행동)   | g (상대 직전행동)
    #   ---------|-----------------|-------------------|-------------------
    #   갱신용   | 관측 opp_{k−1}  | my_{k−2}          | opp_{k−2}
    #   결정용   | 예측 opp_k      | my_{k−1}(=최신)   | opp_{k−1}(=최신)
    #
    # 즉 갱신 경로의 f·g 는 **둘 다 한 시점 더 과거**를 가리킨다(v0.6.6 이 f 에
    # 대해 수정한 것과 동일한 시제 원칙을 g 에 그대로 적용). 호출부(agent.py)가
    # ctx.my_last_action / ctx.their_last_action 에 올바른 시제를 담아 넘긴다.
    # 데이터-수준 일치도는 §7.4 검증 1(TFT·WSLS)에서 필수 확인한다.

    @staticmethod
    def _feature(ctx: Optional[ObservationContext]) -> float:
        """f = 내 직전 행동의 호혜신호 (+1 협력, −1 배신, 0 이력없음)."""
        if ctx is None or ctx.my_last_action is None:
            return 0.0
        return 1.0 - 2.0 * ctx.my_last_action   # C=0 → +1, D=1 → -1

    @staticmethod
    def _feature_g(ctx: Optional[ObservationContext]) -> float:
        """g = 상대 자신의 직전 행동 (+1 협력, −1 배신, 0 이력없음).

        `ObservationContext.their_last_action` 은 v0.6.6 에도 이미 존재했으나
        우도가 사용하지 않았다(§7.2). fg 기저에서 비로소 소비된다.
        """
        if ctx is None or ctx.their_last_action is None:
            return 0.0
        return 1.0 - 2.0 * ctx.their_last_action

    def update(self, opp_action: int, ctx: ObservationContext) -> InversionState:
        """관측된 상대 행동으로 가중치를 갱신."""
        f = self._feature(ctx)
        g = self._feature_g(ctx) if self._fg else 0.0
        pC = self._pC(f, g)
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
        if self._fg:
            self.omega = jit(self.omega, "omega", -3.0, 3.0)
            self.eta = jit(self.eta, "eta", -3.0, 3.0)
        else:
            self.omega = self.omega[idx]
            self.eta = self.eta[idx]
        if self._beta_clamp_val is not None:
            self.beta[:] = self._beta_clamp_val
        else:
            self.beta = jit(self.beta, "beta", 0.05, 12.0)
        self.lambda_j = jit(self.lambda_j, "lambda_j", 0.0, 1.0)
        self.weights = np.ones(self.N) / self.N

    # ------------------------------------------------------------ 예측/요약
    def predict_coop(self, f: float, g: Optional[float] = None) -> float:
        """posterior-weighted 상대 협력확률.

        g : fg 기저의 상대 직전행동 신호(±1). None 이면 0(중립) — f 기저에서는
            ω=η=0 이므로 어차피 무영향이다.
        """
        gg = 0.0 if g is None else float(g)
        return float(np.clip(np.sum(self.weights * self._pC(f, gg)),
                             _EPS, 1 - _EPS))

    def predict_action(self, ctx: Optional[ObservationContext]) -> np.ndarray:
        f = self._feature(ctx)
        g = self._feature_g(ctx) if self._fg else 0.0
        pc = self.predict_coop(f, g)
        return np.array([pc, 1.0 - pc])

    def theta_reward_moments(self, my_action: int, ctx: Optional[ObservationContext],
                             payoff_self) -> Tuple[float, float, float]:
        """
        [v0.11.0 §θ-잔차] 내 행동 a_i 에 대한 **θ-조건부 예상보수**의 입자 분해.

        각 입자 θ_k 로 상대 협력확률 pc_k = P(coop|θ_k, ctx) 을 얻고, 내 행동에서의
        예상보수 r̂_k = pc_k·U(a_i,C) + (1−pc_k)·U(a_i,D) 를 계산한다. 반환:
          · E_θ[r]        = Σ w_k r̂_k               (θ 가 설명하는 기대보수)
          · epistemic_std = √Σ w_k (r̂_k − E_θ[r])²   (입자간 분산 = θ 불확실성)
          · aleatoric_std = √Σ w_k Var_k[r]          (입자내 보수분산 — 참고용, 미사용)
        RPE 잔차화에서 관측보수 r_obs 대비 잔여 mean=r_obs−E_θ[r]→valence,
        epistemic_std→uncertainty 로 귀인된다(입자간 분산만 반영, 사양).
        payoff_self : PAYOFF_SELF 배열 (상태 CC/CD/DC/DD → 내 보수).
        """
        f = self._feature(ctx)
        g = self._feature_g(ctx) if self._fg else 0.0
        pc = self._pC(f, g)                       # (N,) 입자별 상대 협력확률
        # 상태 인덱스: 내 행동(0=C,1=D) × 상대(0=C,1=D) → CC/CD/DC/DD = 0/1/2/3
        if int(my_action) == 0:                    # 내가 협력
            u_if_coop, u_if_def = payoff_self[0], payoff_self[1]   # CC, CD
        else:                                      # 내가 배신
            u_if_coop, u_if_def = payoff_self[2], payoff_self[3]   # DC, DD
        r_hat = pc * float(u_if_coop) + (1.0 - pc) * float(u_if_def)   # (N,)
        w = self.weights
        E_theta = float(np.sum(w * r_hat))
        epi_var = float(np.sum(w * (r_hat - E_theta) ** 2))
        # 입자내 aleatoric: 각 입자의 베르누이 보수분산
        ale_var = float(np.sum(w * pc * (1.0 - pc)
                               * (float(u_if_coop) - float(u_if_def)) ** 2))
        return E_theta, float(np.sqrt(max(epi_var, 0.0))), float(np.sqrt(max(ale_var, 0.0)))

    def posterior_means(self) -> Dict[str, float]:
        out = {
            "alpha": float(np.sum(self.weights * self.alpha)),
            "rho": float(np.sum(self.weights * self.rho)),
            "beta": float(np.sum(self.weights * self.beta)),
            "lambda_j": float(np.sum(self.weights * self.lambda_j)),
        }
        if self._fg:
            out["omega"] = float(np.sum(self.weights * self.omega))
            out["eta"] = float(np.sum(self.weights * self.eta))
        return out

    def posterior_stds(self) -> Dict[str, float]:
        m = self.posterior_means()
        out = {}
        pairs = [("alpha", self.alpha), ("rho", self.rho),
                 ("beta", self.beta), ("lambda_j", self.lambda_j)]
        if self._fg:
            pairs += [("omega", self.omega), ("eta", self.eta)]
        for ax, arr in pairs:
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
        scale = {"alpha": 2.0, "rho": 1.2, "beta": 1.8, "lambda_j": 0.3,
                 "omega": 1.0, "eta": 1.0}
        d = [(cur[a] - prev_means.get(a, cur[a])) / scale[a] for a in self.axes]
        return float(np.sqrt(np.sum(np.square(d))))

    def snapshot(self) -> InversionState:
        return InversionState(
            reliability=self.reliability(),
            entropy=self._param_entropy(),
            ess=float(1.0 / np.sum(self.weights ** 2)),
            means=self.posterior_means(),
        )

    # ------------------------------------------------ [v0.9.0 §5] histogram IG
    # 축별 히스토그램 구간(§5.2 확정 규약).
    #   · 무계 축: 입자 posterior 의 적응적 범위 [q05 − pad, q95 + pad], nb 빈.
    #   · λ_j 는 자연구간 [0,1] 유지.
    #   · pad = pad_frac · SD.  IG 는 이산 상호정보라 비음 자동 보장.
    _HIST_NB = 11          # 빈 수 nb (§10)
    _HIST_PAD_FRAC = 0.5   # pad = 0.5·SD (§10)
    _NAT_RANGE = {"lambda_j": (0.0, 1.0)}   # 자연구간 축

    def _axis_arr(self, axis: str) -> np.ndarray:
        return getattr(self, axis)

    def _weighted_hist_entropy(self, arr: np.ndarray, weights: np.ndarray,
                               axis: str) -> Tuple[float, tuple]:
        """가중 빈질량의 Shannon 엔트로피 −Σ b·log b 와 사용한 구간(edges) 반환."""
        m = float(np.sum(weights * arr))
        v = float(np.sum(weights * (arr - m) ** 2))
        sd = float(np.sqrt(max(v, 1e-12)))
        if axis in self._NAT_RANGE:
            lo, hi = self._NAT_RANGE[axis]
        else:
            q05, q95 = self._weighted_quantiles(arr, weights, (0.05, 0.95))
            pad = self._HIST_PAD_FRAC * sd
            lo, hi = q05 - pad, q95 + pad
        if hi - lo < 1e-9:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, self._HIST_NB + 1)
        idx = np.clip(np.digitize(arr, edges) - 1, 0, self._HIST_NB - 1)
        b = np.zeros(self._HIST_NB)
        np.add.at(b, idx, weights)
        s = b.sum()
        if s <= _EPS:
            return 0.0, (lo, hi)
        b = b / s
        nz = b[b > _EPS]
        return float(-np.sum(nz * np.log(nz))), (lo, hi)

    @staticmethod
    def _weighted_quantiles(arr: np.ndarray, weights: np.ndarray,
                            qs) -> np.ndarray:
        order = np.argsort(arr)
        a = arr[order]
        w = weights[order]
        cw = np.cumsum(w)
        cw = cw / max(cw[-1], _EPS)
        return np.interp(qs, cw, a)

    def _allaxis_entropy(self, weights: np.ndarray) -> float:
        """전 6축(무계는 적응구간, λ_j 자연구간) 주변 히스토그램 엔트로피 합(§5.1)."""
        total = 0.0
        for axis in self.axes:
            h, _ = self._weighted_hist_entropy(self._axis_arr(axis), weights, axis)
            total += h
        return total

    def _current_allaxis_entropy(self) -> float:
        """H0 = _allaxis_entropy(self.weights) 의 메모이제이션.

        플래너가 한 라운드 내 여러 행동가지·정책에 대해 IG 를 반복 평가할 때
        H0 는 (신념 불변) 매번 동일하지만 축별 argsort+히스토그램을 재계산하게
        된다. weights 배열 **객체 자체를 캐시에 참조로 보유**해 id 재활용을 막고,
        동일 객체일 때 재사용한다 — 수치 완전 동일, 순수 속도 개선.
        """
        cache = getattr(self, "_h0_cache", None)
        if cache is not None and cache[0] is self.weights:
            return cache[1]
        h0 = self._allaxis_entropy(self.weights)
        self._h0_cache = (self.weights, h0)   # 참조 보유 → id 재활용 방지
        return h0

    def expected_infogain_allaxis(self, my_action: int, f_next: float,
                                  g_next: float = 0.0) -> float:
        """
        [v0.9.0 §5] 전축 histogram 기대 정보이득.

        내가 my_action 을 둘 때(→ 다음 호혜신호 f_next) 상대 다음 행동 관측이
        상대 θ̂ posterior 를 얼마나 좁히는가의 기대 엔트로피 감소를, **전 6축**
        (α,ρ,ω,η,β,λ_j) 주변 히스토그램으로 계산한다(§5.1~5.2). f 기저에서는
        ω·η 가 상수축이라 자동 0 기여(하위호환).
        """
        pC = self._pC(f_next, g_next)
        p_obsC = float(np.sum(self.weights * pC))
        H0 = self._current_allaxis_entropy()

        def post_entropy(obs_lik):
            w2 = self.weights * obs_lik
            ssum = w2.sum()
            if ssum <= _EPS:
                return H0
            return self._allaxis_entropy(w2 / ssum)

        H_ifC = post_entropy(pC)
        H_ifD = post_entropy(1.0 - pC)
        H_exp = p_obsC * H_ifC + (1.0 - p_obsC) * H_ifD
        # 이산 상호정보라 비음 자동 보장(§5.2) — 수치 잔차만 클램프.
        return max(0.0, H0 - H_exp)

    def observed_infogain_allaxis(self, obs_action: int, f: float,
                                  g: float = 0.0) -> float:
        """
        [v0.9.0 §6.1] **실현** 정보이득 — 특정 관측 행동 obs_action 이 이 필터의
        전축 θ̂ posterior 를 얼마나 좁히는가.

        IG_other 에 쓰인다: self-projection 필터(상대가 나를 추론)에서, 내가
        obs_action 을 두면(내 행동은 결정론적으로 알려짐) 상대의 θ̂_self 가 얼마나
        좁아지는가. 기대(expected)가 아니라 실현 관측 기반이라는 점이 IG_self 와
        다르다("나는 내가 무엇을 둘지 안다").
        """
        pC = self._pC(f, g)
        obs_lik = pC if obs_action == COOP else (1.0 - pC)
        H0 = self._current_allaxis_entropy()
        w2 = self.weights * obs_lik
        ssum = w2.sum()
        if ssum <= _EPS:
            return 0.0
        H1 = self._allaxis_entropy(w2 / ssum)
        return max(0.0, H0 - H1)

    # ------------------------------------------------------------ epistemic
    def expected_infogain(self, my_action: int, f_next: float,
                          g_next: float = 0.0) -> float:
        """
        내가 my_action 을 둘 때(→ 다음 호혜신호 f_next) 상대 다음 행동 관측으로부터
        기대되는 θ 정보이득(엔트로피 감소). social EFE 의 epistemic 항.
        g_next : fg 기저의 상대 직전행동 신호(f 기저에서는 무영향).
        """
        pC = self._pC(f_next, g_next)
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

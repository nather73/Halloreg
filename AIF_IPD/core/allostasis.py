"""
core.allostasis
===============

**HalloReg 의 핵심** : Core Allostatic Belief State 와 그로부터 유도되는 공감
파라미터 λ 의 위계적 항상성 조절(Hierarchical Allostatic Regulation).

[이론적 근거]
  · Kim (2020), "Stability or Plasticity? A Hierarchical Allostatic Regulation
    Model of mPFC Function": mPFC 는 복측→배측 위계로 조직된다.
      - vmPFC : 내부(신체) 신호 기반의 즉각적/반사적 가치계산 (안정성 우선).
      - rmPFC : 내부-외부 가치의 중재(arbitration), 누적된 피드백 이력을 통합한
                정교한 자기보호, 'allostatic self-efficacy', model-free↔model-based 전환.
      - dmPFC : 외부(맥락) 정보 기반의 숙고적 가치계산, mentalizing (가소성 우선).
    더 배측 영역은 외부 정보를 이용해 더 복측 영역(내부 신호)에서 일어날 갈등을
    예측·예방한다. 이 위계가 stability-plasticity 딜레마를 해결한다.
  · Sul et al. (2015): 친사회성 개인차는 vmPFC(자기가치)↔dmPFC(타인가치) 가치표상
    구배로 나타난다 → 본 모형의 κ(귀인 성향) 개인차로 조작화.
  · Yoon/Lee et al. (2018): vmPFC 는 즉각·누적 피드백 모두에, rmPFC 는 *누적* 피드백에만
    관여하여 정교한 자기보호를 매개 → 본 모형의 sophisticated(rmPFC) 토글로 조작화.
  · Barrett/Katsumi (allostasis-first): 뇌의 핵심 기능은 항상성 불균형의 예측적 조절이며,
    예측오차 현저성은 '예측된 allostatic 가치'의 함수다 → core allostatic belief 는
    '무엇이 나의 항상성 불균형을 야기하는가'에 대한 연합학습된 생성적 믿음이다.

[Core Allostatic Belief State 의 정의]
  시행에 걸쳐 축적되는, 항상성 불균형의 *원인 귀속*에 대한 확률분포. 본 IPD 과제에서
  불균형을 야기할 수 있는 변인(원인 후보):
      C1. 예측하지 못한 타인의 defection 으로 인한 단기 손실 (transient / 맥락적)
      C2. 타인의 낮은 협력편향 α  (dispositional intent)
      C3. 타인의 낮은 공감 λ_j    (dispositional intent)
      C4. 타인의 낮은 호혜성 ρ    (dispositional intent)   [확장 변인]
      C5. 타인의 낮은 행동정밀도 β (맥락적 불확실성 / 잡음)
  개인은 자신의 metabolic cost 를 최소화하기 위해 예측못한 피드백을 위 원인들에
  서로 다른 정도로 귀인한다(개인차). 이 귀인 성향이:
    (a) 입자필터 우도 P(a_j=C|h_t,θ_k) 에 대한 *parameter 별 신뢰도 가중치* w_θ 로,
    (b) λ 조절 신호(dispositional grievance vs contextual discount)로
  각각 유도되어, θ 믿음 갱신과 상호작용 전략을 변화시킨다.

[λ 위계적 조절]
  vmPFC 층(즉각): 배신(CD) 관측 시 즉각적 방어 신호.
  rmPFC 층(누적, sophisticated): 누적 피드백 이력과 추론된 β(의도성)를 통합하여
                                 '의도적 배신'에만 선택적으로 방어(정교한 자기보호).
  dmPFC 층(외부/mentalizing): 추론된 θ(α,λ_j,ρ,β)로부터 dispositional 신뢰도를 산출.

  두 개의 누출적분기(leaky integrator):
      grievance g⁻ ∈ [0,1] : dispositional 로 귀인된 배신에 충전, 협력에 방전 → λ 억제
      trust     g⁺ ∈ [0,1] : cooperative 로 귀인된 협력에 충전, 배신에 방전 → λ 상향

      λ_eff = clip( λ_base·(1 - g⁻) + (λ_max - λ_base)·g⁺ , 0, λ_max )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# core allostatic belief 의 원인 후보 축 (파라미터 이름 -> 인덱스)
CAUSE_AXES = ("transient", "alpha", "lambda_j", "rho", "beta")
# 이 중 '내재된 의도(dispositional)'에 해당하는 축
DISPOSITIONAL_AXES = ("alpha", "lambda_j", "rho")
# '맥락적 불확실성(situational)'에 해당하는 축
CONTEXTUAL_AXES = ("transient", "beta")


def _sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + np.exp(-(x - center) / scale))


@dataclass
class CoreAllostaticBeliefState:
    """
    개인별로 상이하게 구현되는 '핵심 항상성 믿음 상태'.

    두 역할을 한다:
      1) 예측못한 사회 피드백을 원인 축(CAUSE_AXES)에 귀인하는 확률분포 `belief` 를
         시행에 걸쳐 누적한다.
      2) 그 믿음으로부터 입자필터가 사용할 *parameter 별 신뢰도 가중치* `reliability_weights`
         를 산출한다. 이 가중치는 어떤 θ 차원이 예측오차를 흡수(설명)할지를 조절한다
         → "귀인이 우도/갱신에 자연스럽게 포함"되는 논리적 방식.

    개인차 파라미터
    --------------
    kappa : float ∈ [0,1]
        dispositional-attribution 성향. κ~0 → 친사회(vmPFC 우세; 예측오차를 맥락에
        관대하게 귀인), κ~1 → 전략(dmPFC 우세; 예측오차를 상대의 의도에 귀인).
        Sul et al.(2015) 의 자기-타인 가치구배 개인차에 대응.
    attribution_target : str ∈ {'all','alpha_only','lambda_only','rho_only','beta_context'}
        예측오차를 어떤 θ 축들에 귀인하도록 허용할지. 'all' 은 정교한(모든 변인) 귀인.
        나머지는 절제(ablation) 조건으로 H9/H10 검증에 사용.
    prior_reliability : dict[str,float] | None
        사전 학습된 core allostatic belief 로 유도되는 축별 초기 신뢰도(선택).
    """

    kappa: float = 0.9
    attribution_target: str = "all"
    prior_reliability: Optional[dict] = None
    learning_rate: float = 0.15          # 원인 믿음 갱신률
    belief: np.ndarray = field(init=False)
    _mask: np.ndarray = field(init=False)

    def __post_init__(self):
        self.kappa = float(np.clip(self.kappa, 0.0, 1.0))
        # 원인 믿음: 초기엔 균등 (무엇이 불균형을 야기하는지 미지)
        self.belief = np.ones(len(CAUSE_AXES)) / len(CAUSE_AXES)
        self._mask = self._build_mask(self.attribution_target)
        self._cause_mask = self._build_cause_mask(self.attribution_target)
        if self.prior_reliability is not None:
            # 사전 학습된 신뢰도를 원인 믿음에 반영
            pr = np.array([self.prior_reliability.get(ax, 0.0) for ax in CAUSE_AXES])
            if pr.sum() > 0:
                self.belief = 0.5 * self.belief + 0.5 * (pr / pr.sum())

    # ---------------------------------------------------------------- 마스크
    #   θ 마스크     : 입자필터가 '갱신할 수 있는' 상대 형질 축 (alpha, rho, beta, lambda_j)
    #   cause 마스크 : core allostatic belief 이 '귀인할 수 있는' 원인 축 (CAUSE_AXES)
    #   두 마스크는 한 쌍의 개인차(귀인 성향)를 서로 다른 층에서 구현한다.
    _THETA_MASKS = {
        "all":          {"alpha": 1.0, "rho": 1.0, "beta": 1.0, "lambda_j": 1.0},
        "intent_only":  {"alpha": 1.0, "rho": 1.0, "beta": 0.0, "lambda_j": 1.0},
        "alpha_only":   {"alpha": 1.0, "rho": 0.0, "beta": 0.0, "lambda_j": 0.0},
        "lambda_only":  {"alpha": 0.0, "rho": 0.0, "beta": 0.0, "lambda_j": 1.0},
        "rho_only":     {"alpha": 0.0, "rho": 1.0, "beta": 0.0, "lambda_j": 0.0},
        "beta_context": {"alpha": 0.4, "rho": 0.0, "beta": 1.0, "lambda_j": 0.0},
    }
    #                    transient, alpha, lambda_j, rho, beta   (= CAUSE_AXES 순서)
    _CAUSE_MASKS = {
        "all":          (1.0, 1.0, 1.0, 1.0, 1.0),
        "intent_only":  (0.0, 1.0, 1.0, 1.0, 0.0),   # 맥락 귀인 불가 → 의도 과대귀인
        "alpha_only":   (0.0, 1.0, 0.0, 0.0, 0.0),
        "lambda_only":  (0.0, 0.0, 1.0, 0.0, 0.0),
        "rho_only":     (0.0, 0.0, 0.0, 1.0, 0.0),
        "beta_context": (1.0, 0.0, 0.0, 0.0, 1.0),   # 맥락 전용 → 의도 과소귀인
    }

    @classmethod
    def _build_mask(cls, target: str) -> np.ndarray:
        """attribution_target -> θ 축(alpha, rho, beta, lambda_j) 갱신 허용 마스크."""
        if target not in cls._THETA_MASKS:
            raise ValueError(f"알 수 없는 attribution_target: {target}")
        m = cls._THETA_MASKS[target]
        return np.array([m["alpha"], m["rho"], m["beta"], m["lambda_j"]])

    @classmethod
    def _build_cause_mask(cls, target: str) -> np.ndarray:
        """attribution_target -> CAUSE_AXES(원인) 귀인 허용 마스크."""
        return np.array(cls._CAUSE_MASKS[target], dtype=float)

    @property
    def theta_update_mask(self) -> np.ndarray:
        """입자필터 차원 (alpha, rho, beta, lambda_j) 갱신 허용/신뢰 가중치."""
        return self._mask.copy()

    # ---------------------------------------------------------------- 갱신
    def update(self, betrayal: bool, opp_cooperated: bool,
               inferred: dict, prediction_surprise: float,
               opp_defected: bool = False) -> None:
        """
        한 라운드 관측으로 core allostatic belief 를 갱신한다.

        betrayal : 내가 협력했는데 상대가 배신(CD) — 항상성 불균형의 주 원천.
        opp_cooperated : 상대가 협력했는가.
        opp_defected : 상대가 배신했으나 나도 배신(DD) — 약한 증거(불균형이 작음).
                       이것이 없으면 자기보호로 전환한 뒤 core belief 이 동결된다.
        inferred : 입자필터 posterior means {'alpha','rho','beta','lambda_j'}.
        prediction_surprise : 상대 행동에 대한 예측 서프라이즈 (-log lik), 현저성 가중.
        """
        defect_evidence = betrayal or opp_defected
        if not (defect_evidence or opp_cooperated):
            return
        # 각 원인 축에 대한 '증거' 벡터를 만든다 (관측을 얼마나 잘 설명하는가).
        E_alpha = inferred.get("alpha", 0.0)
        E_beta = inferred.get("beta", 1.0)
        E_lam = inferred.get("lambda_j", 0.5)
        E_rho = inferred.get("rho", 0.0)

        # dispositional 증거: 낮은 α / 낮은 λ_j / (배신맥락에서) 낮은 ρ + 높은 β(의도성)
        disp_alpha = _sigmoid(-E_alpha, scale=1.0)       # α 낮을수록 ↑
        disp_lambda = 1.0 - float(np.clip(E_lam, 0, 1))  # λ_j 낮을수록 ↑
        disp_rho = _sigmoid(-E_rho, scale=1.0)
        precision = float(np.clip(E_beta / 4.0, 0.0, 1.0))  # 의도성(높은 β)

        evidence = np.zeros(len(CAUSE_AXES))
        if defect_evidence:
            evidence[CAUSE_AXES.index("transient")] = (1 - precision)   # 낮은 β → 맥락
            evidence[CAUSE_AXES.index("alpha")] = disp_alpha * precision
            evidence[CAUSE_AXES.index("lambda_j")] = disp_lambda * precision
            evidence[CAUSE_AXES.index("rho")] = disp_rho * precision
            evidence[CAUSE_AXES.index("beta")] = (1 - precision)        # 낮은 β → 잡음
        elif opp_cooperated:
            # 협력은 dispositional 원인(불균형)의 증거를 약화
            evidence[CAUSE_AXES.index("transient")] = 0.2
            evidence[CAUSE_AXES.index("beta")] = precision * 0.3

        # 귀인 개인차: 허용되지 않은 원인 축은 증거를 받을 수 없다.
        # → 잔여 증거가 허용된 축으로 재분배되며, 이것이 과대/과소귀인의 원천이다
        #   (예: intent_only 는 잡음 배신을 전부 의도로 귀인).
        evidence = evidence * self._cause_mask
        s = float(evidence.sum())
        if s <= 1e-9:
            if not defect_evidence:
                return                       # 협력은 허용축에 증거를 주지 못함 → 갱신 없음
            # 배신인데 허용축이 설명하지 못함 → 허용축에 균등 귀인(강제 귀인)
            evidence = self._cause_mask / self._cause_mask.sum()
        else:
            evidence = evidence / s
        # 서프라이즈(현저성)로 학습률 변조 — allostatic 가치가 큰 예측오차일수록 크게 갱신
        salience = float(np.clip(0.5 + 0.5 * np.tanh(prediction_surprise), 0.1, 1.0))
        # DD(내가 이미 방어 중)는 항상성 불균형이 작으므로 약한 증거로만 반영
        weight = 1.0 if (betrayal or opp_cooperated) else 0.35
        lr = self.learning_rate * salience * weight
        self.belief = (1 - lr) * self.belief + lr * evidence
        self.belief = self.belief / self.belief.sum()

    # ---------------------------------------------------------------- 산출
    def dispositional_credence(self) -> float:
        """현재 믿음에서 '내재된 의도(dispositional)' 원인이 차지하는 총 확률질량."""
        idx = [CAUSE_AXES.index(a) for a in DISPOSITIONAL_AXES]
        return float(self.belief[idx].sum())

    def contextual_credence(self) -> float:
        """'맥락적 불확실성(situational)' 원인이 차지하는 총 확률질량."""
        idx = [CAUSE_AXES.index(a) for a in CONTEXTUAL_AXES]
        return float(self.belief[idx].sum())

    def reliability_weights(self) -> dict:
        """
        입자필터 축별 신뢰도 가중치. 원인 믿음 + attribution_target 마스크의 곱.
        (α, ρ, β, λ_j) 순서의 마스크에 대응하는 믿음 질량을 결합한다.
        """
        base = {
            "alpha": self.belief[CAUSE_AXES.index("alpha")],
            "rho": self.belief[CAUSE_AXES.index("rho")],
            "beta": self.belief[CAUSE_AXES.index("beta")],
            "lambda_j": self.belief[CAUSE_AXES.index("lambda_j")],
        }
        mask = {"alpha": self._mask[0], "rho": self._mask[1],
                "beta": self._mask[2], "lambda_j": self._mask[3]}
        # 마스크로 게이팅하되, 최소 바닥값을 두어 완전 정지 방지
        return {k: float(np.clip(mask[k] * (0.3 + base[k]), 0.0, 1.5))
                for k in base}


@dataclass
class LambdaRegulator:
    """
    Core allostatic belief 로부터 공감 파라미터 λ 를 위계적으로 조절.

    vmPFC(즉각) + rmPFC(누적/정교) + dmPFC(mentalizing) 의 상호작용을 두 누출적분기
    (grievance g⁻, trust g⁺)로 구현한다.

    파라미터
    --------
    lam_base : λ 기저(구조적 사전).
    lam_max  : λ 상한(협력 신뢰가 누적될 때 도달 가능한 최대).
    sophisticated : True 면 rmPFC 층 활성 — 추론된 β(의도성)로 방어를 게이팅
                    (정교한 자기보호). False 면 vmPFC 즉각방어만(β 무시).
    kappa : dispositional-attribution 성향(개인차).
    charge/discharge/decay : leaky integrator 계수.
    """

    lam_base: float = 0.4
    lam_max: float = 0.8
    sophisticated: bool = True
    kappa: float = 0.9
    protective_gain: float = 0.9     # g⁻ 충전 이득 (실현된 배신)
    anticipatory_gain: float = 0.25  # g⁻ 예기적 충전 이득 (allostasis: 예측된 배신)
    tonic_weight: float = 0.55       # 실현 배신 구동 중 '지속(tonic)' 성분
    acute_weight: float = 0.45       # 실현 배신 구동 중 '급성(놀람)' 성분
    forgiveness: float = 0.05        # g⁻ 방전(용서)
    trust_gain: float = 0.10         # g⁺ 충전 이득
    trust_decay_on_betrayal: float = 0.25
    decay: float = 0.92              # g⁻ 누출
    trust_decay: float = 0.97        # g⁺ 누출

    grievance: float = field(default=0.0, init=False)   # g⁻
    trust: float = field(default=0.0, init=False)        # g⁺
    lam: float = field(init=False)

    def __post_init__(self):
        self.lam_base = float(np.clip(self.lam_base, 0.0, 1.0))
        self.lam_max = float(np.clip(self.lam_max, self.lam_base, 1.0))
        self.lam = self.lam_base

    def step(self, betrayal: bool, opp_cooperated: bool,
             inferred: dict, pred_coop_prev: float,
             core: CoreAllostaticBeliefState,
             regulate: bool = True, opp_defected: bool = False) -> dict:
        """
        한 라운드 λ 조절.

        betrayal : 예측못한 배신(CD) 발생.
        opp_cooperated : 상대 협력.
        inferred : 입자필터 posterior means (alpha,rho,beta,lambda_j).
        pred_coop_prev : 직전 예측한 상대 협력확률(배신 강도 = 놀람 정도).
        core : CoreAllostaticBeliefState (귀인 성향 제공).
        regulate : False 면 λ 고정(원 Albarracin 재현).
        opp_defected : 상호배신(DD) — 상대가 배신했으나 나도 방어 중.
            **정교형(rmPFC)** 은 자신이 이미 방어 중인 DD 를 '항상성 불균형이 작은'
            약한 증거로 취급하여 grievance 를 충전하지 않는다(대신 예기적 구동 사용).
            **즉각형(vmPFC)** 은 맥락(내 행동, 상대 β)과 무관하게 상대의 배신 자체를
            기질 증거로 귀인하므로 DD 에서도 grievance 를 계속 충전한다.
            → 착취자(ALLD) 상대에서 즉각형은 grievance 가 포화 상태로 유지되어
              조기·지속적 배신(강한 자원 방어)을 보이고, 잡음 TFT 상대에서는
              보복 나선(DD)의 배신까지 기질로 오귀인하여 협력을 복구하지 못한다
              (H5 의 조작화).
        """
        E_alpha = inferred.get("alpha", 0.0)
        E_beta = inferred.get("beta", 1.0)
        E_lam = inferred.get("lambda_j", 0.5)

        # dmPFC(mentalizing): '귀인이 허용된 축'만으로 기질 판단을 구성한다.
        #   → alpha_only 는 α 로만, lambda_only 는 λ_j 로만 상대를 읽는다(개인차).
        E_rho = inferred.get("rho", 0.0)
        disp_from_bias = _sigmoid(-E_alpha, scale=0.5)      # α 낮을수록 → 1
        disp_from_lambda = 1.0 - float(np.clip(E_lam, 0, 1))  # λ_j 낮을수록 → 1
        disp_from_rho = _sigmoid(-E_rho, scale=0.5)          # ρ 낮을수록 → 1
        w_a, w_r, _, w_l = core.theta_update_mask
        w_sum = w_a + w_r + w_l
        if w_sum <= 1e-9:                                    # beta_context: 기질 판단 없음
            disposition = 0.0
        else:
            disposition = float(
                (1.5 * w_a * disp_from_bias + 1.0 * w_l * disp_from_lambda
                 + 0.7 * w_r * disp_from_rho)
                / (1.5 * w_a + 1.0 * w_l + 0.7 * w_r))

        # rmPFC(정교): 추론된 의도성(β)으로 방어를 게이팅.
        precision_conf = float(np.clip(E_beta / 4.0, 0.0, 1.0))
        disp_credence = core.dispositional_credence()
        if self.sophisticated:
            disposition = disposition * precision_conf
        else:
            # vmPFC 즉각형: 상대의 정밀도(의도성)도, 누적된 원인귀인 신뢰도도 무시하고
            # '지금 당한 배신' 자체에만 반응한다 (Yoon/Lee 2018: 즉각 피드백 층).
            disposition = 1.0
            disp_credence = 1.0

        # ---- 배신 구동: tonic(지속) + acute(놀람) ----
        # 순수 반응적(homeostatic) 설계라면 구동 ∝ 놀람(pred_coop_prev)뿐이며,
        # 상대의 배신이 '예측 가능'해지는 순간 구동이 0으로 꺼져 자기보호가 붕괴한다.
        # 항상성 조절(allostasis)은 예기적이어야 하므로 tonic 성분을 남긴다.
        #
        # 즉각형(vmPFC, sophisticated=False)은 '지금 상대가 배신했다'는 신호 자체에
        # 반응한다. 자신이 방어 중(DD)인지 여부(맥락)를 구분하지 못하므로, 상대의
        # 모든 배신(CD ∪ DD)이 grievance 를 충전한다 (타인의 내재된 의도에만 귀인).
        defect_signal = betrayal if self.sophisticated else (betrayal or opp_defected)
        betrayal_drive = (self.tonic_weight
                          + self.acute_weight * float(pred_coop_prev)) if defect_signal else 0.0
        # κ: 개인의 dispositional 귀인 성향
        attributed_disp = self.kappa * disposition * disp_credence * betrayal_drive

        # ---- 예기적(allostatic) 구동 ----
        # 실제 배신이 관측되지 않아도(예: 내가 먼저 방어해 DD 가 된 경우) 상대가
        # 기질적으로 배신할 것이라 '예측'되면 자기보호 상태를 유지한다.
        anticipated_defect = 1.0 - float(np.clip(pred_coop_prev, 0.0, 1.0))
        anticipatory = (self.kappa * disposition * disp_credence * anticipated_defect
                        if self.sophisticated else 0.0)

        # ---- grievance g⁻ (억제) ----
        g = (self.decay * self.grievance
             + self.protective_gain * attributed_disp
             + self.anticipatory_gain * anticipatory)
        if opp_cooperated:
            g -= self.forgiveness
        self.grievance = float(np.clip(g, 0.0, 1.0))

        # ---- trust g⁺ (상향) ----
        # 협력을 cooperative disposition(높은 α/λ_j)에 귀인할 때 신뢰 충전
        coop_credence = _sigmoid(E_alpha, scale=0.5) * float(np.clip(E_lam + 0.5, 0, 1))
        t = self.trust_decay * self.trust
        if opp_cooperated:
            t += self.trust_gain * coop_credence
        if betrayal:
            t -= self.trust_decay_on_betrayal
        self.trust = float(np.clip(t, 0.0, 1.0))

        # ---- λ 산출 ----
        if regulate:
            self.lam = float(np.clip(
                self.lam_base * (1.0 - self.grievance)
                + (self.lam_max - self.lam_base) * self.trust,
                0.0, self.lam_max))
        else:
            self.lam = self.lam_base

        return {
            "lam": self.lam,
            "grievance": self.grievance,
            "trust": self.trust,
            "disposition": disposition,
            "disp_credence": disp_credence,
        }

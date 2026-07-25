"""
ipd.agent
=========

능동추론 IPD 에이전트.

ToMEmpathicAgent : Albarracin et al. (2026) 재현판. gated ToM 입자필터 + 재귀적
                   social EFE + (선택) sophisticated planning. **λ 는 고정**.
                   → 기저 λ 가 높으면 착취자에게 자기보호를 못 하는 원 논문의 한계를 그대로 지님.

AdaptiveAgent    : **HalloReg**. 위에 더해 CoreAllostaticBeliefState 와 LambdaRegulator 를
                   얹어 λ 를 위계적 항상성 조절(내생적)한다. 착취자에게는 λ 하향(자기보호),
                   noisy 상대에게는 맥락(β) 귀인으로 λ 복원.

두 에이전트 모두 `step(observed_state) -> action` 인터페이스(프로토타입 호환)를 갖는다.

[백엔드]
  기본은 numpy 해석적 EFE(빠름, pymdp 와 등가). `use_pymdp=True` 로 pymdp 1.0.x(JAX)
  백엔드를 base EFE(perception 포함)에 사용할 수 있다(등가; 느림).
"""

from __future__ import annotations

from typing import Optional, Dict, List

import numpy as np

from AIF_IPD.core.constants import (
    CC, CD, DC, DD, COOP, DEFECT,
    opponent_action_from_state, my_action_from_state,
)
from AIF_IPD.core.allostasis import CoreAllostaticBeliefState, LambdaRegulator
from AIF_IPD.core.controllability import ControllabilityAttribution
from AIF_IPD.core.logging_utils import get_logger
from .tom import (
    OpponentInversion, ObservationContext, TheoryOfMind, GatedToM,
    RecursiveSocialEFE, OpponentSimulator, SophisticatedPlanner,
)

LOGGER = get_logger("HalloReg.agent")


class ToMEmpathicAgent:
    """
    고정 λ 공감 에이전트 (Albarracin et al. 2026 재현 + 재귀적 확장).
    """

    def __init__(self, lam_base: float = 0.4, beta_self: float = 4.0,
                 beta_other: float = 4.0, n_particles: int = 400,
                 w_epi_self: float = 0.6, w_epi_other: float = 0.3,
                 recursive_depth: int = 2, planning_horizon: int = 1,
                 use_pymdp: bool = False, prior_opp_coop: float = 0.5,
                 likelihood_basis: str = "f",
                 rollout_reciprocity: bool = False,
                 legacy_efe: bool = False,
                 name: str = "ToMEmpathic", seed: int = 0):
        self.name = name
        self.lam_base = float(lam_base)
        self.lam = float(lam_base)          # 고정
        self.planning_horizon = planning_horizon
        self.use_pymdp = use_pymdp
        # [v0.8.0 §7] 우도 기저 토글 — agent → inversion → tom_core 일관 전파.
        self.likelihood_basis = likelihood_basis
        # [v0.7.1 §3] rollout ρ 전파 토글 (horizon≥2 에서만 유효).
        self.rollout_reciprocity = bool(rollout_reciprocity)
        # [v0.9.0 §4~7] EFE 엄밀화 경로. legacy_efe=True 면 v0.8.2 형 복원.
        self.legacy_efe = bool(legacy_efe)
        self.rng = np.random.default_rng(seed + 991)
        self.beta_self = float(beta_self)

        # ToM 구성요소
        self.inversion = OpponentInversion(n_particles=n_particles, seed=seed,
                                           likelihood_basis=likelihood_basis)
        # [v0.9.0 §6.1] 자기-사영 필터 θ̂_self — 내 행동 이력에 동일한 fg 역추론기를
        # 나 자신에게 적용("상대가 나를 이렇게 추론할 것이다"). IG_other·depth-2 를
        # 원리화한다. 별도 seed 로 독립.
        self.self_inversion = OpponentInversion(
            n_particles=max(n_particles // 2, 200), seed=seed + 7,
            likelihood_basis=likelihood_basis)
        self.tom = TheoryOfMind(beta_other=beta_other)
        self.gated = GatedToM(self.tom, self.inversion)
        self.social_efe = RecursiveSocialEFE(
            self.gated, self.inversion, empathy_factor=self.lam_base,
            beta_self=beta_self, w_epi_self=w_epi_self,
            w_epi_other=w_epi_other, recursive_depth=recursive_depth,
            self_inversion=self.self_inversion, legacy_efe=self.legacy_efe)

        # 선택적 pymdp 백엔드
        self._pymdp = None
        if use_pymdp:
            from AIF_IPD.core.pymdp_backend import PymdpEFE, pymdp_available
            if pymdp_available():
                self._pymdp = PymdpEFE(prior_opp_coop=prior_opp_coop)
                self._empirical_prior = self._pymdp.D
            else:
                LOGGER.warning("pymdp 미가용 — numpy 경로로 대체")
                self.use_pymdp = False

        # 상태
        self.my_last = COOP
        self.my_actions: List[int] = []
        # [v0.7.0] 상대 행동 이력. g(상대 자신의 직전 행동) 추출과 반사실 주변화에
        # 사용된다(§1 §7). opp_actions[k] = 라운드 k 에서 관측된 상대 행동.
        self.opp_actions: List[int] = []
        self.pred_coop_prev = 0.5
        # [v0.9.0 §3.1] 단일 r_pred 계약 — 직전 행동선택 시의 pragmatic 기대보수.
        # 다음 라운드 관측 보수와의 차가 RPE 가 된다. R(상호협력)로 초기화.
        from AIF_IPD.core.constants import R as _R0
        self._last_r_pred = float(_R0)
        self._prev_means = self.inversion.posterior_means()

        self.log: Dict[str, list] = {
            "lam": [], "action": [], "pred_coop": [], "reliability": [],
            "E_alpha": [], "E_rho": [], "E_beta": [], "E_lambda_j": [],
            # [v0.8.0 §7.2] fg 기저 축. f 기저에서는 0 으로 채워진다.
            "E_omega": [], "E_eta": [],
            # [v0.8.1] 입자 사후 SD — **실험 내 식별 상태의 사후 진단**용.
            # recovery 배터리는 이상 조건(T=480·강제 균형 점유)의 상한 검증이므로,
            # 본 실험(T=60/240·on-policy 점유)에서 θ̂ 의존 기제(β-게이팅·반사실
            # 귀인)를 해석할 때는 이 SD 로 θ̂ 가 실제로 좁혀졌는지 함께 본다.
            # 예: sd_beta 가 사전(1.8) 근처에 머물면 β̂ 기반 판정은 약식별 상태.
            "sd_alpha": [], "sd_rho": [], "sd_beta": [], "sd_lambda_j": [],
            "sd_omega": [], "sd_eta": [],
            "grievance": [], "trust": [], "belief_update": [],
            "disp_credence": [], "ctx_credence": [],
            "control": [], "w_other": [], "sPE": [], "oPE": [],
            # [v0.7.0 §1.4] 유발분(provoked) — 반사실 귀인 진단용.
            "provoked": [],
            # [v0.9.0 §2~3] 2계층 allostasis 진단.
            #   lam_base   : SelfModel setpoint (§2.2)
            #   delta_lam  : CoreAffect Δλ_drive (§3.5)
            #   rpe        : 기대보상 부적 예측오차 (§3.1)
            #   deficit    : 기대보상 분포 하락폭 [0,1] (§3.2)
            #   q_disp/q_ctx : 원인 귀인 posterior (§3.3, disp_credence/ctx_credence 재활용)
            "lam_base": [], "delta_lam": [], "rpe": [], "deficit": [],
        }

    # ------------------------------------------------------------ 조절 훅
    def _feature_g_val(self, ctx: ObservationContext) -> float:
        """ctx 의 g 신호(±1). f 기저에서는 ω=η=0 이므로 무영향."""
        if ctx is None or ctx.their_last_action is None:
            return 0.0
        return 1.0 - 2.0 * float(ctx.their_last_action)

    def _opp_coop_rate(self) -> float:
        """관측된 상대 협력율 — 반사실 g 주변화의 P(g=+1) (§7.5)."""
        if not self.opp_actions:
            return 0.5
        return float(np.mean([a == COOP for a in self.opp_actions]))

    def _current_lambda(self) -> float:
        """서브클래스에서 λ 조절을 오버라이드."""
        return self.lam_base

    def _regulate(self, observed_state: int, opp_action: int,
                  inferred: dict) -> dict:
        """서브클래스 훅. 기본은 조절 없음."""
        return {"grievance": 0.0, "trust": 0.0,
                "disp_credence": 0.0, "ctx_credence": 0.0, "provoked": 0.0}

    # ------------------------------------------------------------ 한 라운드
    def step(self, observed_state: Optional[int]) -> int:
        # [v0.8.0 §7.2 시제 표] 결정 경로: 예측 대상은 opp_k 이며, 상대는 내 최신
        # 행동 my_{k−1} 과 자신의 최신 행동 opp_{k−1} 에 조건화해 반응한다.
        #   → f = self.my_last (=my_{k−1}),  g = opp_actions[-1] (=opp_{k−1})
        # (갱신 경로는 아래에서 둘 다 한 시점 더 과거를 쓴다.)
        ctx = ObservationContext(
            my_last_action=self.my_last,
            their_last_action=(self.opp_actions[-1] if self.opp_actions
                               else None),
            round_number=len(self.my_actions),
        )
        reg_info = {"grievance": 0.0, "trust": 0.0,
                    "disp_credence": 0.0, "ctx_credence": 0.0, "provoked": 0.0}
        bu = 0.0

        if observed_state is not None:
            opp_action = opponent_action_from_state(observed_state)
            ctx.their_last_action = opp_action
            ctx.joint_outcome = observed_state
            self._prev_means = self.inversion.posterior_means()

            # (1) 입자필터 갱신 (귀인 개인차가 jitter/사전에 반영됨)
            # [v0.6.6 정렬 수정] 관측된 opp_{k−1} 은 (동시행동 게임의 1라운드
            # 지연으로) 내 행동 a_{k−2} 에 반응한 것이다. 종전에는 ctx 의
            # my_last_action=a_{k−1} 을 그대로 써서 우도가 한 라운드 어긋났고
            # (off-by-one), 반응 전략(TFT 등)의 호혜 신호가 파괴되어 ρ̂→0 으로
            # 붕괴했다(무작위 focal 검증: 현행 ρ̂=−1.10/β̂=0.12 vs 정렬
            # ρ̂=+3.21/β̂=4.92). 예측 경로는 결정 시점에 my_last=a_{k−2} 로
            # opp_{k−1} 을 예측하므로 원래 올바름 — 갱신도 같은 a_{k−2} 를 쓰면
            # 예측–갱신 일관성이 복원된다.
            f_upd = (self.my_actions[-2] if len(self.my_actions) >= 2
                     else None)
            # [v0.8.0 §7.2 시제 표] g 도 f 와 **같은 시제 원칙**을 따른다:
            # 관측 opp_{k−1} 에 대한 g 는 opp_{k−2} 다. 이 시점에서 opp_actions 는
            # 아직 opp_{k−1} 을 append 하기 전이므로 opp_actions[-1] == opp_{k−2}.
            # (append 는 _regulate 직후에 수행 — 순서 의존이므로 변경 금지.)
            g_upd = (self.opp_actions[-1] if self.opp_actions else None)
            ctx_upd = ObservationContext(
                my_last_action=f_upd, their_last_action=g_upd,
                joint_outcome=observed_state,
                round_number=len(self.my_actions))
            self.inversion.update(opp_action, ctx_upd)
            inferred = self.inversion.posterior_means()
            bu = self.inversion.belief_update_magnitude(self._prev_means)

            # (2) λ 위계적 조절 (서브클래스)
            reg_info = self._regulate(observed_state, opp_action, inferred)

            # [v0.7.0] 상대 행동 이력 기록. **순서 주의**: (1) 입자필터 갱신이
            # g_upd=opp_{k−2} 를 참조하므로 반드시 갱신 이후에 append 한다.
            # 여기서 append 하면 opp_actions[-1] 이 방금 관측한 opp_{k−1} 이 된다.
            self.opp_actions.append(opp_action)

            # (3) pymdp 상태추론(옵션; perception 층 — 로깅/충실도용)
            if self.use_pymdp and self._pymdp is not None:
                qs = self._pymdp.infer_states(observed_state, self._empirical_prior)
                self._empirical_prior = self._pymdp.update_empirical_prior(
                    self.my_last, qs)

        # (4) 행동 선택
        lam = self._current_lambda()
        if self.planning_horizon > 1:
            action = self._plan_action(ctx, lam)
            # depth-2 예측 협력확률과 선택행동의 pragmatic 기대보수(r_pred)를
            # social_efe 로 일관 계산(§3.1 단일 r_pred 계약).
            res = self.social_efe.compute(ctx, self.my_last, lam=lam)
            res_info = res.info
            q_coop = res.info["pc"]
            r_pred = float(res.info["pragmatic_self"][action])
        else:
            action, res = self.social_efe.select_action(
                ctx, self.my_last, lam=lam, rng=self.rng)
            res_info = res.info
            q_coop = res.info["pc"]
            r_pred = float(res.info.get("r_pred", res.info["pragmatic_self"][action]))

        # [v0.9.0 §6.1] 자기-사영 필터 갱신 — 내 선택행동을 관측으로, 내가 반응한
        # 상대 직전행동을 호혜신호 f 로, 내 직전행동을 g 로. (상대가 나를 추론하는
        # 필터의 시제와 정합.)
        self_ctx = ObservationContext(
            my_last_action=(self.opp_actions[-1] if self.opp_actions else None),
            their_last_action=self.my_last,
            round_number=len(self.my_actions))
        self.self_inversion.update(action, self_ctx)

        # (5) 상태 갱신 + 로깅
        self.my_last = action
        self.my_actions.append(action)
        self.pred_coop_prev = q_coop
        self._last_r_pred = r_pred
        self.social_efe.my_coop_rate = float(np.mean(self.my_actions))
        self.tom.update_my_policy_belief(self.social_efe.my_coop_rate)
        self.inversion.my_cooperation_rate = self.social_efe.my_coop_rate

        m = self.inversion.posterior_means()
        self.log["lam"].append(lam)
        self.log["action"].append(action)
        self.log["pred_coop"].append(q_coop)
        self.log["reliability"].append(self.inversion.reliability())
        self.log["E_alpha"].append(m["alpha"])
        self.log["E_rho"].append(m["rho"])
        self.log["E_beta"].append(m["beta"])
        self.log["E_lambda_j"].append(m["lambda_j"])
        self.log["E_omega"].append(m.get("omega", 0.0))
        self.log["E_eta"].append(m.get("eta", 0.0))
        sd = self.inversion.posterior_stds()
        self.log["sd_alpha"].append(sd["alpha"])
        self.log["sd_rho"].append(sd["rho"])
        self.log["sd_beta"].append(sd["beta"])
        self.log["sd_lambda_j"].append(sd["lambda_j"])
        self.log["sd_omega"].append(sd.get("omega", 0.0))
        self.log["sd_eta"].append(sd.get("eta", 0.0))
        self.log["grievance"].append(reg_info.get("grievance", 0.0))
        self.log["trust"].append(reg_info.get("trust", 0.0))
        self.log["belief_update"].append(bu)
        self.log["disp_credence"].append(reg_info.get("disp_credence", 0.0))
        self.log["ctx_credence"].append(reg_info.get("ctx_credence", 0.0))
        self.log["control"].append(reg_info.get("control", 1.0))
        self.log["w_other"].append(reg_info.get("w_other", 1.0))
        self.log["sPE"].append(reg_info.get("sPE", 0.0))
        self.log["oPE"].append(reg_info.get("oPE", 0.0))
        self.log["provoked"].append(reg_info.get("provoked", 0.0))
        # [v0.9.0] 2계층 allostasis 진단
        self.log["lam_base"].append(reg_info.get("lam_base", self.lam_base))
        self.log["delta_lam"].append(reg_info.get("delta_lam", 0.0))
        self.log["rpe"].append(reg_info.get("rpe", 0.0))
        self.log["deficit"].append(reg_info.get("deficit", 0.0))
        return action

    def _plan_action(self, ctx: ObservationContext, lam: float) -> int:
        """sophisticated planning horizon 으로 행동 선택."""
        # [v0.7.1 §3] rollout_reciprocity=True 면 시뮬레이터가 step>0 에서 ρ̂ 로
        # 내 가상행동에 조건화된 예측을 낸다(horizon=1 이면 무영향).
        sim = OpponentSimulator(self.tom, self.gated, ctx,
                               rollout_reciprocity=self.rollout_reciprocity,
                               inversion=self.inversion)
        planner = SophisticatedPlanner(
            sim, empathy_factor=lam, horizon=self.planning_horizon,
            beta_self=self.social_efe.beta_self,
            social_efe=self.social_efe, base_ctx=ctx)   # [v0.9.0 §5.3] IG-in-rollout
        q_action, _, _ = planner.plan(lam=lam)
        return COOP if self.rng.random() < q_action[COOP] else DEFECT


class AdaptiveAgent(ToMEmpathicAgent):
    """
    **HalloReg 적응 에이전트.**

    ToMEmpathicAgent 에 CoreAllostaticBeliefState + LambdaRegulator 를 추가하여
    공감 λ 를 위계적 항상성 조절한다.

    추가 파라미터
    -------------
    kappa : dispositional-attribution 성향(개인차; vmPFC↔dmPFC 구배).
    sophisticated : rmPFC 정교한 자기보호(β 의도성 게이팅) 여부.
    attribution_target : 예측오차를 귀인할 θ 축 집합
        {'all','alpha_only','lambda_only','rho_only','beta_context'}.
    regulate_lambda : False 면 λ 고정(원 논문 재현).
    lam_max : 신뢰 누적 시 도달 가능한 λ 상한.
    prior_reliability : 사전 학습된 core allostatic belief(축별 신뢰도) (선택).
    """

    def __init__(self, lam_base: float = 0.4, lam_max: float = 0.8,
                 kappa: float = 0.9, sophisticated: bool = True,
                 attribution_target: str = "all", regulate_lambda: bool = True,
                 forgiveness: float = 0.05, prior_reliability: Optional[dict] = None,
                 dd_charges_grievance: Optional[bool] = None,
                 grievance_decay: Optional[float] = None,
                 beta_clamp: bool = False,
                 controllability: bool = False,
                 controllability_kwargs: Optional[dict] = None,
                 disposition_mode: str = "legacy",
                 cf_g_handling: str = "marginalize",
                 cf_attr_gate_dedup: bool = True,
                 allostasis_legacy: bool = False,
                 selfmodel_kwargs: Optional[dict] = None,
                 identity_memory: bool = True,
                 core_affect_kwargs: Optional[dict] = None,
                 name: str = "Adaptive", **kwargs):
        super().__init__(lam_base=lam_base, name=name, **kwargs)
        self.regulate_lambda = regulate_lambda
        self.disposition_mode = disposition_mode
        # [v0.9.0 §9.1] 조절 계층 선택. 기본 v0.9.0(SelfModel+CoreAffect); legacy 는
        # A/B 용 토글로만 보존.
        self.allostasis_legacy = bool(allostasis_legacy)

        # 자기/타인 통제권 귀인 (Spiering 2025; 선택적 — 기본 off).
        self.controllability = (
            ControllabilityAttribution(**(controllability_kwargs or {}))
            if controllability else None)

        if beta_clamp:                       # H2 절제 대조: β 축 동결(필터 수준)
            self.inversion.clamp_beta()
            self.self_inversion.clamp_beta()

        if self.allostasis_legacy:
            # ---- 구 조절 계층 (leaky 이중 적분기) ----
            self.core = CoreAllostaticBeliefState(
                kappa=kappa, attribution_target=attribution_target,
                prior_reliability=prior_reliability)
            self.inversion.set_reliability(self.core.reliability_weights(),
                                           reinit=True)
            reg_kw = dict(lam_base=lam_base, lam_max=lam_max,
                          sophisticated=sophisticated, kappa=kappa,
                          forgiveness=forgiveness,
                          dd_charges=dd_charges_grievance,
                          disposition_mode=disposition_mode,
                          cf_g_handling=cf_g_handling,
                          cf_attr_gate_dedup=cf_attr_gate_dedup)
            if grievance_decay is not None:
                reg_kw["decay"] = float(grievance_decay)
            self.regulator = LambdaRegulator(**reg_kw)
            self.self_model = None
            self.core_affect = None
        else:
            # ---- v0.9.0 2계층 allostasis ----
            from AIF_IPD.core.allostasis_v9 import SelfModel, CoreAffect
            from AIF_IPD.core.constants import R as _R, T as _T, S as _S, P as _P
            self.self_model = SelfModel(**(selfmodel_kwargs or {}))
            self.core_affect = CoreAffect(
                self.self_model, payoffs=(_R, _T, _S, _P),
                lambda_max=lam_max, regulate=regulate_lambda,
                **(core_affect_kwargs or {}))
            # setpoint 에서 출발 (lam_base 는 이제 학습됨, §9.3).
            self.lam_base = self.self_model.lambda_base()   # 초기: 상대 미상 → self
            self.core = None
            self.regulator = None

        self.lam = self.lam_base
        # [§3.6 재정의] identity 관측 채널: 환경(run_dyad)이 begin_partner 로
        # 상대 identity 를 알려준다. 미통지 시 기본 1 (다이애드 단일 상대).
        self.identity_memory = bool(identity_memory)
        self._partner_identity: int = 1

    def begin_partner(self, ident: int) -> None:
        """
        [§3.6 재정의] 상대 identity 관측 통지(환경 호출). 저장된 boundary-내
        identity 와의 **재조우**면 belief q(z) 를 저장된 잠재 z 로 즉시 초기화
        (조기 보정 실효화). 신규 identity 는 boundary 밖에서 관계 형성 시작.
        """
        self._partner_identity = int(ident)
        if self.identity_memory and self.core_affect is not None:
            self.self_model.observe_identity(int(ident))
            self.core_affect.reset_for_partner(int(ident))

    def _current_lambda(self) -> float:
        return self.lam

    # ------------------------------------------------------------ 조절
    def _regulate(self, observed_state: int, opp_action: int,
                  inferred: dict) -> dict:
        if self.allostasis_legacy:
            return self._regulate_legacy(observed_state, opp_action, inferred)
        return self._regulate_v9(observed_state, opp_action, inferred)

    # ---- [v0.9.0] 2계층 조절 ----
    def _regulate_v9(self, observed_state: int, opp_action: int,
                     inferred: dict) -> dict:
        from AIF_IPD.core.constants import (
            PAYOFF_SELF, R as _R, T as _T, S as _S, P as _P)
        # 가변 페이오프 환경 정합: 생성모형 모수를 현재 보수로 갱신(§11.1, VP).
        self.core_affect.set_payoffs((_R, _T, _S, _P))

        # RPE = 실현보수 − 직전 선택의 pragmatic 기대보수(§3.1 단일 r_pred 계약).
        r_obs = float(PAYOFF_SELF[observed_state])
        beta_sd = float(self.inversion.posterior_stds().get("beta", 0.0))
        # boundary 조기보정용 dispositional 배신확률 근사(legacy §3.6 경로용).
        disp_hint = float(1.0 - np.clip(inferred.get("lambda_j", 0.5), 0, 1))

        # [v0.11.0 §θ-잔차] 직전 행동(my_last)이 만든 관측보수를 θ 로 설명한 잔차.
        #   ctx 는 필터갱신과 동일 시제(f=my_{t-2}, g=opp_{t-2}) — 상대는 1라운드
        #   지연으로 내 직전행동에 반응하므로.
        # [v0.11.2] Z_i(s,a) 의 **결정 시점 상태** s_dec: a_{t−1} 을 고를 때 관측하고
        #   있던 공동상태 = joint(a_{t−2}, opp_{t−2}) = 2·my_{t−2} + opp_{t−2}.
        #   (결과 상태 s_{t−1} 을 쓰면 보수 자신이 조건이 되어 순환이 된다.)
        #   이 시점 opp_actions 는 아직 opp_{t−1} 미append 이므로 [-1]=opp_{t−2}.
        #   이력 부족(초기 1~2R)이면 None → CC(0) 초기상태 관례.
        if len(self.my_actions) >= 2 and self.opp_actions:
            s_decision = int(2 * int(self.my_actions[-2]) + int(self.opp_actions[-1]))
        else:
            s_decision = None

        theta_rmean = theta_epi = None
        theta_coop_now = None
        try:
            from AIF_IPD.ipd.tom.inversion import ObservationContext as _OC
            f_prev = (self.my_actions[-2] if len(self.my_actions) >= 2
                      else self.my_last)
            g_prev = (self.opp_actions[-1] if self.opp_actions else None)
            ctx_prev = _OC(my_last_action=f_prev, their_last_action=g_prev,
                           round_number=len(self.my_actions))
            E_th, epi, _ale = self.inversion.theta_reward_moments(
                self.my_last, ctx_prev, PAYOFF_SELF)
            theta_rmean, theta_epi = E_th, epi
            # θ-예측 상대 협력확률 — identity θ 기록·λ_base 갱신용.
            g_now = (self.opp_actions[-1] if self.opp_actions else None)
            theta_coop_now = float(self.inversion.predict_coop(
                self.my_last, g_now if self.inversion._fg else None))
        except Exception:
            theta_rmean = theta_epi = None
            theta_coop_now = None

        # [§3.6 재정의] identity 경로: 관측된 상대 identity 로 prior·관계형성.
        pid = (self._partner_identity if getattr(self, "identity_memory", True)
               else None)
        out = self.core_affect.step(
            r_pred=self._last_r_pred, r_obs=r_obs, inferred=inferred,
            beta_sd=beta_sd, identity_disp=disp_hint,
            partner_identity=pid,
            opp_defected=(opp_action == DEFECT),
            theta_reward_mean=theta_rmean,
            theta_epistemic_std=theta_epi,
            theta_coop=theta_coop_now,
            theta_lambda_j=float(np.clip(inferred.get("lambda_j", 0.5), 0.0, 1.0)),
            state_prev=s_decision, action_prev=int(self.my_last))
        self.lam = out["lam"]
        # social_efe 의 empathy_factor 도 동기(로깅/재귀예측 일관).
        self.social_efe.lam = self.lam

        # 통제권 귀인(선택) — 로깅 호환용.
        ctrl_info = {"control": 1.0, "w_other": 1.0, "sPE": 0.0, "oPE": 0.0}
        if self.controllability is not None:
            surprise = -np.log(max(
                self.pred_coop_prev if opp_action == COOP
                else (1 - self.pred_coop_prev), 1e-6))
            emitted = my_action_from_state(observed_state)
            ci = self.controllability.update(
                intended_action=self.my_last, emitted_action=emitted,
                observed_state=observed_state, total_pe=surprise)
            ctrl_info.update({k: ci[k] for k in ("control", "w_other", "sPE", "oPE")})

        q_disp = out["q_dispositional"]
        q_ctx = out["q_contextual"]
        deficit = out["deficit"]
        # 구 지표(grievance/trust)로의 사상 — 하위호환 로깅/지표.
        #   grievance ← dispositional 자기보호 압력 (q_disp·deficit)
        #   trust     ← contextual 유지/회복 (q_ctx·(1−deficit))
        return {
            "grievance": float(np.clip(q_disp * deficit, 0.0, 1.0)),
            "trust": float(np.clip(q_ctx * (1.0 - deficit), 0.0, 1.0)),
            "provoked": 0.0,                       # §9.6 재해석: 독립 코드 폐기
            "disp_credence": q_disp,
            "ctx_credence": q_ctx,
            "lam_base": out["lam_base"],
            "delta_lam": out["delta_lambda"],
            "rpe": out["rpe"],
            "deficit": deficit,
            "control": ctrl_info["control"],
            "w_other": ctrl_info["w_other"],
            "sPE": ctrl_info["sPE"],
            "oPE": ctrl_info["oPE"],
        }

    # ---- [legacy] 구 조절 계층 (A/B 토글 보존) ----
    def _regulate_legacy(self, observed_state: int, opp_action: int,
                         inferred: dict) -> dict:
        betrayal = (observed_state == CD)
        opp_cooperated = observed_state in (CC, DC)
        opp_defected = (observed_state == DD)
        surprise = -np.log(max(
            self.pred_coop_prev if opp_action == COOP else (1 - self.pred_coop_prev),
            1e-6))
        attr_gate = 1.0
        ctrl_info = {"control": 1.0, "w_other": 1.0, "sPE": 0.0, "oPE": 0.0,
                     "self_caused": 0.0}
        if self.controllability is not None:
            emitted = my_action_from_state(observed_state)
            ctrl_info = self.controllability.update(
                intended_action=self.my_last, emitted_action=emitted,
                observed_state=observed_state, total_pe=surprise)
            attr_gate = ctrl_info["w_other"]
        self.core.update(betrayal, opp_cooperated, inferred, surprise,
                         opp_defected=opp_defected, attr_gate=attr_gate)
        self.inversion.set_reliability(self.core.reliability_weights())
        out = self.regulator.step(
            betrayal, opp_cooperated, inferred, self.pred_coop_prev,
            self.core, regulate=self.regulate_lambda, opp_defected=opp_defected,
            attr_gate=attr_gate,
            my_coop_rate=self.inversion.my_cooperation_rate,
            g_prob_coop=self._opp_coop_rate())
        self.lam = out["lam"]
        self.social_efe.lam = self.lam
        return {
            "grievance": out["grievance"],
            "trust": out["trust"],
            "provoked": out.get("provoked", 0.0),
            "disp_credence": self.core.dispositional_credence(),
            "ctx_credence": self.core.contextual_credence(),
            "lam_base": self.lam_base,
            "delta_lam": out["lam"] - self.lam_base,
            "rpe": 0.0,
            "deficit": out["grievance"],
            "control": ctrl_info["control"],
            "w_other": ctrl_info["w_other"],
            "sPE": ctrl_info["sPE"],
            "oPE": ctrl_info["oPE"],
        }

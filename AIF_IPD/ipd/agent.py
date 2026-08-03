"""
ipd.agent
=========

능동추론 IPD 에이전트.

  EmpathicAgent : Albarracin et al. (2026) 재현판. 게이팅된 ToM 입자필터 +
                  재귀적 social EFE. **λ 는 외생 고정.**
                  → 기저 λ 가 높으면 착취자에게 자기보호를 못 하는 원 논문의
                    한계를 그대로 지니며, 본 연구의 대조군이자 λ 복원 실험
                    (H1A)의 '알려진 λ 를 가진 상대' 로도 쓰인다.

  HalloRegAgent : **본 연구의 제안 모형.** EmpathicAgent 에 위계적 이상성 조절
                  (SelfModel → CoreAffect / OpponentInversion → Empathy)을 얹어
                  λ 를 내생적으로 산출한다.

[HalloRegAgent 의 한 라운드 정보 흐름]

    SelfModel ──(id, theta)────────────▶ OpponentInversion ──┐
        │                                                     │ λ_ctx = f(α̂, λ̂_j)
        ├──(id, expected reward dist)──▶ CoreAffect ──────────┤ λ_aff = V × A
        │                                     ▲               │
        └──(기저 기대보상 분포 = 설정점)──────┘               ▼
                                                          Empathy
                                                              │ λ_t = λ_{t−1}
                                                              │      + (1−w)λ_aff
                                                              │      + w·λ_ctx
                                                              ▼
                                                    RecursiveSocialEFE → 행동

    갱신된 (theta, dist) 와 (expected reward dist) 는 다시 SelfModel 로 commit 된다.
    즉 SelfModel 은 추론하지 않고 **기억만 한다**.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from AIF_IPD.core.constants import (
    COOP, DEFECT, PAYOFF_SELF, opponent_action_from_state,
)
from AIF_IPD.core.core_affect import CoreAffect
from AIF_IPD.core.empathy import Empathy
from AIF_IPD.core.self_model import SelfModel
from .tom import (
    GatedToM, ObservationContext, OpponentInversion, OpponentSimulator,
    RecursiveSocialEFE, SophisticatedPlanner, TheoryOfMind,
)


class EmpathicAgent:
    """
    고정 λ 공감 에이전트 (원 논문 재현 + 재귀적 확장).

    Parameters
    ----------
    lam : float
        고정 공감 가중치 λ ∈ [0, 1].
    beta_self, beta_other : float
        내/상대 행동선택 정밀도.
    n_particles : int
        상대 θ̂ 입자필터의 입자 수.
    planning_horizon : int
        1 이면 myopic(단일스텝 EFE), ≥2 면 SophisticatedPlanner 사용.
    recursive_depth : int
        2 면 depth-2 조망수용(자기-사영 필터 주입) 활성.
    self_projection : bool
        자기-사영 필터 θ̂_self 사용 여부. False 면 IG_other = 0, depth-2 비활성.
    """

    #: 로깅 채널 목록 — 하위 클래스가 확장한다.
    LOG_KEYS = ("action", "lam", "pred_coop", "reliability",
                "E_alpha", "E_rho", "E_omega", "E_eta", "E_beta", "E_lambda_j",
                "SD_alpha", "SD_rho", "SD_omega", "SD_eta", "SD_beta",
                "SD_lambda_j", "belief_update", "self_pred_coop",
                # 자기-사영 필터 θ̂_self 의 사후 평균 — "상대가 나를 이렇게
                # 추론할 것이다". H6 의 사영 정합성 검증에 쓴다.
                "P_alpha", "P_rho", "P_omega", "P_eta", "P_beta",
                "P_lambda_j")

    def __init__(self, lam: float = 0.4, beta_self: float = 4.0,
                 beta_other: float = 4.0, n_particles: int = 400,
                 planning_horizon: int = 2, recursive_depth: int = 2,
                 self_projection: bool = True, jitter_scale: float = 1.0,
                 lam_schedule: Optional[list] = None,
                 name: str = "Empathic", seed: int = 0):
        self.name = name
        self.lam = float(lam)
        self.planning_horizon = int(planning_horizon)
        # [H1A] λ 전환 스케줄 [(round, lam), ...]. 알려진 λ 가 세션 중간에 바뀌는
        # 상대를 만들어, 추론기가 λ̂_j 의 **변화**를 추적하는지 검증한다.
        self.lam_schedule = sorted(lam_schedule or [], key=lambda x: x[0])
        self.rng = np.random.default_rng(seed + 991)

        # ---- 조망수용 계층 ----
        self.inversion = OpponentInversion(n_particles=n_particles,
                                           jitter_scale=jitter_scale,
                                           seed=seed)
        # 자기-사영 필터: 내 행동 이력에 **동일한 역추론기**를 나 자신에게 적용해
        # "상대가 나를 이렇게 추론할 것이다" 를 구성한다. 입자 수는 절반으로 줄여
        # 비용을 억제(자기 행동은 내가 확정적으로 알므로 불확실성이 작다).
        self.self_inversion = (
            OpponentInversion(n_particles=max(n_particles // 2, 200),
                              jitter_scale=jitter_scale, seed=seed + 7)
            if self_projection else None)

        self.tom = TheoryOfMind(beta_other=beta_other)
        self.gated = GatedToM(self.tom, self.inversion)
        self.social_efe = RecursiveSocialEFE(
            self.gated, self.inversion, empathy_factor=self.lam,
            beta_self=beta_self, recursive_depth=recursive_depth,
            self_inversion=self.self_inversion)

        # ---- 상호작용 상태 ----
        self.my_last = COOP                 # 내 직전 행동 (초기 관례: 협력)
        self.my_actions: List[int] = []
        self.opp_actions: List[int] = []    # 관측한 상대 행동 이력
        self._prev_means = self.inversion.posterior_means()

        self.log: Dict[str, list] = {k: [] for k in self.LOG_KEYS}

    # ============================================================ 훅
    def begin_partner(self, identity: int) -> None:
        """
        상대 identity 통지(환경이 호출). 고정 λ 에이전트는 기억이 없으므로 무시.
        """
        return None

    def note_emitted(self, emitted: int) -> None:
        """
        **환경 계층 실행오류 정합** (환경이 호출).

        run_dyad 는 에이전트가 선택한 행동을 확률적으로 뒤집을 수 있다. 이때
        상대는 **실제 방출된 행동**에 반응하므로, 에이전트 내부의 `my_last`
        (다음 라운드의 호혜신호 f 의 원천)도 방출된 행동으로 맞춰야 한다.
        그러지 않으면 우도의 f 가 상대가 실제로 본 것과 어긋나(off-by-one 이
        아니라 값 자체의 불일치) 호혜성 추정 ρ̂ 이 체계적으로 붕괴한다.

        자기-사영 필터도 마찬가지다 — 상대가 관측한 것은 방출된 행동이므로,
        "상대가 나를 어떻게 볼까" 를 계산하는 필터는 방출 행동을 봐야 한다.
        다만 이미 update() 를 마쳤으므로, 여기서는 이후 라운드의 맥락에 쓰이는
        상태만 정정한다(과거 갱신의 소급 정정은 입자필터에서 불가능하며,
        오류율이 낮으면 그 영향은 무시할 수 있다).
        """
        e = int(emitted)
        if self.my_actions:
            self.my_actions[-1] = e
        self.my_last = e

    def current_lambda(self) -> float:
        """
        행동선택에 쓸 λ.
        기본은 상수이나, lam_schedule 이 있으면 현재 라운드에 해당하는 값으로
        전환한다(마지막으로 도달한 항목이 유효).
        """
        if self.lam_schedule:
            t = len(self.my_actions)
            for r, v in self.lam_schedule:
                if t >= r:
                    self.lam = float(v)
            self.social_efe.lam = self.lam
        return self.lam

    def _regulate(self, observed_state: int, opp_action: int,
                  inferred: dict) -> dict:
        """λ 조절 훅. 기본은 조절 없음 — HalloRegAgent 가 오버라이드."""
        return {}

    # ============================================================ 한 라운드
    def step(self, observed_state: Optional[int]) -> int:
        """
        t−1 의 joint outcome 을 관측하고 t 의 행동을 반환한다.
        observed_state=None 이면 첫 라운드(관측 없음).
        """
        # ---------- (0) 결정용 맥락 ----------
        # 예측 대상은 opp_k 이며, 상대는 최신 정보(my_{k−1}, opp_{k−1})에 반응한다.
        ctx = ObservationContext(
            my_last_action=self.my_last,
            their_last_action=(self.opp_actions[-1] if self.opp_actions else None),
            round_number=len(self.my_actions))

        reg = {}
        belief_update = 0.0

        if observed_state is not None:
            opp_action = opponent_action_from_state(observed_state)
            ctx.their_last_action = opp_action
            ctx.joint_outcome = observed_state
            self._prev_means = self.inversion.posterior_means()

            # ---------- (1) 상대 θ̂ 입자필터 갱신 ----------
            # 갱신용 시제: f = my_{k−2}, g = opp_{k−2}.
            # opp_actions 는 아직 opp_{k−1} 을 append 하기 **전**이므로
            # opp_actions[-1] 이 곧 opp_{k−2} 이다. (append 는 아래에서 수행 —
            # 순서 의존이므로 절대 바꾸지 말 것.)
            f_upd = self.my_actions[-2] if len(self.my_actions) >= 2 else None
            g_upd = self.opp_actions[-1] if self.opp_actions else None
            ctx_upd = ObservationContext(
                my_last_action=f_upd, their_last_action=g_upd,
                joint_outcome=observed_state, round_number=len(self.my_actions))
            self.inversion.update(opp_action, ctx_upd)

            inferred = self.inversion.posterior_means()
            belief_update = self.inversion.belief_update_magnitude(self._prev_means)

            # ---------- (2) λ 조절 (하위 클래스) ----------
            reg = self._regulate(observed_state, opp_action, inferred)

            # ---------- (3) 상대 행동 이력 기록 ----------
            self.opp_actions.append(opp_action)

        # ---------- (4) 행동 선택 ----------
        lam = self.current_lambda()
        res = self.social_efe.compute(ctx, lam=lam)
        if self.planning_horizon > 1:
            sim = OpponentSimulator(self.tom, self.gated, ctx, self.inversion)
            planner = SophisticatedPlanner(
                sim, self.social_efe, base_ctx=ctx, empathy_factor=lam,
                horizon=self.planning_horizon,
                beta_self=self.social_efe.beta_self)
            q_action, _, _ = planner.plan(lam=lam)
            action = COOP if self.rng.random() < q_action[COOP] else DEFECT
        else:
            from AIF_IPD.core.generative import softmax
            q_pi = softmax(-res.G_social,
                           temperature=1.0 / self.social_efe.beta_self)
            action = COOP if self.rng.random() < q_pi[COOP] else DEFECT

        # ---------- (5) 자기-사영 필터 갱신 ----------
        # 상대의 관점에서: 내 행동이 '관측', 상대 자신의 직전 행동이 호혜자극 f,
        # 내 직전 행동이 g 가 된다 (focal 필터와 f/g 역할이 정확히 뒤바뀜).
        self_pc = np.nan
        if self.self_inversion is not None:
            self_ctx = ObservationContext(
                my_last_action=(self.opp_actions[-1] if self.opp_actions else None),
                their_last_action=self.my_last,
                round_number=len(self.my_actions))
            # [H6 projection] **갱신 이전**에 예측 협력확률을 기록한다. 갱신 후에
            # 재면 방금 관측한 행동이 이미 반영되어 있어 예측이 아니라 후험적
            # 적합이 되어버린다(정보 누출).
            f_me = (0.0 if not self.opp_actions
                    else 1.0 - 2.0 * float(self.opp_actions[-1]))
            g_me = 1.0 - 2.0 * float(self.my_last)
            self_pc = float(self.self_inversion.predict_coop(f_me, g_me))
            self.self_inversion.update(action, self_ctx)

        # ---------- (6) 상태 갱신 ----------
        self.my_last = action
        self.my_actions.append(action)
        my_rate = float(np.mean(self.my_actions))
        self.social_efe.my_coop_rate = my_rate
        self.tom.update_my_policy_belief(my_rate)
        self.inversion.my_cooperation_rate = my_rate
        if self.self_inversion is not None:
            # 자기-사영 필터에서 '내 협력률' 자리는 상대의 협력률이 차지한다.
            self.self_inversion.my_cooperation_rate = (
                float(np.mean([a == COOP for a in self.opp_actions]))
                if self.opp_actions else 0.5)

        # ---------- (7) 로깅 ----------
        self._record(action, lam, res, belief_update, reg)
        self.log["self_pred_coop"].append(float(self_pc))
        return action

    # ============================================================ 로깅
    def _record(self, action: int, lam: float, res, belief_update: float,
                reg: dict) -> None:
        m = self.inversion.posterior_means()
        sd = self.inversion.posterior_stds()
        self.log["action"].append(int(action))
        self.log["lam"].append(float(lam))
        self.log["pred_coop"].append(float(res.info["pc"]))
        self.log["reliability"].append(float(res.info["reliability"]))
        for ax in ("alpha", "rho", "omega", "eta", "beta", "lambda_j"):
            self.log[f"E_{ax}"].append(float(m[ax]))
            self.log[f"SD_{ax}"].append(float(sd[ax]))
        self.log["belief_update"].append(float(belief_update))
        # 자기-사영 필터의 사후 평균. 필터가 없으면(self_projection=False) NaN.
        pm = (self.self_inversion.posterior_means()
              if self.self_inversion is not None else None)
        for ax in ("alpha", "rho", "omega", "eta", "beta", "lambda_j"):
            self.log[f"P_{ax}"].append(
                float(pm[ax]) if pm is not None else float("nan"))


class HalloRegAgent(EmpathicAgent):
    """
    **HalloReg — 위계적 이상성 조절 에이전트.**

    EmpathicAgent 에 SelfModel / CoreAffect / Empathy 를 결합해 λ 를 내생 조절한다.

    Parameters
    ----------
    w_cd : float
        Empathy 의 정서–맥락 채널 가중.
    lam_gain : float
        λ 적분 이득 η.
    lam_min, lam_max : float
        λ 의 허용 구간.
    social_lr : float
        SelfModel 사회 기저분포의 학습률(설정점 이동 속도).
    kl_scale : float
        CoreAffect arousal 포화 상수 κ.
    identity_memory : bool
        identity 기반 기억 사용 여부. False 면 매 상대를 신규로 취급(절제 대조).
    regulate : bool
        False 면 λ 를 설정점에 고정 — 조절 기제 자체의 절제 대조.
    """

    LOG_KEYS = EmpathicAgent.LOG_KEYS + (
        "valence", "arousal", "lambda_aff", "lambda_ctx", "rpe", "kl",
        "expected_reward", "baseline_reward")

    def __init__(self, w_cd: float = 0.5, lam_gain: float = 0.05,
                 lam_min: float = 0.0, lam_max: float = 1.0,
                 social_lr: float = 0.06, kl_scale: float = 0.05,
                 lam_floor: float = 0.10, lam_ceil: float = 0.70,
                 identity_memory: bool = True, regulate: bool = True,
                 alpha_scale: float = 2.0,
                 name: str = "HalloReg", **kwargs):
        # 초기 λ 는 SelfModel 설정점이 결정하므로, 부모의 lam 인자는 임시값이다.
        kwargs.pop("lam", None)
        super().__init__(lam=0.4, name=name, **kwargs)

        self.identity_memory = bool(identity_memory)
        self.regulate = bool(regulate)

        # ---- 위계 구성 ----
        self.self_model = SelfModel(social_lr=social_lr,
                                    lam_floor=lam_floor, lam_ceil=lam_ceil)
        self.core_affect = CoreAffect(self.self_model, payoffs=PAYOFF_SELF,
                                      kl_scale=kl_scale)
        lam0 = self.self_model.lambda_setpoint(PAYOFF_SELF)
        self.empathy = Empathy(lam_init=lam0, w_cd=w_cd, gain=lam_gain,
                               lam_min=lam_min, lam_max=lam_max,
                               alpha_scale=alpha_scale)

        self.lam = lam0
        self.social_efe.lam = lam0
        self._identity: Optional[int] = None

    # ============================================================ 상대 전환
    def begin_partner(self, identity: int) -> None:
        """
        환경이 상대 identity 를 통지한다.

        · SelfModel 에 identity 관측을 알리고,
        · 그 identity 의 θ 사전을 OpponentInversion 에 주입하며(재조우면 과거
          관계 지점에서 출발),
        · 기대보상 분포 사전을 CoreAffect 에 주입하고,
        · λ 를 **현재 갱신된 설정점**으로 초기화한다.

        identity_memory=False 면 매번 신규 상대로 취급한다(절제 대조).
        """
        pid = int(identity) if self.identity_memory else None
        self._identity = pid
        if pid is not None:
            self.self_model.observe_identity(pid)
        self.inversion.set_prior(self.self_model.theta_prior(pid), reinit=True)
        self.core_affect.begin_partner(pid)
        lam0 = self.self_model.lambda_setpoint(self.core_affect.payoffs)
        self.lam = self.empathy.reset(lam0)
        self.social_efe.lam = self.lam

    def current_lambda(self) -> float:
        return self.lam

    # ============================================================ λ 조절
    def _regulate(self, observed_state: int, opp_action: int,
                  inferred: dict) -> dict:
        """
        한 라운드의 위계적 λ 조절.

        1. 가변 보수 환경 정합 — CoreAffect 에 현재 보수 벡터를 알린다.
        2. CoreAffect.step(관측) → valence, arousal, λ_aff.
           (내부에서 identity 별 보상분포를 갱신하고 SelfModel 에 commit.)
        3. OpponentInversion 사후 (theta, dist) 를 SelfModel 에 commit.
        4. λ_ctx = Empathy.contextual(α̂, λ̂_j).
        5. λ_t = Empathy.step(λ_aff, λ_ctx).
        """
        # --- 1. 현재 보수 반영 (비정상 보수 스케줄 대응) ---
        self.core_affect.set_payoffs(PAYOFF_SELF)

        # --- 2. 핵심정서 구성 ---
        affect = self.core_affect.step(observed_state)

        # --- 3. 상대 특성 기억 commit ---
        self.self_model.commit_theta(self._identity, inferred,
                                     self.inversion.posterior_stds())

        # --- 4. 맥락적 동기 ---
        lam_ctx = self.empathy.contextual(inferred["alpha"],
                                          inferred["lambda_j"])

        # --- 5. λ 갱신 ---
        if self.regulate:
            self.lam = self.empathy.step(affect["lambda_aff"], lam_ctx)
        else:
            self.lam = self.empathy.lam      # 설정점 고정 (절제 대조)
        self.social_efe.lam = self.lam

        return {"valence": affect["valence"], "arousal": affect["arousal"],
                "lambda_aff": affect["lambda_aff"], "lambda_ctx": lam_ctx,
                "rpe": affect["rpe"], "kl": affect["kl"],
                "expected_reward": self.core_affect.expected_reward(),
                "baseline_reward": affect["r_base"]}

    # ============================================================ 로깅
    def _record(self, action, lam, res, belief_update, reg) -> None:
        super()._record(action, lam, res, belief_update, reg)
        # 첫 라운드는 관측이 없어 reg 가 비어 있다 → 중립값으로 채운다.
        defaults = {"valence": 0.0, "arousal": 0.0, "lambda_aff": 0.0,
                    "lambda_ctx": 0.0, "rpe": 0.0, "kl": 0.0,
                    "expected_reward": self.core_affect.expected_reward(),
                    "baseline_reward": SelfModel.expected_reward(
                        self.self_model.social_reward_prior(),
                        self.core_affect.payoffs)}
        for k, v in defaults.items():
            self.log[k].append(float(reg.get(k, v)))

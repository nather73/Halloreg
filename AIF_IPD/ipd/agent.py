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
    COOP, DEFECT, PAYOFF_SELF, empathy_shift, opponent_action_from_state,
)
from AIF_IPD.core.core_affect import CoreAffect
from AIF_IPD.core.empathy import Empathy
from AIF_IPD.core.constants import joint_index
from AIF_IPD.core.qrtd import QuantileTD, RewardModel, state_index
from AIF_IPD.core.self_model import SelfModel
from .tom.self_policy import SELF_AXES, SelfPolicy
from .tom import (
    GatedToM, ObservationContext, OpponentInversion, RecursiveSocialEFE,
    TheoryOfMind,
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
        **형질공간** rollout 지평 H. 1 이면 근시안(다단계 결과 무시).
    recursive_depth : int
        2 면 depth-2 조망수용(자기-사영 필터 주입) 활성.
    self_projection : bool
        자기-사영 필터 θ̂_self 사용 여부. False 면 IG_other = 0, depth-2 비활성.
    """

    #: 로깅 채널 목록 — 하위 클래스가 확장한다.
    LOG_KEYS = ("action", "lam", "pred_coop", "reliability",
                "S_rho", "S_omega", "S_eta", "SD_S_rho", "SD_S_omega",
                "SD_S_eta", "policy_ess", "policy_G", "q_coop",
                "E_alpha", "E_rho", "E_omega", "E_eta", "E_beta", "E_lambda_j",
                "SD_alpha", "SD_rho", "SD_omega", "SD_eta", "SD_beta",
                "SD_lambda_j", "belief_update", "self_pred_coop",
                # 자기-사영 필터 θ̂_self 의 사후 평균 — "상대가 나를 이렇게
                # 추론할 것이다". H6 의 사영 정합성 검증에 쓴다.
                "P_alpha", "P_rho", "P_omega", "P_eta", "P_beta",
                "P_lambda_j")

    def __init__(self, lam: float = 0.4, beta_self: float = 4.0,
                 policy_particles: int = 64, prop_sd: float = 0.40,
                 policy_gamma: float = 8.0,
                 w_epi_j: float = 10.0, w_epi_r: float = 1.0,
                 w_cplx: float = 0.15,
                 beta_other: float = 4.0, n_particles: int = 400,
                 planning_horizon: int = 6, recursive_depth: int = 2,
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
        # 형질공간 정책 — 행동은 여기서 나온다.
        self.self_policy = SelfPolicy(
            n_particles=policy_particles, prop_sd=prop_sd, gamma=policy_gamma,
            horizon=planning_horizon, w_epi_j=w_epi_j, w_epi_r=w_epi_r,
            w_cplx=w_cplx,
            beta_self=beta_self,
            seed=(None if seed is None else seed * 7717 + 31))
        self._last_policy = {ax: 0.0 for ax in SELF_AXES}
        self._last_qc = 0.5

        self.social_efe = RecursiveSocialEFE(
            self.gated, self.inversion, empathy_factor=self.lam,
            beta_self=beta_self, recursive_depth=recursive_depth,
            self_inversion=self.self_inversion)

        # ---- 상호작용 상태 ----
        # EmpathicAgent(대조군)는 QRTD 를 갖지 않는다. 보수행렬을 아는 것으로
        # 보고 항상 해석해 s(λ,p) 를 쓴다 — HalloRegAgent 가 이를 덮어쓴다.
        self.payoff_access = "oracle"
        self.reward_model = None
        self.qrtd = None
        self._prev_sa = None
        self._shift_used = float("nan")
        self._shift_oracle = float("nan")

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

        # ---------- (4) 형질공간 정책 선택 → 행동 ----------
        # 행동을 직접 고르지 않는다. 자기 형질 θ_i = (ρ, ω, η) 의 입자집합을
        # θ̂_j·λ·ctx 에 조건부로 SMC 갱신하고, 그 사후 혼합에서 행동을 표집한다.
        #
        # 기저 역할 스왑 주의 — focal 정책에서는
        #   f = 상대의 직전 행동 (내가 되갚을 대상 = 호혜 신호)
        #   g = 나 자신의 직전 행동 (자기 관성)
        # 상대 모형에서는 f 가 '내 직전 행동' 이었다.
        lam = self.current_lambda()
        theta_j = self.inversion.posterior_means()

        f_me = (0.0 if not self.opp_actions
                else 1.0 - 2.0 * float(self.opp_actions[-1]))
        g_me = (0.0 if self.my_last is None
                else 1.0 - 2.0 * float(self.my_last))

        # 상대 협력률 p_j (내 우도의 s(λ,p) 인자) 와 내 협력률 p_i (상대 우도용)
        p_other = (float(np.mean(self.opp_actions == 0)) if False
                   else (float(np.mean([1.0 - a for a in self.opp_actions]))
                         if self.opp_actions else 0.5))
        p_self = (float(np.mean([1.0 - a for a in self.my_actions]))
                  if self.my_actions else 0.5)

        # ---- 두 인식항 ----
        # (a) IG_j : 상대의 숨겨진 의도 θ̂_j 에 대한 정보이득 (행위별)
        # (b) IG_R : **환경의 잠재 구조 R̂ 에 대한 정보이득** (행위별)
        # 내 행위는 어떤 결합결과를 관측할지의 분포를 바꾸므로, 겪어보지 않은
        # 칸(협력 정착 관계에서의 DC 등)을 관측하게 하는 행위가 높은 IG 를
        # 받는다. 탐색이 사전이 아니라 목적함수에서 나온다.
        p_coop_j = float(self.inversion.predict_coop(g_me, f_me))
        ig_j = ig_r = None
        if (self.self_policy.w_epi_j != 0.0
                or self.self_policy.w_epi_r != 0.0):
            # 행위 의존성: 내가 a_i 를 두면 상대의 다음 호혜자극이 f' = a_i 가
            # 되므로, 행위마다 상대 관측의 기대 정보이득이 달라진다.
            #   f' = 1 − 2·a_i (내 행동),  g' = 상대 자신의 직전 행동 = f_me
            ig_j = np.array([
                self.inversion.expected_infogain(1.0 - 2.0 * a, f_me)
                for a in (COOP, DEFECT)], dtype=float)
            if self.reward_model is not None:
                unc = self.reward_model.uncertainty()      # (4,) 칸별 불확실성
                span_r = max(float(self.core_affect.payoffs.max()
                                   - self.core_affect.payoffs.min()), 1e-6)
                ig_r = np.array([
                    (p_coop_j * unc[joint_index(a, COOP)]
                     + (1.0 - p_coop_j) * unc[joint_index(a, DEFECT)]) / span_r
                    for a in (COOP, DEFECT)], dtype=float)

        # --- 보수 접근 모드에 따른 공급 ---
        if self.payoff_access == "naive":
            u_s = self.reward_model.payoff_vector("self")
            u_o = self.reward_model.payoff_vector("other")
        else:
            u_s = u_o = None

        # --- 우도 절편: λ-가중 **수익 이점**, (1−γ) 로 라운드당 등가 단위화 ---
        # A_x = Z_x(s,C) − Z_x(s,D) 는 수익 차이라 규모가 보상의 1/(1−γ) 배다.
        # 정규화 없이 쓰면 |절편| ≈ 2.5 로 σ(4·절편) 이 포화해 ρ·ω·η 가 로짓에서
        # 밀려난다(실측). (1−γ) 를 곱하면 '라운드당 평균 이점' 이 되어 1-step
        # 보상과 같은 차원이 되고 형질 항과 견줄 수 있다.
        shift_i = None
        if self.qrtd is not None and self.qrtd.n_obs > 20:
            s_cur = state_index(
                COOP if self.my_last is None else int(self.my_last),
                COOP if not self.opp_actions else int(self.opp_actions[-1]))
            a_self = (self.qrtd.value(s_cur, COOP, "self")
                      - self.qrtd.value(s_cur, DEFECT, "self"))
            a_other = (self.qrtd.value(s_cur, COOP, "other")
                       - self.qrtd.value(s_cur, DEFECT, "other"))
            shift_i = float((1.0 - self.qrtd.gamma)
                            * ((1.0 - lam) * a_self + lam * a_other))
        elif self.payoff_access == "naive":
            shift_i = self.reward_model.shift(lam, p_other)

        term = (self.qrtd.terminal_value
                if (self.qrtd is not None and self.qrtd.n_obs > 20) else None)

        pol = self.self_policy.step(theta_j, lam, f_me, g_me,
                                    p_other, p_self, ig_j=ig_j,
                                    u_self=u_s, u_other=u_o,
                                    terminal=term, shift_i=shift_i,
                                    p_coop_j=p_coop_j, ig_r=ig_r)
        q_c = self.self_policy.coop_prob_mixture(f_me, g_me, lam, p_other,
                                                 shift=shift_i)
        self._shift_used = (shift_i if shift_i is not None
                            else empathy_shift(lam, p_other))
        self._shift_oracle = empathy_shift(lam, p_other)
        action = COOP if self.rng.random() < q_c else DEFECT
        self._last_policy = pol
        self._last_qc = q_c
        # 다음 라운드 TD 전이의 출발점. 상태는 **결정 시점의** (내 직전, 상대 직전).
        my_prev = self.my_last if self.my_last is not None else COOP
        opp_prev = self.opp_actions[-1] if self.opp_actions else COOP
        self._prev_sa = (state_index(my_prev, opp_prev), int(action))
        res = None

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
        # res 는 형질공간 정책 도입 이후 쓰이지 않는다. 예측 협력확률은
        # 상대 필터에서, 신뢰도는 그 필터의 reliability 에서 직접 읽는다.
        self.log["pred_coop"].append(
            float(res.info["pc"]) if res is not None
            else float(self.inversion.predict_coop(
                0.0 if self.my_last is None else 1.0 - 2.0 * float(self.my_last),
                0.0 if not self.opp_actions
                else 1.0 - 2.0 * float(self.opp_actions[-1]))))
        self.log["reliability"].append(float(self.inversion.reliability()))
        # ---- 자기 형질 (ρ, ω, η) 사후 ----
        sp = self.self_policy.posterior_means()
        spd = self.self_policy.posterior_stds()
        for ax in SELF_AXES:
            self.log[f"S_{ax}"].append(float(sp[ax]))
            self.log[f"SD_S_{ax}"].append(float(spd[ax]))
        self.log["policy_ess"].append(float(self._last_policy.get("ess", np.nan)))
        self.log["policy_G"].append(float(self._last_policy.get("G_mean", np.nan)))
        self.log["q_coop"].append(float(self._last_qc))
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
        "valence", "arousal", "lambda_aff", "lambda_ctx", "rpe", "surprise",
        "expected_reward", "baseline_reward", "pessimism", "social_distance",
        "shift_used", "shift_oracle", "qrtd_n", "value",
        "lambda_setpoint")

    def __init__(self, w_cd: float = 0.5, lam_gain: float = 0.05,
                 lam_min: float = 0.0, lam_max: float = 1.0,
                 social_lr: float = 0.02, identity_lr: float = 0.15,
                 kl_scale: float = 0.05,
                 recency_tau: float = 200.0, familiarity_scale: float = 30.0,
                 k_disc: float = 3.0,
                 lam_floor: float = 0.10, lam_ceil: float = 0.70,
                 identity_memory: bool = True, regulate: bool = True,
                 alpha_scale: float = 2.0,
                 payoff_access: str = "naive",
                 qrtd_gamma: float = 0.5, qrtd_lr: float = 0.20,
                 seed_history: bool = True,
                 history_coop_mean: float = 0.55,
                 history_coop_sd: float = 0.18,
                 history_distance_slope: float = 0.0,
                 name: str = "HalloReg", **kwargs):
        # 초기 λ 는 SelfModel 설정점이 결정하므로, 부모의 lam 인자는 임시값이다.
        kwargs.pop("lam", None)
        super().__init__(lam=0.4, name=name, **kwargs)

        self.identity_memory = bool(identity_memory)
        self.regulate = bool(regulate)

        # ---- 위계 구성 ----
        self.self_model = SelfModel(
            social_lr=social_lr, identity_lr=identity_lr,
            recency_tau=recency_tau, familiarity_scale=familiarity_scale,
            k_disc=k_disc, lam_floor=lam_floor, lam_ceil=lam_ceil)
        self.core_affect = CoreAffect(self.self_model, payoffs=PAYOFF_SELF,
                                      kl_scale=kl_scale)

        # ---- 분포적 가치·보상 학습 (QRTD) ----
        # payoff_access — **기본값은 "naive"** (v1.5.2 전환).
        #   "naive"  : 보수행렬을 미리 관측하지 못한다. 관측된 보상만으로
        #              R̂(1-step 보상)·Z(수익)를 학습하고, 우도 절편도 학습된
        #              ŝ(λ,p) 를 쓴다. 환경의 보수 구조가 변동할 수 있으므로
        #              이것이 기본이어야 한다 — 이전 기본값 "oracle" 은
        #              HalloReg 에게만 매 라운드 현재 보수행렬을 넘겨주어
        #              고정전략과의 비교를 교란했다(H4 의 우위 중 얼마가 모형이고
        #              얼마가 정보 접근 특권인지 분리 불가).
        #   "oracle" : 보수행렬 직접 관측. 상한 기준·절제 대조로만 쓴다.
        self.payoff_access = str(payoff_access)
        self.reward_model = RewardModel(lr=max(qrtd_lr * 2, 0.02))
        self.qrtd = QuantileTD(gamma=qrtd_gamma, lr=qrtd_lr,
                               seed=(kwargs.get("seed", 0) or 0) + 5171)
        self.reward_model.set_scale(float(PAYOFF_SELF.max()
                                          - PAYOFF_SELF.min()))
        # Z 의 Huber 임계는 **가치 척도**(보상 범위 / (1−γ))에 맞춘다.
        self.qrtd.set_scale(float(PAYOFF_SELF.max() - PAYOFF_SELF.min())
                            / max(1.0 - qrtd_gamma, 1e-6))

        self._prev_sa = None            # 직전 (상태, 행위) — TD 전이 구성용
        # ---- 사전 사회사 적재 ----
        # 개체는 백지로 사회에 진입하지 않는다. 실험 이전의 관계망(가까운 사람 ·
        # 지인 · 먼 타인)을 미리 적재해야 거리가중 설정점이 실제로 작동한다.
        # 적재하지 않으면 모든 개체의 λ_0 이 0.40 으로 동일해져 사회적 거리가
        # λ 에 아무 영향도 주지 못한다.
        if seed_history:
            self.self_model.seed_social_history(
                PAYOFF_SELF, gamma=qrtd_gamma,
                value_init=self.qrtd.init_value,
                value_spread=self.qrtd.init_spread,
                coop_mean=history_coop_mean,
                coop_sd=history_coop_sd,
                distance_coop_slope=history_distance_slope,
                rng=np.random.default_rng(
                    (kwargs.get("seed", 0) or 0) * 7919 + 104729))

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

        # --- 1b. QRTD 학습 ---
        # 관측된 joint outcome 으로 (i) 1-step 보상모형 R̂ 과 (ii) 수익 가치 Z 를
        # 갱신한다. Z 는 직전 (상태, 행위) 로부터의 전이가 있어야 하므로
        # _prev_sa 가 설정된 이후에만 갱신된다.
        from AIF_IPD.core.constants import PAYOFF_OTHER
        j = int(observed_state)
        r_s, r_o = float(PAYOFF_SELF[j]), float(PAYOFF_OTHER[j])
        self.reward_model.update(j, r_s, r_o)
        v_now = v_shift = None
        v_vec = None
        if self._prev_sa is not None:
            s_prev, a_prev = self._prev_sa
            my_prev = self.my_actions[-1] if self.my_actions else COOP
            s_now = state_index(my_prev, int(opp_action))
            # 갱신 전후의 Z(s,a) 를 비교해 **상황가치 믿음의 이동량**을 얻는다.
            # 이것이 arousal 의 인자다 — 보상 분포의 이동이 아니라 전망의 붕괴다.
            before = self.qrtd.z_self.values[s_prev, a_prev].copy()
            self.qrtd.update(s_prev, a_prev, r_s, r_o, s_now, self._last_qc)
            after = self.qrtd.z_self.values[s_prev, a_prev]
            span_v = float(self.core_affect.payoffs.max()
                           - self.core_affect.payoffs.min())
            span_v = max(span_v / max(1.0 - self.qrtd.gamma, 1e-6), 1e-6)
            v_shift = float(np.mean(np.abs(after - before)) / span_v)
            # valence 의 재료: 방금 겪은 (상태, 행위) 의 Z 분위수 벡터 전체.
            # SelfModel 이 EMA 축약해 '이 상대에 대한 기대 보상 분포' 로 유지
            # 하고, 모집단 참조와의 중앙값 순위가 valence 가 된다.
            v_vec = after.copy()
            v_now = float(np.median(after))
        # 참조(사전 관계들의 가치분포)를 **같은 재귀로 한 스텝** 굴린다.
        # 현재 상대의 Z 와 학습 단계를 맞춰 미수렴 편향을 상쇄한다.
        if self.qrtd is not None:
            self.self_model.tick_reference(
                self.qrtd.gamma, self.qrtd.z_self.lr, self.qrtd.z_self.kappa)

        # --- 2. 핵심정서 구성 ---
        # 상대의 협력 여부를 함께 넘긴다 — SelfModel 이 거리가중 협력률(친사회성
        # 설정점의 원천)을 누적하는 데 쓴다.
        affect = self.core_affect.step(
            observed_state, opponent_cooperated=(int(opp_action) == COOP),
            value_vector=v_vec, value_shift=v_shift,
            z_snapshot=(self.qrtd.z_self.values
                        if self.qrtd is not None else None))

        # --- 3. 상대 특성 기억 commit ---
        # 자기 형질 사후도 함께 보관 — 재조우 시 "이 사람에게는 이런 태세로
        # 대했었다" 가 복원된다. 호혜 경로에도 기억이 생긴다.
        self.self_model.commit_self_theta(
            self._identity, self.self_policy.posterior_means(),
            self.self_policy.posterior_stds())
        self.self_model.commit_theta(self._identity, inferred,
                                     self.inversion.posterior_stds())

        # --- 4. λ 갱신 (v1.5.0: 순수 정서 구동 — λ_ctx 제거) ---
        # OpponentInversion 의 영향은 호혜 경로(ρ,ω,η)로 이관되었다. λ 는
        # 오직 CoreAffect 의 λ_aff 만이 움직인다.
        if self.regulate:
            self.lam = self.empathy.step(affect["lambda_aff"])
        else:
            self.lam = self.empathy.lam      # 설정점 고정 (절제 대조)
        self.social_efe.lam = self.lam

        return {"valence": affect["valence"], "arousal": affect["arousal"],
                "lambda_aff": affect["lambda_aff"], "lambda_ctx": 0.0,
                "rpe": affect["rpe"], "surprise": affect["surprise"],
                "expected_reward": self.core_affect.expected_reward(),
                "baseline_reward": affect["r_base"],
                "pessimism": affect["pessimism"],
                # 실제 쓰인 절편과 오라클 해석해 — naive 모드의 학습 진행을
                # 사후에 검증할 수 있게 둘 다 남긴다.
                "shift_used": getattr(self, "_shift_used", float("nan")),
                "shift_oracle": getattr(self, "_shift_oracle", float("nan")),
                "value": affect["value"],
                "qrtd_n": float(self.qrtd.n_obs
                                if self.qrtd is not None else 0),
                "social_distance": self.self_model.social_distance(
                    self._identity),
                "lambda_setpoint": self.self_model.lambda_setpoint()}

    # ============================================================ 로깅
    def _record(self, action, lam, res, belief_update, reg) -> None:
        super()._record(action, lam, res, belief_update, reg)
        # 첫 라운드는 관측이 없어 reg 가 비어 있다 → 중립값으로 채운다.
        defaults = {"valence": 0.0, "arousal": 0.0, "lambda_aff": 0.0,
                    "lambda_ctx": 0.0, "rpe": 0.0, "surprise": 0.0,
                    "expected_reward": self.core_affect.expected_reward(),
                    "baseline_reward":
                        self.self_model.reference_median(
                            exclude=self._identity),
                    "pessimism": 0.0,
                    "social_distance": self.self_model.social_distance(
                        self._identity),
                    "lambda_setpoint": self.self_model.lambda_setpoint(),
                    "shift_used": getattr(self, "_shift_used", float("nan")),
                    "shift_oracle": getattr(self, "_shift_oracle",
                                            float("nan")),
                    "value": float("nan"),
                    "qrtd_n": float(self.qrtd.n_obs
                                    if self.qrtd is not None else 0)}
        for k, v in defaults.items():
            self.log[k].append(float(reg.get(k, v)))

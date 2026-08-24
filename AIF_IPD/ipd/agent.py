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
from AIF_IPD.core.empathy import empathy_shift_z
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
    #: 하위 클래스가 덮어쓰는 행위 선택 모드 (EmpathicAgent 는 형질 경로).
    policy_mode = "traits"
    beta_es = 45.0

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
        #: SARSA 보류 전이 (s, a, r_self, r_other, s_next) — a' 관측 후 적용.
        self._pending = []
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

    def _occ_adv(self):
        """점유율 가중 행위 이점 (A_self, A_other) — 보상적 λ_sp 의 입력."""
        if self.qrtd is None or self.qrtd.n_obs <= 3:
            return None
        d = self._state_visits / max(self._state_visits.sum(), 1e-9)
        a_s = float(np.sum([d[ss] * (self.qrtd.value(ss, COOP, "self")
                                     - self.qrtd.value(ss, DEFECT, "self"))
                            for ss in range(4)]))
        a_o = float(np.sum([d[ss] * (self.qrtd.value(ss, COOP, "other")
                                     - self.qrtd.value(ss, DEFECT, "other"))
                            for ss in range(4)]))
        return (a_s, a_o)

    def _tom_mirror_es(self, lam_vec, f, g):
        """거울 es_z 공급자 (v3.7.1) — OpponentInversion._pC 가 호출."""
        if f == 0.0 or g == 0.0:
            from AIF_IPD.core.constants import empathy_shift
            return empathy_shift(lam_vec,
                                 self.inversion.my_cooperation_rate)
        my_a = 0 if f > 0 else 1
        th_a = 0 if g > 0 else 1
        m = joint_index(th_a, my_a)        # 상대 시점 상태 (거울)
        a_s = (self.qrtd.value(m, 0, "self")
               - self.qrtd.value(m, 1, "self"))
        a_o = (self.qrtd.value(m, 0, "other")
               - self.qrtd.value(m, 1, "other"))
        return ((1.0 - lam_vec) * a_s + lam_vec * a_o) / self._es_scale

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

        # --- 우도 절편: λ-가중 수익 이점 ---
        #   A_x(s) = Z̄_x(s, C) − Z̄_x(s, D)
        #   shift  = (1−γ)·[ (1−λ)·A_self + λ·A_other ]
        #
        # [v1.9.0 — 즉각항/꼬리항 분리 폐기]
        # v1.7~1.8 은 즉각항을 R̂(1-step)에서, 꼬리항을 Z 에서 가져와 합쳤다.
        # 그러나 Z 자체가 "이 상태에서 이 행위의 장기 가치" 이므로 그 차이가
        # 곧 행위 이점이고, 두 원천을 섞는 것은 인위적이다. R̂ 를 끌어온 이유는
        # 미방문 칸이 초깃값에 머문다는 것이었는데, 그것은 절편의 문제가 아니라
        # **학습률의 문제**였다. QuantileTD 의 방문 횟수 적응 학습률
        # (lr = max(1/(1+n), lr_base))로 원인을 직접 제거했으므로 — 한 번 겪어
        # 보상이 낮으면 그 즉시 그 행위의 기대 보상이 내려간다 — 절편은 Z 만으로
        # 일관되게 구성한다.
        shift_i = None
        # v2.8.0: 게이트 8 → 3. λ 가 5~7R 내에 반응해야 하므로, 이점 추정이
        # 거칠더라도 조기에 쓰는 편이 낫다 (방문 적응 lr 로 첫 관측이 즉시
        # 반영되므로 부호는 대체로 옳다).
        if self.qrtd is not None and self.qrtd.n_obs > 3:
            s_cur = state_index(
                COOP if self.my_last is None else int(self.my_last),
                COOP if not self.opp_actions else int(self.opp_actions[-1]))
            a_self = (self.qrtd.value(s_cur, COOP, "self")
                      - self.qrtd.value(s_cur, DEFECT, "self"))
            a_other = (self.qrtd.value(s_cur, COOP, "other")
                       - self.qrtd.value(s_cur, DEFECT, "other"))
            # **es(λ, s) — Z 만으로 계산되는 공감 절편** (v2.1)
            raw_es = float(empathy_shift_z(lam, a_self, a_other))
            # **절편 정규화** (v3.0) — es 를 고정 참조 산포로 나눈다.
            #   es_norm = es / σ_ref,  σ_ref = (1−γ)·(r_max − r_min)
            # 정규화 Z̃ 에서 한 라운드 행위가 만드는 이점의 이론적 최대 규모가
            # (1−γ)·보상폭 이다. 이 값으로 나누면 es 가 '가능한 최대 이점 대비
            # 몇 배' 라는 무차원량이 된다.
            #
            # [왜 고정 참조인가]
            # 실측 산포로 나누면 되먹임이 생긴다 — 행동이 극단화될수록 산포가
            # 줄고 그러면 절편이 커져 더 극단화된다(v1.9 형질 정규화에서
            # 겪은 것과 같은 구조). 고정 참조는 그 고리를 만들지 않는다.
            #
            # [무엇을 고치는가]
            # on-policy 부트스트랩이 Z̃(s,C) 와 Z̃(s,D) 를 서로를 향해 압착해,
            # ALLD 상대에서 A_self 가 이론값 −0.10 의 1/4(−0.024)에 그쳤다.
            # 그 결과 λ=0.010 을 달성하고도 로짓이 −0.81 에 머물러 협력률
            # 0.344 가 남았다. 정규화는 이 압착을 척도 수준에서 되돌린다.
            # −G_social 의 실용항: w_U · es_norm  (v3.4)
            shift_i = self.w_u * (raw_es / self._es_scale)
            # **인식적 항 복원** (v3.1) — EFE 의 epistemic value 를 절편에
            # 더한다. 두 행위의 차분이므로 정책 로짓에 그대로 얹힌다:
            #     logit = β_es·es_norm + w_j·ΔIG_j + w_r·ΔIG_R
            # ΔIG_x = IG_x(C) − IG_x(D).
            #
            # [왜 절편에 더하는가] λ-단독 전환 후 SelfPolicy 의 EFE 가 호출되지
            # 않아 인식항이 행위 선택에서 소실되어 있었다. 2 행위 문제에서
            # softmax(−G) 는 로짓 차분의 시그모이드와 동치이므로, 차분을 절편에
            # 더하는 것이 EFE 형태를 보존하는 최소 복원이다.
            #
            # [왜 초기에만 작동하는가] IG_R 의 불확실성은 spread/√(1+n) 이라
            # 방문이 쌓이면 0 으로 감쇠하고, IG_j 도 사후가 수축하면 KL 이
            # 줄어든다. 별도 스케줄 없이 **탐색이 저절로 꺼진다**.
            if self.w_ig_r > 0.0 and ig_r is not None:
                shift_i += self.w_ig_r * float(ig_r[COOP] - ig_r[DEFECT])
            if self.w_ig_j > 0.0 and ig_j is not None:
                shift_i += self.w_ig_j * float(ig_j[COOP] - ig_j[DEFECT])

        term = (self.qrtd.terminal_value
                if (self.qrtd is not None and self.qrtd.n_obs > 3) else None)

        if self.policy_mode == "lambda_only":
            # (v2.0 실험) **λ-단독 행위 선택** — 형질 (ρ, ω, η) 폐기.
            #   P(C) = σ( β · s_t(λ) ),  s_t = c·[(1−λ)A_s + λA_o]/scale
            # 상태·맥락 의존성은 A_x(s) = Z̄_x(s,C) − Z̄_x(s,D) 가 담고,
            # 관계 의존성은 λ 의 동적 조절이 담는다. 호혜처럼 보이는 행동은
            # 상태 의존 절편에서, 성향처럼 보이는 것은 λ 궤적에서 나온다.
            # 단일 파라미터 β_es 로 통합 (v2.1). 이전에는 shift_gain(0.30) ×
            # shift_beta(8.0) ÷ trait_scale(1.046) 이 곱해져 실질 이득 2.295 를
            # 세 상수로 중복 표현했고, 분모는 **폐기된 형질**의 사전 규모라
            # 의미가 없었다. 이제 로짓 정밀도 하나만 남는다.
            # **사회적 EFE** (v3.4): 로짓 = α + β·(−G_social(C|λ, s_t)),
            #   −G_social = w_U·es_norm + w_R·ΔIG_R + w_θ·ΔIG_j
            # (es 는 정의상 C−D 차분이고, IG 항도 차분으로 두어 2행위
            #  softmax(−G) 와 정확히 동치인 시그모이드 형이다.)
            _z = self.alpha_bias + (0.0 if shift_i is None
                                    else self.beta_g * shift_i)
            q_c = float(1.0 / (1.0 + np.exp(-_z)))
            pol = {"ess": float("nan"), "theta_mean": {}, "pc": q_c}
        else:
            pol = self.self_policy.step(theta_j, lam, f_me, g_me,
                                    p_other, p_self, ig_j=ig_j,
                                    u_self=u_s, u_other=u_o,
                                    terminal=term, shift_i=shift_i,
                                    p_coop_j=p_coop_j, ig_r=ig_r)
        if self.policy_mode != "lambda_only":
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
        "lam_sp", "allo_phi", "fitness",
        "lambda_setpoint")

    def __init__(self, w_cd: float = 0.5, lam_gain: float = 0.05,
                 lam_min: float = 0.0, lam_max: float = 0.80,
                 social_lr: float = 0.02, identity_lr: float = 0.15,
                 kl_scale: float = 0.05,
                 recency_tau: float = 200.0, familiarity_scale: float = 30.0,
                 k_disc: float = 3.0,
                 lam_floor: float = 0.10, lam_ceil: float = 0.70,
                 identity_memory: bool = True, regulate: bool = True,
                 alpha_scale: float = 2.0,
                 payoff_access: str = "naive",
                 qrtd_gamma: float = 0.9, qrtd_lr: float = 0.20,
                 w_tonic: float = 0.10,
                 aff_gain: float = 0.30,
                 lam_gain_down: float = 0.45,
                 lam_mode: str = "allostatic",
                 group_bias: float = 0.50,
                 lam_lo: float = -0.5, lam_hi: float = 1.0,
                 tom_es_mode: str = "mirror",
                 allo_aff_gain: float = 0.0,
                 policy_mode: str = "lambda_only",
                 beta_g: float = 3.0, w_u: float = 70.0 / 3.0,
                 beta_es: Optional[float] = None,
                 sp_disposition: float = 0.50,
                 bootstrap: str = "sarsa",
                 plan_sweeps: int = 1,
                 plan_update: str = "conf",
                 plan_lr: float = 0.5,
                 n_step: int = 1,
                 reanchor_at: int = 8,
                 w_ig_r: float = 3.5,
                 w_ig_j: float = 3.5,
                 r_surv_fixed: Optional[float] = None,
                 e_source: str = "z",
                 alpha_kappa: float = 0.0,
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
        self.policy_mode = str(policy_mode)
        #: λ 조절 방식 — 'integrator'(현행) | 'allostatic'(직접 사상, v2.9 탐색).
        self.lam_mode = str(lam_mode)
        #: 집단 적합성 편향 g — 생존이 보장될 때 λ 가 도달하는 상한.
        self.group_bias = float(group_bias)
        #: λ 사상 범위 (v3.7 검토): λ = lam_lo + (lam_hi−lam_lo)·clip(φ,0,1).
        #: None 이면 기존 [0, g] (λ = g·φ). 음수 lam_lo 는 결핍 시 **경쟁적
        #: (반공감) 태세** — es = (1−λ)A_self + λ·A_other 에서 λ<0 이면 상대
        #: 이득이 내 로짓에 음(−)으로 들어간다 (처벌/억지).
        self.lam_lo = None if lam_lo is None else float(lam_lo)
        self.lam_hi = None if lam_hi is None else float(lam_hi)
        #: 알로스테시스 모드의 정서 미세조절 이득.
        self.allo_aff_gain = float(allo_aff_gain)
        # λ-단독에서는 절편이 로짓의 **유일한** 항이므로 정밀도를 따로 둔다.
        # 형질 4항이 나눠 갖던 로짓 예산을 절편 하나가 감당해야 한다.
        #: 사회적 EFE 의 정밀도 β 와 실용 가중 w_U (v3.4).
        #:     −G_social(C|λ,s) = w_U·es(λ,s)/σ_ref + w_R·ΔIG_R + w_θ·ΔIG_j
        #:     P(C) = σ(β·(−G_social))
        #: beta_es 를 직접 주면 w_u = beta_es/β 로 환산한다 (하위호환).
        self.beta_g = float(beta_g)
        self.w_u = (float(w_u) if beta_es is None
                    else float(beta_es) / max(self.beta_g, 1e-9))
        #: 파생량 β·w_U — 실용항 단독 정밀도. 계획의 q_c(s') 가 이를 쓴다.
        self.beta_es = self.beta_g * self.w_u
        #: es 정규화의 고정 참조 산포 — (1−γ)·보상폭.
        #: 생존 기준점 고정값 (None = 학습된 R̂ 의 보장수준 maximin).
        self.r_surv_fixed = (None if r_surv_fixed is None
                             else float(r_surv_fixed))
        #: E_t 의 원천 — 'reward' (R̂ 1-step ToM 예측) | 'z' (Z̃ 장기 가치).
        self.e_source = str(e_source)
        self.w_ig_r = float(w_ig_r)
        self.w_ig_j = float(w_ig_j)
        #: ToM es 모드 (v3.7.1): 'mirror'(기본) — 상대 우도의 es 를 내 Z̃ 의
        #: 역할 교환으로 대리한다 (모형 정렬: ToM 이 행위자의 실제 생성과정과
        #: 같은 족의 효용을 상정). 'analytic' 은 구식 해석적 empathy_shift
        #: (레거시 — 인자를 명시해야 활성).
        #:     es_j(λ_j; s) = [(1−λ_j)·A_self(m(s)) + λ_j·A_other(m(s))]/σ_ref
        #:     m(s) = 거울 상태 (상대 시점: 행동쌍 역할 교환)
        #: 이력 없음(f=0 또는 g=0)이면 해석적 형으로 후퇴한다.
        self.tom_es_mode = str(tom_es_mode)
        if self.tom_es_mode == "mirror":
            self.inversion.es_provider = self._tom_mirror_es
        self._es_scale = max((1.0 - float(qrtd_gamma))
                             * float(PAYOFF_SELF.max() - PAYOFF_SELF.min()),
                             1e-6)
        self.alpha_kappa = float(alpha_kappa)
        self.core_affect.sp_disposition = float(sp_disposition)
        # (v2.1) 형질 사전 규모 분모는 폐기되었다 — 폐기된 형질의 규모를
        # 살아있는 절편의 분모로 쓰는 것은 의미가 없었고, β_es 하나로 흡수된다.
        self.reward_model = RewardModel(lr=max(qrtd_lr * 2, 0.02))
        self.qrtd = QuantileTD(gamma=qrtd_gamma, lr=qrtd_lr,
                               seed=(kwargs.get("seed", 0) or 0) + 5171)
        self.reward_model.set_scale(float(PAYOFF_SELF.max()
                                          - PAYOFF_SELF.min()))
        # v2.7.0: Z̃ = (1−γ)Z 정규화이므로 Huber 임계도 **보상 척도**로 잡는다.
        self.qrtd.set_scale(float(PAYOFF_SELF.max() - PAYOFF_SELF.min()))
        self.qrtd.bootstrap = str(bootstrap)
        #: 라운드당 모형 기반 계획 스윕 횟수 (0 이면 순수 TD).
        self.plan_sweeps = int(plan_sweeps)
        self.plan_update = str(plan_update)
        self.plan_lr = float(plan_lr)
        #: ③ on-policy n-step SARSA 의 n (1 = 기존 1-step).
        self.n_step = max(1, int(n_step))
        #: 이 관측 수에서 Z̃ 를 R̂ 기반 값으로 1 회 재기준화한다 (0 이면 없음).
        self.reanchor_at = int(reanchor_at)
        self._reanchored = False

        self._prev_sa = None            # 직전 (상태, 행위) — TD 전이 구성용
        self._state_visits = np.zeros(4)   # 경험적 상태 점유율 d(s)

        # ---- 사전 사회사 적재 ----
        # 개체는 백지로 사회에 진입하지 않는다. 실험 이전의 관계망(가까운 사람 ·
        # 지인 · 먼 타인)을 미리 적재해야 거리가중 설정점이 실제로 작동한다.
        # 적재하지 않으면 모든 개체의 λ_0 이 0.40 으로 동일해져 사회적 거리가
        # λ 에 아무 영향도 주지 못한다.
        if seed_history:
            self.self_model.seed_social_history(
                PAYOFF_SELF, gamma=qrtd_gamma,
                coop_mean=history_coop_mean,
                coop_sd=history_coop_sd,
                distance_coop_slope=history_distance_slope,
                rng=np.random.default_rng(
                    (kwargs.get("seed", 0) or 0) * 7919 + 104729))

        # 사회사가 적재된 뒤 Z 를 '전형적 관계의 가치' 에서 출발시킨다.
        # 참조(기억 속 관계들)와 Z(현재 관계)가 같은 척도에서 시작하므로,
        # 첫 라운드 valence 가 중립이 된다.
        _c, _sp = self.self_model.value_prior()
        self.qrtd.reinit(_c, _sp)

        if self.policy_mode == "lambda_only":
            # (v2.2) λ₀ 폐기 — λ = λ_sp + λ_val 로, 사회사 정보는 α 가 담당한다.
            # λ 는 이제 '이 상대·이 상황' 만을 담는 순수 조절 변수다. 중립값
            # 0.5 에서 출발해 첫 조우 직후 λ_sp 로 이완한다.
            lam0 = 0.5
        else:
            lam0 = self.self_model.lambda_setpoint(PAYOFF_SELF)
        self.empathy = Empathy(lam_init=lam0, w_cd=w_cd, gain=lam_gain,
                               w_tonic=w_tonic, aff_gain=aff_gain, gain_down=lam_gain_down,
                               lam_min=lam_min, lam_max=lam_max,
                               alpha_scale=alpha_scale)

        #: 사회사 기반 협력 편향 α — 사회사 적재 후 1회 확정, 로짓의 가산항.
        #   α ~ N(0, κ²).  '나는 어떤 사회적 세계에서 왔는가' 를 담는다.
        self.alpha_bias = (
            self.self_model.cooperation_bias(kappa=self.alpha_kappa)
            if self.policy_mode == "lambda_only" else 0.0)
        # 보상적 λ_sp 의 목표 절편 I₀ (v2.2).
        #   로짓 = α + β_es·es(λ, s) 이므로, 중립 행동(로짓 0)에 필요한 es 는
        #   es* = −α/β_es 다. λ_sp 가 이 값을 목표로 삼아야 α 와 정합한다.
        #   I₀ = 0 으로 두면 λ_sp 가 α 의 존재를 모른 채 무차별점만 겨냥해,
        #   협력적 사회사(α>0)에서도 공감 요구량을 과대평가하고 그 반대도 같다.
        self.core_affect.sp_i0 = float(-self.alpha_bias / max(self.beta_es,
                                                              1e-6))

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
            # **참 SARSA — 전이를 한 라운드 보류한다** (v2.6.0)
            # 지금 라운드 t 에서 완성되는 전이는 (s_{t-1}, a_{t-1}) → s_t 인데,
            # 부트스트랩이 요구하는 a' = a_t 는 아직 선택되지 않았다. 반면
            # 라운드 t−1 에 보류해 둔 전이 (s_{t-2}, a_{t-2}) → s_{t-1} 의
            # a' = a_{t-1} 은 **이미 관측되었다**. 그것을 지금 적용한다.
            #   보류분의 s_next 는 s_prev 와 같고, a_prev 가 곧 그 상태에서
            #   실제로 선택된 행위다.
            if self._pending is None:
                self._pending = []
            if not self._pending:
                before = after = self.qrtd.z_self.values[s_prev, a_prev].copy()
            elif self.n_step <= 1:
                p_s, p_a, p_rs, p_ro, p_sn = self._pending[0]
                before = self.qrtd.z_self.values[p_s, p_a].copy()
                self.qrtd.update(p_s, p_a, p_rs, p_ro, p_sn, a_prev)
                after = self.qrtd.z_self.values[p_s, p_a]
                self._pending = []
            else:
                # ③ **on-policy n-step SARSA** — 가장 오래된 전이를,
                # 중간 보상들의 할인합과 γⁿ 부트스트랩으로 완결한다:
                #   y = Σ_{k<n} γᵏ(1−γ)r_{t0+k} + γⁿ·Z̃(s_{t0+n}, a_{t0+n})
                # 부트스트랩(임의 초기화 칸이 목표에 들어오는 통로)의
                # 가중이 γ → γⁿ 으로 줄어, 동결 칸의 낙관 주입이 기하적으로
                # 감쇠한다 (Sutton & Barto 7장).
                self.qrtd.note_reward(r_s, r_o)   # n-step 경로의 r̄ 갱신
                before = self.qrtd.z_self.values[
                    self._pending[0][0], self._pending[0][1]].copy()
                after = before
                if len(self._pending) >= self.n_step:
                    p_s, p_a = self._pending[0][0], self._pending[0][1]
                    g_s = g_o = 0.0
                    _wr = 1.0 - self.qrtd.gamma
                    for _k, _e in enumerate(self._pending):
                        g_s += (self.qrtd.gamma ** _k) * _wr * _e[2]
                        g_o += (self.qrtd.gamma ** _k) * _wr * _e[3]
                    _gn = self.qrtd.gamma ** len(self._pending)
                    _bs = self.qrtd.bvec(s_prev, a_prev, "self")
                    _bo = self.qrtd.bvec(s_prev, a_prev, "other")
                    self.qrtd.apply_target(p_s, p_a,
                                           g_s + _gn * _bs,
                                           g_o + _gn * _bo)
                    after = self.qrtd.z_self.values[p_s, p_a]
                    self._pending.pop(0)
            self._pending.append((s_prev, a_prev, r_s, r_o, s_now))

            # --- (C) 초기화 재기준화 ---
            # 사회사 사전(≈2.11)은 '전형적 관계' 의 값이라 나쁜 관계에서는
            # 참값의 3 배에서 출발하는 낙관적 초기화가 된다. R̂ 이 최소한의
            # 관측을 모으면 **이 관계의 값**으로 한 번 다시 앉힌다 — 하강
            # 거리가 1/3 로 줄어 이후 수축이 빨라진다. (Sutton & Barto §2.6)
            if (not self._reanchored and self.reanchor_at > 0
                    and self.reward_model.n_obs >= self.reanchor_at):
                rv_s = self.reward_model.payoff_vector("self")
                rv_o = self.reward_model.payoff_vector("other")
                for _arr, _rv in ((self.qrtd.z_self, rv_s),
                                  (self.qrtd.z_other, rv_o)):
                    _c = float(np.mean(_rv))
                    _arr.values[:] = (_arr.values
                                      - float(np.mean(_arr.values)) + _c)
                self._reanchored = True

            # --- (A) 모형 기반 계획 스윕 ---
            if self.plan_sweeps > 0 and self.reward_model.n_obs > 3:
                # 상태별 p_j — 상태 s = (내 직전 행위, 상대 직전 행위) 가
                # 곧 ToM 의 맥락 (f, g) 이다. 상대의 조건부 전략(TFT 등)이
                # 모형 안에서 보존된다.
                _pj = np.array([
                    float(self.inversion.predict_coop(
                        1.0 - 2.0 * float(ss // 2), 1.0 - 2.0 * float(ss % 2)))
                    for ss in range(4)])
                # 연속정책도 상태별로 — q_c(s') = σ(β_es·es(λ_t, s')/σ_ref).
                # (인식항은 상태별 산출 비용이 커 계획에서는 실용항만 쓴다 —
                #  학습 후반에는 어차피 소멸하는 항이라 근사 오차가 작다.)
                if self.qrtd.n_obs > 8:
                    _qc = np.empty(4)
                    for _s in range(4):
                        _as = (self.qrtd.value(_s, COOP, "self")
                               - self.qrtd.value(_s, DEFECT, "self"))
                        _ao = (self.qrtd.value(_s, COOP, "other")
                               - self.qrtd.value(_s, DEFECT, "other"))
                        _es = ((1.0 - self.lam) * _as + self.lam * _ao) \
                            / self._es_scale
                        _qc[_s] = 1.0 / (1.0 + np.exp(-self.beta_es * _es))
                else:
                    _qc = np.full(4, float(self._last_qc))
                self.qrtd.plan_sweep(
                    self.reward_model,
                    p_coop_j=_pj,
                    p_coop_self=_qc,
                    n_sweeps=self.plan_sweeps,
                    update=self.plan_update, lr=self.plan_lr)
            span_v = float(self.core_affect.payoffs.max()
                           - self.core_affect.payoffs.min())
            span_v = max(span_v, 1e-6)   # v2.7.0: Z̃ 는 보상 척도
            v_shift = float(np.mean(np.abs(after - before)) / span_v)
            # valence 의 재료: 방금 겪은 (상태, 행위) 의 Z 분위수 벡터 전체.
            # SelfModel 이 EMA 축약해 '이 상대에 대한 기대 보상 분포' 로 유지
            # 하고, 모집단 참조와의 중앙값 순위가 valence 가 된다.
            # **정책 하 기대가치로 주변화** (v1.7.0).
            # 이전에는 '방금 밟은 칸' 의 Z 를 그대로 썼다. 그러면 칸 방문의
            # 우연성이 그대로 들어와 valence 가 진동한다(실측: ALLC 상대에서
            # 부호가 네 번 뒤집힘). valence 가 물어야 할 것은 "방금 이 수가
            # 좋았나" 가 아니라 **"이 관계가 좋은가"** 이므로,
            #     Q = Σ_s d(s)·Σ_a π(a|s)·Z(s,a)
            # 로 주변화한다. d(s) 는 **경험적 상태 점유율**(이 관계에서 내가
            # 실제로 놓이는 상황의 빈도), π 는 현재 협력확률이다.
            self._state_visits[s_prev] += 1.0
            d = self._state_visits / max(self._state_visits.sum(), 1e-9)
            pc = float(self._last_qc)
            v_vec = np.zeros_like(after)
            for ss in range(4):
                if d[ss] <= 1e-12:
                    continue
                v_vec += d[ss] * (pc * self.qrtd.z_self.values[ss, 0]
                                  + (1.0 - pc) * self.qrtd.z_self.values[ss, 1])
            v_now = float(np.median(v_vec))
            # **상태 간 적합도** — 이 관계 안에서 지금 상태가 좋은가.
            #   V(s') = π(C)·Z(s',C) + (1−π)·Z(s',D)  (정책 하 상태가치)
            #   valence = 2·Σ_{s': V(s')≤V(s)} d(s') − 1
            # 점유율 d 로 가중된 백분위이므로, 드물게 겪는 상태가 과대평가되지
            # 않는다. WSLS 상대에서 CD 가 최하·DD 가 최상위권으로 뒤집히는
            # 신호가 여기서 나온다(TFT 와 정반대 순서).
            _vs = np.array([pc * self.qrtd.value(ss, COOP, "self")
                            + (1.0 - pc) * self.qrtd.value(ss, DEFECT, "self")
                            for ss in range(4)], dtype=float)
            # **현재 상태는 참조에서 제외한다.** 포함하면 자기 질량이 백분위에
            # 들어가, 가장 자주 머무는 상태가 그 사실만으로 높은 순위를 얻는다
            # (실측: ALLD 상대에서 DD 점유가 커지자 DD 의 valence 가 올라 λ 가
            # 0.79 까지 상승 — 착취자에게 협력률 0.94). 관계 간 비교에서 현재
            # 상대를 제외하는 것과 같은 이유다.
            _mask = np.ones(4, dtype=bool)
            _mask[s_prev] = False
            _den = float(d[_mask].sum())
            if _den > 1e-9:
                _num = float(d[_mask & (_vs <= _vs[s_prev])].sum())
                _st_val = float(2.0 * (_num / _den) - 1.0)
            else:
                _st_val = 0.0

        # --- 2. 핵심정서 구성 ---
        # 상대의 협력 여부를 함께 넘긴다 — SelfModel 이 거리가중 협력률(친사회성
        # 설정점의 원천)을 누적하는 데 쓴다.
        affect = self.core_affect.step(
            observed_state, opponent_cooperated=(int(opp_action) == COOP),
            value_vector=v_vec, value_shift=v_shift,
            state_valence=(_st_val if self._prev_sa is not None else None),
            adv=(self._occ_adv() if self.policy_mode == "lambda_only"
                 else None),
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
        if self.regulate and self.lam_mode == "allostatic":
            # **알로스테시스 직접 사상** (v2.9 탐색) — 적분기 폐기.
            #     E_t     = Σ_s d(s)·[q_c·Z̃(s,C) + (1−q_c)·Z̃(s,D)]
            #               (이 관계에서 예상되는 라운드당 자원 유입)
            #     r_surv  = max_a min_{a_j} R̂(a, a_j)
            #               (자기보호만으로 확보 가능한 생존 자원 — 게임의 보장수준)
            #     top     = R̂(C, C)   (상호협력의 공동 최적)
            #     φ       = (E_t − r_surv) / (top − r_surv)   (잉여율)
            #     λ_t     = g · clip(φ, 0, 1)  (+ 정서 미세 조절)
            # 예상 유입이 생존 기준 아래면 λ → 0 (선제적 자기보호, allostatic
            # deviation 대응), 잉여가 있으면 그에 비례해 λ ↑ (집단 적합성 편향
            # g 가 상한). r_surv 와 top 이 학습된 R̂ 에서 나오므로 보수 레짐이
            # 바뀌면 기준점이 함께 재계산된다 — 비정상 환경 적응의 직접 기제.
            if self.lam_lo is not None:
                _hi = self.lam_hi if self.lam_hi is not None else 1.0
                lam_allo = self.lam_lo + (_hi - self.lam_lo) * 0.5
            else:
                lam_allo = self.group_bias * 0.5   # 무정보 기본값
            self._allo_phi = np.nan
            if self.reward_model.n_obs > 3:
                # E_t 는 **ToM 기반 1-step 예측**으로 계산한다 (Z̃ 가 아니라).
                #   E_t = q_c·[p_j·R̂(C,C) + (1−p_j)·R̂(C,D)]
                #       + (1−q_c)·[p_j·R̂(D,C) + (1−p_j)·R̂(D,D)]
                # Z̃ 점유가중을 쓰면 미방문 협력 분기가 사회사 사전(≈2.1)에
                # 머물러 E 가 낙관 오염된다 — ALLD 상대에서 λ 가 0.65 까지
                # 올라 착취당하는 것을 실측으로 확인했다. ToM 예측은 p_j 가
                # 미방문 분기의 가중치를 지우므로(착취자면 p_j → 0) 오염이
                # 없고, 상대 모형의 갱신 속도로 **선제적으로** 반응한다 —
                # 이것이 반응적(항상성)이 아닌 예측적(이상성) 조절이다.
                qc = float(self._last_qc)
                rv = self.reward_model.payoff_vector("self")
                if (self.e_source == "z" and self.qrtd is not None
                        and self.qrtd.n_obs > 3):
                    # **Z̃ 기반 E_t** (v3.2) — 장기 가치가 λ 를 주도한다.
                    #   E_t = q_c·Z̃(s_t, C) + (1−q_c)·Z̃(s_t, D)
                    # Z̃ 는 라운드당 평균 수익 척도라 r_surv 와 직접 비교된다.
                    # 현재 상태로 조건화되므로 λ 가 **상태 의존적**이 된다 —
                    # 같은 관계 안에서도 나쁜 국면에서는 λ 가 내려간다.
                    # (v2.9 의 Z 점유가중 실패는 계획 스윕 도입 전의 일이다.
                    #  당시엔 미방문 분기가 사전값 2.1 에 머물러 낙관 순환을
                    #  만들었지만, 이제 매 라운드 8칸 전부가 R̂·p_j(s) 로
                    #  백업되므로 그 오염원이 없다.)
                    e_t = (qc * self.qrtd.value(observed_state, COOP, "self")
                           + (1 - qc) * self.qrtd.value(observed_state,
                                                        DEFECT, "self"))
                else:
                    _f = 1.0 - 2.0 * float(self.my_last
                                           if self.my_last is not None else 0)
                    _g = (1.0 - 2.0 * float(self.opp_actions[-1])
                          if self.opp_actions else 0.0)
                    pj = float(self.inversion.predict_coop(_f, _g))
                    e_t = (qc * (pj * rv[0] + (1 - pj) * rv[1])
                           + (1 - qc) * (pj * rv[2] + (1 - pj) * rv[3]))
                r_surv = (self.r_surv_fixed
                          if self.r_surv_fixed is not None else
                          float(max(min(rv[0], rv[1]), min(rv[2], rv[3]))))
                top = float(rv[0])
                if top - r_surv > 1e-6:
                    phi_a = (e_t - r_surv) / (top - r_surv)
                    self._allo_phi = float(phi_a)
                    _ph = float(np.clip(phi_a, 0.0, 1.0))
                    if self.lam_lo is not None:
                        lam_allo = (self.lam_lo
                                    + ((self.lam_hi if self.lam_hi is not None
                                        else 1.0) - self.lam_lo) * _ph)
                    else:
                        lam_allo = self.group_bias * _ph
                    # 정서 미세 조절 (상태 처방 통로 유지, 소이득)
                    _clo = (self.lam_lo if self.lam_lo is not None else 0.0)
                    _chi = (self.lam_hi if self.lam_lo is not None
                            else self.group_bias)
                    lam_allo = float(np.clip(
                        lam_allo + self.allo_aff_gain * affect["lambda_aff"],
                        _clo, _chi))
            self.lam = lam_allo
            self.empathy.lam = self.lam
        elif self.regulate:
            self.lam = self.empathy.step(
                affect["lambda_aff"], valence=affect["valence"],
                lam_sp=affect.get("lam_sp"),
                # 위협 = 관계 간 적합도가 음수 (모집단 참조보다 나쁜 관계).
                threat=bool(np.isfinite(affect.get("fitness", np.nan))
                            and affect["fitness"] < 0.0))
        else:
            self.lam = self.empathy.lam      # 설정점 고정 (절제 대조)
        self.social_efe.lam = self.lam

        return {"valence": affect["valence"], "arousal": affect["arousal"],
                "lambda_aff": affect["lambda_aff"], "lambda_ctx": 0.0,
                "lam_sp": affect.get("lam_sp", float("nan")),
                "fitness": affect.get("fitness", float("nan")),
                "allo_phi": float(getattr(self, "_allo_phi", float("nan"))),
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
                    "lam_sp": float("nan"), "fitness": float("nan"),
                    "allo_phi": float("nan"),
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

"""
core.allostasis_v9
==================

**HalloReg v0.9.0 항상성 조절 계층의 이론적 재정초.**

이 모듈은 v0.2.1~v0.8.2 의 ``CoreAllostaticBeliefState`` + ``LambdaRegulator``
(leaky 이중 적분기)를 **폐기**하고, 두 계층으로 대체한다(명세서 §1, §9.1):

    SelfModel   — 느린 계층. 발달적으로 학습된 사회 환경 믿음에서 λ 의 setpoint
                  (λ_base)을 형성한다. IPD 수행 중에는 Self boundary 밖의 먼 Other
                  만, 오랜 반복 시행 후에만, 사소하게 갱신된다(setpoint drift).
    CoreAffect  — 빠른 계층. 항상성 불균형 사건을 **행동 라벨이 아니라 기대보상에
                  대한 부적 예측오차(RPE)** 로 재정의하고(§3.1), Distributional RL
                  로 기대보상 분포를 추적하며(§3.2), **prior 를 지닌 베이지안 belief**
                  로 RPE 를 {contextual, dispositional} 두 원인에 귀인한다(§3.3).
                  λ 는 적분기가 아니라 이 belief 에서 **직접 사상**된다(§3.5).

두 계층은 predictive-coding 의 상·하위로 연결된다: SelfModel 이 λ_base 예측을
내려보내고, CoreAffect 가 그에 대한 PE 를 올려보낸다(§2.4).

이론적 근거
-----------
  · Sterling/Barrett-Katsumi (allostasis-first): 뇌의 핵심 기능은 항상성 불균형의
    *예측적* 조절이며, 현저성은 '예측된 allostatic 가치'(=기대보상)의 함수다.
    → 사건을 CD 라벨이 아니라 기대보상 RPE 로 정의하는 것이 정합적이다(§8).
  · predictive coding: setpoint(SelfModel) 예측과 편차(CoreAffect) PE 의 위계.
  · social distance in prosociality (Sul et al. 2015 등): Self boundary·거리가중
    λ_base 형성.
  · Distributional RL (Dabney et al. 2020 등): quantile 표현으로 기대보상 분포를
    추적 — '낮아졌다'와 '불확실해졌다'를 구분(§3.2).

모든 기본값은 명세서 §10 "구현 결정 잠금표" 를 따른다. 상수는 코드에 박지 않고
명명 상수/기본 인자로 노출한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

# =====================================================================§10 상수
# 구현 결정 잠금표 (명세서 §10). 값은 초기 제안이며 명명 상수로 노출한다.
D_BOUNDARY_DEFAULT = 1.0          # Self boundary 거리 임계
TAU_DIST_DEFAULT = 0.5            # 거리 커널 폭
LAMBDA_FLOOR_DEFAULT = 0.1        # λ_base 하한
LAMBDA_CEIL_DEFAULT = 0.7         # λ_base 상한
LAMBDA_MAX_DEFAULT = 0.8          # λ 상한
W_ADMIT_DEFAULT = 40.0            # 먼 Other 편입 누적 상호작용 임계(라운드)
ETA_SLOW_DEFAULT = 0.02           # 느린 갱신 학습률
M_QUANTILE_DEFAULT = 21           # distributional RL expectile 개수(격자)
ALPHA_DIST_DEFAULT = 0.1          # distributional RL 학습률
G_DISP_DEFAULT = 0.6              # [DEPRECATED] 구 dispositional λ 이득(legacy 경로)
G_CTX_DEFAULT = 0.2               # [DEPRECATED] 구 contextual λ 이득(legacy 경로)
BETA_REF_DEFAULT = 4.0            # [DEPRECATED] 구 β 스위치 기준(legacy 경로)
HUBER_KAPPA_DEFAULT = 1.0         # [DEPRECATED] 구 quantile Huber κ(legacy 경로)

# [v0.9.3 재정초] valence+uncertainty affect 로 λ 조절 (§3.5 개정).
#   기대보상 분포(expectile code)의 두 모먼트가 직접 λ 를 구동한다:
#     V = (E[r] − E0)/(R−P)         valence  — setpoint 함의 기대 대비 부호
#     U = downside_semidev/σ_ref    uncertainty — 하방 반편차(방어 비대칭 내생)
#     Δλ = k_V·V − k_U·U
#   구 β-deficit·q_z 귀인 게이팅을 폐기(affect_mode="valence_uncertainty" 기본).
CODE_EXPECTILE = "expectile"      # 분포 코드: expectile(신경근거, Dabney 2020) 기본
CODE_QUANTILE = "quantile"        # legacy quantile+Huber A/B
K_VALENCE_DEFAULT = 0.58          # valence λ 이득 (실측 최소자승 캘리브레이션, v0.10.0)
K_UNCERTAINTY_DEFAULT = 0.24      # 하방 불확실성 λ 이득 (방어는 U 항이 주도)
# [v0.11.0 §θ-잔차] θ-설명 잔차 affect + uncertainty 게이팅 Δλ.
#   Δλ = k_affect · V · (1 − U_norm)  — V 부호가 방향, U 가 확신도 게이트.
#   k_U 제거(단일 스케일). V=θ 설명 후 잔여 RPE mean, U=θ 입자간(epistemic) std.
K_AFFECT_DEFAULT = 0.7            # 잔차 valence→Δλ 단일 스케일(게이팅형)
U_GATE_REF_DEFAULT = 0.5          # U 게이트 정규화 기준 = frac·(R−P)
SIGMA_REF_FRAC_DEFAULT = 0.5      # σ_ref = frac·(R−P) (U 정규화 기준)

# 발달적 사전(§10): Self(d=0, p_noncoop=0.2) + 먼 Other 3개.
#   각 엔트리: (distance, p_noncoop, contextual_share)
#   contextual_share 는 그 Other 에 대한 불확실성이 맥락에 귀인되는 정도.
DEV_PRIOR_ENTRIES = (
    # (identity, distance, p_noncoop, contextual_share)
    (0, 0.0, 0.20, 0.30),         # Self
    (1, 0.5, 0.30, 0.40),
    (2, 0.8, 0.40, 0.50),
    (3, 1.5, 0.50, 0.60),
)

# 중간-앵커 Other 구조 (target_lambda_base 캘리브레이션 전용, §10 개정).
#   목적: floor/ceil 상수를 제거(항등 사상)하고, λ_base 의 경계·진폭을 **거리
#   구조 자체**가 내생적으로 결정하게 한다. Self(d=0)가 setpoint 를 지배(자기 EFE
#   우선)하되, 현행 DEV(0.5/0.8/1.5)보다 조금 더 민 Other 들이 setpoint 를 소폭
#   흔든다(drift 대역폭). 최근접 Other 는 boundary(≤1.0) 내에 유지 → §3.6 조기보정·
#   §6.2 boundary-공감 경로 보존.
#   각 항: (identity, distance, coop=1−p_noncoop, contextual_share)
MID_ANCHOR_OTHERS = (
    (1, 0.9, 0.70, 0.40),         # 최근접 — boundary 내(§3.6 매칭 후보 유지)
    (2, 1.3, 0.60, 0.50),
    (3, 1.9, 0.50, 0.60),
)


def _phi(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """가우시안 밀도 (수치 안정, 벡터화)."""
    sigma = max(float(sigma), 1e-6)
    z = (np.asarray(x, float) - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def _INV_PHI(p: np.ndarray) -> np.ndarray:
    """표준정규 역CDF Φ⁻¹ 근사(Acklam) — 초기 격자 형태 부여용(벡터화, 무SciPy)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.zeros_like(p)
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo])) if lo.any() else np.array([])
    if lo.any():
        out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return out


# =====================================================================SelfModel
@dataclass
class SelfEntry:
    """SelfModel 의 agent 엔트리(§2.1)."""
    identity: int
    distance: float                       # social distance ≥ 0 (Self=0)
    p_noncoop: float                      # dispositional 비협력률 ∈ [0,1]
    contextual_share: float               # 불확실성의 맥락 귀인 정도 ∈ [0,1]
    weight: float = 0.0                   # 누적 상호작용량(갱신 관성)
    dynamic: bool = False                 # [§3.6 재정의] 세션 중 형성된 identity 엔트리
    n_obs: float = 0.0                    # 그 identity 와의 누적 상호작용 라운드
    learned_reward: Optional[tuple] = None  # [v0.9.3] 그 상대에 대해 학습한 (E,σ)
    theta_coop: Optional[float] = None      # [v0.11.1] θ-예측 협력확률 ĉ_θ (즉시 반영)
    theta_lambda_j: Optional[float] = None  # [v0.11.1] 상대가 나를 향한 공감 λ̂_j


# [§3.6 재정의] identity 관계형성 상수.
#   identity 는 유사성 추론이 아니라 **관측 가능한 표지**다. 새 상대의 identity 는
#   boundary 밖(D_INIT)에서 시작하고, 반복 상호작용으로 거리가 좁혀져 W_admit
#   라운드에 boundary 를 넘으며(편입), 이후 D_FINAL 까지 접근한다. 편입된 엔트리에
#   그 상대에 대해 학습한 잠재 z(p_noncoop, contextual_share)가 저장된다.
D_INIT_IDENTITY_DEFAULT = 1.6     # [v0.11.1] 낯선 identity 시작 거리(먼 타자)
D_MIN_IDENTITY_DEFAULT = 0.5      # [v0.11.1] 친숙화 포화 하한 거리
TAU_FAMILIARITY_DEFAULT = 60.0    # [v0.11.1] 거리 지수포화 시상수(trial 수 only)
ETA_IDENTITY_DEFAULT = 0.05       # per-identity 보조 EMA(맥락지분 등) 학습률

# [v0.11.1] λ_base = (1−w_cd)·λ_baseSelf + w_cd·λ_baseContext(θ_j)
#   w_cd  : focal agent 고유 context-dependency(맥락 민감성 개인특질).
#   m_recip: λ_baseContext 상호성 혼합 — ĉ_θ 와 λ̂_j(상대의 나를 향한 공감) 가중.
W_CD_DEFAULT = 0.5                # 맥락 의존성(0=순수 self trait, 1=순수 상대맞춤)
M_RECIPROCAL_DEFAULT = 0.35       # 상호성 혼합비(λ̂_j 반영 정도)


class SelfModel:
    """
    느린 계층 — λ 의 setpoint(λ_base)를 형성하는 발달적 사회환경 믿음(§2).

    · Self + social distance·trait 를 지닌 Other 들의 집합을 유지.
    · Self boundary(distance ≤ D_boundary) 내 Other 만 focal 행위선택의 G_social 에
      λ 가중으로 진입(§6.2 요구는 CoreAffect/agent 가 소비).
    · λ_base = 거리 역가중 trait 집계(§2.2). 가까울수록 지배적.
    · IPD 중 갱신은 **오직 boundary 밖 먼 Other 만**, 누적 weight 가 W_admit 를
      넘은 뒤에만, η_slow 로 사소하게(§2.3) — allostatic setpoint drift.
    """

    def __init__(self,
                 entries: Optional[List[Tuple]] = None,
                 d_boundary: float = D_BOUNDARY_DEFAULT,
                 tau_dist: float = TAU_DIST_DEFAULT,
                 lambda_floor: float = LAMBDA_FLOOR_DEFAULT,
                 lambda_ceil: float = LAMBDA_CEIL_DEFAULT,
                 w_admit: float = W_ADMIT_DEFAULT,
                 eta_slow: float = ETA_SLOW_DEFAULT,
                 enable_drift: bool = True,
                 target_lambda_base: Optional[float] = None,
                 anchor_others: Optional[List[Tuple]] = None,
                 w_cd: float = W_CD_DEFAULT,
                 m_reciprocal: float = M_RECIPROCAL_DEFAULT):
        self.w_cd = float(w_cd)                 # [v0.11.1] context-dependency 특질
        self.m_reciprocal = float(m_reciprocal)  # [v0.11.1] 상호성 혼합비
        self.d_boundary = float(d_boundary)
        self.tau_dist = float(tau_dist)
        self.lambda_floor = float(lambda_floor)
        self.lambda_ceil = float(lambda_ceil)
        self.w_admit = float(w_admit)
        self.eta_slow = float(eta_slow)
        self.enable_drift = bool(enable_drift)

        if target_lambda_base is not None:
            # [§10 개정] Self-앵커 + 항등 사상 재정식화. floor/ceil 상수 제거,
            # 경계는 거리 구조가 전담. Self 가 setpoint 를 앵커(자기 EFE 우선),
            # Other 거리가 흔들림 대역폭을 정한다. entries 를 명시하면 그 구조
            # 위에서 Self coop 만 푼다; 없으면 중간-앵커 기본 구조를 쓴다.
            self.entries = self._build_anchored(
                float(target_lambda_base),
                explicit_entries=entries,
                anchor_others=anchor_others)
        else:
            # 기본 경로 — 현행 affine 사상·DEV prior 그대로(하위호환·기준선 불변).
            src = DEV_PRIOR_ENTRIES if entries is None else entries
            self.entries = [
                SelfEntry(identity=int(e[0]), distance=float(e[1]),
                          p_noncoop=float(np.clip(e[2], 0.0, 1.0)),
                          contextual_share=float(np.clip(e[3], 0.0, 1.0)))
                for e in src
            ]

        # 먼-Other 후보 누적 통계 (아직 편입되지 않은 상대의 관측 축적)
        self._pending_weight = 0.0
        self._pending_noncoop_sum = 0.0

    def _build_anchored(self, target: float,
                        explicit_entries: Optional[List[Tuple]] = None,
                        anchor_others: Optional[List[Tuple]] = None
                        ) -> List["SelfEntry"]:
        """
        Self-앵커 항등 사상으로 λ_base=target 을 **폐형해**로 정확히 맞춘다(§10 개정).

        항등 사상(floor=0, ceil=1)에서
            λ_base = [coop_self + Σ_oth w·coop_oth] / [1 + W_oth],  W_oth = Σ_oth w
        이므로 Self 의 협력성만 미지수로 두면 닫힌 해:
            coop_self = target·(1 + W_oth) − Σ_oth w·coop_oth
        Self(d=0)의 가중치 1 이 Other 총합 W_oth 를 지배(자기 EFE 우선)하고, Other
        거리가 W_oth(=흔들림 진폭)를 정한다. 해가 [0,1] 밖이면 그 Other 구조로는
        target 도달 불가 → ValueError(조용한 근사 금지; Other 거리 조정 안내).
        """
        # 항등 사상 강제 — 경계 상수 제거.
        self.lambda_floor, self.lambda_ceil = 0.0, 1.0

        # Other 구조 결정: 명시 entries(비-Self) > anchor_others 인자 > 중간앵커 기본.
        if explicit_entries is not None:
            others = [(int(e[0]), float(e[1]),
                       1.0 - float(np.clip(e[2], 0.0, 1.0)),   # p_noncoop→coop
                       float(np.clip(e[3], 0.0, 1.0)))
                      for e in explicit_entries if int(e[0]) != 0]
        else:
            src = MID_ANCHOR_OTHERS if anchor_others is None else anchor_others
            others = [(int(o[0]), float(o[1]),
                       float(np.clip(o[2], 0.0, 1.0)),          # coop 직접
                       float(np.clip(o[3], 0.0, 1.0))) for o in src]

        w = np.array([np.exp(-d / self.tau_dist) for _, d, _, _ in others])
        coop_oth = np.array([c for _, _, c, _ in others])
        w_oth = float(w.sum())
        coop_self = float(target) * (1.0 + w_oth) - float((w * coop_oth).sum())
        if not (-1e-9 <= coop_self <= 1.0 + 1e-9):
            raise ValueError(
                f"target λ_base={target} 는 현 Other 구조(W_oth={w_oth:.3f}, "
                f"Σw·coop={float((w*coop_oth).sum()):.3f})로 도달 불가 — 필요한 "
                f"coop_self={coop_self:.3f}∉[0,1]. Other 거리(anchor_others)를 조정하라.")
        coop_self = float(np.clip(coop_self, 0.0, 1.0))

        entries = [SelfEntry(identity=0, distance=0.0,
                             p_noncoop=1.0 - coop_self, contextual_share=0.30)]
        for ident, dist, c, cs in others:
            entries.append(SelfEntry(identity=ident, distance=dist,
                                     p_noncoop=1.0 - c, contextual_share=cs))
        return entries

    # ---------------------------------------------------------- boundary/λ_base
    def boundary_entries(self) -> List[SelfEntry]:
        """Self boundary(distance ≤ D_boundary) 내 엔트리(§2.1)."""
        return [e for e in self.entries if e.distance <= self.d_boundary + 1e-9]

    def lambda_base_self(self) -> float:
        """
        [v0.11.1] λ_baseSelf — **발달 엔트리만**의 거리가중 협력성 집계.
        상대와 무관한 자기 고유 공감 성향(trait setpoint). 동적 identity 엔트리는
        제외한다(그쪽은 λ_baseContext 가 전담) — 이로써 '거리 소멸 → 방어 자기제한'
        문제가 원천 제거된다.
        """
        devs = [e for e in self.entries if not e.dynamic]
        if not devs:
            devs = self.entries
        w = np.array([np.exp(-e.distance / self.tau_dist) for e in devs])
        coop = np.array([1.0 - e.p_noncoop for e in devs])
        denom = float(w.sum())
        frac = 0.5 if denom <= 1e-12 else float(np.sum(w * coop) / denom)
        return float(self.lambda_floor
                     + (self.lambda_ceil - self.lambda_floor) * frac)

    def lambda_base_context(self, ident: Optional[int] = None) -> Optional[float]:
        """
        [v0.11.1] λ_baseContext(θ_j) — 현재 상대 identity 의 θ 에 맞춤형 setpoint.

        **상호적(reciprocal) 사상**: 상대의 예측 협력확률 ĉ_θ 와, 상대가 나를 향해
        갖는 공감 λ̂_j 를 함께 반영한다("나에게 공감하는 이에게 공감한다"):

            base_j = (1 − m_recip)·ĉ_θ + m_recip·λ̂_j
            λ_ctx  = λ_floor + (λ_ceil − λ_floor)·base_j

        거리와 **무관**하게 θ 를 직접 반영하므로, 예측된 착취자에 대해 setpoint 가
        즉시 낮아진다(anticipatory 방어 — affect 가 θ 로 설명돼 침묵해도 작동).
        미등록·θ 미기록이면 None(→ 호출측에서 self 로 fallback).
        """
        if ident is None:
            return None
        e = self.entry_for_identity(int(ident))
        if e is None or e.theta_coop is None:
            return None
        c_hat = float(np.clip(e.theta_coop, 0.0, 1.0))
        lam_j = e.theta_lambda_j
        if lam_j is None:
            base_j = c_hat
        else:
            m = float(self.m_reciprocal)
            base_j = (1.0 - m) * c_hat + m * float(np.clip(lam_j, 0.0, 1.0))
        return float(self.lambda_floor
                     + (self.lambda_ceil - self.lambda_floor) * base_j)

    def lambda_base(self, ident: Optional[int] = None) -> float:
        """
        [v0.11.1] self/context 중재 setpoint (§2.2 개정).

            λ_base = (1 − w_cd)·λ_baseSelf + w_cd·λ_baseContext(θ_j)

        w_cd = focal agent 고유의 **context-dependency**(맥락 민감성 개인특질).
        상대 θ 정보가 없으면 λ_baseSelf 로 자연 축약된다.
        """
        lam_self = self.lambda_base_self()
        lam_ctx = self.lambda_base_context(ident)
        if lam_ctx is None:
            return lam_self
        w = float(np.clip(self.w_cd, 0.0, 1.0))
        return float((1.0 - w) * lam_self + w * lam_ctx)

    # ---------------------------------------------------------- identity 매칭
    def match_entry(self, disp_prob: float,
                    within_boundary: bool = True) -> Optional[SelfEntry]:
        """
        추론된 dispositional 배신확률에 가장 가까운 boundary 내 엔트리(§3.6, §10).

        다이애드에서는 명시 ID 가 없으므로 θ̂ 근접(p_noncoop 최근접)으로 식별한다.
        boundary 밖/미매칭이면 None → CoreAffect 는 중립 prior 를 쓴다.
        """
        cand = self.boundary_entries() if within_boundary else self.entries
        cand = [e for e in cand if e.identity != 0]     # Self 는 상대 매칭 대상 아님
        if not cand:
            return None
        d = float(np.clip(disp_prob, 0.0, 1.0))
        best = min(cand, key=lambda e: abs(e.p_noncoop - d))
        # 근접 임계: 차이가 0.35 를 넘으면 미매칭으로 간주(중립 prior)
        return best if abs(best.p_noncoop - d) <= 0.35 else None

    # ---------------------------------------------------------- 느린 갱신(§2.3)
    def slow_update(self, disp_prob: float) -> None:
        """
        boundary 밖 먼 Other 의 반복 상호작용을 누적한다. 누적 weight 가 W_admit 를
        넘으면 **비로소** 먼 엔트리의 p_noncoop 를 η_slow 로 소폭 갱신 → λ_base drift.

        boundary 내 엔트리(Self 포함)의 trait 는 IPD 중 변하지 않는다(발달 고정).
        """
        if not self.enable_drift:
            return
        self._pending_weight += 1.0
        self._pending_noncoop_sum += float(np.clip(disp_prob, 0.0, 1.0))
        if self._pending_weight < self.w_admit:
            return
        # 임계 도달 — 누적 관측 비협력률로 '가장 먼' 엔트리를 소폭 이동.
        obs_noncoop = self._pending_noncoop_sum / max(self._pending_weight, 1e-9)
        far = max((e for e in self.entries if e.distance > self.d_boundary + 1e-9),
                  key=lambda e: e.distance, default=None)
        if far is not None:
            far.p_noncoop = float(np.clip(
                (1 - self.eta_slow) * far.p_noncoop + self.eta_slow * obs_noncoop,
                0.0, 1.0))
            far.weight += self._pending_weight
        # 창을 리셋(누적을 소진) — 다음 W_admit 라운드 후 다시 갱신.
        self._pending_weight = 0.0
        self._pending_noncoop_sum = 0.0

    # ------------------------------------------- [§3.6 재정의] identity 레지스트리
    def implied_reward_belief(self, R: float, P: float) -> Tuple[float, float]:
        """
        [지시 2] 사회환경 믿음이 함의하는 초기 기대보상 (E0, spread).

        중심 E0 = P + λ_base·(R−P): λ_base(=사회환경 협력성 집계)를 기대보수로 사상
        (협력적 환경 → 높은 기대, 험한 환경 → 낮은 기대). 퍼짐 spread 는 엔트리
        협력성의 거리가중 표준편차를 보수 스케일로 환산 — 사회환경이 일관되면 좁고
        이질적이면 넓다. 이것이 CoreAffect 분포의 무관측 초기 σ 를 공급한다.
        """
        lb = self.lambda_base()
        E0 = float(P) + lb * (float(R) - float(P))
        w = np.array([np.exp(-e.distance / self.tau_dist) for e in self.entries])
        coop = np.array([1.0 - e.p_noncoop for e in self.entries])
        wsum = float(w.sum()) if w.sum() > 0 else 1.0
        mean_coop = float((w * coop).sum() / wsum)
        var_coop = float((w * (coop - mean_coop) ** 2).sum() / wsum)
        spread = float(np.sqrt(max(var_coop, 0.0)) * (float(R) - float(P)))
        return E0, spread

    def entry_for_identity(self, ident: int) -> Optional[SelfEntry]:
        """관측된 identity 의 동적 엔트리(발달 엔트리와 네임스페이스 분리)."""
        for e in self.entries:
            if e.dynamic and e.identity == int(ident):
                return e
        return None

    def observe_identity(self, ident: int) -> SelfEntry:
        """
        identity 관측(§3.6 재정의). 미등록이면 **boundary 밖**(D_INIT)에 신규 등록.

        낯선 identity 의 초기 trait 는 현재 λ_base 에 **중립**이 되도록
        p_noncoop = 1 − λ_base 로 둔다(등록 자체가 setpoint 를 흔들지 않음 —
        이후 이동은 순수하게 그 상대에 대한 학습이 만든다). contextual_share 는
        무정보 0.5.
        """
        e = self.entry_for_identity(ident)
        if e is not None:
            return e
        e = SelfEntry(identity=int(ident),
                      distance=D_INIT_IDENTITY_DEFAULT,
                      p_noncoop=float(np.clip(1.0 - self.lambda_base(), 0.0, 1.0)),
                      contextual_share=0.5,
                      dynamic=True, n_obs=0.0)
        self.entries.append(e)
        return e

    def update_identity(self, ident: int, opp_defected: bool,
                        q_ctx: float, theta_coop: Optional[float] = None,
                        theta_lambda_j: Optional[float] = None) -> SelfEntry:
        """
        반복 trial 에 의한 관계 형성(§3.6, v0.11.1 개정).

        trial 에 따라 변하는 것은 두 가지, 그리고 **서로 독립**이다:
        · **θ 믿음(즉시)**: theta_coop/theta_lambda_j 를 입자필터 사후에서 그대로
          기록한다. EMA 평활을 **제거** — 입자필터가 이미 원리적 순차 베이지안
          추론기이므로 그 출력에 EMA 를 덧씌우는 것은 이중 평활이고 근거가 없다.
          (p_noncoop 는 λ_baseSelf 집계용 호환 필드로만 동기화.)
        · **social distance(친숙도)**: **오직 trial 수의 함수**. 상대의 협력확률·
          성향은 거리에 일절 관여하지 않는다 — 친숙도와 호오(好惡)의 분리.
              d(n) = D_min + (D_start − D_min)·exp(−n / τ_fam)
          '오래 본 착취자'는 가깝되(친숙) 신뢰하지 않는(λ_baseContext 낮음) 상태로
          표현된다. 성향에 따른 방어는 전적으로 λ_baseContext(θ)가 담당한다.

        발달 엔트리(dynamic=False)는 절대 갱신되지 않는다(발달 고정).
        """
        e = self.observe_identity(ident)
        e.n_obs += 1.0
        e.weight += 1.0
        # (1) θ 믿음 — 즉시 반영(EMA 없음)
        if theta_coop is not None:
            e.theta_coop = float(np.clip(theta_coop, 0.0, 1.0))
            e.p_noncoop = float(1.0 - e.theta_coop)     # 집계 호환 필드 동기화
        if theta_lambda_j is not None:
            e.theta_lambda_j = float(np.clip(theta_lambda_j, 0.0, 1.0))
        # 맥락 지분(진단·legacy prior 용)만 완만 EMA 유지
        eta = ETA_IDENTITY_DEFAULT
        e.contextual_share = float(np.clip(
            (1 - eta) * e.contextual_share + eta * float(q_ctx), 0, 1))
        # (2) social distance — trial 수만의 지수포화(성향 무관)
        e.distance = float(
            D_MIN_IDENTITY_DEFAULT
            + (D_INIT_IDENTITY_DEFAULT - D_MIN_IDENTITY_DEFAULT)
            * np.exp(-e.n_obs / max(TAU_FAMILIARITY_DEFAULT, 1e-6)))
        return e

    def snapshot(self) -> Dict[str, float]:
        return {
            "lambda_base": self.lambda_base(),
            "n_entries": float(len(self.entries)),
            "n_boundary": float(len(self.boundary_entries())),
            "far_p_noncoop": float(max(
                (e.p_noncoop for e in self.entries
                 if e.distance > self.d_boundary + 1e-9), default=0.0)),
        }


# ================================================================Distributional
class QuantileValue:
    """
    기대보상 분포의 quantile 표현(§3.2). QR-DQN 방식의 Huber quantile 회귀.
    · 고정 개수 M 격자 {z_1..z_M} 를 관측 r_obs 로 당긴다(학습률 α_dist).
    · **expectile code 기본**(Dabney 2020 신경근거: DAN tuning 이 heaviside(quantile)
      보다 bilinear(expectile)에 부합). η=0.5 expectile 이 정확히 평균이라 valence
      신호가 편향 없이 나온다. legacy quantile+Huber 는 code="quantile" 로 A/B.
    · categorical/C51 대신 expectile/quantile — 지지구간 무가정(가변 페이오프 강건).
    · 출력: E[r](valence), σ[r], **하방 반편차**(downside semi-deviation) — 방어
      비대칭을 분포 하방구조에서 유도(§3.5 재정초).
    """

    def __init__(self, m: int = M_QUANTILE_DEFAULT,
                 alpha: float = ALPHA_DIST_DEFAULT,
                 kappa: float = HUBER_KAPPA_DEFAULT,
                 init_value: float = 0.0,
                 code: str = CODE_EXPECTILE,
                 init_center: Optional[float] = None,
                 init_spread: float = 0.0):
        self.M = int(m)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.code = str(code)
        # 레벨 η_i(=τ_i) = (i+0.5)/M ∈ (0,1)
        self.taus = (np.arange(self.M) + 0.5) / self.M
        # [지시 2] SelfModel 함의 기대(init_center)와 퍼짐(init_spread)로 격자 배치.
        #   center 미지정 시 구 동작(단순 init_value)으로 하위호환.
        c = float(init_value if init_center is None else init_center)
        self._E0 = c                                  # setpoint 함의 기대(valence 기준)
        # η 격자를 center 중심으로 대칭 배치: z_i = c + spread·Φ⁻¹근사(η_i)
        # (등간격 레벨의 표준정규 분위로 초기 형태 부여 — 무관측시 σ≈spread)
        zscore = _INV_PHI((np.arange(self.M) + 0.5) / self.M)
        self.z = c + float(init_spread) * zscore

    def update(self, r_obs: float) -> None:
        """관측 r_obs 로 각 격자점을 비대칭 회귀로 당긴다(expectile 기본)."""
        r = float(r_obs)
        u = r - self.z                                # (M,) 오차
        w = np.abs(self.taus - (u < 0).astype(float)) # 비대칭 가중 |η−1{u<0}|
        if self.code == CODE_QUANTILE:
            # legacy: quantile Huber — 부호만(크기 clip)
            grad = np.clip(u, -self.kappa, self.kappa)
        else:
            # expectile: 오차 크기 u 를 곱함(L2 비대칭) — η=0.5 → 평균
            grad = u
        self.z = self.z + self.alpha * w * grad

    @property
    def mean(self) -> float:
        """
        기대보상 E[r]. **expectile code 에서는 η=0.5 expectile 이 정확히 평균**이므로
        중앙 격자점을 읽는다(격자 산술평균은 치우친 분포서 편향). quantile code 는
        격자평균이 ∫F⁻¹≈E[r] 의 리만근사라 그대로 사용.
        """
        if self.code == CODE_QUANTILE:
            return float(self.z.mean())
        mid = self.M // 2                       # M 홀수 → τ=(mid+0.5)/M=0.5 정확
        return float(self.z[mid])

    @property
    def downside_std(self) -> float:
        """하위 절반 격자의 퍼짐."""
        lower = self.z[: max(self.M // 2, 1)]
        return float(np.std(lower))

    @property
    def downside_semidev(self) -> float:
        """
        [§3.5 재정초] 하방 반편차 √(E[(E0−r)_+²]) 근사 — 기대 E0 아래로 벌어진
        격자만의 RMS 편차. 방어 비대칭(나쁜 쪽 불확실성)을 대칭 σ 대신 이 값으로
        재므로, 구 g_disp>g_ctx 비대칭이 분포 하방구조에서 내생적으로 유도된다.
        """
        below = np.clip(self._E0 - self.z, 0.0, None)   # E0 아래로 벌어진 폭만
        return float(np.sqrt(np.mean(below ** 2)))

    @property
    def std(self) -> float:
        return float(np.std(self.z))


# ==================================================================CoreAffect
class CoreAffect:
    """
    빠른 계층 — RPE 귀인 기반 λ 동적 조절(§3).

    사건 = 기대보상에 대한 부적 RPE(§3.1). Distributional RL 로 기대보상 분포를
    추적(§3.2)하고, prior 를 지닌 베이지안 belief 로 RPE 를 {contextual,
    dispositional} 에 귀인(§3.3~3.4)하며, λ 를 belief 에서 직접 사상한다(§3.5).

    적분기 상태변수가 없다 — λ_k = clip(λ_base + Δλ_drive, 0, λ_max).
    """

    def __init__(self,
                 self_model: SelfModel,
                 payoffs: Tuple[float, float, float, float] = (3.0, 5.0, 0.0, 1.0),
                 lambda_max: float = LAMBDA_MAX_DEFAULT,
                 m_quantile: int = M_QUANTILE_DEFAULT,
                 alpha_dist: float = ALPHA_DIST_DEFAULT,
                 g_disp: float = G_DISP_DEFAULT,
                 g_ctx: float = G_CTX_DEFAULT,
                 beta_ref: float = BETA_REF_DEFAULT,
                 belief_forget: float = 0.90,
                 regulate: bool = True,
                 affect_mode: str = "theta_residual",
                 code: str = CODE_EXPECTILE,
                 k_valence: float = K_VALENCE_DEFAULT,
                 k_uncertainty: float = K_UNCERTAINTY_DEFAULT,
                 sigma_ref_frac: float = SIGMA_REF_FRAC_DEFAULT,
                 k_affect: float = K_AFFECT_DEFAULT,
                 u_gate_ref_frac: float = U_GATE_REF_DEFAULT):
        self.self_model = self_model
        self.lambda_max = float(lambda_max)
        self.g_disp = float(g_disp)                 # legacy 경로용
        self.g_ctx = float(g_ctx)                   # legacy 경로용
        self.beta_ref = float(beta_ref)             # legacy 경로용
        self.belief_forget = float(belief_forget)
        self.regulate = bool(regulate)
        # [v0.9.3 재정초] affect_mode: "valence_uncertainty"(기본) vs "legacy_attrib".
        self.affect_mode = str(affect_mode)
        self.code = str(code)
        self.k_valence = float(k_valence)
        self.k_uncertainty = float(k_uncertainty)
        self.sigma_ref_frac = float(sigma_ref_frac)
        self.k_affect = float(k_affect)
        self.u_gate_ref_frac = float(u_gate_ref_frac)
        self.m_quantile = int(m_quantile)
        self.alpha_dist = float(alpha_dist)
        self.set_payoffs(payoffs)

        # [지시 2] 기대보상 분포를 **SelfModel 함의 기대**로 초기화(협력적 환경→
        # 높은 기대, 험한 환경→낮은 기대). 중심 E0·퍼짐 spread 모두 사회믿음에서.
        self.value = self._make_value()

        # 원인 belief q(z)(legacy_attrib 경로에서만 사용)
        self.q_z = np.array([0.5, 0.5])
        self._prior_z = np.array([0.5, 0.5])

        # 진단용 최근 상태
        self.last_rpe = 0.0
        self.last_deficit = 0.0
        self.last_delta = 0.0
        self.last_valence = 0.0
        self.last_uncertainty = 0.0
        self.lam = self.self_model.lambda_base()
        self.lam_base = self.lam

    def _make_value(self) -> "QuantileValue":
        """SelfModel 함의 (E0, spread)로 분포 추적기 구성(§3.5 재정초·지시 2).

        legacy_attrib 경로는 구 골든 재현을 위해 R-init(퍼짐 0)로 되돌린다.
        """
        if self.affect_mode == "legacy_attrib":
            return QuantileValue(m=self.m_quantile, alpha=self.alpha_dist,
                                 code=self.code, init_value=self.R)
        E0, spread = self.self_model.implied_reward_belief(self.R, self.P)
        return QuantileValue(m=self.m_quantile, alpha=self.alpha_dist,
                             code=self.code, init_center=E0, init_spread=spread)

    # ------------------------------------------------------------ 보수/생성모형
    def set_payoffs(self, payoffs: Tuple[float, float, float, float]) -> None:
        """생성모형 P(RPE|z) 의 모수를 현재 보수 (R,T,S,P) 로부터 잠금(§10)."""
        self.R, self.T, self.S, self.P = (float(x) for x in payoffs)
        # P(RPE|contextual)  = N(0, σ_c²),  σ_c = 1.5·|S − P|
        self.sigma_c = 1.5 * abs(self.S - self.P)
        if self.sigma_c < 1e-6:
            self.sigma_c = 1.5
        # P(RPE|dispositional) = N(μ_d, σ_d²), μ_d = −(T − R)/2, σ_d = 0.5·σ_c
        self.mu_d = -(self.T - self.R) / 2.0
        self.sigma_d = 0.5 * self.sigma_c
        # deficit 정규화 기준: 협력 기대 R 에서 상호배신 바닥 P 까지.
        self._ref_high = self.R
        self._ref_low = self.P

    # ------------------------------------------------------------ prior(§3.6)
    def set_prior_from_identity(self, ident: Optional[int]) -> None:
        """
        [§3.6 재정의] 상대의 **관측된 identity** 가 boundary 안에 편입되어
        있으면(반복 trial 로 형성된 관계), 그 identity 에 저장된 잠재 z
        (contextual_share)로 q(z) prior 를 편향한다. 미편입/미등록이면 중립.
        유사성 매칭이 아니라 identity 동일성 — 그 특정 상대와의 저장된 경험이다.
        """
        if ident is None:
            self._prior_z = np.array([0.5, 0.5])
            return
        e = self.self_model.entry_for_identity(int(ident))
        if e is not None and e.distance <= self.self_model.d_boundary + 1e-9:
            c = float(np.clip(e.contextual_share, 0.05, 0.95))
            self._prior_z = np.array([c, 1.0 - c])
        else:
            self._prior_z = np.array([0.5, 0.5])

    def reset_for_partner(self, ident: Optional[int]) -> None:
        """
        [v0.9.3 재정초] 재조우 즉시 보정 — 저장 대상이 q_z 가 아니라 **기대보상
        분포(E,σ)** 다(지시 3·identity 재배선). 편입된 identity 를 다시 만나면
        value 추적기를 그 상대에 대해 학습해 둔 (E,σ)로 재초기화한다 — '아는
        상대는 학습된 기대분포로 시작'. 미편입/미등록/미저장이면 무보정.
        """
        if ident is None:
            return
        e = self.self_model.entry_for_identity(int(ident))
        if e is None or e.distance > self.self_model.d_boundary + 1e-9:
            return
        er = getattr(e, "learned_reward", None)
        if er is not None:
            E_star, sig_star = float(er[0]), float(er[1])
            self.value = QuantileValue(
                m=self.m_quantile, alpha=self.alpha_dist, code=self.code,
                init_center=E_star, init_spread=sig_star)
        # legacy_attrib 경로 호환: q_z 도 저장 c 로(있으면).
        if self.affect_mode == "legacy_attrib":
            c = float(np.clip(e.contextual_share, 0.05, 0.95))
            self.q_z = np.array([c, 1.0 - c])

    def _store_reward_belief(self, ident: Optional[int]) -> None:
        """편입된 identity 엔트리에 현재 학습된 기대보상 분포(E,σ)를 저장."""
        if ident is None:
            return
        e = self.self_model.entry_for_identity(int(ident))
        if e is not None:
            e.learned_reward = (float(self.value.mean), float(self.value.std))

    def set_prior_from_selfmodel(self, disp_prob: float) -> None:
        """
        [DEPRECATED — §3.6 재정의로 대체] 구 trait-근접 매칭 prior. identity 는
        유사성 추론이 아니므로 v0.9.2 부터 identity 경로가 표준. legacy A/B 용.
        """
        entry = self.self_model.match_entry(disp_prob, within_boundary=True)
        if entry is not None:
            c = float(np.clip(entry.contextual_share, 0.05, 0.95))
            self._prior_z = np.array([c, 1.0 - c])
        else:
            self._prior_z = np.array([0.5, 0.5])

    # ------------------------------------------------------------ θ-증거(§3.4)
    def _theta_evidence(self, inferred: dict, beta_sd: float = 0.0) -> np.ndarray:
        """
        입자필터 θ̂ 로부터 {contextual, dispositional} 우도 보강 증거를 산출한다.

        **명세서 §3.4 [확정] 규약** — 부호 정합(원안 −ρ/β 오류 교정):
          dispositional ∝ clip(−α,0) + clip(−ρ,0) + (1−λ_j) + |η|·[착취적]
          contextual    ∝ (1 − clip(β/β_ref,0,1)) + β-불확실성 + |ω| 자기일관 관성
          β 는 스위치: 높으면 나머지를 dispositional 로, 낮으면 contextual 로.
        반환: 정규화된 [ctx, disp] 우도(양수).
        """
        a = float(inferred.get("alpha", 0.0))
        rho = float(inferred.get("rho", 0.0))
        lam_j = float(np.clip(inferred.get("lambda_j", 0.5), 0.0, 1.0))
        beta = float(inferred.get("beta", 1.0))
        omega = float(inferred.get("omega", 0.0))
        eta = float(inferred.get("eta", 0.0))

        disp = (max(-a, 0.0)
                + max(-rho, 0.0)
                + (1.0 - lam_j)
                + abs(eta) * (1.0 if eta < 0.0 else 0.0))  # η<0 = 조건적 착취 규칙
        ctx = ((1.0 - float(np.clip(beta / self.beta_ref, 0.0, 1.0)))
               + float(np.clip(beta_sd / self.beta_ref, 0.0, 1.0))
               + abs(omega))

        # β 스위치: 정밀(높은 β)이면 dispositional 쪽으로 증거를 라우팅.
        precision = float(np.clip(beta / self.beta_ref, 0.0, 1.0))
        disp *= (0.5 + 0.5 * precision)
        ctx *= (0.5 + 0.5 * (1.0 - precision))

        ev = np.array([ctx, disp]) + 1e-6
        return ev / ev.sum()

    # ------------------------------------------------------------ deficit(§3.2)
    def _deficit(self) -> float:
        """
        기대보상 분포의 하락폭([0,1] 정규화). 협력 기대 R 에서 배신 바닥 P 까지의
        정규화 gap. 분포 평균이 낮을수록 deficit↑.
        """
        span = max(self._ref_high - self._ref_low, 1e-6)
        return float(np.clip((self._ref_high - self.value.mean) / span, 0.0, 1.0))

    # ------------------------------------------------------------ 한 라운드(§3)
    def step(self, r_pred: float, r_obs: float, inferred: dict,
             beta_sd: float = 0.0, identity_disp: Optional[float] = None,
             partner_identity: Optional[int] = None,
             opp_defected: Optional[bool] = None,
             theta_reward_mean: Optional[float] = None,
             theta_epistemic_std: Optional[float] = None,
             theta_coop: Optional[float] = None,
             theta_lambda_j: Optional[float] = None) -> dict:
        """
        한 라운드 λ 조절.

        Parameters
        ----------
        r_pred : 선택 행동의 pragmatic 기대보수 (§3.1 단일 r_pred 계약 — G_self 의
                 pragmatic 키와 동일 값. 별도 재계산 금지).
        r_obs  : 실현 보수 payoff_self(a_i, a_j).
        inferred : 입자필터 posterior means (α,ρ,β,λ_j[,ω,η]).
        beta_sd  : β 사후 표준편차(β-불확실성 → contextual 증거).
        identity_disp : [DEPRECATED legacy 경로] 구 trait-근접 매칭용 배신확률 힌트.
        partner_identity : [§3.6 재정의] 관측된 상대 identity. 제공되면 prior 는
                 identity 동일성으로, SelfModel 갱신은 관계형성(update_identity)으로.
        opp_defected : 이번 라운드 상대의 배신 여부(identity 잠재 z 학습용).
        """
        # ---- (0) SelfModel setpoint 를 예측신호로 수신(§2.4) ----
        self.lam_base = self.self_model.lambda_base(partner_identity)

        # ---- (1) RPE 사건(§3.1) ----
        rpe = float(r_obs) - float(r_pred)
        self.last_rpe = rpe

        # ---- (2) 기대보상 분포 갱신(§3.2, expectile code 기본) ----
        self.value.update(r_obs)
        deficit = self._deficit()
        self.last_deficit = deficit

        # ============= [v0.11.0 §θ-잔차] θ-설명 잔차 affect + U 게이팅 =========
        if self.affect_mode == "theta_residual":
            span = max(self.R - self.P, 1e-6)
            # (2') RPE 를 θ 로 잔차화: 관측보수에서 θ-조건부 기대보수를 회귀 제거.
            #   θ 예측이 없으면(초기·미제공) SelfModel setpoint E0 로 fallback.
            E_theta = (float(theta_reward_mean) if theta_reward_mean is not None
                       else float(getattr(self.value, "_E0", self.R)))
            epi_std = (float(theta_epistemic_std)
                       if theta_epistemic_std is not None else 0.0)
            # 잔여 RPE mean → valence, θ 입자간(epistemic) std → uncertainty.
            valence = float((r_obs - E_theta) / span)
            u_ref = max(self.u_gate_ref_frac * span, 1e-6)
            U = float(np.clip(epi_std / u_ref, 0.0, 1.0))
            self.last_valence, self.last_uncertainty = valence, U

            # (5') Δλ = k · V · (1 − U): V 부호=방향, U=확신도 게이트.
            #   확실(U↓)한 협력/배신 → |Δλ|↑ ; 불확실(U↑) → |Δλ|↓(조심).
            delta = self.k_affect * valence * (1.0 - U)
            self.last_delta = delta
            if self.regulate:
                self.lam = float(np.clip(self.lam_base + delta, 0.0, self.lambda_max))
            else:
                self.lam = float(self.lam_base)

            # (6') SelfModel 상향: identity θ 기록·거리 극완만 접근 (agent 가 θ 주입).
            if partner_identity is not None:
                self.self_model.update_identity(
                    int(partner_identity),
                    bool(opp_defected) if opp_defected is not None
                    else bool(valence < 0.0),
                    float(np.clip(0.5 + 0.5 * valence, 0.0, 1.0)),
                    theta_coop=theta_coop, theta_lambda_j=theta_lambda_j)
                self._store_reward_belief(int(partner_identity))
            else:
                disp_hint = float(1.0 - np.clip(inferred.get("lambda_j", 0.5), 0, 1))
                self.self_model.slow_update(disp_hint)

            return {
                "lam": self.lam, "lam_base": self.lam_base,
                "delta_lambda": delta, "rpe": rpe, "deficit": deficit,
                "valence": valence, "uncertainty": U,
                "theta_reward_mean": E_theta, "theta_epistemic_std": epi_std,
                "q_dispositional": float(np.clip(-valence, 0.0, 1.0)),
                "q_contextual": float(np.clip(valence, 0.0, 1.0)),
                "value_mean": self.value.mean,
                "value_downside_std": self.value.downside_std,
            }

        # ================= [v0.10.0] valence+uncertainty affect ================
        if self.affect_mode != "legacy_attrib":
            # 분포의 두 모먼트가 직접 λ 를 구동(§3.5 재정초). 귀인(q_z)·β-deficit
            # 게이팅 폐기. E0 = SelfModel 함의 기대(분포 초기중심).
            span = max(self.R - self.P, 1e-6)
            E0 = float(getattr(self.value, "_E0", self.R))
            valence = float((self.value.mean - E0) / span)          # V ∈ 대략 [−1,1]
            sigma_ref = max(self.sigma_ref_frac * span, 1e-6)
            uncertainty = float(self.value.downside_semidev / sigma_ref)  # U ≥ 0, 하방
            self.last_valence, self.last_uncertainty = valence, uncertainty

            delta = self.k_valence * valence - self.k_uncertainty * uncertainty
            self.last_delta = delta
            if self.regulate:
                self.lam = float(np.clip(self.lam_base + delta, 0.0, self.lambda_max))
            else:
                self.lam = float(self.lam_base)

            # (6) SelfModel 상향 갱신 — identity 관계형성 or legacy drift.
            #     opp 배신 여부는 관측(opp_defected) 우선, 없으면 음 valence 로 대용.
            if partner_identity is not None:
                self.self_model.update_identity(
                    int(partner_identity),
                    bool(opp_defected) if opp_defected is not None
                    else bool(valence < 0.0),
                    float(np.clip(0.5 + 0.5 * valence, 0.0, 1.0)))  # ctx 대용(양 valence=맥락)
                self._store_reward_belief(int(partner_identity))
            else:
                disp_hint = float(1.0 - np.clip(inferred.get("lambda_j", 0.5), 0, 1))
                self.self_model.slow_update(disp_hint)

            return {
                "lam": self.lam, "lam_base": self.lam_base,
                "delta_lambda": delta, "rpe": rpe, "deficit": deficit,
                "valence": valence, "uncertainty": uncertainty,
                # 하위호환 로그키(귀인 폐기 → valence 로 사상):
                "q_dispositional": float(np.clip(-valence, 0.0, 1.0)),
                "q_contextual": float(np.clip(valence, 0.0, 1.0)),
                "value_mean": self.value.mean,
                "value_downside_std": self.value.downside_std,
                "value_downside_semidev": self.value.downside_semidev,
            }
        # ================= legacy_attrib: 구 베이지안 귀인 경로(A/B) ===============

        # ---- (3) prior 편향(§3.6 재정의): identity 동일성 기반 조기 보정 ----
        disp_hint = (identity_disp if identity_disp is not None
                     else float(1.0 - np.clip(inferred.get("lambda_j", 0.5), 0, 1)))
        if partner_identity is not None:
            self.set_prior_from_identity(partner_identity)
        else:
            self.set_prior_from_selfmodel(disp_hint)   # legacy trait-근접 경로

        # ---- (4) 베이지안 belief 갱신(§3.3): RPE 우도 × θ-증거 ----
        lik_ctx = float(_phi(np.array([rpe]), 0.0, self.sigma_c)[0])
        lik_disp = float(_phi(np.array([rpe]), self.mu_d, self.sigma_d)[0])
        rpe_lik = np.array([lik_ctx, lik_disp]) + 1e-12
        theta_lik = self._theta_evidence(inferred, beta_sd=beta_sd)
        f = self.belief_forget
        post = (rpe_lik * theta_lik
                * (self.q_z ** f) * (self._prior_z ** (1.0 - f)))
        s = float(post.sum())
        self.q_z = post / s if s > 1e-300 else np.array([0.5, 0.5])
        q_ctx, q_disp = float(self.q_z[0]), float(self.q_z[1])

        # ---- (5) λ 사상(§3.5): belief 에서 직접, 적분기 없음 ----
        delta = (-self.g_disp * q_disp * deficit
                 + self.g_ctx * q_ctx * (1.0 - deficit))
        self.last_delta = delta
        if self.regulate:
            self.lam = float(np.clip(self.lam_base + delta, 0.0, self.lambda_max))
        else:
            self.lam = float(self.lam_base)

        # ---- (6) SelfModel 상향 갱신(§2.4): 관계형성 또는 legacy drift ----
        if partner_identity is not None:
            self.self_model.update_identity(
                int(partner_identity),
                bool(opp_defected) if opp_defected is not None
                else bool(rpe < 0.0),
                q_ctx)
        else:
            self.self_model.slow_update(disp_hint)

        return {
            "lam": self.lam,
            "lam_base": self.lam_base,
            "delta_lambda": delta,
            "rpe": rpe,
            "deficit": deficit,
            "q_dispositional": q_disp,
            "q_contextual": q_ctx,
            "value_mean": self.value.mean,
            "value_downside_std": self.value.downside_std,
        }

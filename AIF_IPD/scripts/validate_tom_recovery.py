#!/usr/bin/env python
"""
validate_tom_recovery.py — ToM 파라미터 복원 연구 (v0.6.4 전면 개정).

**양방향 복원**을 모두 검증한다.

  · 정방향(추정): 입자필터가 상정하는 것과 동일한 생성모형으로 θ=(α, ρ, β, λ_j)
    를 넓은 구간에서 무작위 표집한 합성 상대를 만들고, 상호작용 후 사후평균이
    참값을 얼마나 회복하는가 — 편향·RMSE·R²·회복기울기·90% 커버리지, 그리고
    **파라미터 혼동행렬** corr(참_i, 추정_j).
  · 역방향(생성 재현): 추정된 θ̂ 로 상대를 **재생성**했을 때 원래 상대와 같은
    행동(맥락별 협력확률)이 나오는가 — 사후예측 r·RMSE·커버리지.

[설계 배터리 — 회귀자 중심화]
로짓 = β·(α + ρ·f + empathy_shift(λ_j, p)),
       empathy_shift(λ,p) = (T−S)·λ + (R−T+P−S)·p + (S−P)
이므로 λ 의 회귀자는 **(T−S)** 이다. 기존 과제는 단일 맥락(T=5,S=0 고정)이라
(T−S)=5 가 **상수** → 로짓에서 λ 열이 절편(α)과 공선 → α·λ 개별 식별 불가.
(set_payoffs 는 R 만 바꾸므로 R-블록화로도 이 축퇴는 깨지지 않는다.)

  → `centered` 배터리: u=(T−S) ∈ {+2, 0, −2} 로 **중심화**(평균 0)하고, β 를
    식별하는 기지 오프셋 v = (R−T+P−S)p + (S−P) 를 u 와 **직교 요인설계**로 교차
    한다. 회귀에서 절편과 기울기 추정은 회귀자 평균이 0 일 때 무상관이 되므로
    중심화가 α·λ 혼동을 제거한다. 배터리는 식별을 위해 PD 가 아닌 2×2 게임
    (T<S 등)을 의도적으로 포함한다(복원 과제 전용 — 본 실험에는 미사용).

비교군으로 `legacy`(단일 맥락) 를 함께 돌려 개선을 문서화한다.

사용:
    python scripts/validate_tom_recovery.py                  # legacy + centered
    python scripts/validate_tom_recovery.py --quick
    python scripts/validate_tom_recovery.py --designs centered --agents 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from AIF_IPD.core.logging_utils import get_logger, set_korean_font
from AIF_IPD.core.constants import (
    empathy_shift as C_empathy_shift, set_payoff_matrix, reset_payoffs,
)
from AIF_IPD.ipd.tom.inversion import OpponentInversion, ObservationContext

LOGGER = get_logger("HalloReg.recovery")

RESULTS = _PKG_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

AXES = ("alpha", "rho", "beta", "lambda_j")
AX_LABEL = {"alpha": "α", "rho": "ρ", "beta": "β", "lambda_j": "λ_j",
            "omega": "ω", "eta": "η"}
# [v0.8.0 §7.3] fg 기저의 6축
AXES_FG = ("alpha", "rho", "omega", "eta", "beta", "lambda_j")

# θ 표집 구간 (복원 지표는 이 구간에 의존하므로 반드시 함께 보고한다)
THETA_RANGES = {"alpha": (-1.2, 1.2), "rho": (0.2, 1.6),
                "beta": (0.5, 6.0), "lambda_j": (0.05, 0.95),
                # [v0.8.0 §7.2/§7.3] ω(관성), η(결과-조건성). ρ 와 동급 스케일.
                # WSLS 류(큰 양의 η)를 포함하도록 ± 대칭 구간.
                "omega": (-1.2, 1.2), "eta": (-1.6, 1.6)}

# ---------------------------------------------------------------- 설계 배터리
# 블록 = (R, T, S, P, p_focal).  u=(T−S)=λ 회귀자,  v=(R−T+P−S)p+(S−P)=기지 오프셋
def _ctx(u: float, v: float, p: float = 0.5):
    """(u,v) → (R,T,S,P,p).  u=(T−S)∈{+2,0,−2} 는 (T,S)∈{(2,0),(1,1),(0,2)},
    P=1 고정. v=(R−T+P−S)·p+(S−P) 를 R 로 역산:
        R = (v−(S−P))/p + T − P + S      (p=0.5 에서 R = 2v + u + 1)
    """
    T, S = {2.0: (2.0, 0.0), 0.0: (1.0, 1.0), -2.0: (0.0, 2.0)}[float(u)]
    P = 1.0
    R = (v - (S - P)) / p + T - P + S
    return (R, T, S, P, p)


DESIGNS = {
    # 현행(비교군): 단일 맥락. u=5 상수 → α·λ 공선.
    "legacy": {"blocks": [(3.0, 5.0, 0.0, 1.0, 0.5)]},
    # v0.6.4: u ∈ {+2,0,−2} (평균 0, 중심화) × v ∈ {−0.8,+0.8} 직교 요인설계.
    "centered": {"blocks": [
        _ctx(+2.0, -0.8), _ctx(+2.0, +0.8),
        _ctx(0.0, -0.8), _ctx(0.0, +0.8),
        _ctx(-2.0, -0.8), _ctx(-2.0, +0.8),
    ]},
    # v0.6.5: 능동 설계 — 메뉴 u∈{+2,0,−2} × v∈{−1.6,−0.8,0,+0.8,+1.6} (15 맥락)
    # × focal 행동 f∈{±1} = 30 후보. 매 라운드 입자 사후로 (ρ,β) 기대 분산감소를
    # 계산해 최대 정보 자극을 선택(ε=0.15 탐색, 초기 2메뉴 라운드로빈 번인).
    #   · 기지 오프셋 v 의 계단(5수준): Δlogit=β·Δv 에서 Δv 가 알려져 있으므로
    #     β 가 βρ 곱이 아닌 **단독으로** 식별 → ρ=(βρ)/β 자동 해소.
    #   · 능동 선택: 예측 |logit| 이 포화 밖(정보 대역)에 머무는 자극을 우선
    #     — 큰 β 상대의 시그모이드 포화 문제를 회피(적응적 난이도).
    "adaptive": {"menu": [_ctx(u, v)
                          for u in (+2.0, 0.0, -2.0)
                          for v in (-1.6, -0.8, 0.0, +0.8, +1.6)],
                 "adaptive": True},
    # ---------------------------------------------------------------- v0.8.0 §7.3
    # fg 기저의 식별 설계. **경고(§7.3)**: g 는 상대 자신의 행동이라 직접 조작이
    # 불가능하다 — f 만 변조하는 기존 배터리로는 ω·η 가 식별되지 않는다.
    # 해결: 합성 상대이므로 **상대 행동열을 조건부 강제**할 수 있다. 매 라운드
    # (f, g) ∈ {±1}² 4셀을 균형 순회하도록 (i) focal 행동 f 를 설계하고
    # (ii) 상대의 '직전 행동' g 를 강제 주입한다. 강제된 g 는 우도의 조건일 뿐
    # 관측(상대의 이번 행동)은 여전히 상대 모형이 생성하므로 추론은 편향되지 않는다.
    #   · 설계행렬 (1, f, g, fg) full-rank + 중심화(각 열 평균 0) 달성.
    #   · v 계단(기지 오프셋)과 교차 → β 단독 식별 유지.
    #   · η 는 f·g 곱 회귀자이므로 f·g 주효과와 직교화 필요 → 4셀 균형이 이를 보장.
    "fg_centered": {
        "blocks": [_ctx(+2.0, -0.8), _ctx(+2.0, +0.8),
                   _ctx(0.0, -0.8), _ctx(0.0, +0.8),
                   _ctx(-2.0, -0.8), _ctx(-2.0, +0.8)],
        "fg_cells": True,      # (f,g) 4셀 균형 순회 + g 조건부 강제
        "basis": "fg"},
}

# 역방향(생성 재현) 공통 평가 맥락 — 설계 간 비교 가능하도록 고정
EVAL_CONTEXTS = DESIGNS["centered"]["blocks"]


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def design_diagnostics(design):
    """배터리의 λ 회귀자 u=(T−S) 와 기지 오프셋 v 의 평균·SD·상관(직교성 진단).

    [v0.8.0 §7.3] fg 설계에서는 다회귀자 진단으로 확장한다:
    (u, v, f-빈도, g-빈도, fg-균형). (1,f,g,fg) 설계행렬의 full-rank 와 중심화
    (각 열 평균 0)를 수치로 확인한다 — η 는 f·g 곱 회귀자라 주효과와 직교화가
    필수이며, 4셀 균형이 이를 보장한다.
    """
    blocks = design["menu"] if "menu" in design else design["blocks"]
    u = np.array([T - S for (_, T, S, _, _) in blocks], float)
    v = np.array([(R - T + P - S) * p + (S - P) for (R, T, S, P, p) in blocks],
                 float)
    corr = (float(np.corrcoef(u, v)[0, 1])
            if len(u) > 1 and u.std() > 1e-9 and v.std() > 1e-9 else 0.0)
    out = {"u_mean": float(u.mean()), "u_sd": float(u.std()),
           "v_mean": float(v.mean()), "v_sd": float(v.std()),
           "corr_u_v": corr, "n_blocks": len(blocks)}
    if design.get("fg_cells"):
        # 균형 순회 (f,g) 4셀: focal/opp 행동코드 0=C,1=D → 신호 ±1
        cells = np.array([(0, 0), (0, 1), (1, 0), (1, 1)], float)
        f = 1.0 - 2.0 * cells[:, 0]
        g = 1.0 - 2.0 * cells[:, 1]
        X = np.column_stack([np.ones(4), f, g, f * g])   # 설계행렬 (1,f,g,fg)
        out.update({
            "f_mean": float(f.mean()), "g_mean": float(g.mean()),
            "fg_mean": float((f * g).mean()),
            "corr_f_g": float(np.corrcoef(f, g)[0, 1]),
            "design_rank": int(np.linalg.matrix_rank(X)),
            "design_full_rank": bool(np.linalg.matrix_rank(X) == 4),
            # 중심화: 상수항을 제외한 각 열 평균이 0 이어야 함
            "centered": bool(np.allclose(X[:, 1:].mean(axis=0), 0.0)),
            "n_cells": 4,
        })
    return out


class SyntheticOpponent:
    """입자필터의 우도와 동일한 협력확률 모형으로 행동하는 합성 상대 (λ_j 자유).

    [v0.8.0 §7] fg 기저에서는 ω(관성)·η(결과-조건성)도 갖는다:
        P(C | f, g) = σ(β(α + ρf + ωg + η·fg + s(λ_j, p)))
    ω=η=0 이면 f 기저와 동일(하위호환).
    """

    def __init__(self, alpha, rho, beta, lambda_j, seed=0,
                 omega=0.0, eta=0.0):
        self.alpha, self.rho, self.beta = alpha, rho, beta
        self.lambda_j = lambda_j
        self.omega, self.eta = float(omega), float(eta)
        self.rng = np.random.default_rng(seed)
        self.my_coop_rate = 0.5          # focal 협력율 p (맥락 블록마다 설정)

    def coop_prob(self, focal_last, own_last=None):
        f = 0.0 if focal_last is None else (1.0 - 2.0 * focal_last)
        g = 0.0 if own_last is None else (1.0 - 2.0 * own_last)
        shift = C_empathy_shift(self.lambda_j, self.my_coop_rate)
        return float(_logistic(self.beta * (
            self.alpha + self.rho * f + self.omega * g + self.eta * f * g
            + shift)))

    def act(self, focal_last, own_last=None):
        return 0 if self.rng.random() < self.coop_prob(focal_last, own_last) else 1


def _wq(vals, weights, q):
    """가중 분위수 (입자 사후예측 구간용)."""
    idx = np.argsort(vals)
    v = np.asarray(vals, float)[idx]
    w = np.asarray(weights, float)[idx]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.percentile(v, 100 * q))
    c = c / c[-1]
    return float(np.interp(q, c, v))


def recover_once(theta, blocks, rounds, seed, n_particles=400):
    """
    한 합성 상대에 대해 설계 배터리를 순회하며 관측 후 사후를 반환.
    반환: (inv, means_final, stds_final, stds_initial)
    """
    opp = SyntheticOpponent(theta["alpha"], theta["rho"], theta["beta"],
                            theta["lambda_j"], seed=seed)
    inv = OpponentInversion(n_particles=n_particles, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    focal_last = None
    st0 = None
    per_block = max(1, rounds // len(blocks))
    for (R_, T_, S_, P_, p_) in blocks:
        set_payoff_matrix(R_, T_, S_, P_)     # 맥락 전환(상대·필터 모두 반영)
        opp.my_coop_rate = p_
        inv.my_cooperation_rate = p_
        for t in range(per_block):
            opp_a = opp.act(focal_last)
            ctx = ObservationContext(my_last_action=focal_last,
                                     their_last_action=None,
                                     joint_outcome=None, round_number=t)
            inv.update(opp_a, ctx)
            if st0 is None:
                st0 = dict(inv.posterior_stds())
            # focal 은 블록의 협력율 p_ 로 행동(외생 설계; 상대 지속성 혼입 차단)
            focal_last = int(rng.random() >= p_)
    return inv, dict(inv.posterior_means()), dict(inv.posterior_stds()), st0


def recover_once_fg(theta, blocks, rounds, seed, n_particles=600):
    """
    [v0.8.0 §7.3] **fg 기저 복원**: (f, g) 4셀 균형 순회 프로토콜.

    g(상대 자신의 직전 행동)는 직접 조작 불가하므로, 합성 상대에 한해 **조건부
    강제**한다: 매 라운드 설계가 지정한 (f, g) 셀에 대해
      · f = focal 의 직전 행동   → focal 행동을 설계값으로 강제
      · g = 상대의 직전 행동     → 상대에게 '네 직전 행동은 g 였다'고 주입
    상대의 **이번** 행동은 여전히 상대 모형이 P(C|f,g) 로 생성하므로 관측은
    오염되지 않는다 — 조작되는 것은 조건(설계)이지 반응이 아니다.

    설계행렬 (1, f, g, fg): 4셀을 균등 순회하면 f·g·fg 열 평균이 모두 0 → 중심화
    달성, 상호 직교 → full-rank. 여기에 v 계단(기지 오프셋)을 교차해 β 단독 식별.
    셀당 최소 ~30 관측 권장(§7.3) → rounds ≥ 480 검토.
    """
    opp = SyntheticOpponent(theta["alpha"], theta["rho"], theta["beta"],
                            theta["lambda_j"], seed=seed,
                            omega=theta.get("omega", 0.0),
                            eta=theta.get("eta", 0.0))
    inv = OpponentInversion(n_particles=n_particles, seed=seed + 1,
                            likelihood_basis="fg")
    st0 = None
    # (f, g) 4셀 × 블록(v 계단) 교차 — 균형 순회
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]     # (focal_last, opp_last) 행동코드
    per_block = max(1, rounds // len(blocks))
    t = 0
    for (R_, T_, S_, P_, p_) in blocks:
        set_payoff_matrix(R_, T_, S_, P_)     # 맥락 전환(상대·필터 모두 반영)
        opp.my_coop_rate = p_
        inv.my_cooperation_rate = p_
        for k in range(per_block):
            focal_last, opp_last = cells[k % 4]   # 균형 순회 (중심화 보장)
            opp_a = opp.act(focal_last, opp_last)
            # 우도 조건: f=focal_last, g=opp_last (설계가 아는 값)
            inv.update(opp_a, ObservationContext(
                my_last_action=focal_last, their_last_action=opp_last,
                joint_outcome=None, round_number=t))
            if st0 is None:
                st0 = dict(inv.posterior_stds())
            t += 1
    return inv, dict(inv.posterior_means()), dict(inv.posterior_stds()), st0


def _expected_var_reduction(inv, u_c, v_c, f_c, axes=("rho", "beta")):
    """
    후보 자극 (u,v,f) 배열에 대한 (ρ,β) 기대 사후분산 감소(정규화 합) — 능동 설계의
    획득함수. 입자 가중 갱신을 가상으로 수행(재표집 없음). 모두 벡터화 (N, C).
    """
    w = inv.weights
    args = (inv.alpha[:, None] + inv.rho[:, None] * f_c[None, :]
            + inv.lambda_j[:, None] * u_c[None, :] + v_c[None, :])
    pC = 1.0 / (1.0 + np.exp(-np.clip(inv.beta[:, None] * args, -30, 30)))
    pbar = w @ pC                                   # (C,) 예측 협력확률
    score = np.zeros(len(u_c))
    for ax in axes:
        x = getattr(inv, ax if ax != "lambda" else "lambda_j")
        m0 = float(w @ x)
        var0 = float(w @ (x - m0) ** 2) + 1e-12
        for lik, pobs in ((pC, pbar), (1.0 - pC, 1.0 - pbar)):
            w2 = w[:, None] * lik                   # (N,C)
            ssum = w2.sum(axis=0) + 1e-12
            w2 = w2 / ssum
            m1 = (w2 * x[:, None]).sum(axis=0)
            var1 = (w2 * (x[:, None] - m1) ** 2).sum(axis=0)
            score += pobs * (var0 - var1) / var0
    return score


def recover_once_adaptive(theta, menu, rounds, seed, n_particles=400,
                          eps=0.15):
    """
    **능동(adaptive) 복원**: 매 라운드 입자 사후로 30개 후보 자극(15 맥락 × f∈{±1})
    의 (ρ,β) 기대 분산감소를 계산해 최대 정보 자극을 선택한다(ε-탐색, 초기에는
    메뉴 2회 라운드로빈 번인으로 전 수준을 관측). 포화 영역(정보 0)의 자극은
    획득함수가 자동으로 회피한다 — 적응적 난이도.
    """
    opp = SyntheticOpponent(theta["alpha"], theta["rho"], theta["beta"],
                            theta["lambda_j"], seed=seed)
    inv = OpponentInversion(n_particles=n_particles, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    # 후보 배열 (30,)
    cands = [(R_, T_, S_, P_, p_, f) for (R_, T_, S_, P_, p_) in menu
             for f in (+1.0, -1.0)]
    u_c = np.array([T_ - S_ for (_, T_, S_, _, _, _) in cands])
    v_c = np.array([(R_ - T_ + P_ - S_) * p_ + (S_ - P_)
                    for (R_, T_, S_, P_, p_, _) in cands])
    f_c = np.array([f for (*_, f) in cands])
    st0 = None
    burnin = 2 * len(cands)
    for t in range(rounds):
        if t < burnin:
            k = t % len(cands)
        elif rng.random() < eps:
            k = int(rng.integers(len(cands)))
        else:
            k = int(np.argmax(_expected_var_reduction(inv, u_c, v_c, f_c)))
        R_, T_, S_, P_, p_, f = cands[k]
        set_payoff_matrix(R_, T_, S_, P_)
        opp.my_coop_rate = p_
        inv.my_cooperation_rate = p_
        focal_last = 0 if f > 0 else 1          # f=+1 ↔ 직전 협력(0)
        opp_a = opp.act(focal_last)
        inv.update(opp_a, ObservationContext(my_last_action=focal_last,
                                             their_last_action=None,
                                             joint_outcome=None,
                                             round_number=t))
        if st0 is None:
            st0 = dict(inv.posterior_stds())
    return inv, dict(inv.posterior_means()), dict(inv.posterior_stds()), st0


def predictive_check(inv, theta, blocks=None, fg=False):
    """
    **역방향(생성 재현)**: 추정 θ̂ 로 상대를 재생성하면 원래 상대와 같은 행동이
    나오는가. 맥락(블록 × 호혜신호 f)마다 참 협력확률 vs θ̂ 협력확률을 비교하고,
    입자 사후예측 90% 구간이 참값을 포함하는지(커버리지) 본다.
    설계 간 비교 가능하도록 **공통 평가 맥락**(EVAL_CONTEXTS)을 기본 사용한다.
    """
    blocks = EVAL_CONTEXTS if blocks is None else blocks
    m = inv.posterior_means()
    rows = []
    # [v0.8.0 §7.3] fg 기저의 역방향 평가맥락은 (f,g) 4셀을 **전부** 포함하도록
    # 확장한다 — g 를 무시하면 ω·η 의 재현 오차가 평가에서 누락된다.
    g_levels = (+1.0, -1.0) if fg else (0.0,)
    for (R_, T_, S_, P_, p_) in blocks:
        set_payoff_matrix(R_, T_, S_, P_)
        inv.my_cooperation_rate = p_
        for f in (+1.0, -1.0):
            for g in g_levels:
                pc_true = _logistic(theta["beta"] * (
                    theta["alpha"] + theta["rho"] * f
                    + theta.get("omega", 0.0) * g
                    + theta.get("eta", 0.0) * f * g
                    + C_empathy_shift(theta["lambda_j"], p_)))
                pc_hat = _logistic(m["beta"] * (
                    m["alpha"] + m["rho"] * f
                    + m.get("omega", 0.0) * g + m.get("eta", 0.0) * f * g
                    + C_empathy_shift(m["lambda_j"], p_)))
                part = inv._pC(f, g)            # 입자별 예측 협력확률
                lo = _wq(part, inv.weights, 0.05)
                hi = _wq(part, inv.weights, 0.95)
                rows.append((pc_true, pc_hat, lo, hi))
    a = np.array(rows, float)
    return {"pc_true": a[:, 0], "pc_hat": a[:, 1],
            "covered": (a[:, 0] >= a[:, 2]) & (a[:, 0] <= a[:, 3])}


def run_design(name, design, n_agents, rounds, seed0=0, n_particles=400):
    """설계 하나에 대해 무작위 θ 표집 → 복원 → 정·역방향 지표.

    [v0.8.0 §7.3] design["basis"]=="fg" 면 6축(α,ρ,ω,η,β,λ_j) 복원 배터리로
    전환한다. 입자 수는 차원 증가를 보상해 기본 600 이상 권장(§7.2).
    """
    rng = np.random.default_rng(seed0)
    is_adaptive = bool(design.get("adaptive"))
    is_fg = design.get("basis") == "fg"
    axes = AXES_FG if is_fg else AXES
    blocks = design["menu"] if is_adaptive else design["blocks"]
    if is_fg:
        n_particles = max(int(n_particles), 600)   # §7.2 차원 증가 보상
    tru, est, sd_fin, sd_ini = [], [], [], []
    pc_t, pc_h, cov_pp = [], [], []
    for i in range(n_agents):
        theta = {ax: float(rng.uniform(*THETA_RANGES[ax])) for ax in axes}
        if is_fg:
            inv, m, st, st0 = recover_once_fg(
                theta, blocks, rounds, seed=seed0 + 1000 * i + 3,
                n_particles=n_particles)
        elif is_adaptive:
            inv, m, st, st0 = recover_once_adaptive(
                theta, blocks, rounds, seed=seed0 + 1000 * i + 3,
                n_particles=n_particles)
        else:
            inv, m, st, st0 = recover_once(theta, blocks, rounds,
                                           seed=seed0 + 1000 * i + 3,
                                           n_particles=n_particles)
        tru.append([theta[a] for a in axes])
        est.append([m[a] for a in axes])
        sd_fin.append([st[a] for a in axes])
        sd_ini.append([(st0 or st)[a] for a in axes])
        pp = predictive_check(inv, theta, fg=is_fg)   # 공통 평가 맥락
        pc_t.append(pp["pc_true"]); pc_h.append(pp["pc_hat"])
        cov_pp.append(pp["covered"])
    reset_payoffs()
    Tr, E = np.array(tru), np.array(est)
    SDf, SDi = np.array(sd_fin), np.array(sd_ini)
    pc_t = np.concatenate(pc_t); pc_h = np.concatenate(pc_h)
    cov_pp = np.concatenate(cov_pp)

    summary = {}
    for i, ax in enumerate(axes):
        err = E[:, i] - Tr[:, i]
        lo = E[:, i] - 1.645 * SDf[:, i]
        hi = E[:, i] + 1.645 * SDf[:, i]
        denom = float(np.sum((Tr[:, i] - Tr[:, i].mean()) ** 2))
        summary[ax] = {
            "bias": float(err.mean()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r": float(np.corrcoef(Tr[:, i], E[:, i])[0, 1]),
            "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan"),
            "recovery_slope": float(np.polyfit(Tr[:, i], E[:, i], 1)[0]),
            "coverage_90": float(np.mean((lo <= Tr[:, i]) & (Tr[:, i] <= hi))),
            "shrinkage": float(np.mean(SDi[:, i] - SDf[:, i])),
            "true_sd": float(Tr[:, i].std()), "est_sd": float(E[:, i].std()),
        }
    # 파라미터 혼동행렬 corr(참_i, 추정_j) — 대각 지배해야 개별 식별
    # [v0.8.0 §7.3] fg 기저에서는 6×6 혼동행렬.
    n_ax = len(axes)
    conf = np.array([[float(np.corrcoef(Tr[:, i], E[:, j])[0, 1])
                      for j in range(n_ax)] for i in range(n_ax)])
    offdiag = float(np.max(np.abs(conf - np.diag(np.diag(conf)))))
    diag_dominant = bool(all(abs(conf[i, i]) > np.max(
        np.abs(np.delete(conf[:, i], i))) for i in range(n_ax)))
    predictive = {
        "r": float(np.corrcoef(pc_t, pc_h)[0, 1]),
        "rmse": float(np.sqrt(np.mean((pc_t - pc_h) ** 2))),
        "coverage_90": float(np.mean(cov_pp)),
        "mean_abs_coop_err": float(np.mean(np.abs(pc_t - pc_h))),
    }
    return {"name": name, "summary": summary, "axes": list(axes),
            "confusion": conf.tolist(), "confusion_max_offdiag": offdiag,
            "confusion_diagonal_dominant": diag_dominant,
            "predictive": predictive, "design": design_diagnostics(design),
            "_true": Tr, "_est": E, "_pc_true": pc_t, "_pc_hat": pc_h}


# ------------------------------------------------------------------- 그림
def make_figure(res_by_design, rounds, n_agents):
    try:
        set_korean_font(plt, font_manager)
    except Exception:
        pass
    order = [k for k in ("legacy", "centered", "adaptive", "fg_centered")
             if k in res_by_design]
    main = res_by_design[order[-1]]
    base = res_by_design[order[0]] if len(order) > 1 else None
    COLORS = {"legacy": "0.75", "centered": "C0", "adaptive": "C2",
              "fg_centered": "C4"}
    # [v0.8.0 §7.3] 축 수는 설계에 따라 4(f) 또는 6(fg).
    PAX = list(main.get("axes", AXES))
    ncol = max(len(PAX), 4)
    fig, ax = plt.subplots(2, ncol, figsize=(4.7 * ncol, 9))

    # (a-d/f) 정방향 복원 산점: 참 vs 추정 (해당 축을 가진 설계만 중첩)
    for i, axn in enumerate(PAX):
        a = ax[0, i]
        Tr, E = main["_true"][:, i], main["_est"][:, i]
        for k in order[:-1]:
            r_ = res_by_design[k]
            kax = list(r_.get("axes", AXES))
            if axn not in kax:          # ω·η 는 f-기저 설계에 없음
                continue
            j_ = kax.index(axn)
            a.scatter(r_["_true"][:, j_], r_["_est"][:, j_], s=14,
                      color=COLORS[k], alpha=0.55,
                      label=f"{k} (r={r_['summary'][axn]['r']:.2f})")
        a.scatter(Tr, E, s=20, color=COLORS[order[-1]], alpha=0.8,
                  label=f"{order[-1]} (r={main['summary'][axn]['r']:.2f})")
        lo = float(min(Tr.min(), E.min())); hi = float(max(Tr.max(), E.max()))
        a.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y=x (완전 복원)")
        sl, ic = np.polyfit(Tr, E, 1)
        xs = np.linspace(lo, hi, 20)
        a.plot(xs, ic + sl * xs, color="C3", lw=1.5, label=f"회복기울기={sl:.2f}")
        s = main["summary"][axn]
        a.set_title(f"({'abcdef'[i]}) {AX_LABEL[axn]} 복원\n"
                    f"R²={s['r2']:.2f}, bias={s['bias']:+.2f}", fontsize=10)
        a.set_xlabel(f"참 {AX_LABEL[axn]}"); a.set_ylabel(f"추정 {AX_LABEL[axn]}")
        a.legend(fontsize=6.5, loc="best")

    for i in range(len(PAX), ncol):
        ax[0, i].axis("off")

    # (e,f) 혼동행렬 히트맵 (기준 설계 vs 주 설계)
    for k, (tag, res) in enumerate([(order[0], base), (order[-1], main)]):
        a = ax[1, k]
        if res is None:
            a.axis("off"); continue
        M = np.array(res["confusion"])
        rax = list(res.get("axes", AXES))
        im = a.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(rax)):
            for j in range(len(rax)):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       fontsize=7, color="k")
        a.set_xticks(range(len(rax)))
        a.set_xticklabels([f"추정 {AX_LABEL[x]}" for x in rax], fontsize=6.5,
                          rotation=45)
        a.set_yticks(range(len(rax)))
        a.set_yticklabels([f"참 {AX_LABEL[x]}" for x in rax], fontsize=6.5)
        ok = "대각 지배 ✓" if res["confusion_diagonal_dominant"] else "혼동 잔존 ✗"
        a.set_title(f"({'ef'[k]}) 혼동행렬 — {tag}\n"
                    f"최대 비대각={res['confusion_max_offdiag']:.2f} [{ok}]",
                    fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046)

    # (g) 90% 커버리지 (전 설계)
    a = ax[1, 2]
    x = np.arange(len(PAX)); w = 0.8 / max(len(order), 1)
    for j, k in enumerate(order):
        r_ = res_by_design[k]
        kax = list(r_.get("axes", AXES))
        vals = [r_["summary"][s]["coverage_90"] if s in kax else np.nan
                for s in PAX]
        a.bar(x + (j - (len(order) - 1) / 2) * w, vals, w,
              color=COLORS[k], label=k)
    a.axhline(0.90, color="r", ls="--", label="목표 0.90")
    a.set_xticks(x); a.set_xticklabels([AX_LABEL[s] for s in PAX])
    a.set_ylim(0, 1.05); a.legend(fontsize=7)
    a.set_title("(g) 90% 신용구간 경험적 커버리지\n(목표선 근처여야 사후가 정직)",
                fontsize=10)

    # (h) 역방향(생성 재현): 참 협력확률 vs θ̂ 재생성 협력확률 (공통 평가맥락)
    a = ax[1, 3]
    for i in range(4, ncol):
        ax[1, i].axis("off")
    for k in order:
        r_ = res_by_design[k]
        a.scatter(r_["_pc_true"], r_["_pc_hat"], s=8, color=COLORS[k],
                  alpha=0.45, label=f"{k} (r={r_['predictive']['r']:.2f})")
    a.plot([0, 1], [0, 1], "k--", lw=1.2, label="y=x")
    p = main["predictive"]
    a.set_title(f"(h) 역방향 — θ̂ 로 재생성한 상대의 행동\n"
                f"RMSE={p['rmse']:.3f}, 사후예측 커버리지={p['coverage_90']:.2f}",
                fontsize=10)
    a.set_xlabel("참 상대의 협력확률 (맥락별)")
    a.set_ylabel("θ̂ 상대의 협력확률")
    a.legend(fontsize=7)

    fig.suptitle(f"ToM 파라미터 복원 (양방향) — 회귀자 중심화 배터리 | "
                 f"n={n_agents} 상대, {rounds} 라운드", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "recovery_tom.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "recovery_tom.pdf", bbox_inches="tight")
    with open(RESULTS / "recovery_tom.caption.json", "w", encoding="utf-8") as f:
        json.dump({"figure": "recovery_tom",
                   "caption":
                       "ToM 파라미터 양방향 복원. 정방향(a–d): 참 vs 추정 산점 "
                       "(회색=legacy 단일맥락, 파랑=중심화 배터리). 혼동행렬"
                       "(e,f): legacy 는 참 λ_j 가 추정 α 를 강하게 예측(α·λ 축퇴)"
                       "하나, λ 회귀자 (T−S) 를 평균 0 으로 중심화하면 대각이 "
                       "지배한다. (g) 사후 보정. 역방향(h): θ̂ 로 재생성한 상대의 "
                       "맥락별 협력확률이 원 상대와 일치하는지(사후예측).",
                   "rounds": rounds, "n_agents": n_agents,
                   "theta_ranges": THETA_RANGES}, f, ensure_ascii=False, indent=2)
    plt.close(fig)
    LOGGER.info("그림 저장: %s", RESULTS / "recovery_tom.png")


def main():
    ap = argparse.ArgumentParser()
    # [v0.8.0 §7.3] fg_centered 를 기본 배터리에 편입 — g·fg 도입은 식별 요구를
    # 근본적으로 늘리므로 복원 검증이 **필수 동반 작업**이다.
    ap.add_argument("--designs", nargs="*",
                    default=["legacy", "centered", "adaptive", "fg_centered"],
                    choices=list(DESIGNS))
    ap.add_argument("--agents", type=int, default=60)
    # §7.3: (f,g) 4셀 × v 계단 → 셀당 최소 ~30 관측 권장 → rounds ≥ 480
    ap.add_argument("--rounds", type=int, default=480)
    ap.add_argument("--particles", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.agents, args.rounds, args.particles = 12, 60, 200

    LOGGER.info("복원 연구(양방향): designs=%s agents=%d rounds=%d",
                args.designs, args.agents, args.rounds)
    out = {}
    for name in args.designs:
        d = design_diagnostics(DESIGNS[name])
        # (adaptive 는 메뉴 기준 진단)
        LOGGER.info("[%s] 블록=%d | λ회귀자 u=(T−S): 평균=%.2f SD=%.2f | "
                    "기지오프셋 v: 평균=%.2f SD=%.2f | corr(u,v)=%.2f",
                    name, d["n_blocks"], d["u_mean"], d["u_sd"],
                    d["v_mean"], d["v_sd"], d["corr_u_v"])
        res = run_design(name, DESIGNS[name], args.agents, args.rounds,
                         seed0=args.seed, n_particles=args.particles)
        out[name] = res
        for ax_ in AXES:
            s = res["summary"][ax_]
            LOGGER.info("  [%s/%s] r=%.2f R²=%+.2f 편향=%+.3f 기울기=%.2f "
                        "커버리지=%.2f", name, ax_, s["r"], s["r2"], s["bias"],
                        s["recovery_slope"], s["coverage_90"])
        LOGGER.info("  [%s] 혼동 최대 비대각=%.2f (%s) | 역방향 r=%.2f "
                    "RMSE=%.3f 사후예측 커버리지=%.2f", name,
                    res["confusion_max_offdiag"],
                    "대각 지배" if res["confusion_diagonal_dominant"] else "혼동 잔존",
                    res["predictive"]["r"], res["predictive"]["rmse"],
                    res["predictive"]["coverage_90"])

    make_figure(out, args.rounds, args.agents)
    main_name = ("adaptive" if "adaptive" in out
                 else "centered" if "centered" in out else args.designs[0])
    payload = {
        "designs": {k: {kk: vv for kk, vv in v.items()
                        if not kk.startswith("_")} for k, v in out.items()},
        # 하위호환: 주 설계(centered)의 축별 요약
        "summary": out[main_name]["summary"],
        "theta_ranges": {k: list(v) for k, v in THETA_RANGES.items()},
        "config": vars(args),
    }
    with open(RESULTS / "recovery_tom.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    LOGGER.info("결과 저장: %s", RESULTS / "recovery_tom.json")


if __name__ == "__main__":
    main()

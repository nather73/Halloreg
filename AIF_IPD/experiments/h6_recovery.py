"""
experiments.h6_recovery
=======================

**H6 — Parameter recovery 와 projection 은 강건하게 이루어지는가?**

두 개의 논리적으로 독립적인 하위 과제로 나눈다.

────────────────────────────────────────────────────────────────────────
과제 A — 파라미터 복원 (parameter recovery)
────────────────────────────────────────────────────────────────────────
"모형이 자기가 가정한 데이터 생성과정을 되돌릴 수 있는가" 라는 복원 타당성
(recovery validity)의 표준 검사다.

  1. 참 θ* = (α, ρ, ω, η, β, λ_j) 를 사전 지지범위에서 무작위 추출한다.
  2. `LikelihoodAgent(θ*)` 가 추론기의 생성 우도로부터 직접 행동을 생성한다.
  3. focal 은 `ProbeAgent` — 매 라운드 독립적으로 확률 0.5 로 협력한다.
  4. `OpponentInversion` 을 관측열에 적용해 θ̂ 를 얻고 θ* 와 대조한다.

[왜 ProbeAgent 인가 — 식별가능성]
복원의 성패는 설계행렬 (1, f, g, f·g) 가 실제로 스팬되는지에 달렸다. focal 이
정책적으로 행동하면(예: 늘 협력) f 가 상수가 되어 α 와 ρ 가 공선이 되고,
ρ·ω·η 는 식별 불가능해진다. 무작위 probe 는 f 를 두 수준에 균등 배치하고,
상대 자기 이력 g 와의 조합도 네 칸을 모두 채운다.

[사전 예측 — 반드시 명시할 한계]
λ_j 는 우도에 `s(λ_j, p) = (T−S)·λ_j + …` 라는 **절편** 형태로만 들어간다.
고정 보수에서는 (T−S)=5 가 상수이므로 λ_j 항은 α 와 완전 공선이 되고, 두 축은
**개별 식별이 불가능**하다. 따라서 두 조건을 나란히 돌린다.

  · 조건 fixed    : 고정 PD. λ_j 복원이 실패할 것으로 **사전 예측**한다.
  · 조건 varied   : 블록마다 (T, S) 를 변조해 λ_j 회귀자 (T−S) 를 중심화한다.
                    → 공선성이 깨져 λ_j 가 식별 가능해질 것으로 예측한다.

이 대조는 "복원 실패" 를 결함이 아니라 **식별가능성의 구조적 결과**로 진단하며,
그 진단이 옳다면 varied 조건에서 복원이 회복되어야 한다.

  · 확증 A1 : fixed 조건에서 (α, ρ, ω, η, β) 5축 각각 corr(θ̂, θ*) > 0.
  · 확증 A2 : varied 조건에서 λ_j 의 corr > fixed 조건의 corr.
  · 탐색   : 축별 편향·RMSE, 라운드에 따른 수렴.

[β 축에 대한 추가 사전 예측]
β 는 선형예측자 **전체에 곱해지는** 정밀도이므로, 선형항의 부호·순서를 바꾸지
않고 오직 선택확률의 **극단성**만 조절한다. 따라서 β 는
  · α·ρ·ω·η 와 곱셈적으로 얽혀 있고(β·α 만 식별되는 방향이 존재),
  · 중간 크기의 β(3~6)에서는 σ(·) 가 이미 포화에 가까워 관측 우도가 β 변화에
    둔감하다.
그 결과 **β 의 복원 상관은 다른 축보다 낮을 것으로 사전 예측한다.** 이는
입자필터의 결함이 아니라 로지스틱 선택모형의 알려진 식별 한계이며, 예측대로
낮게 나오더라도 그대로 보고한다(억지로 지지 판정을 만들지 않는다).

────────────────────────────────────────────────────────────────────────
과제 B — 자기-사영 (self-projection)
────────────────────────────────────────────────────────────────────────
HalloReg 의 depth-2 조망수용은 "상대가 나를 어떻게 볼까" 를 자기-사영 필터
θ̂_self 로 구성한다(같은 역추론기를 자기 행동 이력에 적용). 이 사영이 타당하려면
**실제 외부 관찰자가 나에 대해 추론하는 것과 일치**해야 한다.

  주 지표 — **사영 정합성(projection fidelity)**
      HalloReg A 와 HalloReg B 를 맞붙인다.
        · A 의 θ̂_self  = A 가 생각하는 "B 가 본 나"
        · B 의 θ̂        = B 가 실제로 추론한 A
      두 벡터를 축별로 대조한다. 상관 r > 0 이면 사영이 veridical 하다.
      이것이 depth-2 재귀의 타당성 조건이며, 사영이 자기 위안적 환상이 아니라
      **타자의 관점을 실제로 근사**하는지를 직접 검정한다.

  보조 지표 — 예측 보정
      자기-사영 필터가 매 라운드 (갱신 전) 내놓는 예측 협력확률과 실제 focal
      행동의 AUC·Brier 기술점수.

      **사전 예측(중요).** focal 의 라운드별 행동은 softmax(−G_social) 에서
      표집되므로, 그 변동의 상당 부분은 **정책 표집 잡음**이다. 잡음은 원리적으로
      예측 불가능하므로 AUC 는 0.5 근처일 것으로 **사전 예측한다**. 이는 사영의
      실패가 아니라 "무엇이 예측 가능한 대상인가" 의 문제다. 따라서 이 지표는
      확증이 아니라 **탐색**으로 등록한다.

  · 확증 B1 : 사영 정합성 — (α, ρ, β, λ_j) 축 각각 corr(θ̂_self, θ̂_partner) > 0.
  · 탐색   : ω·η 축 정합성, AUC, Brier 기술점수, 상대 유형별 예측 보정.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from AIF_IPD.core.constants import (
    COOP, DEFECT, reset_payoffs, set_payoff_matrix,
)
from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.env import ALL_TYPES, LikelihoodAgent, ProbeAgent, TYPE_LABEL_KO
from AIF_IPD.ipd.metrics import (
    boot_mean_ci, corr_boot, one_sample_perm, perm_test,
)
from AIF_IPD.ipd.population import type_spec
from AIF_IPD.ipd.sim import run_many
from AIF_IPD.ipd.tom.inversion import (
    ObservationContext, OpponentInversion, THETA_AXES,
)
from .common import Config, Registry, TYPE_COLORS, save_fig, save_json

LOGGER = get_logger("HalloReg.H6")

#: 참 θ 를 뽑을 범위 (사전 지지범위의 안쪽 — 경계 클리핑 인공물 회피)
TRUE_RANGE = {
    "alpha": (-2.0, 2.0), "rho": (-1.5, 2.5), "omega": (-1.5, 1.5),
    "eta": (-1.5, 1.5), "beta": (1.0, 6.0), "lambda_j": (0.05, 0.95),
}

#: varied 조건의 블록별 (R, T, S, P).
#: λ_j 의 회귀자는 (T − S) 이므로 블록마다 이 값을 크게 바꾸고 **평균을 0 근처로**
#: 만들어 α(절편)와의 공선성을 깬다. 세 번째 블록은 T < S 인 비-PD 게임인데,
#: 이는 식별을 위한 의도적 설계다(본 실험 외의 결과에는 영향이 없다).
VARIED_BLOCKS = [
    (3.0, 5.0, 0.0, 1.0),      # T−S = +5  (표준 PD)
    (3.0, 2.0, 2.0, 1.0),      # T−S =  0  (λ_j 회귀자 소거)
    (3.0, 0.0, 5.0, 1.0),      # T−S = −5  (부호 반전)
    (3.0, 4.0, 1.0, 1.0),      # T−S = +3
]

AXIS_LABEL_KO = {"alpha": "α 협력편향", "rho": "ρ 호혜성", "omega": "ω 관성",
                 "eta": "η 결과조건성", "beta": "β 정밀도",
                 "lambda_j": "λⱼ 상대공감"}
IDENTIFIED_AXES = ("alpha", "rho", "omega", "eta", "beta")


# ==================================================================== 과제 A
def _recover_one(theta_true: Dict[str, float], n_rounds: int, seed: int,
                 varied: bool, n_particles: int) -> Tuple[Dict[str, float], np.ndarray]:
    """
    한 번의 복원 시행. 반환 (최종 θ̂, 라운드별 θ̂ 궤적 (T, D)).

    시뮬레이션 루프를 직접 돌린다 — focal 이 능동추론 에이전트가 아니라
    무작위 probe 이므로 run_dyad 의 AIF 경로가 필요 없고, 추론기 하나만
    관측열에 적용하면 되기 때문이다.
    """
    rng_gen = LikelihoodAgent(**theta_true, seed=seed)
    probe = ProbeAgent(0.5, seed=seed + 1)
    inv = OpponentInversion(n_particles=n_particles, seed=seed + 2)

    my_actions: List[int] = []
    opp_actions: List[int] = []
    traj = np.zeros((n_rounds, len(THETA_AXES)))

    for t in range(n_rounds):
        # ---- varied 조건: 블록마다 보수를 바꿔 λ_j 회귀자를 변조 ----
        if varied:
            b = min(len(VARIED_BLOCKS) - 1,
                    int(t * len(VARIED_BLOCKS) / max(n_rounds, 1)))
            set_payoff_matrix(*VARIED_BLOCKS[b])

        # ---- 행동 방출 ----
        a_j = rng_gen.act()          # 상대(생성 프로세스)
        a_i = probe.act()            # focal (무작위 자극)
        rng_gen.observe(a_i)

        # ---- 추론기 갱신 (갱신용 시제: f = my_{t−1}, g = opp_{t−1}) ----
        ctx = ObservationContext(
            my_last_action=(my_actions[-1] if my_actions else None),
            their_last_action=(opp_actions[-1] if opp_actions else None),
            round_number=t)
        inv.my_cooperation_rate = (float(np.mean([a == COOP for a in my_actions]))
                                   if my_actions else 0.5)
        inv.update(a_j, ctx)

        my_actions.append(a_i)
        opp_actions.append(a_j)
        m = inv.posterior_means()
        traj[t] = [m[ax] for ax in THETA_AXES]

    if varied:
        reset_payoffs()
    return inv.posterior_means(), traj


def run_recovery(cfg: Config, reg: Registry) -> dict:
    """과제 A — 파라미터 복원 (fixed vs varied 조건)."""
    n_sim = cfg.seeds
    rng = np.random.default_rng(41)
    truths = []
    for i in range(n_sim):
        truths.append({ax: float(rng.uniform(*TRUE_RANGE[ax]))
                       for ax in THETA_AXES})

    out = {}
    for cond, varied in (("fixed", False), ("varied", True)):
        LOGGER.info("  [H6-A] 복원 조건 '%s' — %d 시행 × %d 라운드",
                    cond, n_sim, cfg.rounds)
        T = np.zeros((n_sim, len(THETA_AXES)))     # 참값
        H = np.zeros((n_sim, len(THETA_AXES)))     # 추정
        trajs = np.zeros((n_sim, cfg.rounds, len(THETA_AXES)))
        for i, th in enumerate(truths):
            est, tr = _recover_one(th, cfg.rounds, seed=20_000 + i * 13,
                                   varied=varied, n_particles=cfg.n_particles)
            T[i] = [th[ax] for ax in THETA_AXES]
            H[i] = [est[ax] for ax in THETA_AXES]
            trajs[i] = tr
        out[cond] = {"true": T, "hat": H, "traj": trajs}

    # ---- 확증 A1: fixed 조건 5축 복원 ----
    corrs_fixed = {}
    for d, ax in enumerate(THETA_AXES):
        c = corr_boot(out["fixed"]["true"][:, d], out["fixed"]["hat"][:, d],
                      seed=d)
        corrs_fixed[ax] = c
        if ax in IDENTIFIED_AXES:
            reg.confirm("H6", f"복원(fixed) {AXIS_LABEL_KO[ax]}: r > 0",
                        c["p"], direction_ok=bool(c["r"] > 0),
                        effect=f"r={c['r']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]")
        else:
            reg.explore("H6", f"복원(fixed) {AXIS_LABEL_KO[ax]} "
                              f"(α 와 공선 — 실패 예측됨)", c["p"],
                        effect=f"r={c['r']:.3f}")

    # ---- 확증 A2: varied 조건에서 λ_j 복원 개선 ----
    corrs_varied = {}
    for d, ax in enumerate(THETA_AXES):
        corrs_varied[ax] = corr_boot(out["varied"]["true"][:, d],
                                     out["varied"]["hat"][:, d], seed=100 + d)
    d_lj = THETA_AXES.index("lambda_j")
    # 두 조건의 상관 차이를 부트스트랩으로 검정 (같은 참값 집합을 공유하므로 짝지음)
    rng2 = np.random.default_rng(77)
    n_boot = 3000
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng2.integers(0, n_sim, n_sim)
        tf, hf = out["fixed"]["true"][i, d_lj], out["fixed"]["hat"][i, d_lj]
        tv, hv = out["varied"]["true"][i, d_lj], out["varied"]["hat"][i, d_lj]
        rf = np.corrcoef(tf, hf)[0, 1] if tf.std() > 0 and hf.std() > 0 else 0.0
        rv = np.corrcoef(tv, hv)[0, 1] if tv.std() > 0 and hv.std() > 0 else 0.0
        diffs[b] = rv - rf
    p_diff = (np.sum(diffs <= 0) + 1) / (n_boot + 1)
    obs_diff = corrs_varied["lambda_j"]["r"] - corrs_fixed["lambda_j"]["r"]
    reg.confirm("H6", "λⱼ 복원: varied(보수변조) > fixed — 식별성 진단",
                p_diff, direction_ok=bool(obs_diff > 0),
                effect=f"Δr={obs_diff:+.3f} "
                       f"(fixed r={corrs_fixed['lambda_j']['r']:.3f} → "
                       f"varied r={corrs_varied['lambda_j']['r']:.3f})")

    # ---- 탐색: 축별 RMSE / 편향 ----
    for cond in ("fixed", "varied"):
        for d, ax in enumerate(THETA_AXES):
            err = out[cond]["hat"][:, d] - out[cond]["true"][:, d]
            t = one_sample_perm(err, 0.0)
            reg.explore("H6", f"복원({cond}) {AXIS_LABEL_KO[ax]} 편향≠0", t["p"],
                        effect=f"편향={err.mean():+.3f}, "
                               f"RMSE={np.sqrt(np.mean(err ** 2)):.3f}")

    out["corr_fixed"] = corrs_fixed
    out["corr_varied"] = corrs_varied
    return out


# ==================================================================== 과제 B
def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """
    이진 라벨 y(1=협력)에 대한 예측확률 p 의 AUC.
    Mann-Whitney U 통계량으로 계산한다(동점은 평균순위로 처리).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # 동점 보정
    uniq, inv_idx, cnt = np.unique(p, return_inverse=True, return_counts=True)
    for u_i in np.where(cnt > 1)[0]:
        m = inv_idx == u_i
        ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


#: 사영 정합성의 확증 대상 축. ω·η 는 자기 행동 이력만으로는 약하게 식별되므로
#: (focal 은 자기 정책상 g 축 변동을 스스로 만들지 않는다) 탐색으로 둔다.
FIDELITY_AXES = ("alpha", "rho", "beta", "lambda_j")


def run_projection(cfg: Config, reg: Registry) -> dict:
    """
    과제 B — 자기-사영의 타당성.

    B1(확증) : HalloReg×HalloReg 다이애드에서 A 의 θ̂_self 와 B 의 θ̂(A) 의 정합성.
    B2(탐색) : 자기-사영 필터의 자기 행동 예측 보정(AUC·Brier).
    """
    hk = cfg.halloreg_kwargs()

    # ---------- B1: 사영 정합성 (HalloReg vs HalloReg) ----------
    specs_f = []
    for sd in range(cfg.seeds):
        specs_f.append({
            "agent": {"type": "halloreg", "seed": 40_000 + sd * 43, **hk},
            "opponent": {"type": "halloreg", "seed": 41_000 + sd * 43, **hk},
            "env_err_agent": cfg.env_error, "env_err_opponent": cfg.env_error,
            "noise_seed": 820_000 + sd * 109})
    res_f = run_many(specs_f, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                     desc="H6 사영정합성 다이애드")

    axes = list(THETA_AXES)
    proj = np.zeros((cfg.seeds, len(axes)))     # A 의 θ̂_self
    partner = np.zeros((cfg.seeds, len(axes)))  # B 가 추론한 A
    for sd in range(cfg.seeds):
        la = res_f[sd]["agent_log"]
        lb = res_f[sd]["opponent_log"]
        for d, ax in enumerate(axes):
            proj[sd, d] = float(la[f"P_{ax}"][-1])
            partner[sd, d] = float(lb[f"E_{ax}"][-1])

    fidelity = {}
    for d, ax in enumerate(axes):
        c = corr_boot(proj[:, d], partner[:, d], seed=200 + d)
        fidelity[ax] = c
        label = f"사영 정합성 {AXIS_LABEL_KO[ax]}: corr(θ̂_self, θ̂_partner) > 0"
        if ax in FIDELITY_AXES:
            reg.confirm("H6", label, c["p"], direction_ok=bool(c["r"] > 0),
                        effect=f"r={c['r']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]")
        else:
            reg.explore("H6", label + " (자기 행동만으로는 약식별)", c["p"],
                        effect=f"r={c['r']:.3f}")
    # 축별 절대 편차 (해석 보조)
    for d, ax in enumerate(axes):
        err = proj[:, d] - partner[:, d]
        t = one_sample_perm(err, 0.0)
        reg.explore("H6", f"사영 편향 {AXIS_LABEL_KO[ax]} ≠ 0", t["p"],
                    effect=f"편향={err.mean():+.3f}, "
                           f"RMSE={np.sqrt(np.mean(err ** 2)):.3f}")

    # ---------- B2: 예측 보정 (탐색) ----------
    specs, registry = [], {}
    for ti, tname in enumerate(ALL_TYPES):
        for sd in range(cfg.seeds):
            opp = dict(type_spec(tname))
            opp["seed"] = 31_000 + sd * 41 + ti
            if tname == "halloreg":
                opp.update(hk)
            registry[(ti, sd)] = len(specs)
            specs.append({
                "agent": {"type": "halloreg", "seed": 30_000 + sd * 41 + ti, **hk},
                "opponent": opp,
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 810_000 + sd * 107 + ti})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H6 사영보정 다이애드")

    auc_by_type: Dict[str, np.ndarray] = {}
    skill_by_type: Dict[str, np.ndarray] = {}
    for ti, tname in enumerate(ALL_TYPES):
        aucs = np.full(cfg.seeds, np.nan)
        skills = np.full(cfg.seeds, np.nan)
        for sd in range(cfg.seeds):
            log = res[registry[(ti, sd)]]["agent_log"]
            p = np.asarray(log["self_pred_coop"], dtype=float)[1:]
            y = (np.asarray(log["action"], dtype=int) == COOP).astype(int)[1:]
            ok = np.isfinite(p)
            p, y = p[ok], y[ok]
            if len(y) < 5 or y.sum() in (0, len(y)):
                continue      # 행동이 한쪽으로 완전히 몰리면 AUC 가 미정의
            aucs[sd] = _auc(y, p)
            # 인과적 기저율 기준선: 그 시점까지의 누적 협력률
            base = np.concatenate([[0.5], np.cumsum(y)[:-1]
                                   / np.arange(1, len(y))])
            bm = float(np.mean((p - y) ** 2))
            bb = float(np.mean((base - y) ** 2))
            skills[sd] = 1.0 - bm / max(bb, 1e-9)
        auc_by_type[tname] = aucs
        skill_by_type[tname] = skills

    all_auc = np.concatenate([auc_by_type[t] for t in ALL_TYPES])
    all_auc = all_auc[np.isfinite(all_auc)]
    all_skill = np.concatenate([skill_by_type[t] for t in ALL_TYPES])
    all_skill = all_skill[np.isfinite(all_skill)]
    t1 = one_sample_perm(all_auc, 0.5, alternative="greater")
    c1 = boot_mean_ci(all_auc)
    reg.explore("H6", "자기사영 예측 AUC > 0.5 "
                      "(라운드별 변동의 상당분은 정책 표집잡음 — 0.5 근방 예측됨)",
                t1["p"],
                effect=f"AUC={c1['mean']:.3f} "
                       f"[{c1['ci'][0]:.3f}, {c1['ci'][1]:.3f}]")
    t2 = one_sample_perm(all_skill, 0.0, alternative="greater")
    c2 = boot_mean_ci(all_skill)
    reg.explore("H6", "자기사영 Brier 기술점수 > 0 (기저율 예측 대비)", t2["p"],
                effect=f"기술점수={c2['mean']:+.3f} "
                       f"[{c2['ci'][0]:+.3f}, {c2['ci'][1]:+.3f}]")

    return {"axes": axes, "proj": proj, "partner": partner,
            "fidelity": fidelity,
            "auc": auc_by_type, "skill": skill_by_type,
            "mean_auc": float(c1["mean"]), "mean_skill": float(c2["mean"])}


# ==================================================================== 실행
def run(cfg: Config, reg: Registry) -> dict:
    LOGGER.info("[H6] 파라미터 복원과 자기-사영")
    a = run_recovery(cfg, reg)
    b = run_projection(cfg, reg)
    out = {"recovery": a, "projection": b}
    _plot(cfg, a, b)
    save_json({"recovery": {
        "corr_fixed": {k: {kk: v[kk] for kk in ("r", "ci", "p", "n")}
                       for k, v in a["corr_fixed"].items()},
        "corr_varied": {k: {kk: v[kk] for kk in ("r", "ci", "p", "n")}
                        for k, v in a["corr_varied"].items()}},
        "projection": {
            "fidelity": {k: {kk: v[kk] for kk in ("r", "ci", "p", "n")}
                         for k, v in b["fidelity"].items()},
            "mean_auc": b["mean_auc"], "mean_skill": b["mean_skill"],
            "auc_by_type": {t: float(np.nanmean(b["auc"][t]))
                            for t in ALL_TYPES}}},
        cfg.results / "H6.json")
    return out


# ==================================================================== 시각화
def _plot(cfg: Config, a: dict, b: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import annotate_n, band_plot

    axes = list(THETA_AXES)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)

    # (a~c) fixed 조건 주요 축 산점
    for i, ax_name in enumerate(("alpha", "rho", "eta")):
        ax = fig.add_subplot(gs[0, i])
        d = axes.index(ax_name)
        x = a["fixed"]["true"][:, d]; y = a["fixed"]["hat"][:, d]
        ax.scatter(x, y, s=16, alpha=0.6, color="#4C72B0")
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, ls="--", color="#888888", lw=1)
        ax.set_xlabel(f"참 {AXIS_LABEL_KO[ax_name]}")
        ax.set_ylabel(f"추정 {AXIS_LABEL_KO[ax_name]}")
        r = a["corr_fixed"][ax_name]["r"]
        ax.set_title(f"({'abc'[i]}) 복원(fixed) — {AXIS_LABEL_KO[ax_name]}\nr={r:.3f}")
        annotate_n(ax, len(x))

    # (d) 축별 복원 상관: fixed vs varied
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(axes)); w = 0.36
    rf = [a["corr_fixed"][k]["r"] for k in axes]
    rv = [a["corr_varied"][k]["r"] for k in axes]
    ax.bar(x - w / 2, rf, w, color="#4C72B0", label="fixed (고정 PD)", alpha=0.9)
    ax.bar(x + w / 2, rv, w, color="#C44E52", label="varied (보수 변조)",
           alpha=0.9)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABEL_KO[k] for k in axes], rotation=40,
                       ha="right", fontsize=7)
    ax.set_ylabel("복원 상관 r"); ax.set_ylim(-0.2, 1.05)
    ax.set_title("(d) 축별 복원 상관\nλⱼ 는 고정 보수에서 α 와 공선 → 변조로 식별")
    ax.legend(fontsize=7)

    # (e) λ_j 복원: 두 조건 산점
    ax = fig.add_subplot(gs[1, 1])
    d = axes.index("lambda_j")
    ax.scatter(a["fixed"]["true"][:, d], a["fixed"]["hat"][:, d], s=15,
               alpha=0.55, color="#4C72B0", label="fixed")
    ax.scatter(a["varied"]["true"][:, d], a["varied"]["hat"][:, d], s=15,
               alpha=0.55, color="#C44E52", label="varied")
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1)
    ax.set_xlabel("참 λⱼ"); ax.set_ylabel("추정 λ̂ⱼ")
    ax.set_title("(e) λⱼ 복원 — 식별성 대조")
    ax.legend(fontsize=7)

    # (f) 복원 수렴 궤적 (참값과의 절대오차)
    ax = fig.add_subplot(gs[1, 2])
    for ax_name, col in (("alpha", "#4C72B0"), ("rho", "#55A868"),
                         ("beta", "#C44E52"), ("eta", "#8172B2")):
        d = axes.index(ax_name)
        err = np.abs(a["fixed"]["traj"][:, :, d]
                     - a["fixed"]["true"][:, None, d])
        # 축마다 스케일이 다르므로 참값 범위로 정규화
        rng_ = TRUE_RANGE[ax_name][1] - TRUE_RANGE[ax_name][0]
        ax.plot(np.median(err, axis=0) / rng_, color=col,
                label=AXIS_LABEL_KO[ax_name], lw=1.3)
    ax.set_xlabel("라운드"); ax.set_ylabel("정규화 절대오차 (중앙값)")
    ax.set_title("(f) 복원 수렴")
    ax.legend(fontsize=7)

    # (g) 사영 정합성 — α̂ 축 산점 (θ̂_self vs θ̂_partner)
    ax = fig.add_subplot(gs[2, 0])
    d = b["axes"].index("alpha")
    ax.scatter(b["partner"][:, d], b["proj"][:, d], s=16, alpha=0.6,
               color="#8172B2")
    lim = [min(b["partner"][:, d].min(), b["proj"][:, d].min()),
           max(b["partner"][:, d].max(), b["proj"][:, d].max())]
    ax.plot(lim, lim, ls="--", color="#888888", lw=1)
    ax.set_xlabel("상대가 추론한 나의 α̂"); ax.set_ylabel("나의 자기-사영 α̂")
    ax.set_title(f"(g) 사영 정합성 — α 축\nr={b['fidelity']['alpha']['r']:.3f}")
    annotate_n(ax, b["proj"].shape[0])

    # (h) 축별 사영 정합성 상관
    ax = fig.add_subplot(gs[2, 1])
    ax_names = b["axes"]
    x = np.arange(len(ax_names))
    rs = [b["fidelity"][k]["r"] for k in ax_names]
    cols = ["#8172B2" if k in FIDELITY_AXES else "#BBBBBB" for k in ax_names]
    ax.bar(x, rs, color=cols, alpha=0.9)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABEL_KO[k] for k in ax_names], rotation=40,
                       ha="right", fontsize=7)
    ax.set_ylabel("corr(θ̂_self, θ̂_partner)"); ax.set_ylim(-0.4, 1.05)
    ax.set_title("(h) 축별 사영 정합성\n(회색 = 자기 행동만으로는 약식별)")

    # (i) 축별 RMSE 요약
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(len(axes)); w = 0.36
    for off, cond, col in ((-w / 2, "fixed", "#4C72B0"),
                           (+w / 2, "varied", "#C44E52")):
        rm = []
        for d, k in enumerate(axes):
            e = a[cond]["hat"][:, d] - a[cond]["true"][:, d]
            rng_ = TRUE_RANGE[k][1] - TRUE_RANGE[k][0]
            rm.append(float(np.sqrt(np.mean(e ** 2)) / rng_))
        ax.bar(x + off, rm, w, color=col, label=cond, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABEL_KO[k] for k in axes], rotation=40,
                       ha="right", fontsize=7)
    ax.set_ylabel("정규화 RMSE")
    ax.set_title("(i) 축별 복원 오차"); ax.legend(fontsize=7)

    fig.suptitle("H6 — 파라미터 복원과 자기-사영의 강건성", fontsize=12, y=0.985)
    save_fig(fig, cfg, "H6_recovery_projection")

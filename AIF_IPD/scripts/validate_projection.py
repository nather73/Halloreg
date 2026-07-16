#!/usr/bin/env python
"""
validate_projection.py — 사영(projection) 분리성·복원성 연구 (v0.6.6).

AdaptiveAgent 가 TFT/GTFT/GTFT_count/WSLS/ALLC/ALLD/Adaptive/Adaptive_imm 를
상대할 때, 입자필터가 추정한 θ̂=(α̂,ρ̂,β̂,λ̂) 는 **모형 밖 행위자를 파라미터
공간으로 사영**한 것이다(참 θ 부재 — 의사-참값/QMLE 틀). 본 스크립트는 네
질문을 분리 측정한다.

  Q1 사영 복원성  : θ̂ 이 의사-참 사영점 θ*_s(장지평 극한)로 수렴·안정하는가
                    — RMSE(θ̂,θ*), 분할-반 상관, θ* 기준 90% 사후 커버리지.
  Q2 사영 분리성  : θ̂ 에서 상대 유형을 해독할 수 있는가 — LOO 마할라노비스
                    최근접-중심 해독 + **라벨 순열검정**, 유형 혼동행렬,
                    쌍별 Bhattacharyya 거리. 원 좌표(4D)와 식별 좌표
                    (α+5λ, βρ, β; 3D) 병행.
  Q3 사영 충실도  : P_θ̂ 가 상대의 기억-1 시그니처 P(C|직전 joint outcome) 를
                    재현하는가. 모형 우도는 focal 직전 행동 f 에만 조건하므로
                    CC≡CD, DC≡DD 로 붕괴 — joint outcome 조건 전략(WSLS)은
                    **원리적으로 표현 불가**(구조적 잔차의 정직한 정량화).
  Q4 동치류 분해  : 분리 실패가 '정책 탓(정보 미자극)'인지 '모형 탓(사영
                    불가분)'인지 — focal 정책 {onpolicy, probe(ε 강제 탐침
                    배신)} 2조건 비교. 협력 균형에서 TFT/GTFT/ALLC 는 행동적
                    동치류이므로 onpolicy 혼동은 원리적 한계이고, probe 가
                    조건부 구조를 자극해 분리를 회복해야 한다.

부수 검사: adaptive 상대는 λ 조절로 θ 가 시변 — θ̂ 드리프트(중간 vs 최종)로
정적 사영의 유의미성을 점검한다.

사용:
    python AIF_IPD/scripts/validate_projection.py            # 전체 (직렬)
    python AIF_IPD/scripts/validate_projection.py --quick
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
    COOP, DEFECT, joint_index, mirror_state, set_coop_index,
    empathy_shift as C_empathy_shift,
)
from AIF_IPD.ipd.agent import AdaptiveAgent
from AIF_IPD.ipd.env import make_opponent

LOGGER = get_logger("HalloReg.projection")
RESULTS = _PKG_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

AXES = ("alpha", "rho", "beta", "lambda_j")
AX_LABEL = {"alpha": "α", "rho": "ρ", "beta": "β", "lambda_j": "λ_j"}
STATE_NAMES = ["CC", "CD", "DC", "DD"]

STRATS = ["tit_for_tat", "generous_tft", "generous_tft_count",
          "wsls", "allc", "alld"]
LABELS = STRATS + ["adaptive", "adaptive_imm"]


def _make_opp(label, seed):
    if label == "adaptive":
        return AdaptiveAgent(seed=seed, kappa=0.9, sophisticated=True)
    if label == "adaptive_imm":
        return AdaptiveAgent(seed=seed, kappa=0.9, sophisticated=False)
    # 전략 내부 error=0 — 잡음은 환경 계층(env error)으로 양측 대칭 부과
    return make_opponent(label, seed=seed, error=0.0)


def _is_aif(x):
    return hasattr(x, "step")


def _force_focal_action(ad, action):
    """
    focal 의 이번 라운드 행동을 정책 수준에서 **덮어쓴다** (probe = focal 정책).

    단순히 방출 행동만 바꾸면(환경 잡음처럼) 에이전트의 자기 기록(my_last,
    협력율)이 선택 행동을 유지해, ToM 이 상대의 보복을 f=+1 에 오귀속시켜
    ρ̂ 를 희석시킨다. probe 는 '고의 배신'이므로 agent.step 말미의 자기 상태
    갱신(my_last / my_actions / my_coop_rate / ToM·inversion 의 p)을 덮어쓴
    행동으로 재수행한다. 환경 실행오류(의도≠방출)와 구별되는 지점이다.
    """
    ad.my_last = action
    ad.my_actions[-1] = action
    rate = float(np.mean(ad.my_actions))
    ad.social_efe.my_coop_rate = rate
    ad.tom.update_my_policy_belief(rate)
    ad.inversion.my_cooperation_rate = rate


def run_probe_dyad(agent, opponent, n_rounds, env_err, probe_eps,
                   noise_seed):
    """
    run_dyad 의 역학을 미러링하되, focal 을 확률 probe_eps 로 **정책 수준
    강제 배신**시키는 탐침 조건을 지원한다(probe_eps=0 → onpolicy 와 동역학
    동일). 탐침은 focal 의 자기 모형에 반영되고(_force_focal_action), 환경
    실행오류는 이후 별도로(비가시적으로) 부과된다 — 양측 대칭·사전 생성(CRN).
    """
    hist = {"my_act": [], "opp_act": [], "state": []}
    prev_state = None
    prev_mirror = None
    nrng = np.random.default_rng(int(noise_seed))
    flips_a = nrng.random(n_rounds) < env_err
    flips_b = nrng.random(n_rounds) < env_err
    probes = nrng.random(n_rounds) < probe_eps
    for t in range(n_rounds):
        my_a = agent.step(prev_state)
        if _is_aif(opponent):
            opp_a = opponent.step(prev_mirror)
        else:
            opp_a = opponent.act()
        if probes[t] and my_a != DEFECT:
            my_a = DEFECT                      # 탐침: 정책 수준 강제 배신
            _force_focal_action(agent, my_a)
        if flips_a[t]:
            my_a = 1 - my_a
        if flips_b[t]:
            opp_a = 1 - opp_a
        if not _is_aif(opponent):
            opponent.observe(my_a)
        state = joint_index(my_a, opp_a)
        hist["my_act"].append(my_a)
        hist["opp_act"].append(opp_a)
        hist["state"].append(state)
        prev_state = state
        prev_mirror = mirror_state(state)
    return {k: np.array(v) for k, v in hist.items()}


def one_dyad(label, policy, T, seed, env_err, probe_eps):
    """다이애드 1회: (θ̂_final, θ̂_mid, 사후sd_final, hist)."""
    ad = AdaptiveAgent(seed=100 + seed, kappa=0.9, sophisticated=True)
    opp = _make_opp(label, 900 + seed)
    eps = probe_eps if policy == "probe" else 0.0
    hist = run_probe_dyad(ad, opp, T, env_err, eps, noise_seed=seed)
    th_f = np.array([ad.log["E_alpha"][-1], ad.log["E_rho"][-1],
                     ad.log["E_beta"][-1], ad.log["E_lambda_j"][-1]])
    th_m = np.array([ad.log["E_alpha"][T // 2], ad.log["E_rho"][T // 2],
                     ad.log["E_beta"][T // 2], ad.log["E_lambda_j"][T // 2]])
    sd = ad.inversion.posterior_stds()
    sd_f = np.array([sd[a] for a in AXES])
    return th_f, th_m, sd_f, hist


# ------------------------------------------------------------------ Q2 도구
def _ident_coords(X):
    """식별 좌표계: (α+5λ, βρ, β)."""
    return np.column_stack([X[:, 0] + 5.0 * X[:, 3], X[:, 2] * X[:, 1],
                            X[:, 2]])


def loo_decode(Z, y, n_cls):
    """풀드-공분산 마할라노비스 최근접-중심 LOO 해독 (정확도, 혼동행렬)."""
    cov = np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1])
    ic = np.linalg.inv(cov)
    n = len(y)
    conf = np.zeros((n_cls, n_cls))
    correct = 0
    for i in range(n):
        mask = np.arange(n) != i
        d = np.full(n_cls, np.inf)
        for c in range(n_cls):
            sel = mask & (y == c)
            if not sel.any():
                continue
            mu = Z[sel].mean(axis=0)
            diff = Z[i] - mu
            d[c] = diff @ ic @ diff
        pred = int(np.argmin(d))
        conf[y[i], pred] += 1
        correct += int(pred == y[i])
    return correct / n, conf


def perm_pvalue(Z, y, n_cls, acc_obs, n_perm, seed=0):
    """라벨 순열 널분포 대비 해독 정확도의 p 값."""
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = rng.permutation(y)
        null[b], _ = loo_decode(Z, yp, n_cls)
    p = (1.0 + np.sum(null >= acc_obs)) / (n_perm + 1.0)
    return float(p), null


def bhattacharyya(Xa, Xb):
    """두 θ̂ 구름 간 가우시안 근사 Bhattacharyya 거리."""
    ma, mb = Xa.mean(0), Xb.mean(0)
    Ca = np.cov(Xa.T) + 1e-6 * np.eye(Xa.shape[1])
    Cb = np.cov(Xb.T) + 1e-6 * np.eye(Xb.shape[1])
    Cm = 0.5 * (Ca + Cb)
    d = ma - mb
    t1 = 0.125 * d @ np.linalg.inv(Cm) @ d
    t2 = 0.5 * np.log(np.linalg.det(Cm)
                      / np.sqrt(np.linalg.det(Ca) * np.linalg.det(Cb)))
    return float(t1 + t2)


# ------------------------------------------------------------------ Q3 도구
def memory_one_signature(hists):
    """경험적 기억-1 시그니처: P(상대 C at t | state_{t-1}=k), k∈{CC,CD,DC,DD}."""
    cnt = np.zeros(4)
    coop = np.zeros(4)
    for h in hists:
        s_prev = h["state"][:-1]
        o_next = h["opp_act"][1:]
        for k in range(4):
            m = s_prev == k
            cnt[k] += m.sum()
            coop[k] += np.sum(o_next[m] == COOP)
    with np.errstate(invalid="ignore"):
        sig = np.where(cnt > 0, coop / np.maximum(cnt, 1), np.nan)
    return sig, cnt


def model_implied_signature(theta_hat, p_focal):
    """P_θ̂ 함의 조건확률 — f 는 focal 직전 행동에만 의존(CC,CD→+1; DC,DD→−1)."""
    a, r, b, l = theta_hat
    out = np.empty(4)
    for k in range(4):
        f = +1.0 if k in (0, 1) else -1.0
        out[k] = 1.0 / (1.0 + np.exp(-b * (a + r * f
                                           + C_empathy_shift(l, p_focal))))
    return out


# ------------------------------------------------------------------- 그림
def make_figure(res, args):
    try:
        set_korean_font(plt, font_manager)
    except Exception:
        pass
    n_cls = len(LABELS)
    cols = plt.cm.tab10(np.linspace(0, 1, n_cls))
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))

    # (a) probe 조건 식별 좌표 산점 (μ=α+5λ vs βρ)
    a = ax[0, 0]
    Xp = np.array(res["X"]["probe"]); yp = np.array(res["y"]["probe"])
    Zp = _ident_coords(Xp)
    for c in range(n_cls):
        m = yp == c
        a.scatter(Zp[m, 0], Zp[m, 1], s=22, color=cols[c], alpha=0.75,
                  label=LABELS[c])
    a.set_xlabel("식별 좌표 μ = α̂ + 5λ̂")
    a.set_ylabel("식별 좌표 β̂·ρ̂")
    a.set_title("(a) θ̂ 사영 구름 (probe 조건, 식별 좌표)", fontsize=10)
    a.legend(fontsize=6, ncol=2)

    # (b) probe 혼동행렬 (원 좌표 4D)
    a = ax[0, 1]
    M = np.array(res["confusion"]["probe_raw"]) / args.seeds
    im = a.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(n_cls):
        for j in range(n_cls):
            if M[i, j] > 0.005:
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       fontsize=6.2,
                       color="w" if M[i, j] > 0.5 else "k")
    a.set_xticks(range(n_cls)); a.set_yticks(range(n_cls))
    a.set_xticklabels(LABELS, rotation=90, fontsize=6.5)
    a.set_yticklabels(LABELS, fontsize=6.5)
    acc = res["decode"]["probe_raw"]["acc"]
    pval = res["decode"]["probe_raw"]["p_perm"]
    a.set_title(f"(b) 유형 혼동행렬 (probe, 원 θ̂ 4D)\n"
                f"LOO 해독={acc:.2f} (우연 {1/n_cls:.3f}, 순열 p={pval:.3g})",
                fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.046)

    # (c) Q4: 해독 정확도 — 정책 × 좌표계 + 순열 널 95%
    a = ax[0, 2]
    keys = [("onpolicy_raw", "on/원4D"), ("onpolicy_ident", "on/식별3D"),
            ("probe_raw", "probe/원4D"), ("probe_ident", "probe/식별3D")]
    accs = [res["decode"][k]["acc"] for k, _ in keys]
    null95 = max(res["decode"][k]["null95"] for k, _ in keys)
    a.bar(range(4), accs, color=["0.6", "0.6", "C0", "C0"])
    a.axhline(1 / n_cls, color="k", ls=":", label=f"우연={1/n_cls:.3f}")
    a.axhline(null95, color="r", ls="--", label="순열 널 95백분위")
    a.set_xticks(range(4)); a.set_xticklabels([t for _, t in keys],
                                              rotation=20, fontsize=8)
    a.set_ylabel("LOO 해독 정확도")
    a.set_title("(c) [Q4] 동치류 분해 — 정책·좌표계별 분리성\n"
                "probe−onpolicy 차이 = 정책이 남긴 정보 손실", fontsize=10)
    a.legend(fontsize=7)

    # (d) Q1: θ* 기준 복원 — 표준화 RMSE + 커버리지 + 분할-반
    a = ax[1, 0]
    rm = [res["q1"][lab]["rmse_std"] for lab in LABELS]
    cv = [res["q1"][lab]["coverage_90"] for lab in LABELS]
    x = np.arange(n_cls)
    a.bar(x - 0.2, rm, 0.4, color="C0", label="표준화 RMSE(θ̂, θ*)")
    a.bar(x + 0.2, cv, 0.4, color="C2", label="90% 커버리지(θ* 기준)")
    a.axhline(0.90, color="r", ls="--", lw=1)
    a.set_xticks(x); a.set_xticklabels(LABELS, rotation=90, fontsize=6.5)
    sh = res["q1_splithalf"]
    a.set_title(f"(d) [Q1] 사영 복원성 (probe)\n분할-반 상관 "
                f"(α,ρ,β,λ) = {', '.join(f'{v:.2f}' for v in sh)}",
                fontsize=10)
    a.legend(fontsize=7)

    # (e) Q3: 충실도 — 유형별 시그니처 RMSE (+ WSLS 구조적 잔차 강조)
    a = ax[1, 1]
    fid = [res["q3"][lab]["rmse"] for lab in LABELS]
    bars = a.bar(range(n_cls), fid,
                 color=["C3" if lab == "wsls" else "C0" for lab in LABELS])
    a.set_xticks(range(n_cls)); a.set_xticklabels(LABELS, rotation=90,
                                                  fontsize=6.5)
    a.set_ylabel("시그니처 RMSE (경험 vs 모형함의)")
    a.set_title("(e) [Q3] 사영 충실도 — 기억-1 시그니처\n"
                "모형은 CC≡CD, DC≡DD 붕괴 (joint 조건 WSLS 표현 불가 → 적색)",
                fontsize=10)

    # (f) 드리프트: |θ̂_final − θ̂_mid| 표준화 노름 (adaptive 강조)
    a = ax[1, 2]
    dr = [res["drift"][lab] for lab in LABELS]
    a.bar(range(n_cls), dr,
          color=["C1" if lab.startswith("adaptive") else "0.6"
                 for lab in LABELS])
    a.set_xticks(range(n_cls)); a.set_xticklabels(LABELS, rotation=90,
                                                  fontsize=6.5)
    a.set_ylabel("‖θ̂(T)−θ̂(T/2)‖ (축 표준화)")
    a.set_title("(f) 정적 사영 전제 점검 — θ̂ 드리프트\n"
                "(adaptive 상대는 λ 조절로 θ 시변 → 주황)", fontsize=10)

    fig.suptitle(f"모형 밖 상대의 θ̂ 사영: 복원성·분리성·충실도·동치류 | "
                 f"{len(LABELS)}유형 × seeds={args.seeds} × T={args.rounds} "
                 f"× 정책 2 (probe ε={args.probe_eps})", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "projection_tom.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "projection_tom.pdf", bbox_inches="tight")
    with open(RESULTS / "projection_tom.caption.json", "w",
              encoding="utf-8") as f:
        json.dump({"figure": "projection_tom",
                   "caption":
                       "모형 밖 상대(고정전략·adaptive)의 θ̂ 사영 검증. "
                       "(a) probe 조건 식별 좌표 구름, (b) 혼동행렬+순열검정, "
                       "(c) 정책×좌표계 해독 — probe 가 행동적 동치류"
                       "(협력 균형의 TFT/GTFT/ALLC)를 가른다, (d) 의사-참 "
                       "사영점 θ* 기준 복원성, (e) 기억-1 시그니처 충실도"
                       "(WSLS 는 joint 조건이라 구조적 잔차), (f) adaptive "
                       "상대의 θ̂ 드리프트(정적 사영 전제 점검)."},
                  f, ensure_ascii=False, indent=2)
    plt.close(fig)
    LOGGER.info("그림 저장: %s", RESULTS / "projection_tom.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=240)
    ap.add_argument("--star-rounds", type=int, default=1200)
    ap.add_argument("--star-seeds", type=int, default=5)
    ap.add_argument("--probe-eps", type=float, default=0.25)
    ap.add_argument("--error", type=float, default=0.05)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.rounds = 5, 120
        args.star_rounds, args.star_seeds, args.perms = 400, 2, 60

    set_coop_index(None)
    n_cls = len(LABELS)
    LOGGER.info("사영 연구: %d유형 × seeds=%d × T=%d × 정책2 | θ*: T=%d×%d",
                n_cls, args.seeds, args.rounds, args.star_rounds,
                args.star_seeds)

    # ---- 본 조건 수집 ----
    X = {"onpolicy": [], "probe": []}
    Xmid, SD, HISTS = {"onpolicy": [], "probe": []}, {"onpolicy": [], "probe": []}, {}
    y = {"onpolicy": [], "probe": []}
    pfoc = {}
    for li, lab in enumerate(LABELS):
        HISTS[lab] = []
        pf = []
        for policy in ("onpolicy", "probe"):
            for sd in range(args.seeds):
                th, thm, sdv, h = one_dyad(lab, policy, args.rounds, sd,
                                           args.error, args.probe_eps)
                X[policy].append(th); Xmid[policy].append(thm)
                SD[policy].append(sdv); y[policy].append(li)
                if policy == "probe":
                    HISTS[lab].append(h)
                    pf.append(float(np.mean(h["my_act"] == COOP)))
        pfoc[lab] = float(np.mean(pf))
        LOGGER.info("  [%s] 수집 완료 (focal 협력율 probe=%.2f)", lab, pfoc[lab])
    for k in X:
        X[k] = np.array(X[k]); Xmid[k] = np.array(Xmid[k])
        SD[k] = np.array(SD[k]); y[k] = np.array(y[k])

    # ---- θ* (의사-참 사영점; probe·장지평) ----
    theta_star = {}
    for lab in LABELS:
        ths = [one_dyad(lab, "probe", args.star_rounds, 7000 + s,
                        args.error, args.probe_eps)[0]
               for s in range(args.star_seeds)]
        theta_star[lab] = np.mean(ths, axis=0)
    LOGGER.info("θ* 추정 완료 (T=%d × %d시드)", args.star_rounds,
                args.star_seeds)

    # ---- Q1: 복원성 (probe) ----
    ax_sd = X["probe"].std(axis=0) + 1e-9           # 축 표준화 스케일
    q1 = {}
    for li, lab in enumerate(LABELS):
        m = y["probe"] == li
        E = X["probe"][m]; S = SD["probe"][m]
        err = (E - theta_star[lab][None, :]) / ax_sd[None, :]
        # 커버리지 = (다이애드 × 축) 전체에서 90% 사후구간이 θ* 를 포함한 비율
        cov = float(np.mean(
            np.abs(E - theta_star[lab][None, :]) <= 1.645 * S))
        q1[lab] = {"rmse_std": float(np.sqrt(np.mean(err ** 2))),
                   "coverage_90": float(cov),
                   "theta_star": theta_star[lab].tolist()}
    sh = [float(np.corrcoef(X["probe"][:, k], Xmid["probe"][:, k])[0, 1])
          for k in range(4)]

    # ---- Q2/Q4: 분리성 ----
    decode, confusion = {}, {}
    for policy in ("onpolicy", "probe"):
        for tag, Z in (("raw", X[policy]), ("ident", _ident_coords(X[policy]))):
            acc, conf = loo_decode(Z, y[policy], n_cls)
            p, null = perm_pvalue(Z, y[policy], n_cls, acc, args.perms,
                                  seed=11)
            key = f"{policy}_{tag}"
            decode[key] = {"acc": float(acc), "p_perm": p,
                           "null95": float(np.percentile(null, 95))}
            confusion[key] = conf.tolist()
            LOGGER.info("  [Q2] %s: 해독=%.2f (우연=%.3f, 순열 p=%.3g)",
                        key, acc, 1 / n_cls, p)
    # 쌍별 Bhattacharyya (probe, 식별 좌표)
    Zp = _ident_coords(X["probe"])
    bd = {}
    for i in range(n_cls):
        for j in range(i + 1, n_cls):
            bd[f"{LABELS[i]}|{LABELS[j]}"] = bhattacharyya(
                Zp[y["probe"] == i], Zp[y["probe"] == j])

    # ---- Q3: 충실도 (probe) ----
    q3 = {}
    for li, lab in enumerate(LABELS):
        sig, cnt = memory_one_signature(HISTS[lab])
        th_mean = X["probe"][y["probe"] == li].mean(axis=0)
        mod = model_implied_signature(th_mean, pfoc[lab])
        ok = ~np.isnan(sig)
        q3[lab] = {"empirical": [None if np.isnan(v) else float(v)
                                 for v in sig],
                   "model": mod.tolist(),
                   "counts": cnt.tolist(),
                   "rmse": float(np.sqrt(np.mean((sig[ok] - mod[ok]) ** 2)))}
        LOGGER.info("  [Q3] %-18s 시그니처 RMSE=%.3f", lab, q3[lab]["rmse"])

    # ---- 드리프트 ----
    drift = {}
    for li, lab in enumerate(LABELS):
        m = y["probe"] == li
        d = (X["probe"][m] - Xmid["probe"][m]) / ax_sd[None, :]
        drift[lab] = float(np.mean(np.linalg.norm(d, axis=1)))

    res = {"X": {k: v.tolist() for k, v in X.items()},
           "y": {k: v.tolist() for k, v in y.items()},
           "decode": decode, "confusion": confusion,
           "bhattacharyya_probe_ident": bd,
           "q1": q1, "q1_splithalf": sh, "q3": q3, "drift": drift,
           "focal_coop_rate_probe": pfoc,
           "labels": LABELS,
           "config": vars(args)}
    # 그림은 원 배열 필요 — res 를 그대로 넘김
    res_fig = dict(res)
    make_figure(res_fig, args)
    with open(RESULTS / "projection_tom.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    LOGGER.info("결과 저장: %s", RESULTS / "projection_tom.json")
    LOGGER.info("요약: probe 해독(원4D)=%.2f p=%.3g | onpolicy=%.2f | "
                "Q4 이득=%+.2f",
                decode["probe_raw"]["acc"], decode["probe_raw"]["p_perm"],
                decode["onpolicy_raw"]["acc"],
                decode["probe_raw"]["acc"] - decode["onpolicy_raw"]["acc"])


if __name__ == "__main__":
    main()

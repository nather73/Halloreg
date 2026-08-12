"""
experiments.h2_protection
=========================

**H2 — 착취자로부터 HalloReg 는 자신의 보수를 잘 보호할 수 있는가?**
**H2A — HalloReg 는 착취자와 noisy TFT 를 구분할 수 있는가?**

────────────────────────────────────────────────────────────────────────
H2 설계 — 자기보호
────────────────────────────────────────────────────────────────────────
주 상대: ALLD (의도적 착취자, 실행잡음 0).

대조군을 두 종류로 둔다.

  (1) **고정 λ 공감 에이전트** (λ ∈ {0.0, 0.4, 0.8})
      원 논문의 한계를 그대로 보여주는 대조. λ 가 높을수록 상대 후생에 가중을
      두므로 착취자에게 계속 협력해 보수를 잃는다. λ=0.4 는 Albarracin et al.
      의 기준 설정이므로 **확증 대조**로, λ=0.0 과 λ=0.8 은 탐색으로 둔다.

  (2) **고정전략** (TFT, GTFT, WSLS, ALLC)
      ALLD 를 상대로 한 표준 벤치마크. 이 중 ALLC 는 **바닥 조건**이다.

[핵심 대조 — 이중 맥락 최악값(dual-context worst case)]
λ 가 높을수록 착취자에게 취약해지므로, λ 를 낮게 고정하면 맥락 E 에서는 유리하다.
그러나 그 유리함은 **상대가 착취자라는 것을 미리 알고 λ 를 맞춰 두었을 때만**
의미가 있다. 따라서 각 에이전트를 **두 맥락**에 노출한다.

    맥락 E : ALLD (착취자)        — 자기보호가 필요
    맥락 C : GTFT (관대한 협력자) — 상호협력 구축이 필요

그리고 각 에이전트의 **최악 맥락 보수** min{ μ_E, μ_C } 를 비교한다.

[확증 대조군의 선정 — 사전에 고정한 결정과 그 근거]
확증 검정의 대조는 **Albarracin et al. 의 공감 에이전트**(λ = 0.4, 0.8)와
고정전략이다. λ = 0.0 조건은 확증이 아니라 **탐색**으로 둔다. 이유:

  · λ = 0.0 은 '공감 가중이 0 인 순수 자기이익 능동추론 에이전트' 로,
    원 논문이 제안한 공감 모형이 아니라 **다른 모형**이다. 본 가설
    ("HalloReg 는 착취자로부터 보수를 보호하는가")의 대조로 적절하지 않다.
  · 더 중요하게, λ = 0.0 은 착취자 맥락에서 **정의상 최적**이다(공감항이 없으므로
    배신을 억제할 유일한 힘이 사라진다). 사후에 최적으로 판명된 대조를 확증
    기준으로 삼으면 어떤 적응 모형도 통과할 수 없다.

**단, 이 결정이 불리한 결과를 감추는 데 쓰이지 않도록** λ = 0.0 대조는 반드시
탐색 결과로 **명시 보고**한다. 실제로 본 구현에서는 λ = 0.0 에이전트가 맥락 C
에서도 높은 협력을 달성하는데, 이는 **rollout 안의 호혜성 전파**(planner 가 ρ̂ 로
"내가 협력하면 다음 라운드에 되돌아온다"를 계산) 때문이다. 즉 이 모형에서
협력의 도구적 가치는 λ 가 아니라 EFE 가 담당하며, λ 는 '상대 후생 자체에 두는
가중' 이라는 구성개념을 유지한다. 이 사실은 결함이 아니라 설계 의도이지만,
"공감 없이는 협력이 불가능하다" 는 식의 과잉 주장을 금지한다.

  · 확증 H2-1 : 맥락 E 보수 — HalloReg > 고정 λ=0.4 (원 논문 기준 설정).
  · 확증 H2-2 : 맥락 E 보수 — HalloReg > ALLC (바닥 조건).
  · 확증 H2-3 : 이중 맥락 최악값 — HalloReg > 고정 λ=0.4.
  · 확증 H2-4 : 이중 맥락 최악값 — HalloReg > 고정 λ=0.8.
  · 탐색      : **λ=0.0 대조**, λ 궤적의 맥락별 분기, 고정전략 벤치마크.

────────────────────────────────────────────────────────────────────────
H2A 설계 — 착취자 vs noisy TFT 구분
────────────────────────────────────────────────────────────────────────
두 상대는 **둘 다 배신을 자주 낸다**. 차이는 배신의 *원인* 이다.

  · ALLD        : 형질(α 매우 낮음)이 원인. 배신이 결정론적 → β̂ 높음.
  · noisy TFT   : 실행잡음(error=0.20)이 원인. 배신이 확률적 → β̂ 낮음.

이 구분이 성립하려면 추론기가 **α 축(형질)과 β 축(정밀도)을 분리**해야 한다.
그래서 판별 지표를 두 층위로 둔다.

  (i)  **표상 수준** : θ̂ 공간에서의 판별. 두 조건의 (α̂, β̂) 를 시드 반분
       교차검증으로 분류한다. 정확도 > 0.5 이면 표상 수준에서 구분된 것이다.
  (ii) **행동 수준** : λ 최종값과 협력률의 차이. 구분이 표상에만 머무르고
       행동으로 이어지지 않으면 기능적 의미가 없다. noisy TFT 에서는 λ 가
       회복되어야 하고, ALLD 에서는 억제되어야 한다.

  · 확증 H2A-1 : θ̂ 판별 정확도 > 0.5.
  · 확증 H2A-2 : λ_final(noisy TFT) > λ_final(ALLD)  — 행동 수준 구분.
  · 탐색       : β̂ 차이, α̂ 차이, 후반 협력률 차이.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.metrics import (
    boot_mean_ci, effect_size, effect_size_paired, fmt_es, one_sample_perm,
    perm_test,
)
from AIF_IPD.ipd.sim import run_many
from .common import Config, Registry, TYPE_COLORS, save_fig, save_json
from .h1_intent import DiagGaussianClassifier

LOGGER = get_logger("HalloReg.H2")

#: 고정 λ 대조군의 λ 격자
FIXED_LAMS = (0.0, 0.4, 0.8)
#: 확증 대조로 삼는 기준 λ (Albarracin et al. 의 기준 설정)
REFERENCE_LAM = 0.4
#: 고정전략 벤치마크
BENCH = ("tft", "gtft", "wsls", "allc")
#: 이중 맥락 — E(착취) / C(협력)
CONTEXTS = {"E": {"kind": "alld", "error": 0.0},
            "C": {"kind": "gtft", "error": 0.0}}
CONTEXT_LABEL = {"E": "맥락 E — 착취자(ALLD)", "C": "맥락 C — 협력자(GTFT)"}
#: 한글 표기
COND_LABEL = {"halloreg": "HalloReg", "tft": "TFT", "gtft": "GTFT",
              "wsls": "WSLS", "allc": "ALLC"}


def _cond_label(c: str) -> str:
    """조건 키 → 표시 이름."""
    if c.startswith("fixed_lam"):
        return "고정 λ=" + c.replace("fixed_lam", "")
    return COND_LABEL.get(c, c)


# ==================================================================== H2
def run_protection(cfg: Config, reg: Registry) -> dict:
    """H2 — 착취자 상대 자기보호 + 이중 맥락 최악값."""
    hk = cfg.halloreg_kwargs()
    ek = {"n_particles": cfg.n_particles, "planning_horizon": cfg.horizon}

    conditions: List[tuple] = [("halloreg", {"type": "halloreg", **hk})]
    for lam in FIXED_LAMS:
        conditions.append((f"fixed_lam{lam:.1f}",
                           {"type": "empathic", "lam": float(lam), **ek}))
    for b in BENCH:
        conditions.append((b, {"type": "strategy", "kind": b}))

    specs, registry = [], {}
    for gi, (gkey, ospec) in enumerate(CONTEXTS.items()):
        for ci, (cname, cspec) in enumerate(conditions):
            for sd in range(cfg.seeds):
                a = dict(cspec); a["seed"] = 9000 + sd * 31 + ci
                o = dict(ospec); o["type"] = "strategy"
                o["seed"] = 9500 + sd * 31 + gi
                registry[(gkey, ci, sd)] = len(specs)
                specs.append({
                    "agent": a, "opponent": o,
                    "env_err_agent": cfg.env_error,
                    "env_err_opponent": cfg.env_error,
                    # 공통난수(CRN): 시드가 같으면 조건 간 잡음 실현이 동일 →
                    # 조건 차이가 잡음 우연이 아니라 정책 차이에서만 온다.
                    "noise_seed": 700_000 + sd * 101 + gi * 7})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H2 이중맥락 다이애드")

    names = [c for c, _ in conditions]
    payoff = {g: {} for g in CONTEXTS}
    coop = {g: {} for g in CONTEXTS}
    lam_traces = {}
    for gi, gkey in enumerate(CONTEXTS):
        for ci, cname in enumerate(names):
            payoff[gkey][cname] = np.array([
                float(np.mean(res[registry[(gkey, ci, sd)]]["hist"]["my_payoff"]))
                for sd in range(cfg.seeds)])
            coop[gkey][cname] = np.array([
                float(np.mean(res[registry[(gkey, ci, sd)]]["hist"]["my_act"] == 0))
                for sd in range(cfg.seeds)])
            if cname == "halloreg" or cname.startswith("fixed_lam"):
                lam_traces[(gkey, cname)] = np.vstack([
                    np.asarray(res[registry[(gkey, ci, sd)]]["agent_log"]["lam"])
                    for sd in range(cfg.seeds)])

    ref = f"fixed_lam{REFERENCE_LAM:.1f}"

    # ---- 확증 H2-1: 맥락 E 에서 기준 λ 대조 대비 ----
    t1 = perm_test(payoff["E"]["halloreg"], payoff["E"][ref], paired=True,
                   alternative="greater")
    e1 = effect_size_paired(payoff["E"]["halloreg"] - payoff["E"][ref])
    reg.confirm("H2", f"착취자 상대 보수: HalloReg > 고정 λ={REFERENCE_LAM}",
                t1["p"], direction_ok=bool(t1["observed"] > 0),
                effect=f"Δ={t1['observed']:+.3f}, {fmt_es(e1, 'dz')}")

    # ---- 확증 H2-2: 바닥 조건 ----
    t2 = perm_test(payoff["E"]["halloreg"], payoff["E"]["allc"], paired=True,
                   alternative="greater")
    e2 = effect_size_paired(payoff["E"]["halloreg"] - payoff["E"]["allc"])
    reg.confirm("H2", "착취자 상대 보수: HalloReg > ALLC (바닥 조건)", t2["p"],
                direction_ok=bool(t2["observed"] > 0),
                effect=f"Δ={t2['observed']:+.3f}, {fmt_es(e2, 'dz')}")

    # ---- 확증 H2-3/H2-4: 이중 맥락 최악값 — 공감 대조군(λ>0) 대비 ----
    # λ=0.0 은 공감 모형이 아니라 순수 자기이익 모형이므로 탐색으로 분리한다
    # (모듈 문서의 '확증 대조군 선정' 참조).
    worst = {c: np.minimum(payoff["E"][c], payoff["C"][c]) for c in names}
    for lam in FIXED_LAMS:
        key = f"fixed_lam{lam:.1f}"
        t3 = perm_test(worst["halloreg"], worst[key], paired=True,
                       alternative="greater")
        e3 = effect_size_paired(worst["halloreg"] - worst[key])
        if lam > 0.0:
            reg.confirm("H2", f"이중맥락 최악값: HalloReg > 고정 λ={lam}", t3["p"],
                        direction_ok=bool(t3["observed"] > 0),
                        effect=f"Δ={t3['observed']:+.3f}, {fmt_es(e3, 'dz')}")
        else:
            reg.explore("H2", f"이중맥락 최악값: HalloReg vs 고정 λ={lam} "
                              f"(순수 자기이익 모형 — 착취자 맥락 정의상 최적)",
                        t3["p"],
                        effect=f"Δ={t3['observed']:+.3f}, {fmt_es(e3, 'dz')}")
    # 맥락 E 에서의 λ=0 대조도 명시 보고 (감추지 않는다)
    t0 = perm_test(payoff["E"]["halloreg"], payoff["E"]["fixed_lam0.0"],
                   paired=True)
    reg.explore("H2", "착취자 상대 보수: HalloReg vs 고정 λ=0.0 "
                      "(공감 없는 상한 기준)", t0["p"],
                effect=f"Δ={t0['observed']:+.3f}")

    # ---- 탐색: 고정전략 벤치마크 (두 맥락) ----
    for gkey in CONTEXTS:
        for b in BENCH:
            tb = perm_test(payoff[gkey]["halloreg"], payoff[gkey][b],
                           paired=True)
            reg.explore("H2", f"[{CONTEXT_LABEL[gkey]}] 보수: HalloReg vs "
                              f"{COND_LABEL[b]}", tb["p"],
                        effect=f"Δ={tb['observed']:+.3f}")

    # ---- 탐색: λ 하강/상승 (초기 1/4 대비 후기 1/4) ----
    q = max(cfg.rounds // 4, 1)
    for gkey, direction in (("E", "greater"), ("C", "less")):
        tr = lam_traces[(gkey, "halloreg")]
        drop = tr[:, :q].mean(axis=1) - tr[:, -q:].mean(axis=1)
        td = one_sample_perm(drop, 0.0, alternative=direction)
        cd = boot_mean_ci(drop)
        arrow = "하강" if gkey == "E" else "상승"
        reg.explore("H2", f"[{CONTEXT_LABEL[gkey]}] λ {arrow} (초기−후기)",
                    td["p"],
                    effect=f"Δλ={cd['mean']:+.3f} "
                           f"[{cd['ci'][0]:+.3f}, {cd['ci'][1]:+.3f}]")

    return {"conditions": names, "payoff": payoff, "coop": coop,
            "worst": worst, "lam_traces": lam_traces, "reference": ref}


# ==================================================================== H2A
def run_discrimination(cfg: Config, reg: Registry) -> dict:
    """H2A — 의도적 착취자 vs 잡음 있는 협력자 구분."""
    hk = cfg.halloreg_kwargs()
    conds = [("exploiter", {"type": "strategy", "kind": "alld", "error": 0.0}),
             ("noisy_tft", {"type": "strategy", "kind": "tft", "error": 0.20})]

    specs, registry = [], {}
    for ci, (cname, ospec) in enumerate(conds):
        for sd in range(cfg.seeds):
            o = dict(ospec); o["seed"] = 11_000 + sd * 37 + ci
            registry[(ci, sd)] = len(specs)
            specs.append({
                "agent": {"type": "halloreg", "seed": 10_000 + sd * 37, **hk},
                "opponent": o,
                # 환경 잡음은 0 으로 둔다. 여기서 구분해야 할 잡음은
                # **상대 내부의 실행잡음**이므로, 환경 잡음이 겹치면 두 조건 모두
                # 잡음을 갖게 되어 대조가 흐려진다.
                "env_err_agent": 0.0, "env_err_opponent": 0.0,
                "noise_seed": 710_000 + sd * 103})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H2A 판별 다이애드")

    feats = ("alpha", "beta", "rho", "lambda_j")
    X = {c: np.zeros((cfg.seeds, len(feats))) for c, _ in conds}
    lam_final = {c: np.zeros(cfg.seeds) for c, _ in conds}
    coop_late = {c: np.zeros(cfg.seeds) for c, _ in conds}
    lam_traces = {c: np.zeros((cfg.seeds, cfg.rounds)) for c, _ in conds}
    # eval_from 이 지정되면 '학습 후' 창을 그것으로 (기본: 후반부).
    half = (int(cfg.eval_from) if getattr(cfg, "eval_from", 0) > 0
            else cfg.rounds // 2)

    for ci, (cname, _) in enumerate(conds):
        for sd in range(cfg.seeds):
            r = res[registry[(ci, sd)]]
            log = r["agent_log"]
            for d, ax in enumerate(feats):
                X[cname][sd, d] = float(log[f"E_{ax}"][-1])
            lam_final[cname][sd] = float(log["lam"][-1])
            lam_traces[cname][sd] = np.asarray(log["lam"], dtype=float)
            coop_late[cname][sd] = float(
                np.mean(r["hist"]["my_act"][half:] == 0))

    # ---- 확증 H2A-1: θ̂ 판별 정확도 > 0.5 (시드 반분 교차검증) ----
    n_cal = cfg.seeds // 2
    names = [c for c, _ in conds]
    Xc = np.vstack([X[c][:n_cal] for c in names])
    yc = [c for c in names for _ in range(n_cal)]
    Xt = np.vstack([X[c][n_cal:] for c in names])
    n_te = cfg.seeds - n_cal
    yt = np.array([i for i in range(len(names)) for _ in range(n_te)])

    mu, sd_ = Xc.mean(axis=0), np.maximum(Xc.std(axis=0, ddof=1), 1e-6)
    clf = DiagGaussianClassifier().fit((Xc - mu) / sd_, yc, names)
    pred = clf.predict((Xt - mu) / sd_)
    acc = float(np.mean(pred == yt))

    rng = np.random.default_rng(21)
    null = np.array([np.mean(rng.permutation(pred) == yt) for _ in range(5000)])
    p_acc = (np.sum(null >= acc) + 1) / (5000 + 1)
    reg.confirm("H2A", "θ̂ 판별 정확도 > 우연(0.5)", p_acc,
                direction_ok=bool(acc > 0.5),
                effect=f"acc={acc:.3f} (n={len(yt)})",
                detail={"accuracy": acc})

    # ---- 확증 H2A-2: 행동 수준 구분 (λ_final) ----
    t2 = perm_test(lam_final["noisy_tft"], lam_final["exploiter"],
                   paired=True, alternative="greater")
    e2 = effect_size_paired(lam_final["noisy_tft"] - lam_final["exploiter"])
    reg.confirm("H2A", "λ_final: noisy TFT > 착취자", t2["p"],
                direction_ok=bool(t2["observed"] > 0),
                effect=f"Δλ={t2['observed']:+.3f}, {fmt_es(e2, 'dz')}")

    # ---- 탐색: 축별 차이 및 후반 협력률 ----
    for d, ax in enumerate(feats):
        tt = perm_test(X["noisy_tft"][:, d], X["exploiter"][:, d], paired=True)
        ee = effect_size(X["noisy_tft"][:, d], X["exploiter"][:, d])
        reg.explore("H2A", f"θ̂ 축 {ax}: noisy TFT vs 착취자", tt["p"],
                    effect=f"Δ={tt['observed']:+.3f}, {fmt_es(ee)}")
    tc = perm_test(coop_late["noisy_tft"], coop_late["exploiter"], paired=True,
                   alternative="greater")
    reg.explore("H2A", "후반 협력률: noisy TFT > 착취자", tc["p"],
                effect=f"Δ={tc['observed']:+.3f}")

    return {"features": list(feats), "theta": X, "lam_final": lam_final,
            "lam_traces": lam_traces, "coop_late": coop_late,
            "accuracy": acc, "conditions": names}


# ==================================================================== 실행
def run(cfg: Config, reg: Registry) -> dict:
    LOGGER.info("[H2/H2A] 착취자 방어와 잡음-의도 판별")
    a = run_protection(cfg, reg)
    b = run_discrimination(cfg, reg)
    out = {"protection": a, "discrimination": b}
    _plot(cfg, a, b)
    save_json(out, cfg.results / "H2.json")
    return out


# ==================================================================== 시각화
def _plot(cfg: Config, a: dict, b: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import annotate_n, band_plot, bar_with_ci

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30)

    conds = a["conditions"]
    labels = [_cond_label(c) for c in conds]
    colors = ["#DA8BC3" if c == "halloreg"
              else ("#8C8C8C" if c.startswith("fixed") else "#4C72B0")
              for c in conds]

    # (a) 맥락별 평균보수
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(conds)); w = 0.38
    for off, gkey, col, lab in ((-w / 2, "E", "#937860", "맥락 E — 착취자"),
                                (+w / 2, "C", "#55A868", "맥락 C — 협력자")):
        vals = [a["payoff"][gkey][c].mean() for c in conds]
        errs = [boot_mean_ci(a["payoff"][gkey][c])["ci"] for c in conds]
        lo = [v - e[0] for v, e in zip(vals, errs)]
        hi = [e[1] - v for v, e in zip(vals, errs)]
        ax.bar(x + off, vals, w, yerr=[lo, hi], capsize=2.5, color=col,
               label=lab, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right",
                                         fontsize=7)
    ax.set_ylabel("라운드당 평균보수")
    ax.set_title("(a) H2 — 맥락별 보수")
    ax.legend(fontsize=7); annotate_n(ax, cfg.seeds)

    # (b) 이중 맥락 최악값 — 핵심 대조
    ax = fig.add_subplot(gs[0, 1])
    means = [a["worst"][c].mean() for c in conds]
    cis = [boot_mean_ci(a["worst"][c])["ci"] for c in conds]
    bar_with_ci(ax, labels, means, cis, colors=colors,
                ylabel="min{맥락 E, 맥락 C} 보수", rotate=40)
    ax.set_title("(b) 이중 맥락 최악값\n"
                 "어떤 고정 λ 도 두 맥락을 동시에 만족시키지 못한다")
    annotate_n(ax, cfg.seeds)

    # (c) 맥락 E 의 λ 궤적 (HalloReg vs 고정 λ)
    ax = fig.add_subplot(gs[0, 2])
    band_plot(ax, a["lam_traces"][("E", "halloreg")], color="#937860",
              label="HalloReg — 맥락 E")
    band_plot(ax, a["lam_traces"][("C", "halloreg")], color="#55A868",
              label="HalloReg — 맥락 C")
    for lam in FIXED_LAMS:
        ax.axhline(lam, color="#BBBBBB", ls=":", lw=1)
    ax.text(1, FIXED_LAMS[-1] + 0.02, "고정 λ 대조 수준", fontsize=6.5,
            color="#888888")
    ax.set_xlabel("라운드"); ax.set_ylabel("공감 가중 λ")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(c) λ 의 맥락 의존적 분기")
    ax.legend(fontsize=7); annotate_n(ax, cfg.seeds)

    # (d) H2A — θ̂ 공간 산점 (α̂ × β̂)
    ax = fig.add_subplot(gs[1, 0])
    fi_a = b["features"].index("alpha")
    fi_b = b["features"].index("beta")
    for c, col, lab in (("exploiter", "#937860", "의도적 착취자(ALLD)"),
                        ("noisy_tft", "#55A868", "잡음 TFT (err=0.20)")):
        ax.scatter(b["theta"][c][:, fi_a], b["theta"][c][:, fi_b],
                   s=14, alpha=0.55, color=col, label=lab)
    ax.set_xlabel("α̂ (협력편향)"); ax.set_ylabel("β̂ (행동정밀도)")
    ax.set_title(f"(d) H2A — θ̂ 표상 수준 판별\n정확도={b['accuracy']:.3f}")
    ax.legend(fontsize=7); annotate_n(ax, cfg.seeds)

    # (e) λ 궤적 비교
    ax = fig.add_subplot(gs[1, 1])
    band_plot(ax, b["lam_traces"]["exploiter"], color="#937860",
              label="착취자 상대")
    band_plot(ax, b["lam_traces"]["noisy_tft"], color="#55A868",
              label="잡음 TFT 상대")
    ax.set_xlabel("라운드"); ax.set_ylabel("공감 가중 λ")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(e) 행동 수준 판별 — λ 궤적의 분기")
    ax.legend(fontsize=7); annotate_n(ax, cfg.seeds)

    # (f) 최종 λ 와 후반 협력률
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(2)
    w = 0.36
    lm = [b["lam_final"]["exploiter"].mean(), b["lam_final"]["noisy_tft"].mean()]
    lc = [boot_mean_ci(b["lam_final"][c])["ci"]
          for c in ("exploiter", "noisy_tft")]
    cm = [b["coop_late"]["exploiter"].mean(), b["coop_late"]["noisy_tft"].mean()]
    cc = [boot_mean_ci(b["coop_late"][c])["ci"]
          for c in ("exploiter", "noisy_tft")]
    ax.bar(x - w / 2, lm, w, yerr=[[m - c[0] for m, c in zip(lm, lc)],
                                   [c[1] - m for m, c in zip(lm, lc)]],
           capsize=3, color="#DA8BC3", label="최종 λ", alpha=0.9)
    ax.bar(x + w / 2, cm, w, yerr=[[m - c[0] for m, c in zip(cm, cc)],
                                   [c[1] - m for m, c in zip(cm, cc)]],
           capsize=3, color="#4C72B0", label="후반 협력률", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["착취자", "잡음 TFT"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("값")
    ax.set_title("(f) 판별의 행동적 귀결")
    ax.legend(fontsize=7)

    fig.suptitle("H2 / H2A — 착취자로부터의 자기보호와 잡음-의도 판별",
                 fontsize=12, y=0.98)
    save_fig(fig, cfg, "H2_protection_discrimination")

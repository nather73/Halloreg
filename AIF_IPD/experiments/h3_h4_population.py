"""
experiments.h3_h4_population
============================

**H3  — 정상 보수구조의 혼합 집단에서 HalloReg 는 다른 고정전략 대비 높은 보상을 얻는가?**
**H3A — 같은 조건에서 집단 상호협력률 상승에 유의하게 더 많이 기여하는가?**
**H4  — 비정상 보수구조에서 H3 와 같은가?**
**H4A — 비정상 보수구조에서 H3A 와 같은가?**

집단 구성: {TFT=a, GTFT=b, WSLS=c, ALLC=d, ALLD=e, HalloReg=f}, 총합 30.
"최대한 많은 조합을 시뮬레이션한다" → `ipd.population` 의 정확한 분해식으로
**모든** 조합(C(35,5) = 324,632개, 각 유형 최소 1명 조건에서는 C(29,5) = 118,755개)
을 평가한다. 근사가 아니라 항등식임은 population.py 의 문서를 참조.

────────────────────────────────────────────────────────────────────────
지표 1 — 보상 (H3 / H4)
────────────────────────────────────────────────────────────────────────
조합 n 에서 유형 i 의 라운드당 평균보수 μ_i(n). HalloReg 의 우위를 다음으로 본다.

    Δ_i(n) = μ_HalloReg(n) − μ_i(n),   i ∈ {TFT, GTFT, WSLS, ALLC, ALLD}

**중요 — ALLD 비교의 해석적 함정.** ALLD 는 협력자가 많은 조합에서 착취로 높은
보수를 얻는다. 그러나 이는 '강건한 성과' 가 아니라 타인의 보수를 이전받은
결과다. 따라서 ALLD 와의 비교는 확증이 아니라 **탐색**으로만 등록하고,
확증 검정은 협력 계열 4종(TFT/GTFT/WSLS/ALLC) 에 대해서만 수행한다.
이 사전 결정은 결과를 보기 전에 고정한다.

    · 확증 : 협력 계열 4종 각각에 대해 Δ_i > 0.
      검정 단위는 **시드**다 — 조합은 서로 독립이 아니므로(같은 Π 에서 파생)
      조합을 replicate 로 쓰면 p 값이 인공적으로 작아진다. 시드별로 독립
      추정된 Π_s 로 전 조합 평균 Δ̄_i(s) 를 만들고, 그 시드 벡터에 부호뒤집기
      순열검정을 적용한다.

────────────────────────────────────────────────────────────────────────
지표 2 — 협력 기여 (H3A / H4A)
────────────────────────────────────────────────────────────────────────
"집단 내 상호협력률 상승에 **더 많이 기여**" 는 개체 하나의 **한계 기여도**로
조작화한다. 조합 n 에서 HalloReg 개체 하나를 유형 i 로 치환했을 때의 CC율 변화:

    Δ^CC_i(n) = CC(n) − CC(n − e_HalloReg + e_i)

Δ^CC_i > 0 이면 그 자리에 HalloReg 가 있는 편이 유형 i 가 있는 것보다 집단
협력을 더 끌어올린다는 뜻이다. 치환 설계를 쓰는 이유는, 단순히 "HalloReg 가
많은 조합의 CC 가 높다" 는 상관은 조합 크기·구성 교란에 취약하기 때문이다.
치환은 집단 크기 30 을 고정한 채 한 자리만 바꾸므로 그 교란이 제거된다.

    · 확증 : 협력 계열 4종 각각에 대해 Δ^CC_i > 0 (시드 단위 검정).
    · 탐색 : ALLD 치환, 조합 공간에서의 용량-반응(HalloReg 수 vs CC율) 기울기.

────────────────────────────────────────────────────────────────────────
H4 / H4A — 비정상 보수
────────────────────────────────────────────────────────────────────────
동일한 절차를 5개 비정상 레짐(blocks / oscillate / aba / drift / shock) 각각에서
반복하고, 레짐 전체를 통합한 검정을 확증으로 둔다. 레짐별 결과는 탐색이다.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.env import ALL_TYPES, TYPE_LABEL_KO
from AIF_IPD.ipd.metrics import (
    boot_mean_ci, effect_size_paired, fmt_es, one_sample_perm, slope_boot,
)
from AIF_IPD.ipd.payoff_schedule import NONSTATIONARY_REGIMES, REGIME_LABEL_KO
from AIF_IPD.ipd.population import (
    default_type_specs, enumerate_compositions, estimate_pair_matrices,
    population_cc, run_round_robin, subsample_compositions,
    substitution_delta_cc, type_payoffs,
)
from .common import Config, Registry, TYPE_COLORS, save_fig, save_json

LOGGER = get_logger("HalloReg.H3H4")

HR = ALL_TYPES.index("halloreg")
#: 확증 비교 대상 — 협력 계열 4종 (ALLD 는 탐색으로만)
COOP_RIVALS = ("tft", "gtft", "wsls", "allc")
TOTAL_AGENTS = 30


# ==================================================================== 핵심 계산
def analyze_regime(pair: dict, comps: np.ndarray) -> dict:
    """
    한 레짐의 유형쌍 행렬로부터 조합 전체의 보수·협력 지표를 계산한다.

    시드별 Π_s / CCm_s 를 각각 쓰므로, 반환되는 배열은 (seeds,) 벡터다 —
    통계검정의 replicate 단위가 시드가 되도록 하기 위함이다.
    """
    names = pair["names"]
    S = pair["Pi_raw"].shape[2]
    k = len(names)

    # (seeds,) — 각 시드의 전 조합 평균 보수차 Δ_i
    d_pay = {n: np.zeros(S) for n in names}
    # (seeds,) — 각 시드의 전 조합 평균 치환 CC 기여 Δ^CC_i
    d_cc = {n: np.zeros(S) for n in names}
    # 용량-반응 기울기 (HalloReg 수 → CC율)
    dose = np.zeros(S)

    for s in range(S):
        Pi = pair["Pi_raw"][:, :, s]
        CCm = pair["CCm_raw"][:, :, s]
        mu = type_payoffs(Pi, comps)                 # (M, k)
        for i, n in enumerate(names):
            if i == HR:
                continue
            d_pay[n][s] = float(np.mean(mu[:, HR] - mu[:, i]))
            d_cc[n][s] = float(np.nanmean(
                substitution_delta_cc(CCm, comps, HR, i)))
        cc = population_cc(CCm, comps)
        sl = np.polyfit(comps[:, HR].astype(float), cc, 1)[0]
        dose[s] = float(sl)

    # 대표값(전 시드 평균 Π)에서의 조합 수준 지표 — 시각화용
    mu_mean = type_payoffs(pair["Pi"], comps)
    cc_mean = population_cc(pair["CCm"], comps)

    return {"names": names, "delta_payoff": d_pay, "delta_cc": d_cc,
            "dose_slope": dose, "mu_by_comp": mu_mean, "cc_by_comp": cc_mean,
            "n_comps": int(len(comps))}


def _confirm_block(reg: Registry, hyp: str, tag: str,
                   deltas: Dict[str, np.ndarray], unit: str) -> None:
    """협력 계열 4종에 대한 확증 검정 + ALLD 탐색 검정을 등록."""
    for rival in COOP_RIVALS:
        d = deltas[rival]
        t = one_sample_perm(d, 0.0, alternative="greater")
        e = effect_size_paired(d)
        ci = boot_mean_ci(d)
        reg.confirm(hyp, f"{tag} vs {TYPE_LABEL_KO[rival]}", t["p"],
                    direction_ok=bool(ci["mean"] > 0),
                    effect=f"Δ{unit}={ci['mean']:+.4f} "
                           f"[{ci['ci'][0]:+.4f}, {ci['ci'][1]:+.4f}], "
                           f"{fmt_es(e, 'dz')}")
    d = deltas["alld"]
    t = one_sample_perm(d, 0.0, alternative="greater")
    ci = boot_mean_ci(d)
    reg.explore(hyp, f"{tag} vs ALLD (착취 이전 효과로 해석 주의)", t["p"],
                effect=f"Δ{unit}={ci['mean']:+.4f}")


# ==================================================================== 실행
def _prepare_comps(cfg: Config) -> np.ndarray:
    """
    조합 열거. 각 유형 최소 1명을 요구한다 — 유형 i 가 0명인 조합에서는
    μ_i 와 치환 대비가 정의되지 않기 때문이다.
    """
    comps = enumerate_compositions(TOTAL_AGENTS, len(ALL_TYPES), min_each=1)
    if cfg.max_compositions and len(comps) > cfg.max_compositions:
        LOGGER.info("  조합 부분표집: %d → %d", len(comps), cfg.max_compositions)
        comps = subsample_compositions(comps, cfg.max_compositions, seed=7)
    return comps


def run_stationary(cfg: Config, reg: Registry, comps: np.ndarray) -> dict:
    """H3 / H3A — 정상 보수구조."""
    LOGGER.info("[H3/H3A] 정상 보수 — 조합 %d개, 유형쌍 %d개 추정",
                len(comps), len(ALL_TYPES) * (len(ALL_TYPES) + 1) // 2)
    specs = default_type_specs(cfg.halloreg_kwargs())
    pair = estimate_pair_matrices(specs, cfg.rounds, cfg.seeds, cfg.jobs,
                                  regime=None, env_error=cfg.env_error,
                                  seed_offset=0)
    an = analyze_regime(pair, comps)

    _confirm_block(reg, "H3", "보수", an["delta_payoff"], "보수")
    _confirm_block(reg, "H3A", "CC 한계기여", an["delta_cc"], "CC")

    # 탐색: 용량-반응 기울기
    t = one_sample_perm(an["dose_slope"], 0.0, alternative="greater")
    ci = boot_mean_ci(an["dose_slope"])
    reg.explore("H3A", "용량-반응: HalloReg 수 → 집단 CC율 기울기 > 0", t["p"],
                effect=f"기울기={ci['mean']:+.5f}/명")

    # ---- 해석적 분해식의 직접 검증 ----
    check = _verify_decomposition(cfg, specs, pair, comps, regime=None)
    reg.explore("H3", "해석적 분해식 vs 직접 라운드로빈 (검증)",
                1.0, effect=f"최대 |오차| CC={check['max_abs_cc_err']:.4f}, "
                            f"보수={check['max_abs_pay_err']:.4f}")

    return {"pair": pair, "analysis": an, "verify": check}


def run_nonstationary(cfg: Config, reg: Registry, comps: np.ndarray) -> dict:
    """H4 / H4A — 비정상 보수구조 (5개 레짐)."""
    regimes = NONSTATIONARY_REGIMES
    specs = default_type_specs(cfg.halloreg_kwargs())
    per_regime = {}

    for ri, rg in enumerate(regimes):
        LOGGER.info("[H4/H4A] 레짐 '%s' (%d/%d)", rg, ri + 1, len(regimes))
        pair = estimate_pair_matrices(specs, cfg.rounds, cfg.seeds, cfg.jobs,
                                      regime=rg, env_error=cfg.env_error,
                                      seed_offset=1000 * (ri + 1))
        per_regime[rg] = {"pair": pair, "analysis": analyze_regime(pair, comps)}

    # ---- 확증: 레짐 통합 (시드 × 레짐을 replicate 로) ----
    pooled_pay = {n: np.concatenate(
        [per_regime[rg]["analysis"]["delta_payoff"][n] for rg in regimes])
        for n in ALL_TYPES}
    pooled_cc = {n: np.concatenate(
        [per_regime[rg]["analysis"]["delta_cc"][n] for rg in regimes])
        for n in ALL_TYPES}

    _confirm_block(reg, "H4", "보수(비정상 통합)", pooled_pay, "보수")
    _confirm_block(reg, "H4A", "CC 한계기여(비정상 통합)", pooled_cc, "CC")

    # ---- 탐색: 레짐별 ----
    for rg in regimes:
        an = per_regime[rg]["analysis"]
        for rival in COOP_RIVALS:
            t = one_sample_perm(an["delta_payoff"][rival], 0.0,
                                alternative="greater")
            ci = boot_mean_ci(an["delta_payoff"][rival])
            reg.explore("H4", f"[{REGIME_LABEL_KO[rg]}] 보수 vs "
                              f"{TYPE_LABEL_KO[rival]}", t["p"],
                        effect=f"Δ={ci['mean']:+.4f}")
        t = one_sample_perm(an["dose_slope"], 0.0, alternative="greater")
        ci = boot_mean_ci(an["dose_slope"])
        reg.explore("H4A", f"[{REGIME_LABEL_KO[rg]}] 용량-반응 기울기", t["p"],
                    effect=f"기울기={ci['mean']:+.5f}/명")

    return {"regimes": regimes, "per_regime": per_regime,
            "pooled_payoff": pooled_pay, "pooled_cc": pooled_cc}


def _verify_decomposition(cfg: Config, specs: dict, pair: dict,
                          comps: np.ndarray, regime: Optional[str],
                          n_check: int = 3) -> dict:
    """
    해석적 분해식이 실제 라운드로빈과 일치하는지 직접 검증한다.

    무작위로 뽑은 몇 개 조합에 대해 30명 라운드로빈(435 다이애드)을 실행하고,
    분해식 예측과 대조한다. 계산이 무거우므로 조합 수와 시드를 최소로 둔다.
    """
    rng = np.random.default_rng(3)
    idx = rng.choice(len(comps), size=min(n_check, len(comps)), replace=False)
    rows = []
    for j, i in enumerate(idx):
        c = comps[i]
        direct = run_round_robin(c, specs, cfg.rounds, seed=100 + j,
                                 regime=regime, env_error=cfg.env_error,
                                 persistent=False, n_jobs=cfg.jobs)
        pred_cc = float(population_cc(pair["CCm"], c))
        pred_mu = type_payoffs(pair["Pi"], c)
        rows.append({
            "counts": c.tolist(),
            "cc_direct": direct["cc_rate"], "cc_analytic": pred_cc,
            "cc_err": direct["cc_rate"] - pred_cc,
            "payoff_direct": direct["by_type_payoff"],
            "payoff_analytic": {n: float(pred_mu[i2])
                                for i2, n in enumerate(pair["names"])},
        })
        LOGGER.info("  검증 %d: CC 직접=%.4f 해석=%.4f (차=%+.4f)",
                    j + 1, direct["cc_rate"], pred_cc,
                    direct["cc_rate"] - pred_cc)
    max_cc = max(abs(r["cc_err"]) for r in rows)
    max_pay = max(
        max(abs(r["payoff_direct"][n] - r["payoff_analytic"][n])
            for n in r["payoff_analytic"])
        for r in rows)
    return {"rows": rows, "max_abs_cc_err": max_cc, "max_abs_pay_err": max_pay}


def run(cfg: Config, reg: Registry) -> dict:
    comps = _prepare_comps(cfg)
    LOGGER.info("[H3~H4A] 조합 공간 크기 = %d (총 %d명, 유형별 최소 1명)",
                len(comps), TOTAL_AGENTS)
    stat = run_stationary(cfg, reg, comps)
    nons = run_nonstationary(cfg, reg, comps)
    out = {"n_compositions": int(len(comps)),
           "stationary": stat, "nonstationary": nons}
    _plot_stationary(cfg, stat, comps)
    _plot_nonstationary(cfg, nons)
    save_json({"n_compositions": int(len(comps)),
               "stationary": {"analysis": _summarize(stat["analysis"]),
                              "Pi": stat["pair"]["Pi"],
                              "CCm": stat["pair"]["CCm"],
                              "verify": stat["verify"]},
               "nonstationary": {
                   "regimes": nons["regimes"],
                   "per_regime": {rg: {
                       "Pi": nons["per_regime"][rg]["pair"]["Pi"],
                       "CCm": nons["per_regime"][rg]["pair"]["CCm"],
                       "analysis": _summarize(
                           nons["per_regime"][rg]["analysis"])}
                       for rg in nons["regimes"]}}},
              cfg.results / "H3_H4.json")
    return out


def _summarize(an: dict) -> dict:
    """
    JSON 직렬화용 요약.

    `mu_by_comp` / `cc_by_comp` 는 **조합 수만큼 긴 배열**이다(전수 열거 시
    118,755행). 그대로 저장하면 레짐마다 수백 MB 가 되어 결과 파일이 쓸모없어
    지므로, 조합 수준 원자료는 그림에만 쓰고 JSON 에는 분포 요약만 남긴다.
    시드 수준 벡터(delta_payoff / delta_cc / dose_slope)는 통계검정의 replicate
    단위라 그대로 보존한다.
    """
    out = {k: v for k, v in an.items()
           if k not in ("mu_by_comp", "cc_by_comp")}
    mu = an["mu_by_comp"]
    cc = an["cc_by_comp"]
    out["mu_by_comp_summary"] = {
        name: {"mean": float(mu[:, i].mean()),
               "sd": float(mu[:, i].std(ddof=1)),
               "q05": float(np.percentile(mu[:, i], 5)),
               "q50": float(np.percentile(mu[:, i], 50)),
               "q95": float(np.percentile(mu[:, i], 95))}
        for i, name in enumerate(an["names"])}
    out["cc_by_comp_summary"] = {
        "mean": float(cc.mean()), "sd": float(cc.std(ddof=1)),
        "q05": float(np.percentile(cc, 5)), "q50": float(np.percentile(cc, 50)),
        "q95": float(np.percentile(cc, 95))}
    return out


# ==================================================================== 시각화
def _plot_stationary(cfg: Config, res: dict, comps: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from .common import annotate_n, bar_with_ci

    pair, an = res["pair"], res["analysis"]
    labels = [TYPE_LABEL_KO[t] for t in ALL_TYPES]
    rivals = [t for t in ALL_TYPES if t != "halloreg"]
    rlabels = [TYPE_LABEL_KO[t] for t in rivals]

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

    # (a) 보수행렬 Π
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(pair["Pi"], cmap="viridis")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("(a) 유형쌍 보수행렬 Π\n(행 유형의 라운드당 보수)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{pair['Pi'][i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (b) 상호협력행렬 CCm
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(pair["CCm"], cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("(b) 유형쌍 상호협력률 CCm")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = pair["CCm"][i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.55 else "black")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (c) H3 — 보수 우위
    ax = fig.add_subplot(gs[0, 2])
    means = [an["delta_payoff"][r].mean() for r in rivals]
    cis = [boot_mean_ci(an["delta_payoff"][r])["ci"] for r in rivals]
    cols = ["#4C72B0" if r in COOP_RIVALS else "#937860" for r in rivals]
    bar_with_ci(ax, rlabels, means, cis, colors=cols,
                ylabel="Δ 라운드당 보수", rotate=30)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title(f"(c) H3 — HalloReg − 상대유형 보수\n"
                 f"(전 조합 {an['n_comps']:,}개 평균, 시드 n={cfg.seeds})")

    # (d) H3A — CC 한계기여
    ax = fig.add_subplot(gs[1, 0])
    means = [an["delta_cc"][r].mean() for r in rivals]
    cis = [boot_mean_ci(an["delta_cc"][r])["ci"] for r in rivals]
    bar_with_ci(ax, rlabels, means, cis, colors=cols,
                ylabel="Δ 집단 CC율 (치환 1명당)", rotate=30)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title("(d) H3A — 치환 한계기여\nCC(n) − CC(HalloReg 1명 → 해당유형)")

    # (e) 용량-반응: HalloReg 수 vs CC율
    ax = fig.add_subplot(gs[1, 1])
    nh = comps[:, HR]
    cc = an["cc_by_comp"]
    binned = [cc[nh == v] for v in range(1, min(int(nh.max()), 25) + 1)]
    xs = [v for v in range(1, min(int(nh.max()), 25) + 1) if len(binned[v - 1])]
    ax.boxplot([binned[v - 1] for v in xs], positions=xs, widths=0.6,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#DA8BC3", alpha=0.7))
    ax.set_xlabel("집단 내 HalloReg 개체 수"); ax.set_ylabel("집단 CC율")
    ax.set_title(f"(e) 용량-반응 (기울기 "
                 f"{an['dose_slope'].mean():+.5f}/명)")
    ax.set_xticks(xs[::4]); ax.set_xticklabels([str(v) for v in xs[::4]])

    # (f) 유형별 평균보수의 조합 분포
    ax = fig.add_subplot(gs[1, 2])
    data = [an["mu_by_comp"][:, i] for i in range(len(ALL_TYPES))]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                    showfliers=False)
    for patch, t in zip(bp["boxes"], ALL_TYPES):
        patch.set_facecolor(TYPE_COLORS[t]); patch.set_alpha(0.8)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("라운드당 평균보수")
    ax.set_title("(f) 전 조합에 걸친 유형별 보수 분포")

    fig.suptitle("H3 / H3A — 정상 보수구조 혼합 집단 (30명, 전 조합 해석적 평가)",
                 fontsize=12, y=0.98)
    save_fig(fig, cfg, "H3_stationary_population")


def _plot_nonstationary(cfg: Config, res: dict) -> None:
    import matplotlib.pyplot as plt
    from AIF_IPD.ipd.payoff_schedule import regime_trace
    from .common import bar_with_ci

    regimes = res["regimes"]
    rivals = [t for t in ALL_TYPES if t != "halloreg"]
    rlabels = [TYPE_LABEL_KO[t] for t in rivals]
    cols = ["#4C72B0" if r in COOP_RIVALS else "#937860" for r in rivals]

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32,
                          height_ratios=[0.75, 1, 1])

    # (a) 레짐별 CI 궤적
    ax = fig.add_subplot(gs[0, :])
    for rg in regimes:
        ax.plot(regime_trace(rg, cfg.rounds), lw=1.3, label=REGIME_LABEL_KO[rg])
    ax.axhline(0.4, color="#888888", ls=":", lw=1, label="기본 PD (CI=0.4)")
    ax.axhline(0.0, color="crimson", ls="--", lw=0.9)
    ax.set_xlabel("라운드"); ax.set_ylabel("협력지수 CI")
    ax.set_title("(a) 비정상 보수 레짐 — CI = (R−P)/(T−S) 궤적\n"
                 "CI<0 교착 · CI=0.4 기본 PD · CI≥1 조화")
    ax.legend(ncol=3, fontsize=7)

    # (b) H4 — 통합 보수 우위
    ax = fig.add_subplot(gs[1, 0])
    means = [res["pooled_payoff"][r].mean() for r in rivals]
    cis = [boot_mean_ci(res["pooled_payoff"][r])["ci"] for r in rivals]
    bar_with_ci(ax, rlabels, means, cis, colors=cols,
                ylabel="Δ 라운드당 보수", rotate=30)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title("(b) H4 — 비정상 통합 보수 우위")

    # (c) H4A — 통합 CC 기여
    ax = fig.add_subplot(gs[1, 1])
    means = [res["pooled_cc"][r].mean() for r in rivals]
    cis = [boot_mean_ci(res["pooled_cc"][r])["ci"] for r in rivals]
    bar_with_ci(ax, rlabels, means, cis, colors=cols,
                ylabel="Δ 집단 CC율", rotate=30)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title("(c) H4A — 비정상 통합 CC 한계기여")

    # (d) 레짐 × 상대유형 보수 우위 히트맵
    ax = fig.add_subplot(gs[1, 2])
    M = np.array([[res["per_regime"][rg]["analysis"]["delta_payoff"][r].mean()
                   for r in rivals] for rg in regimes])
    vmax = np.max(np.abs(M)) if np.max(np.abs(M)) > 0 else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(rivals))); ax.set_xticklabels(rlabels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([REGIME_LABEL_KO[r] for r in regimes], fontsize=7)
    for i in range(len(regimes)):
        for j in range(len(rivals)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=6.5)
    ax.set_title("(d) 레짐별 보수 우위 Δ")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (e) 레짐 × 상대유형 CC 기여 히트맵
    ax = fig.add_subplot(gs[2, 0])
    M = np.array([[res["per_regime"][rg]["analysis"]["delta_cc"][r].mean()
                   for r in rivals] for rg in regimes])
    vmax = np.max(np.abs(M)) if np.max(np.abs(M)) > 0 else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(rivals))); ax.set_xticklabels(rlabels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([REGIME_LABEL_KO[r] for r in regimes], fontsize=7)
    for i in range(len(regimes)):
        for j in range(len(rivals)):
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                    fontsize=6)
    ax.set_title("(e) 레짐별 CC 한계기여 Δ")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (f) 레짐별 용량-반응 기울기
    ax = fig.add_subplot(gs[2, 1])
    means = [res["per_regime"][rg]["analysis"]["dose_slope"].mean()
             for rg in regimes]
    cis = [boot_mean_ci(res["per_regime"][rg]["analysis"]["dose_slope"])["ci"]
           for rg in regimes]
    bar_with_ci(ax, [REGIME_LABEL_KO[r] for r in regimes], means, cis,
                colors="#55A868", ylabel="CC율 기울기 (/명)", rotate=35)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title("(f) 레짐별 용량-반응 기울기")

    # (g) 레짐별 유형 보수 (전 조합 평균)
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(len(regimes))
    w = 0.13
    for i, t in enumerate(ALL_TYPES):
        vals = [res["per_regime"][rg]["analysis"]["mu_by_comp"][:, i].mean()
                for rg in regimes]
        ax.bar(x + (i - 2.5) * w, vals, w, color=TYPE_COLORS[t],
               label=TYPE_LABEL_KO[t], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL_KO[r] for r in regimes], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel("라운드당 평균보수")
    ax.set_title("(g) 레짐 × 유형 평균보수")
    ax.legend(ncol=2, fontsize=6)

    fig.suptitle("H4 / H4A — 비정상 보수구조 혼합 집단", fontsize=12, y=0.985)
    save_fig(fig, cfg, "H4_nonstationary_population")

"""
experiments.h5_evolution
========================

**H5 — 비정상 보수구조의 혼합 집단에서, 복제자 방정식(RE) 및 최적 복제자
방정식(ORE) 시뮬레이션 결과 HalloReg 는 세대에 걸쳐 생존하는가?**

────────────────────────────────────────────────────────────────────────
설계
────────────────────────────────────────────────────────────────────────
H4 에서 이미 추정한 레짐별 보수행렬 Π 를 재사용한다(재시뮬레이션 불필요).
초기 조성 x₀ 은 H3/H4 와 **같은 조합 공간**에서 온다: x₀ = counts / 30.

각 (레짐 × 시드 × 조합) 에 대해:
  1. RE  를 적분해 종착 조성 x_RE(τ) 를 얻는다.
  2. ORE 를 적분해 종착 조성 x_ORE(τ) 를 얻는다.
  3. 생존 여부 = [ x_HalloReg(τ) > 1/30 ]  (개체 1명분 임계).

────────────────────────────────────────────────────────────────────────
지표와 검정
────────────────────────────────────────────────────────────────────────
· 생존 유역 점유율 (survival fraction) — 초기 조성 중 생존하는 비율.
· 우세 유역 점유율 (dominance fraction) — 종착에서 최대 빈도를 갖는 비율.
· 종착 평균 빈도.

  · 확증 H5-1 : RE 하 HalloReg 생존율 > 우연 기준선.
  · 확증 H5-2 : ORE 하 HalloReg 생존율 > 우연 기준선.

**우연 기준선의 정의.** "생존" 은 유형마다 난이도가 다르므로 절대 임계만으로는
비교가 어렵다. 여기서는 6개 유형의 생존율 평균(= 1/6 이 아니라 실제 관측된
평균 생존율)을 기준선으로 삼는다. 즉 "HalloReg 는 평균적 유형보다 더 넓은
초기조건에서 살아남는가" 를 묻는다. 이 기준선은 데이터 의존적이지만 검정은
시드 단위 짝지음이므로 타당하다.

  · 탐색 : 레짐별 생존율, ALLD 대비 생존율, RE→ORE 변화량, 종착 CC율.

────────────────────────────────────────────────────────────────────────
계산량 관리
────────────────────────────────────────────────────────────────────────
조합이 10만 개 이상이므로 모든 조합에 ORE 를 돌리면 비현실적이다. ORE 는
초기점 축이 완전 벡터화되어 있으므로 배치 처리가 가능하지만, FBSM 이 sweep ×
steps 만큼 반복하므로 비용이 RE 의 수십 배다. 따라서:

  · RE  : 조합 공간에서 `n_init` 개를 무작위 표집(레짐·시드 간 **동일 표집** 공유
          — 공통난수(CRN) 로 짝지은 비교 성립).
  · ORE : 같은 초기점 집합을 쓴다(RE 와 ORE 의 짝지음 유지).

n_init 은 기본 1500 이며, 조합 공간의 층화 표본으로 충분한 유역 추정치를 준다.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.env import ALL_TYPES, TYPE_LABEL_KO
from AIF_IPD.ipd.evolution import (
    dominance_fraction, mean_terminal_frequency, ore_ends_batch,
    replicator_ends_batch, survival_fraction, terminal_cc,
)
from AIF_IPD.ipd.metrics import boot_mean_ci, effect_size_paired, fmt_es, one_sample_perm
from AIF_IPD.ipd.payoff_schedule import REGIME_LABEL_KO
from .common import Config, Registry, TYPE_COLORS, save_fig, save_json

LOGGER = get_logger("HalloReg.H5")

HR = ALL_TYPES.index("halloreg")
SURVIVAL_THRESHOLD = 1.0 / 30.0     # 30명 집단의 개체 1명분

#: 유역 추정에 쓰는 초기점 수 (조합 공간에서의 표집)
N_INIT_FULL = 1500
N_INIT_QUICK = 200

#: 진화 적분 파라미터
RE_STEPS = 600
ORE_TAU, ORE_STEPS, ORE_SWEEPS = 1.0, 200, 24


def _sample_initials(comps: np.ndarray, n: int, seed: int = 5) -> np.ndarray:
    """조합 공간에서 초기 조성 x₀ = counts/30 을 균등 표집한다."""
    rng = np.random.default_rng(seed)
    if len(comps) <= n:
        sel = comps
    else:
        sel = comps[rng.choice(len(comps), size=n, replace=False)]
    X0 = sel.astype(float)
    return X0 / X0.sum(axis=1, keepdims=True)


def run(cfg: Config, reg: Registry, h34: dict, comps: np.ndarray) -> dict:
    """
    H5 실행. h34 는 h3_h4_population.run() 의 반환값 — 보수행렬을 재사용한다.
    """
    n_init = N_INIT_QUICK if cfg.quick else N_INIT_FULL
    X0 = _sample_initials(comps, n_init)
    LOGGER.info("[H5] 진화 동역학 — 초기점 %d개 × 레짐 %d개 × 시드 %d개",
                len(X0), len(h34["nonstationary"]["regimes"]), cfg.seeds)

    regimes = h34["nonstationary"]["regimes"]
    k = len(ALL_TYPES)

    # 시드 수가 많으면 진화 적분이 과도해지므로, 검정에 필요한 만큼만 쓴다.
    # (Π 자체가 seeds 개 다이애드의 평균이므로 시드 수준 변동은 이미 작다.)
    n_seed_eval = min(cfg.seeds, 24 if not cfg.quick else 4)

    results: Dict[str, dict] = {}
    for rg in regimes:
        pair = h34["nonstationary"]["per_regime"][rg]["pair"]
        surv_re = np.zeros((n_seed_eval, k))
        surv_ore = np.zeros((n_seed_eval, k))
        dom_re = np.zeros((n_seed_eval, k))
        dom_ore = np.zeros((n_seed_eval, k))
        freq_re = np.zeros((n_seed_eval, k))
        freq_ore = np.zeros((n_seed_eval, k))
        cc_re = np.zeros(n_seed_eval)
        cc_ore = np.zeros(n_seed_eval)

        for s in range(n_seed_eval):
            Pi = pair["Pi_raw"][:, :, s]
            CCm = pair["CCm_raw"][:, :, s]
            ends_re = replicator_ends_batch(Pi, X0, steps=RE_STEPS)
            ends_ore = ore_ends_batch(Pi, X0, tau=ORE_TAU, steps=ORE_STEPS,
                                      sweeps=ORE_SWEEPS)
            for i in range(k):
                surv_re[s, i] = survival_fraction(ends_re, i, SURVIVAL_THRESHOLD)
                surv_ore[s, i] = survival_fraction(ends_ore, i, SURVIVAL_THRESHOLD)
                dom_re[s, i] = dominance_fraction(ends_re, i)
                dom_ore[s, i] = dominance_fraction(ends_ore, i)
                freq_re[s, i] = mean_terminal_frequency(ends_re, i)
                freq_ore[s, i] = mean_terminal_frequency(ends_ore, i)
            cc_re[s] = terminal_cc(CCm, ends_re)
            cc_ore[s] = terminal_cc(CCm, ends_ore)

        # 대표 궤적 (시각화용) — 균등 초기 조성에서 출발
        from AIF_IPD.ipd.evolution import ore_trajectory, replicator_trajectory
        x_unif = np.full(k, 1.0 / k)
        traj_re = replicator_trajectory(pair["Pi"], x_unif, steps=RE_STEPS)
        traj_ore = ore_trajectory(pair["Pi"], x_unif, tau=ORE_TAU,
                                  steps=400, sweeps=40)["x"]

        results[rg] = {
            "surv_re": surv_re, "surv_ore": surv_ore,
            "dom_re": dom_re, "dom_ore": dom_ore,
            "freq_re": freq_re, "freq_ore": freq_ore,
            "cc_re": cc_re, "cc_ore": cc_ore,
            "traj_re": traj_re, "traj_ore": traj_ore,
        }
        LOGGER.info("  [%s] 생존율 RE: HalloReg=%.3f (전유형 평균=%.3f) | "
                    "ORE: HalloReg=%.3f (평균=%.3f)",
                    rg, surv_re[:, HR].mean(), surv_re.mean(),
                    surv_ore[:, HR].mean(), surv_ore.mean())

    # ---- 확증: 레짐 통합, HalloReg 생존율 > 전유형 평균 ----
    for tag, key in (("RE", "surv_re"), ("ORE", "surv_ore")):
        hr = np.concatenate([results[rg][key][:, HR] for rg in regimes])
        avg = np.concatenate([results[rg][key].mean(axis=1) for rg in regimes])
        d = hr - avg
        t = one_sample_perm(d, 0.0, alternative="greater")
        e = effect_size_paired(d)
        ci = boot_mean_ci(d)
        reg.confirm("H5", f"{tag} 생존 유역: HalloReg > 전유형 평균", t["p"],
                    direction_ok=bool(ci["mean"] > 0),
                    effect=f"Δ={ci['mean']:+.4f} "
                           f"[{ci['ci'][0]:+.4f}, {ci['ci'][1]:+.4f}], "
                           f"{fmt_es(e, 'dz')}; "
                           f"HalloReg={hr.mean():.3f}",
                    detail={"halloreg": float(hr.mean()),
                            "all_type_mean": float(avg.mean())})

    # ---- 탐색: 레짐별, 유형별 대조, ORE−RE 차이, 종착 CC ----
    for rg in regimes:
        for tag, key in (("RE", "surv_re"), ("ORE", "surv_ore")):
            hr = results[rg][key][:, HR]
            avg = results[rg][key].mean(axis=1)
            t = one_sample_perm(hr - avg, 0.0, alternative="greater")
            reg.explore("H5", f"[{REGIME_LABEL_KO[rg]}] {tag} 생존율", t["p"],
                        effect=f"HalloReg={hr.mean():.3f} vs 평균={avg.mean():.3f}")
    for i, t_ in enumerate(ALL_TYPES):
        if i == HR:
            continue
        d = np.concatenate([results[rg]["surv_ore"][:, HR]
                            - results[rg]["surv_ore"][:, i] for rg in regimes])
        tt = one_sample_perm(d, 0.0, alternative="greater")
        reg.explore("H5", f"ORE 생존율: HalloReg vs {TYPE_LABEL_KO[t_]}",
                    tt["p"], effect=f"Δ={np.mean(d):+.4f}")
    d_cc = np.concatenate([results[rg]["cc_ore"] - results[rg]["cc_re"]
                           for rg in regimes])
    t = one_sample_perm(d_cc, 0.0, alternative="greater")
    reg.explore("H5", "종착 CC율: ORE > RE (집단수준 선택의 협력 촉진)", t["p"],
                effect=f"Δ={np.mean(d_cc):+.4f}")

    out = {"regimes": regimes, "results": results, "n_init": int(len(X0)),
           "n_seed_eval": n_seed_eval, "threshold": SURVIVAL_THRESHOLD}
    _plot(cfg, out)
    save_json({"regimes": regimes, "n_init": int(len(X0)),
               "summary": {rg: {
                   "surv_re": results[rg]["surv_re"].mean(axis=0),
                   "surv_ore": results[rg]["surv_ore"].mean(axis=0),
                   "dom_ore": results[rg]["dom_ore"].mean(axis=0),
                   "freq_ore": results[rg]["freq_ore"].mean(axis=0),
                   "cc_re": float(results[rg]["cc_re"].mean()),
                   "cc_ore": float(results[rg]["cc_ore"].mean())}
                   for rg in regimes},
               "types": list(ALL_TYPES)},
              cfg.results / "H5.json")
    return out


# ==================================================================== 시각화
def _plot(cfg: Config, out: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import bar_with_ci

    regimes = out["regimes"]
    R = out["results"]
    labels = [TYPE_LABEL_KO[t] for t in ALL_TYPES]

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)

    # (a) RE 생존 유역
    ax = fig.add_subplot(gs[0, 0])
    m = np.array([[R[rg]["surv_re"][:, i].mean() for i in range(len(ALL_TYPES))]
                  for rg in regimes]).mean(axis=0)
    ci = [boot_mean_ci(np.concatenate([R[rg]["surv_re"][:, i]
                                       for rg in regimes]))["ci"]
          for i in range(len(ALL_TYPES))]
    bar_with_ci(ax, labels, m, ci,
                colors=[TYPE_COLORS[t] for t in ALL_TYPES],
                ylabel="생존 유역 점유율", rotate=35)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"(a) RE 생존 유역 (레짐 통합)\n임계 = {out['threshold']:.4f} (1/30)")

    # (b) ORE 생존 유역
    ax = fig.add_subplot(gs[0, 1])
    m = np.array([[R[rg]["surv_ore"][:, i].mean()
                   for i in range(len(ALL_TYPES))] for rg in regimes]).mean(axis=0)
    ci = [boot_mean_ci(np.concatenate([R[rg]["surv_ore"][:, i]
                                       for rg in regimes]))["ci"]
          for i in range(len(ALL_TYPES))]
    bar_with_ci(ax, labels, m, ci,
                colors=[TYPE_COLORS[t] for t in ALL_TYPES],
                ylabel="생존 유역 점유율", rotate=35)
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) ORE 생존 유역 (레짐 통합)")

    # (c) 종착 평균 빈도 (ORE)
    ax = fig.add_subplot(gs[0, 2])
    m = np.array([[R[rg]["freq_ore"][:, i].mean()
                   for i in range(len(ALL_TYPES))] for rg in regimes]).mean(axis=0)
    ci = [boot_mean_ci(np.concatenate([R[rg]["freq_ore"][:, i]
                                       for rg in regimes]))["ci"]
          for i in range(len(ALL_TYPES))]
    bar_with_ci(ax, labels, m, ci,
                colors=[TYPE_COLORS[t] for t in ALL_TYPES],
                ylabel="종착 평균 빈도", rotate=35)
    ax.axhline(1.0 / len(ALL_TYPES), color="crimson", ls="--", lw=1.1,
               label="초기 균등 (1/6)")
    ax.set_title("(c) ORE 종착 조성"); ax.legend(fontsize=7)

    # (d) 레짐별 HalloReg 생존율 (RE vs ORE)
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(regimes)); w = 0.36
    for off, key, col, lab in ((-w / 2, "surv_re", "#4C72B0", "RE"),
                               (+w / 2, "surv_ore", "#C44E52", "ORE")):
        vals = [R[rg][key][:, HR].mean() for rg in regimes]
        errs = [np.array(boot_mean_ci(R[rg][key][:, HR])["ci"]) for rg in regimes]
        lo = [v - e[0] for v, e in zip(vals, errs)]
        hi = [e[1] - v for v, e in zip(vals, errs)]
        ax.bar(x + off, vals, w, yerr=[lo, hi], capsize=3, color=col,
               label=lab, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL_KO[r] for r in regimes], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel("HalloReg 생존 유역"); ax.set_ylim(0, 1.05)
    ax.set_title("(d) 레짐별 HalloReg 생존율"); ax.legend(fontsize=7)

    # (e) 대표 RE 궤적 (첫 레짐)
    rg0 = regimes[0]
    ax = fig.add_subplot(gs[1, 1])
    for i, t in enumerate(ALL_TYPES):
        ax.plot(R[rg0]["traj_re"][:, i], color=TYPE_COLORS[t],
                label=TYPE_LABEL_KO[t], lw=1.3)
    ax.set_xlabel("세대(적분 스텝)"); ax.set_ylabel("조성 빈도")
    ax.set_title(f"(e) RE 궤적 — {REGIME_LABEL_KO[rg0]}\n(균등 초기조성)")
    ax.legend(ncol=2, fontsize=6)

    # (f) 대표 ORE 궤적 (첫 레짐)
    ax = fig.add_subplot(gs[1, 2])
    for i, t in enumerate(ALL_TYPES):
        ax.plot(R[rg0]["traj_ore"][:, i], color=TYPE_COLORS[t],
                label=TYPE_LABEL_KO[t], lw=1.3)
    ax.set_xlabel("적분 스텝 (역방향 공상태 결합)"); ax.set_ylabel("조성 빈도")
    ax.set_title(f"(f) ORE 궤적 — {REGIME_LABEL_KO[rg0]}")
    ax.legend(ncol=2, fontsize=6)

    # (g) 레짐 × 유형 ORE 생존율 히트맵
    ax = fig.add_subplot(gs[2, 0])
    M = np.array([[R[rg]["surv_ore"][:, i].mean()
                   for i in range(len(ALL_TYPES))] for rg in regimes])
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([REGIME_LABEL_KO[r] for r in regimes], fontsize=7)
    for i in range(len(regimes)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if M[i, j] > 0.55 else "black")
    ax.set_title("(g) 레짐 × 유형 ORE 생존율"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045)

    # (h) 우세 유역 (ORE)
    ax = fig.add_subplot(gs[2, 1])
    m = np.array([[R[rg]["dom_ore"][:, i].mean()
                   for i in range(len(ALL_TYPES))] for rg in regimes]).mean(axis=0)
    ci = [boot_mean_ci(np.concatenate([R[rg]["dom_ore"][:, i]
                                       for rg in regimes]))["ci"]
          for i in range(len(ALL_TYPES))]
    bar_with_ci(ax, labels, m, ci,
                colors=[TYPE_COLORS[t] for t in ALL_TYPES],
                ylabel="우세 유역 점유율", rotate=35)
    ax.set_title("(h) ORE 우세 유역\n(종착에서 최대 빈도를 갖는 비율)")

    # (i) 종착 CC율: RE vs ORE
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(len(regimes)); w = 0.36
    for off, key, col, lab in ((-w / 2, "cc_re", "#4C72B0", "RE"),
                               (+w / 2, "cc_ore", "#C44E52", "ORE")):
        vals = [R[rg][key].mean() for rg in regimes]
        errs = [np.array(boot_mean_ci(R[rg][key])["ci"]) for rg in regimes]
        lo = [v - e[0] for v, e in zip(vals, errs)]
        hi = [e[1] - v for v, e in zip(vals, errs)]
        ax.bar(x + off, vals, w, yerr=[lo, hi], capsize=3, color=col,
               label=lab, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL_KO[r] for r in regimes], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel("종착 조성의 기대 CC율")
    ax.set_title("(i) 종착 행동적 협력률"); ax.legend(fontsize=7)

    fig.suptitle("H5 — 복제자(RE) / 최적 복제자(ORE) 동역학에서의 생존",
                 fontsize=12, y=0.985)
    save_fig(fig, cfg, "H5_evolution")

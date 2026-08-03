"""
experiments.h1_intent
=====================

**H1 — OpponentInversion 은 타인의 전략적 의도를 강건하게 추론하는가?**

대상 유형: TFT, GTFT, WSLS, ALLC, ALLD, HalloReg (6종).

────────────────────────────────────────────────────────────────────────
설계
────────────────────────────────────────────────────────────────────────
1. HalloReg focal 이 6개 유형 각각을 상대로 `seeds` 개 다이애드를 수행한다.
2. 각 다이애드의 **최종 사후 평균** θ̂ = (α̂, ρ̂, ω̂, η̂, β̂, λ̂_j) 를 6차원
   특징벡터로 삼는다.
3. 시드를 **보정(calibration)** 과 **검정(test)** 으로 반분한다.
4. 보정 세트에서 유형별 대각공분산 가우시안(=가우시안 나이브베이즈)을 적합하고,
   검정 세트를 분류한다.

[왜 템플릿 대조가 아니라 학습 분류기인가]
TFT·WSLS·ALLC·ALLD 는 닫힌 형태의 θ 시그니처를 갖는다(예: WSLS ⇒ 순수 η).
그러나 **HalloReg 는 적응적 에이전트**라 선험적 템플릿이 존재하지 않는다.
6개 유형을 같은 기준으로 다루려면, 각 유형의 θ̂ 분포를 데이터로부터 추정한
뒤 베이즈 규칙으로 분류하는 편이 정당하다. 보정/검정 시드를 분리하므로
과적합이 아니며, 이는 신경 디코딩 문헌의 표준 교차검증 판독과 동일한 절차다.

────────────────────────────────────────────────────────────────────────
두 조건 — 행동적 등가류(behavioral equivalence class) 문제
────────────────────────────────────────────────────────────────────────
**사전에 예측되는 구조적 한계.** focal 이 협력적으로 행동하는 한, 상대 유형
중 일부는 **관측상 구별 불가능**하다.

  · TFT 와 GTFT : GTFT 의 관대함은 "focal 이 배신했을 때 그럼에도 협력한다"
                  로만 발현된다. focal 이 배신하지 않으면 두 전략은 **동일한
                  행동열**을 낸다.
  · ALLC 와 GTFT : 마찬가지로 focal 이 배신하지 않으면 둘 다 항상 협력한다.

즉 정책적으로(on-policy) 행동하는 관찰자는 자신이 만들어내지 않은 조건에 대한
정보를 얻을 수 없다. 이는 추론기의 결함이 아니라 **능동 자극(active sampling)의
부재**로 인한 식별 문제다. 따라서 두 조건을 나란히 돌린다.

  조건 on_policy : focal 이 자기 정책대로만 행동 (자연스러운 상호작용).
  조건 probe     : 환경 계층 실행오류율을 높여(ε = 0.20) focal 의 행동을 강제로
                   흔든다. 배신이 간헐적으로 발생하므로 상대의 **용서 구조**가
                   관측 가능해진다.

probe 조건은 모형의 일부가 아니라 **실험적 조작**이다(환경이 부과하는 잡음이며
에이전트의 정책을 바꾸지 않는다). 두 조건의 대조가 곧 "무엇이 식별을 가능하게
하는가" 에 대한 진단이 된다.

[영가설과 검정]
· 확증 1 : on_policy 균형정확도 > 우연수준(1/6). 라벨 순열검정.
· 확증 2 : probe 균형정확도 > 우연수준.
· 확증 3 : **최소 유형별 재현율** — probe > on_policy.
           "강건하게" 는 평균이 높은 것이 아니라 **어떤 유형도 실패하지 않는
           것**을 뜻하므로 최소값을 지표로 삼고, 능동 자극이 그 최소값을
           끌어올리는지를 검정한다.
· 탐색   : 조건별·유형별 재현율, θ 축별 판별력, 혼동 구조.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.env import ALL_TYPES, TYPE_LABEL_KO
from AIF_IPD.ipd.metrics import wilson_ci
from AIF_IPD.ipd.population import type_spec
from AIF_IPD.ipd.sim import run_many
from AIF_IPD.ipd.tom.inversion import THETA_AXES
from .common import Config, Registry, TYPE_COLORS, save_fig, save_json

LOGGER = get_logger("HalloReg.H1")

#: 특징으로 쓰는 θ 축 (사후 평균)
FEATURES = list(THETA_AXES)

#: 축 한글 표기
AXIS_LABEL_KO = {"alpha": "α 협력편향", "rho": "ρ 호혜성", "omega": "ω 관성",
                 "eta": "η 결과조건성", "beta": "β 정밀도",
                 "lambda_j": "λⱼ 상대공감"}


# ==================================================================== 분류기
class DiagGaussianClassifier:
    """
    대각공분산 가우시안 분류기 (가우시안 나이브베이즈).

    6차원 특징에 유형당 수십 개 표본뿐이므로 완전공분산은 추정 분산이 너무 크다.
    대각 근사는 편향을 약간 늘리는 대신 분산을 크게 줄여, 이 표본 크기에서
    일반화 성능이 더 좋다(편향–분산 절충).

    사전확률은 균등으로 고정한다 — 유형별 시드 수가 같으므로 자연스럽고,
    다수 유형에 유리한 편향이 생기지 않는다.
    """

    def __init__(self, var_floor: float = 1e-3):
        self.var_floor = float(var_floor)
        self.classes: List[str] = []
        self.mu = None      # (K, D)
        self.var = None     # (K, D)

    def fit(self, X: np.ndarray, y: List[str], classes: List[str]):
        self.classes = list(classes)
        K, D = len(self.classes), X.shape[1]
        self.mu = np.zeros((K, D))
        self.var = np.zeros((K, D))
        for k, c in enumerate(self.classes):
            Xc = X[np.asarray(y) == c]
            self.mu[k] = Xc.mean(axis=0)
            # 분산 하한: 어떤 축이 한 유형에서 거의 상수일 때 우도가 발산하는 것을 막는다
            self.var[k] = np.maximum(Xc.var(axis=0, ddof=1), self.var_floor)
        return self

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """(N, K) 로그우도."""
        # −½ Σ_d [ log(2πσ²) + (x−μ)²/σ² ]
        diff = X[:, None, :] - self.mu[None, :, :]           # (N, K, D)
        return -0.5 * np.sum(np.log(2 * np.pi * self.var)[None, :, :]
                             + diff ** 2 / self.var[None, :, :], axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 라벨 인덱스 (N,)."""
        return np.argmax(self.log_likelihood(X), axis=1)


def balanced_accuracy(y_true_idx: np.ndarray, y_pred_idx: np.ndarray,
                      K: int) -> float:
    """유형별 재현율의 평균 (불균형에 강건한 정확도)."""
    recalls = []
    for k in range(K):
        m = y_true_idx == k
        if m.sum() > 0:
            recalls.append(float(np.mean(y_pred_idx[m] == k)))
    return float(np.mean(recalls)) if recalls else np.nan


def per_class_recall(y_true_idx: np.ndarray, y_pred_idx: np.ndarray,
                     K: int) -> np.ndarray:
    out = np.full(K, np.nan)
    for k in range(K):
        m = y_true_idx == k
        if m.sum() > 0:
            out[k] = float(np.mean(y_pred_idx[m] == k))
    return out


def confusion(y_true_idx: np.ndarray, y_pred_idx: np.ndarray,
              K: int) -> np.ndarray:
    """행=참, 열=예측. 행 기준 정규화."""
    M = np.zeros((K, K))
    for t, p in zip(y_true_idx, y_pred_idx):
        M[t, p] += 1
    return M / np.maximum(M.sum(axis=1, keepdims=True), 1)


# ==================================================================== 실행
#: probe 조건의 focal 실행오류율 (능동 자극의 조작화)
PROBE_EPS = 0.20
CONDITION_LABEL = {"on_policy": "on-policy (자연 상호작용)",
                   "probe": f"probe (focal 잡음 ε={PROBE_EPS})"}


def collect_theta(cfg: Config, focal_eps: float, tag: int
                  ) -> Dict[str, np.ndarray]:
    """
    6개 유형 각각에 대해 seeds 개 다이애드를 돌리고 최종 θ̂ 를 수집한다.

    focal_eps : focal 측 환경 계층 실행오류율. probe 조건에서 크게 준다.
    tag       : 조건 간 시드 독립화를 위한 오프셋.
    반환      : {유형명: (seeds, D) 배열}.
    """
    hk = cfg.halloreg_kwargs()
    specs, registry = [], {}
    for ti, tname in enumerate(ALL_TYPES):
        for sd in range(cfg.seeds):
            agent = {"type": "halloreg", "seed": 1000 + tag + sd * 13 + ti, **hk}
            opp = dict(type_spec(tname))
            opp["seed"] = 2000 + tag + sd * 13 + ti
            if tname == "halloreg":
                opp.update(hk)
            registry[(ti, sd)] = len(specs)
            specs.append({"agent": agent, "opponent": opp,
                          "env_err_agent": focal_eps,
                          "env_err_opponent": cfg.env_error,
                          "noise_seed": 500_000 + tag + sd * 97 + ti})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H1 의도추론 다이애드")

    out = {}
    for ti, tname in enumerate(ALL_TYPES):
        F = np.zeros((cfg.seeds, len(FEATURES)))
        for sd in range(cfg.seeds):
            log = res[registry[(ti, sd)]]["agent_log"]
            for d, ax in enumerate(FEATURES):
                F[sd, d] = float(log[f"E_{ax}"][-1])     # 최종 라운드 사후평균
        out[tname] = F
    return out


def classify(theta: Dict[str, np.ndarray], seeds: int) -> dict:
    """시드 반분 보정/검정으로 6유형 분류를 수행하고 지표를 반환."""
    K = len(ALL_TYPES)
    n_cal = seeds // 2
    Xc, yc, Xt, yt = [], [], [], []
    for tname in ALL_TYPES:
        F = theta[tname]
        Xc.append(F[:n_cal]); yc += [tname] * n_cal
        Xt.append(F[n_cal:]); yt += [tname] * (len(F) - n_cal)
    Xc = np.vstack(Xc); Xt = np.vstack(Xt)

    # 표준화는 **보정 세트 통계만** 사용한다 (검정 세트 정보 누출 방지).
    mu, sd = Xc.mean(axis=0), np.maximum(Xc.std(axis=0, ddof=1), 1e-6)
    clf = DiagGaussianClassifier().fit((Xc - mu) / sd, yc, list(ALL_TYPES))
    idx_of = {c: i for i, c in enumerate(ALL_TYPES)}
    yt_idx = np.array([idx_of[c] for c in yt])
    pred = clf.predict((Xt - mu) / sd)

    return {"y_true": yt_idx, "y_pred": pred,
            "balanced_accuracy": balanced_accuracy(yt_idx, pred, K),
            "recalls": per_class_recall(yt_idx, pred, K),
            "confusion": confusion(yt_idx, pred, K),
            "n_test_per_class": len(yt) // K}


def _perm_null(y_true: np.ndarray, y_pred: np.ndarray, K: int,
               n_perm: int = 2000, seed: int = 11):
    """예측 라벨을 무작위 재배열해 균형정확도·최소재현율의 영분포를 만든다."""
    rng = np.random.default_rng(seed)
    nb = np.empty(n_perm); nm = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(y_pred)
        nb[i] = balanced_accuracy(y_true, p, K)
        nm[i] = np.nanmin(per_class_recall(y_true, p, K))
    return nb, nm


def run(cfg: Config, reg: Registry) -> dict:
    LOGGER.info("[H1] 전략적 의도 추론 — 6유형 × %d 시드 × %d 라운드 × 2 조건",
                cfg.seeds, cfg.rounds)
    K = len(ALL_TYPES)
    chance = 1.0 / K

    conds = {}
    for tag, (cname, eps) in enumerate(
            (("on_policy", cfg.env_error), ("probe", PROBE_EPS))):
        LOGGER.info("  조건 '%s' (focal ε=%.2f)", cname, eps)
        th = collect_theta(cfg, focal_eps=eps, tag=tag * 77)
        cl = classify(th, cfg.seeds)
        conds[cname] = {"theta": th, **cl}

    # ---- 확증 1·2: 조건별 균형정확도 > 우연 ----
    for cname in ("on_policy", "probe"):
        c = conds[cname]
        nb, nm = _perm_null(c["y_true"], c["y_pred"], K,
                            seed=11 if cname == "on_policy" else 12)
        p = (np.sum(nb >= c["balanced_accuracy"]) + 1) / (len(nb) + 1)
        c["null_min"] = nm
        reg.confirm("H1", f"[{CONDITION_LABEL[cname]}] 균형정확도 > 우연(1/6)",
                    p, direction_ok=bool(c["balanced_accuracy"] > chance),
                    effect=f"BA={c['balanced_accuracy']:.3f} (우연={chance:.3f})",
                    detail={"balanced_accuracy": c["balanced_accuracy"]})

    # ---- 확증 3: 능동 자극이 최소 재현율을 끌어올리는가 ----
    # 두 조건의 검정 표본은 독립이므로 순열검정으로 최소재현율 차이를 검정한다.
    min_on = float(np.nanmin(conds["on_policy"]["recalls"]))
    min_pr = float(np.nanmin(conds["probe"]["recalls"]))
    obs = min_pr - min_on
    rng = np.random.default_rng(13)
    n_perm = 2000
    # 영가설: 두 조건의 예측이 교환가능. 조건 라벨을 섞어 차이의 영분포 생성.
    yt = conds["on_policy"]["y_true"]
    pool = np.stack([conds["on_policy"]["y_pred"], conds["probe"]["y_pred"]])
    null = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(pool.shape[1]) < 0.5
        a = np.where(swap, pool[1], pool[0])
        b = np.where(swap, pool[0], pool[1])
        null[i] = (np.nanmin(per_class_recall(yt, b, K))
                   - np.nanmin(per_class_recall(yt, a, K)))
    p3 = (np.sum(null >= obs) + 1) / (n_perm + 1)
    worst_on = ALL_TYPES[int(np.nanargmin(conds["on_policy"]["recalls"]))]
    reg.confirm("H1", "최소 유형별 재현율: probe > on-policy (능동 자극의 효과)",
                p3, direction_ok=bool(obs > 0),
                effect=f"min recall {min_on:.3f} ({TYPE_LABEL_KO[worst_on]}) "
                       f"→ {min_pr:.3f} (Δ={obs:+.3f})")

    # ---- 탐색: 조건 × 유형별 재현율 ----
    from math import comb
    for cname in ("on_policy", "probe"):
        c = conds[cname]
        n_test = c["n_test_per_class"]
        for k, tname in enumerate(ALL_TYPES):
            hit = int(round(c["recalls"][k] * n_test))
            ci = wilson_ci(hit, n_test)
            p_bin = sum(comb(n_test, i) * chance ** i
                        * (1 - chance) ** (n_test - i)
                        for i in range(hit, n_test + 1))
            reg.explore("H1", f"[{cname}] {TYPE_LABEL_KO[tname]} 재현율 > 우연",
                        p_bin,
                        effect=f"recall={c['recalls'][k]:.3f} "
                               f"[{ci['ci'][0]:.3f}, {ci['ci'][1]:.3f}]")

    result = {
        "chance": chance, "types": list(ALL_TYPES), "features": FEATURES,
        "conditions": {
            cname: {
                "balanced_accuracy": conds[cname]["balanced_accuracy"],
                "recalls": conds[cname]["recalls"],
                "confusion": conds[cname]["confusion"],
                "n_test_per_class": conds[cname]["n_test_per_class"],
                "theta_mean": {t: conds[cname]["theta"][t].mean(axis=0)
                               for t in ALL_TYPES},
                "theta_sd": {t: conds[cname]["theta"][t].std(axis=0, ddof=1)
                             for t in ALL_TYPES},
            } for cname in conds},
        "min_recall": {"on_policy": min_on, "probe": min_pr},
    }
    _plot(cfg, result, conds)
    save_json(result, cfg.results / "H1.json")
    return result


# ==================================================================== 시각화
def _plot(cfg: Config, r: dict, conds: dict) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.0],
                          hspace=0.62, wspace=0.34)
    labels = [TYPE_LABEL_KO[t] for t in ALL_TYPES]
    order = ("on_policy", "probe")

    # (a, b) 조건별 혼동행렬
    for ci, cname in enumerate(order):
        c = r["conditions"][cname]
        ax = fig.add_subplot(gs[0, ci])
        im = ax.imshow(c["confusion"], cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("예측 유형"); ax.set_ylabel("참 유형")
        ax.set_title(f"({'ab'[ci]}) 혼동행렬 — {CONDITION_LABEL[cname]}\n"
                     f"균형정확도 {c['balanced_accuracy']:.3f} "
                     f"(우연 {r['chance']:.3f}, 검정 n={c['n_test_per_class']}/유형)",
                     fontsize=8.5)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = c["confusion"][i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if v > 0.55 else "black")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045)

    # (c) 조건별 유형 재현율 비교
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(labels)); w = 0.38
    for off, cname, col in ((-w / 2, "on_policy", "#4C72B0"),
                            (+w / 2, "probe", "#C44E52")):
        ax.bar(x + off, r["conditions"][cname]["recalls"], w, color=col,
               label=CONDITION_LABEL[cname], alpha=0.9)
    ax.axhline(r["chance"], color="crimson", ls="--", lw=1.2, label="우연수준")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right",
                                         fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("재현율")
    ax.set_title(f"(c) 유형별 재현율\n최소값 "
                 f"{r['min_recall']['on_policy']:.2f} → "
                 f"{r['min_recall']['probe']:.2f}", fontsize=8.5)
    ax.legend(fontsize=6.5)

    # (d) θ 시그니처 히트맵 (probe 조건, 축별 표준화)
    ax = fig.add_subplot(gs[1, 0])
    M = np.vstack([r["conditions"]["probe"]["theta_mean"][t]
                   for t in ALL_TYPES])
    Mz = (M - M.mean(axis=0)) / np.maximum(M.std(axis=0, ddof=1), 1e-9)
    im = ax.imshow(Mz, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([AXIS_LABEL_KO[a] for a in FEATURES], rotation=45,
                       ha="right", fontsize=6.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("(d) 유형별 θ̂ 시그니처 (probe, 축별 표준화)", fontsize=8.5)
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (e~i) 주요 판별축의 유형별 분포 (probe 조건)
    panels = ["e", "f", "g", "h", "i"]
    for p_i, ax_name in enumerate(("rho", "eta", "alpha", "beta", "lambda_j")):
        row, col = divmod(p_i + 1, 3)
        ax = fig.add_subplot(gs[1 + row, col])
        d = FEATURES.index(ax_name)
        data = [conds["probe"]["theta"][t][:, d] for t in ALL_TYPES]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        widths=0.6, showfliers=False)
        for patch, t in zip(bp["boxes"], ALL_TYPES):
            patch.set_facecolor(TYPE_COLORS[t]); patch.set_alpha(0.75)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        ax.set_ylabel(AXIS_LABEL_KO[ax_name], fontsize=8)
        ax.set_title(f"({panels[p_i]}) {AXIS_LABEL_KO[ax_name]}", fontsize=8.5)

    fig.suptitle("H1 — 전략적 의도의 강건한 추론 (입자필터 사후 θ̂ 판독)",
                 fontsize=12, y=0.985)
    save_fig(fig, cfg, "H1_intent_inference")

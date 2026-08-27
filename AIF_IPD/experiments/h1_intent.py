"""
experiments.h1_intent
=====================

**H1 — does OpponentInversion robustly infer another agent's
strategic intent?**

Target types: TFT, GTFT, WSLS, ALLC, ALLD, HalloReg (6).

Design
------
1. A HalloReg focal runs `seeds` dyads against each of the 6 types.
2. The **final posterior mean** theta-hat = (alpha, rho, omega, eta,
   beta, lambda_j) of each dyad is a 6-dimensional feature vector.
3. Seeds are split in half into **calibration** and **test** sets.
4. A diagonal-covariance Gaussian (i.e. Gaussian naive Bayes) is
   fitted per type on calibration and used to classify the test set.

Why a learned classifier rather than template matching: TFT, WSLS,
ALLC and ALLD have closed-form theta signatures (e.g. WSLS implies
pure eta), but **HalloReg is adaptive** and has no a-priori template.
Treating all six on equal footing therefore means estimating each
type's theta-hat distribution from data and classifying by Bayes
rule. Calibration and test seeds are disjoint, so this is not
overfitting — it is the standard cross-validated decoding readout of
the neural decoding literature.

Two conditions — the behavioural equivalence class problem
----------------------------------------------------------
**A structural limit predicted in advance.** As long as the focal
behaves cooperatively, some opponent types are **observationally
indistinguishable**:

  - TFT vs GTFT : GTFT's generosity only manifests as "cooperates
    even after the focal defected". Without focal defections the two
    strategies emit **identical action sequences**.
  - ALLC vs GTFT : likewise, both always cooperate absent focal
    defections.

An on-policy observer cannot obtain information about conditions it
never creates. This is not an inference defect but an identification
problem caused by the **absence of active sampling**. Hence two
conditions run side by side:

  on_policy : the focal acts purely on its own policy (natural
              interaction).
  probe     : the environment-layer execution error is raised
              (eps = 0.20), forcing the focal's behaviour to wobble.
              Intermittent defections make the opponent's
              **forgiveness structure** observable.

The probe condition is an **experimental manipulation**, not part of
the model (environment-imposed noise; the agent's policy is
unchanged). Contrasting the conditions is itself the diagnostic of
"what makes identification possible".

[Null hypotheses and tests]
- Confirmatory 1 : on_policy balanced accuracy > chance (1/6);
                   label permutation test.
- Confirmatory 2 : probe balanced accuracy > chance.
- Confirmatory 3 : **minimum per-type recall** — probe > on_policy.
                   "Robustly" means no type fails, not that the mean
                   is high, so the minimum is the index and the test
                   asks whether active sampling raises it.
- Exploratory    : recall by condition and type, per-axis
                   discriminability, confusion structure.
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

#: theta axes used as features (posterior means)
FEATURES = list(THETA_AXES)

#: Axis display labels
AXIS_LABEL_KO = {"alpha": "alpha (cooperation bias)",
                 "rho": "rho (reciprocity)",
                 "omega": "omega (inertia)",
                 "eta": "eta (outcome contingency)",
                 "beta": "beta (precision)",
                 "lambda_j": "lambda_j (their empathy)"}


# ================================================================ Classifier
class DiagGaussianClassifier:
    """
    Diagonal-covariance Gaussian classifier (Gaussian naive Bayes).

    With 6 features and only tens of samples per type, a full
    covariance has far too much estimation variance; the diagonal
    approximation trades a little bias for a large variance reduction
    and generalises better at this sample size.

    Priors are fixed uniform — natural because seed counts are equal
    per type, and it avoids a bias favouring majority types.
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
            # Variance floor: prevents a diverging likelihood when an
            # axis is nearly constant within a type
            self.var[k] = np.maximum(Xc.var(axis=0, ddof=1), self.var_floor)
        return self

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """(N, K) log-likelihood."""
        # −½ Σ_d [ log(2πσ²) + (x−μ)²/σ² ]
        diff = X[:, None, :] - self.mu[None, :, :]           # (N, K, D)
        return -0.5 * np.sum(np.log(2 * np.pi * self.var)[None, :, :]
                             + diff ** 2 / self.var[None, :, :], axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted label indices (N,)."""
        return np.argmax(self.log_likelihood(X), axis=1)


def balanced_accuracy(y_true_idx: np.ndarray, y_pred_idx: np.ndarray,
                      K: int) -> float:
    """Mean of per-type recalls (accuracy robust to imbalance)."""
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
    """Rows = true, columns = predicted; row-normalised."""
    M = np.zeros((K, K))
    for t, p in zip(y_true_idx, y_pred_idx):
        M[t, p] += 1
    return M / np.maximum(M.sum(axis=1, keepdims=True), 1)


# =================================================================== Run
#: Focal execution-error rate in the probe condition (the
#: operationalisation of active sampling)
PROBE_EPS = 0.20
CONDITION_LABEL = {"on_policy": "on-policy (natural interaction)",
                   "probe": f"probe (focal noise eps={PROBE_EPS})"}


def collect_theta(cfg: Config, focal_eps: float, tag: int
                  ) -> Dict[str, np.ndarray]:
    """
    Run `seeds` dyads against each of the 6 types and collect the
    final theta-hat.

    focal_eps : focal-side environment execution error, raised in the
                probe condition.
    tag       : offset that decorrelates seeds across conditions.
    Returns   : {type name: (seeds, D) array}.
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
                   desc="H1 intent-inference dyads")

    out = {}
    for ti, tname in enumerate(ALL_TYPES):
        F = np.zeros((cfg.seeds, len(FEATURES)))
        for sd in range(cfg.seeds):
            log = res[registry[(ti, sd)]]["agent_log"]
            for d, ax in enumerate(FEATURES):
                F[sd, d] = float(log[f"E_{ax}"][-1])     # final-round posterior mean
        out[tname] = F
    return out


def classify(theta: Dict[str, np.ndarray], seeds: int) -> dict:
    """Classify the 6 types with a half-split calibration/test and
    return the metrics."""
    K = len(ALL_TYPES)
    n_cal = seeds // 2
    Xc, yc, Xt, yt = [], [], [], []
    for tname in ALL_TYPES:
        F = theta[tname]
        Xc.append(F[:n_cal]); yc += [tname] * n_cal
        Xt.append(F[n_cal:]); yt += [tname] * (len(F) - n_cal)
    Xc = np.vstack(Xc); Xt = np.vstack(Xt)

    # Standardise using **calibration statistics only** (no test-set
    # information leakage).
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
    """Shuffle predicted labels to build the null distribution of
    balanced accuracy and minimum recall."""
    rng = np.random.default_rng(seed)
    nb = np.empty(n_perm); nm = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(y_pred)
        nb[i] = balanced_accuracy(y_true, p, K)
        nm[i] = np.nanmin(per_class_recall(y_true, p, K))
    return nb, nm


def run(cfg: Config, reg: Registry) -> dict:
    LOGGER.info("[H1] strategic intent inference — 6 types x %d seeds "
                "x %d rounds x 2 conditions",
                cfg.seeds, cfg.rounds)
    K = len(ALL_TYPES)
    chance = 1.0 / K

    conds = {}
    for tag, (cname, eps) in enumerate(
            (("on_policy", cfg.env_error), ("probe", PROBE_EPS))):
        LOGGER.info("  condition '%s' (focal eps=%.2f)", cname, eps)
        th = collect_theta(cfg, focal_eps=eps, tag=tag * 77)
        cl = classify(th, cfg.seeds)
        conds[cname] = {"theta": th, **cl}

    # ---- Confirmatory 1-2: balanced accuracy > chance per condition
    for cname in ("on_policy", "probe"):
        c = conds[cname]
        nb, nm = _perm_null(c["y_true"], c["y_pred"], K,
                            seed=11 if cname == "on_policy" else 12)
        p = (np.sum(nb >= c["balanced_accuracy"]) + 1) / (len(nb) + 1)
        c["null_min"] = nm
        reg.confirm("H1", f"[{CONDITION_LABEL[cname]}] balanced accuracy "
                    f"> chance (1/6)",
                    p, direction_ok=bool(c["balanced_accuracy"] > chance),
                    effect=f"BA={c['balanced_accuracy']:.3f} "
                           f"(chance={chance:.3f})",
                    detail={"balanced_accuracy": c["balanced_accuracy"]})

    # ---- Confirmatory 3: does active sampling raise the minimum
    # recall? The two conditions' test samples are independent, so the
    # difference in minimum recall is tested by permutation.
    min_on = float(np.nanmin(conds["on_policy"]["recalls"]))
    min_pr = float(np.nanmin(conds["probe"]["recalls"]))
    obs = min_pr - min_on
    rng = np.random.default_rng(13)
    n_perm = 2000
    # Null: predictions of the two conditions are exchangeable —
    # shuffle condition labels to build the null of the difference.
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
    reg.confirm("H1", "minimum per-type recall: probe > on-policy "
                "(effect of active sampling)",
                p3, direction_ok=bool(obs > 0),
                effect=f"min recall {min_on:.3f} ({TYPE_LABEL_KO[worst_on]}) "
                       f"→ {min_pr:.3f} (Δ={obs:+.3f})")

    # ---- Exploratory: recall by condition x type ----
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
            reg.explore("H1", f"[{cname}] {TYPE_LABEL_KO[tname]} recall "
                        f"> chance",
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


# ============================================================== Figures
def _plot(cfg: Config, r: dict, conds: dict) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.0],
                          hspace=0.62, wspace=0.34)
    labels = [TYPE_LABEL_KO[t] for t in ALL_TYPES]
    order = ("on_policy", "probe")

    # (a, b) confusion matrix per condition
    for ci, cname in enumerate(order):
        c = r["conditions"][cname]
        ax = fig.add_subplot(gs[0, ci])
        im = ax.imshow(c["confusion"], cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("predicted type"); ax.set_ylabel("true type")
        ax.set_title(f"({'ab'[ci]}) confusion — {CONDITION_LABEL[cname]}\n"
                     f"balanced accuracy {c['balanced_accuracy']:.3f} "
                     f"(chance {r['chance']:.3f}, "
                     f"test n={c['n_test_per_class']}/type)",
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

    # (c) per-type recall across conditions
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(labels)); w = 0.38
    for off, cname, col in ((-w / 2, "on_policy", "#4C72B0"),
                            (+w / 2, "probe", "#C44E52")):
        ax.bar(x + off, r["conditions"][cname]["recalls"], w, color=col,
               label=CONDITION_LABEL[cname], alpha=0.9)
    ax.axhline(r["chance"], color="crimson", ls="--", lw=1.2,
               label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right",
                                         fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("recall")
    ax.set_title(f"(c) per-type recall\nminimum "
                 f"{r['min_recall']['on_policy']:.2f} → "
                 f"{r['min_recall']['probe']:.2f}", fontsize=8.5)
    ax.legend(fontsize=6.5)

    # (d) theta signature heatmap (probe condition, per-axis z)
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
    ax.set_title("(d) theta-hat signature by type "
                 "(probe, per-axis z-scored)", fontsize=8.5)
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (e-i) distribution of the main discriminative axes (probe)
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

    fig.suptitle("H1 — robust inference of strategic intent "
                 "(particle-filter posterior theta-hat readout)",
                 fontsize=12, y=0.985)
    save_fig(fig, cfg, "H1_intent_inference")

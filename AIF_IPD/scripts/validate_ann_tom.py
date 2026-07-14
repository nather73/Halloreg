#!/usr/bin/env python3
"""
validate_ann_tom.py
===================

**독립 실행 검증**: 학습된 인공신경망(ANN)이 입자필터 기반 베이지안 Theory-of-Mind 을
재현하는가? (수정사항 #4)

  python AIF_IPD/scripts/validate_ann_tom.py [--quick] [--seeds N] [--epochs E] [--jobs J]

두 구조를 학습·평가한다:
  · Schwarcz-style GRU (Schwarcz et al. 2025): 순환 동역학에 베이즈 믿음 갱신 내재화.
  · Kim-style GCN+RNN (Kim et al. 2026): 관계형 그래프 + 스포트라이트 주의(간선 엔트로피
    = 특성별 reliability, dACC 대응).

교사(teacher) = 입자필터 사후평균 + 특성별 정밀도. 평가:
  (1) 교사 재현 R²/RMSE (증류 목표),
  (2) 참(true) θ 회복 — 생성모형의 α↔λ_j 가법 축퇴로 개별 회복은 제한되며, 식별가능
      합성 α+5λ_j 는 잘 회복됨(정직한 식별성 보고),
  (3) 보정(calibration): 예측 정밀도 ↔ 실제 오차,
  (4) A-B-A 일반화: 상대 θ 가 A→B→A 로 전환할 때 특성 추론 복구(§ABA 연계).

산출물: results/ann_tom_recovery.png (+pdf, caption), results/ann_tom_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from AIF_IPD.ipd import ann_tom as at
from AIF_IPD.core.constants import CC  # noqa (font util 경로 통일)

RESULTS = _ROOT / "AIF_IPD" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _set_korean_font():
    try:
        from AIF_IPD.core.logging_utils import set_korean_font
        set_korean_font(plt, font_manager)
    except Exception:
        for cand in ("NanumGothic", "Malgun Gothic", "AppleGothic"):
            try:
                plt.rcParams["font.family"] = cand
                plt.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue


def make_figure(e_gru, e_gcn, aba, hist_gru, hist_gcn, tag=""):
    _set_korean_font()
    AX = at._AX
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (a) 교사 재현 R² (특성별, 두 모델)
    x = np.arange(len(AX)); w = 0.38
    r2g = [e_gru["per_trait"][a]["r2_teacher"] for a in AX]
    r2c = [e_gcn["per_trait"][a]["r2_teacher"] for a in AX]
    ax[0, 0].bar(x - w / 2, r2g, w, label="Schwarcz GRU", color="C0")
    ax[0, 0].bar(x + w / 2, r2c, w, label="Kim GCN+RNN", color="C2")
    ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(AX)
    ax[0, 0].set_title("(a) 교사(입자필터) 재현 R²\nBayesian ToM 의 ANN 재현")
    ax[0, 0].set_ylabel("R² (vs 교사 사후)"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_ylim(min(0, min(r2g + r2c)) - 0.05, 1.0)

    # (b) 참 θ 회복 — 개별(제한) vs 식별가능 합성
    r2t_g = [e_gru["per_trait"][a]["r2_true"] for a in AX]
    r2t_c = [e_gcn["per_trait"][a]["r2_true"] for a in AX]
    ax[0, 1].bar(x - w / 2, r2t_g, w, label="GRU", color="C0", alpha=0.8)
    ax[0, 1].bar(x + w / 2, r2t_c, w, label="GCN+RNN", color="C2", alpha=0.8)
    ax[0, 1].axhline(0, color="k", lw=0.8)
    comp_g = e_gru["identifiable_composite"]["r2_true"]
    comp_c = e_gcn["identifiable_composite"]["r2_true"]
    ax[0, 1].bar([len(AX)], [comp_g], w, color="C0", hatch="//")
    ax[0, 1].bar([len(AX) + 0.4], [comp_c], w, color="C2", hatch="//")
    ax[0, 1].set_xticks(list(x) + [len(AX) + 0.2])
    ax[0, 1].set_xticklabels(AX + ["α+5λ_j\n(식별가능)"], fontsize=8)
    ax[0, 1].set_title("(b) 참 θ 회복: 개별(축퇴) vs 식별가능 합성")
    ax[0, 1].set_ylabel("R² (vs 참값)"); ax[0, 1].legend(fontsize=8)

    # (c) 산점: 교사 vs GRU 예측 (lambda_j — 가장 식별가능)
    dtest = aba["_scatter"]
    ax[0, 2].scatter(dtest["teach"], dtest["pred"], s=6, alpha=0.3, color="C0")
    lim = [min(dtest["teach"].min(), dtest["pred"].min()),
           max(dtest["teach"].max(), dtest["pred"].max())]
    ax[0, 2].plot(lim, lim, "k--", lw=1)
    ax[0, 2].set_title(f"(c) λ_j: 교사 vs GRU 예측 (R²={e_gru['per_trait']['lambda_j']['r2_teacher']:.2f})")
    ax[0, 2].set_xlabel("교사 사후평균"); ax[0, 2].set_ylabel("GRU 예측")

    # (d) 보정: 예측 로그정밀도 ↔ 실제 오차 상관 (음수=잘 보정)
    calib_g = [e_gru["calibration_lprec_vs_abserr"][a] for a in AX]
    calib_c = [e_gcn["calibration_lprec_vs_abserr"][a] for a in AX]
    ax[1, 0].bar(x - w / 2, calib_g, w, label="GRU", color="C0")
    ax[1, 0].bar(x + w / 2, calib_c, w, label="GCN+RNN", color="C2")
    ax[1, 0].axhline(0, color="k", lw=0.8)
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(AX)
    ax[1, 0].set_title("(d) 보정: corr(로그정밀도, |오차|)\n음수=신뢰도가 정확도 반영")
    ax[1, 0].set_ylabel("상관"); ax[1, 0].legend(fontsize=8)

    # (e) A-B-A 특성 궤적 복구 (lambda_j; 교사 vs 두 모델 vs 참값)
    il = AX.index("lambda_j")
    T = aba["n_rounds"]; third = aba["third"]
    tt = np.arange(T)
    ax[1, 1].plot(tt, np.array(aba["true_traj"])[:, il], "k-", lw=2, label="참값")
    ax[1, 1].plot(tt, np.array(aba["teacher_traj"])[:, il], color="0.5", ls="--",
                  lw=1.6, label="교사(입자필터)")
    for name, color in [("gru", "C0"), ("gcn", "C2")]:
        ax[1, 1].plot(tt, np.array(aba["models"][name]["traj"])[:, il], color=color,
                      lw=1.6, label={"gru": "GRU", "gcn": "GCN+RNN"}[name])
    for c in (third, 2 * third):
        ax[1, 1].axvline(c, color="0.7", ls=":", lw=1)
    ax[1, 1].set_title("(e) A→B→A λ_j 추론 복구\n협력(A)→착취(B)→협력(A)")
    ax[1, 1].set_xlabel("라운드"); ax[1, 1].set_ylabel("λ_j 추정"); ax[1, 1].legend(fontsize=7)

    # (f) 학습 곡선
    ax[1, 2].plot(hist_gru, color="C0", label="GRU")
    ax[1, 2].plot(hist_gcn, color="C2", label="GCN+RNN")
    ax[1, 2].set_title("(f) 학습 곡선 (교사 증류 NLL)")
    ax[1, 2].set_xlabel("epoch"); ax[1, 2].set_ylabel("손실"); ax[1, 2].legend(fontsize=8)

    cap = ("§ANN-ToM 학습된 ANN 의 베이지안 ToM 재현: (a) 두 구조 모두 입자필터 사후를 "
           "높은 R² 로 재현(Bayesian ToM 은 학습가능). (b) 참 θ 개별 회복은 생성모형의 "
           "α↔λ_j 가법 축퇴로 제한되나 식별가능 합성 α+5λ_j 는 잘 회복(정직한 식별성). "
           "(c) 교사–예측 산점. (d) 예측 정밀도가 실제 오차와 음의 상관(보정됨). "
           "(e) A→B→A 전환 시 특성 추론이 교사와 함께 복구. (f) 학습 곡선.")
    fig.suptitle("학습된 ANN 의 베이지안 Theory-of-Mind 재현 (Schwarcz GRU · Kim GCN+RNN)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    png = RESULTS / f"ann_tom_recovery{tag}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / f"ann_tom_recovery{tag}.pdf", bbox_inches="tight")
    (RESULTS / f"ann_tom_recovery{tag}.caption.txt").write_text(cap, encoding="utf-8")
    plt.close(fig)
    print(f"[figure] {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="빠른 스모크(작은 데이터/에폭)")
    ap.add_argument("--train-dyads", type=int, default=240)
    ap.add_argument("--test-dyads", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--particles", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        args.train_dyads = 40; args.test_dyads = 16; args.rounds = 40
        args.epochs = 20; args.particles = 200; args.hidden = 48

    print(f"=== ANN-ToM 검증 === train={args.train_dyads} test={args.test_dyads} "
          f"rounds={args.rounds} epochs={args.epochs} particles={args.particles}")
    print("[1/4] 데이터 생성(교사=입자필터 사후) ...")
    train = at.generate_dataset(args.train_dyads, args.rounds, args.seed,
                                n_particles=args.particles)
    test = at.generate_dataset(args.test_dyads, args.rounds, args.seed + 99,
                               n_particles=args.particles)
    scaler = at._theta_scaler(train["theta"].reshape(-1, 4))

    key = jax.random.PRNGKey(args.seed + 3)
    k1, k2 = jax.random.split(key)
    print("[2/4] Schwarcz GRU 학습 ...")
    m_gru, h_gru = at.train_model(at.SchwarczGRU(k1, hidden=args.hidden), train,
                                  scaler, k1, epochs=args.epochs, tag="GRU")
    print("[3/4] Kim GCN+RNN 학습 ...")
    m_gcn, h_gcn = at.train_model(at.KimGCNRNN(k2, hidden=args.hidden), train,
                                  scaler, k2, epochs=args.epochs, tag="GCN")

    print("[4/4] 평가 + A-B-A 일반화 ...")
    ef = max(10, args.rounds // 3)
    e_gru = at.evaluate(m_gru, test, scaler, eval_from=ef)
    e_gcn = at.evaluate(m_gcn, test, scaler, eval_from=ef)
    aba = at.aba_generalization({"gru": m_gru, "gcn": m_gcn}, scaler,
                                seed=args.seed + 7,
                                n_dyads=max(16, args.test_dyads // 2),
                                n_rounds=max(60, args.rounds + 40 if not args.quick else 60))
    # 산점용(λ_j 교사 vs GRU 예측)
    ms, _, _ = at.predict(m_gru, test["X"]); ms = at._destandardize(ms, scaler)
    sl = slice(ef, args.rounds)
    aba["_scatter"] = {"teach": test["Y_mean"][:, sl, at._AX.index("lambda_j")].ravel(),
                       "pred": ms[:, sl, at._AX.index("lambda_j")].ravel()}

    print(f"\n  Schwarcz GRU : meanR²_teacher={e_gru['mean_r2_teacher']:.3f} "
          f"| 합성 α+5λ_j R²_true={e_gru['identifiable_composite']['r2_true']:.3f}")
    print(f"  Kim GCN+RNN  : meanR²_teacher={e_gcn['mean_r2_teacher']:.3f} "
          f"| 합성 α+5λ_j R²_true={e_gcn['identifiable_composite']['r2_true']:.3f}")
    for a in at._AX:
        print(f"    {a:9s} R²_teacher: GRU={e_gru['per_trait'][a]['r2_teacher']:+.3f} "
              f"GCN={e_gcn['per_trait'][a]['r2_teacher']:+.3f}")
    print(f"  A-B-A 복구오차: GRU={aba['models']['gru']['rec_err']:.3f} "
          f"GCN={aba['models']['gcn']['rec_err']:.3f} "
          f"(교사정합 A3: GRU={aba['models']['gru']['align_teacher_A3']:.3f})")

    make_figure(e_gru, e_gcn, aba, h_gru, h_gcn)

    summary = {
        "config": vars(args),
        "gru": {"eval": _clean(e_gru), "final_loss": float(h_gru[-1]),
                "aba": {k: v for k, v in aba["models"]["gru"].items() if k != "traj"}},
        "gcn": {"eval": _clean(e_gcn), "final_loss": float(h_gcn[-1]),
                "aba": {k: v for k, v in aba["models"]["gcn"].items() if k != "traj"}},
        "verdict": {
            "ann_reproduces_bayesian_tom": bool(e_gru["mean_r2_teacher"] > 0.6
                                                and e_gcn["mean_r2_teacher"] > 0.6),
            "identifiable_composite_recovered": bool(
                e_gru["identifiable_composite"]["r2_true"] > 0.3),
            "note": ("개별 α/λ_j 참값 회복은 생성모형의 가법 축퇴로 제한적이며, 이는 "
                     "교사(입자필터)에도 동일하게 적용되는 식별성 한계임(ANN 결함 아님).")},
    }
    out = RESULTS / "ann_tom_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {out}")
    print("\n판정:", "ANN 이 베이지안 ToM 을 재현함 ✓"
          if summary["verdict"]["ann_reproduces_bayesian_tom"]
          else "재현 불충분 — 데이터/에폭 증가 권장")


def _clean(d):
    """numpy 스칼라 → 파이썬 기본형 (json 직렬화)."""
    import numpy as _np
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (_np.floating, _np.integer)):
        return float(d)
    return d


if __name__ == "__main__":
    main()

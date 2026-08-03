"""
experiments.arch_validation
===========================

**아키텍처 검증 (ARCH) — 가설검증 이전에 반드시 통과해야 하는 기제 확인.**

가설 결과가 의미를 가지려면, 그 이전에 구현된 위계가 **설계대로 작동하는지**를
직접 확인해야 한다. 이 모듈은 가설이 아니라 **구현 정합성**을 검증한다.
여기서 실패하면 H1~H6 의 해석 자체가 무의미하므로, 가장 먼저 실행된다.

검증 항목
─────────
V1. **정보 흐름** — SelfModel 이 사전을 공급하고 두 모듈이 갱신 결과를 되돌리는가.
    상대를 바꿔가며 여러 번 조우시킨 뒤 SelfModel.memory 에 identity 별 항목이
    쌓이고, 각 항목의 θ 가 상대 유형을 반영하는지 확인한다.

V2. **valence 의 부호** — RPE 의 부호와 valence 의 부호가 일치하는가.
    설계상 valence = tanh(RPE / σ) 이므로 두 부호는 **완전히 일치**해야 한다.
    불일치율이 0 이 아니면 구현 결함이다.

V3. **arousal 과 믿음 갱신** — arousal 이 KL 의 단조 증가함수인가.
    설계상 arousal = 1 − exp(−KL/κ) 이므로 Spearman 상관이 정확히 +1 이어야 한다.

V4. **λ 갱신식의 정확성** — 기록된 λ_t 가 λ_{t−1} + η[(1−w)λ_aff + w λ_ctx] 를
    클리핑한 값과 일치하는가. 수치적으로 재구성해 최대 편차를 확인한다.

V5. **할로스타틱 설정점의 이동** — 사회적 기저 기대보상이 경험에 따라
    **느리게** 이동하는가. 착취적 환경에서는 하강, 협력적 환경에서는 상승해야
    하며, 그 속도가 identity 별 갱신보다 느려야 한다(이상성 vs 항상성).

V6. **λ 의 조건별 분기** — 착취자 상대에서는 λ 가 하강하고 협력자 상대에서는
    상승하는가. 이것이 성립하지 않으면 조절 기제가 방향성을 갖지 않는 것이다.

각 항목은 **결정적(deterministic) 검증**이거나 방향성 검정이며, 통계적 가설이
아니므로 확증/탐색 레지스트리에는 등록하지 않고 별도 판정표로 보고한다.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.constants import PAYOFF_SELF, COOP
from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.core.self_model import SelfModel
from AIF_IPD.ipd.agent import HalloRegAgent
from AIF_IPD.ipd.env import make_opponent
from AIF_IPD.ipd.sim import run_dyad, run_many
from .common import Config, save_fig, save_json

LOGGER = get_logger("HalloReg.ARCH")

#: 검증에 쓰는 상대 유형 (협력적 ↔ 착취적 스펙트럼)
PROBE_TYPES = ("allc", "gtft", "tft", "wsls", "alld")
PROBE_LABEL = {"allc": "ALLC", "gtft": "GTFT", "tft": "TFT",
               "wsls": "WSLS", "alld": "ALLD"}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """순위 상관 (동점은 평균순위)."""
    def rank(v):
        o = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        r[o] = np.arange(1, len(v) + 1)
        uq, iv, ct = np.unique(v, return_inverse=True, return_counts=True)
        for u in np.where(ct > 1)[0]:
            m = iv == u
            r[m] = r[m].mean()
        return r
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def run(cfg: Config) -> dict:
    LOGGER.info("[ARCH] 아키텍처 정합성 검증")
    checks: List[dict] = []
    n_probe = min(cfg.seeds, 30)

    # ================================================== V1. 정보 흐름 / 기억
    # 하나의 에이전트가 5개 유형을 순차 조우한다. run_dyad 는 begin_partner 로
    # identity 를 통지하므로, SelfModel.memory 에 5개 항목이 쌓여야 한다.
    agent = HalloRegAgent(seed=1, **cfg.halloreg_kwargs())
    theta_by_partner = {}
    reward_by_partner = {}
    for pid, kind in enumerate(PROBE_TYPES, start=1):
        opp = make_opponent(kind, seed=100 + pid)
        run_dyad(agent, opp, cfg.rounds, partner_id_a=pid, partner_id_b=999)
        ent = agent.self_model.memory[pid]
        theta_by_partner[kind] = dict(ent.theta)
        reward_by_partner[kind] = SelfModel.expected_reward(ent.reward,
                                                            PAYOFF_SELF)
    n_mem = len(agent.self_model.memory)
    checks.append({
        "id": "V1", "name": "SelfModel 기억 축적 (identity 별 θ·보상분포)",
        "passed": bool(n_mem == len(PROBE_TYPES)),
        "detail": f"기억 항목 {n_mem}개 / 조우 {len(PROBE_TYPES)}개",
    })
    # 기억된 α̂ 가 상대의 협력성과 단조 관계인가 (ALLC 가장 높고 ALLD 가장 낮음)
    alpha_order = [theta_by_partner[k]["alpha"] for k in PROBE_TYPES]
    checks.append({
        "id": "V1b", "name": "기억된 α̂ 가 상대 협력성 순서를 반영 (ALLC > ALLD)",
        "passed": bool(alpha_order[0] > alpha_order[-1]),
        "detail": " | ".join(f"{PROBE_LABEL[k]}={theta_by_partner[k]['alpha']:+.2f}"
                             for k in PROBE_TYPES),
    })

    # ============================== V2~V4. 정서·λ 갱신식의 수치적 정합성
    specs = []
    for ti, kind in enumerate(PROBE_TYPES):
        for sd in range(n_probe):
            specs.append({
                "agent": {"type": "halloreg", "seed": 200 + sd * 11 + ti,
                          **cfg.halloreg_kwargs()},
                "opponent": {"type": "strategy", "kind": kind,
                             "seed": 300 + sd * 11 + ti},
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 400_000 + sd * 13 + ti})
    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="ARCH 검증 다이애드")

    sign_mismatch = 0
    sign_total = 0
    rho_ka: List[float] = []
    lam_max_dev = 0.0
    lam_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    base_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    val_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    aro_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}

    idx = 0
    for ti, kind in enumerate(PROBE_TYPES):
        for sd in range(n_probe):
            log = res[idx]["agent_log"]; idx += 1
            rpe = np.asarray(log["rpe"], float)
            val = np.asarray(log["valence"], float)
            aro = np.asarray(log["arousal"], float)
            kl = np.asarray(log["kl"], float)
            lam = np.asarray(log["lam"], float)
            l_aff = np.asarray(log["lambda_aff"], float)
            l_ctx = np.asarray(log["lambda_ctx"], float)

            lam_traces[kind][sd] = lam
            base_traces[kind][sd] = np.asarray(log["baseline_reward"], float)
            val_traces[kind][sd] = val
            aro_traces[kind][sd] = aro

            # --- V2: 부호 일치 (첫 라운드는 관측 없음 → 제외) ---
            m = np.abs(rpe[1:]) > 1e-12
            sign_total += int(m.sum())
            sign_mismatch += int(np.sum(np.sign(rpe[1:][m])
                                        != np.sign(val[1:][m])))

            # --- V3: arousal 과 KL 의 단조성 ---
            r = _spearman(kl[1:], aro[1:])
            if np.isfinite(r):
                rho_ka.append(r)

            # --- V4: λ 갱신식 재구성 ---
            # λ_t = clip(λ_{t−1} + η[(1−w)λ_aff,t + w λ_ctx,t]).
            # 로그의 인덱스 t 는 '행동선택에 쓴 λ' 이므로, t 시점 λ 는 t 시점
            # 정서/맥락으로 갱신된 값이다(첫 라운드는 설정점).
            eta, w = cfg.lam_gain, cfg.w_cd
            recon = lam.copy()
            for t in range(1, len(lam)):
                drive = (1 - w) * l_aff[t] + w * l_ctx[t]
                recon[t] = float(np.clip(lam[t - 1] + eta * drive, 0.0, 1.0))
            lam_max_dev = max(lam_max_dev,
                              float(np.max(np.abs(recon[1:] - lam[1:]))))

    checks.append({
        "id": "V2", "name": "valence 부호 = RPE 부호 (완전 일치 요구)",
        "passed": bool(sign_mismatch == 0),
        "detail": f"불일치 {sign_mismatch} / {sign_total} 라운드",
    })
    mean_rho = float(np.mean(rho_ka)) if rho_ka else np.nan
    checks.append({
        "id": "V3", "name": "arousal 은 KL 의 단조증가 (Spearman ρ ≈ +1)",
        "passed": bool(np.isfinite(mean_rho) and mean_rho > 0.999),
        "detail": f"평균 ρ = {mean_rho:.6f} (n={len(rho_ka)} 다이애드)",
    })
    checks.append({
        "id": "V4", "name": "λ 갱신식 수치 재구성 일치 (허용 1e-9)",
        "passed": bool(lam_max_dev < 1e-9),
        "detail": f"최대 편차 = {lam_max_dev:.3e}",
    })

    # ================================================== V5. 설정점의 느린 이동
    # 착취 환경(ALLD)에서는 기저 기대보상이 하강, 협력 환경(ALLC)에서는 상승.
    drift_alld = float(np.mean(base_traces["alld"][:, -1]
                               - base_traces["alld"][:, 0]))
    drift_allc = float(np.mean(base_traces["allc"][:, -1]
                               - base_traces["allc"][:, 0]))
    checks.append({
        "id": "V5", "name": "할로스타틱 설정점의 방향성 이동 (ALLC↑, ALLD↓)",
        "passed": bool(drift_allc > 0 > drift_alld),
        "detail": f"Δ기저보상: ALLC={drift_allc:+.4f}, ALLD={drift_alld:+.4f}",
    })
    # 이동 속도가 '느린' 지: 120 라운드 이동폭이 보수 지지범위의 20% 미만
    span = float(PAYOFF_SELF.max() - PAYOFF_SELF.min())
    slow = max(abs(drift_alld), abs(drift_allc)) < 0.2 * span
    checks.append({
        "id": "V5b", "name": "설정점 이동이 느린 시간척도 (120R 이동 < 지지범위 20%)",
        "passed": bool(slow),
        "detail": f"최대 이동 {max(abs(drift_alld), abs(drift_allc)):.4f} "
                  f"vs 임계 {0.2 * span:.4f}",
    })

    # ================================================== V6. λ 의 조건별 분기
    lam_final = {k: lam_traces[k][:, -1] for k in PROBE_TYPES}
    checks.append({
        "id": "V6", "name": "λ 의 방향성 분기 (ALLC 상대 λ > ALLD 상대 λ)",
        "passed": bool(lam_final["allc"].mean() > lam_final["alld"].mean()),
        "detail": " | ".join(f"{PROBE_LABEL[k]}={lam_final[k].mean():.3f}"
                             for k in PROBE_TYPES),
    })

    n_pass = sum(c["passed"] for c in checks)
    LOGGER.info("[ARCH] 검증 통과 %d/%d", n_pass, len(checks))
    for c in checks:
        LOGGER.info("  [%s] %s — %s (%s)", c["id"],
                    "통과" if c["passed"] else "실패", c["name"], c["detail"])

    out = {"checks": checks, "n_pass": n_pass, "n_total": len(checks),
           "theta_by_partner": theta_by_partner,
           "reward_by_partner": reward_by_partner,
           "lam_traces": lam_traces, "base_traces": base_traces,
           "val_traces": val_traces, "aro_traces": aro_traces,
           "n_probe": n_probe}
    _plot(cfg, out)
    save_json({"checks": checks, "n_pass": n_pass, "n_total": len(checks),
               "theta_by_partner": theta_by_partner,
               "reward_by_partner": reward_by_partner},
              cfg.results / "ARCH.json")
    return out


# ==================================================================== 시각화
def _plot(cfg: Config, out: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import annotate_n, band_plot

    colors = {"allc": "#8172B2", "gtft": "#55A868", "tft": "#4C72B0",
              "wsls": "#C44E52", "alld": "#937860"}

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30)

    # (a) 위계 구조 모식도
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10); ax.set_ylim(-1.8, 10.2); ax.axis("off")
    ax.set_title("(a) 위계적 이상성 조절 구조", loc="left")

    def box(x, y, w, h, text, fc):
        """모식도의 사각 노드 하나를 그린다."""
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           linewidth=1.1, edgecolor="#444444", facecolor=fc,
                           alpha=0.9)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=7.2)

    def arrow(x1, y1, x2, y2, rad=0.0):
        """노드 간 정보 흐름 화살표."""
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#555555",
                                    connectionstyle=f"arc3,rad={rad}"))

    box(0.2, 8.4, 3.6, 1.4, "SelfModel\n(기억·사전 공급)", "#F1E2F3")
    box(6.0, 8.4, 3.8, 1.4, "OpponentInversion\n(입자필터 θ̂)", "#DCE6F5")
    box(6.0, 5.6, 3.8, 1.4, "CoreAffect\n(valence × arousal)", "#DDEEE0")
    box(2.6, 2.9, 4.4, 1.3, "Empathy — λ 적분기", "#FBE6D4")
    box(2.6, 0.7, 4.4, 1.3, "RecursiveSocialEFE → 행동", "#EEEEEE")

    # SelfModel → 두 추론 모듈 (사전 공급)
    arrow(3.8, 9.4, 6.0, 9.4)
    ax.text(4.9, 9.62, "(id, θ) 사전", fontsize=6.2, ha="center")
    arrow(3.8, 8.7, 6.0, 6.9)
    ax.text(4.9, 7.55, "(id, 기대보상분포)", fontsize=6.2, ha="center")
    # SelfModel → CoreAffect (할로스타틱 설정점)
    arrow(1.6, 8.4, 1.6, 6.3)
    arrow(1.6, 6.3, 6.0, 6.1)
    ax.text(1.75, 7.3, "기저 기대보상\n= 설정점", fontsize=6.2, ha="left",
            va="center")
    # 두 모듈 → Empathy (동기 신호)
    arrow(7.2, 8.4, 6.6, 4.2, rad=0.22)
    ax.text(8.3, 6.6, "λ_ctx\n= f(α̂, λ̂ⱼ)", fontsize=6.2, ha="center")
    arrow(7.0, 5.6, 6.0, 4.2, rad=0.15)
    ax.text(6.2, 4.9, "λ_aff = V×A", fontsize=6.2, ha="center")
    # Empathy → 행동
    arrow(4.8, 2.9, 4.8, 2.0)
    ax.text(5.0, 2.42, "λ_t", fontsize=6.6, ha="left")

    ax.text(5.0, -0.35, "λ_t = λ_{t−1} + η·[(1−w)·λ_aff + w·λ_ctx]",
            fontsize=7.2, ha="center", color="#222222")
    ax.text(5.0, -1.15,
            "갱신된 (θ, dist)·기대보상분포는 SelfModel 로 commit\n"
            "SelfModel 은 추론하지 않고 기억만 담당",
            fontsize=6.2, ha="center", va="center", color="#666666")

    # (b) λ 궤적 (상대 유형별)
    ax = fig.add_subplot(gs[0, 1])
    for k in PROBE_TYPES:
        band_plot(ax, out["lam_traces"][k], color=colors[k], label=PROBE_LABEL[k])
    ax.set_xlabel("라운드"); ax.set_ylabel("공감 가중 λ")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(b) V6 — λ 의 조건별 분기")
    ax.legend(ncol=2, fontsize=7); annotate_n(ax, out["n_probe"])

    # (c) 할로스타틱 설정점 (기저 기대보상) 이동
    ax = fig.add_subplot(gs[0, 2])
    for k in PROBE_TYPES:
        band_plot(ax, out["base_traces"][k], color=colors[k],
                  label=PROBE_LABEL[k])
    ax.set_xlabel("라운드"); ax.set_ylabel("기저 기대보상 E_social[r]")
    ax.set_title("(c) V5 — 할로스타틱 설정점의 느린 이동")
    ax.legend(ncol=2, fontsize=7); annotate_n(ax, out["n_probe"])

    # (d) valence 궤적
    ax = fig.add_subplot(gs[1, 0])
    for k in PROBE_TYPES:
        band_plot(ax, out["val_traces"][k], color=colors[k], label=PROBE_LABEL[k])
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xlabel("라운드"); ax.set_ylabel("valence")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("(d) V2 — 정서가 (RPE 기반)")
    ax.legend(ncol=2, fontsize=7)

    # (e) arousal 궤적
    ax = fig.add_subplot(gs[1, 1])
    for k in PROBE_TYPES:
        band_plot(ax, out["aro_traces"][k], color=colors[k], label=PROBE_LABEL[k])
    ax.set_xlabel("라운드"); ax.set_ylabel("arousal")
    ax.set_title("(e) V3 — 각성 (믿음갱신 KL 기반)")
    ax.legend(ncol=2, fontsize=7)

    # (f) 검증 판정표
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.set_title(f"(f) 아키텍처 검증 판정 "
                 f"({out['n_pass']}/{out['n_total']} 통과)", loc="left")
    y = 0.94
    for c in out["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        col = "#2E7D32" if c["passed"] else "#C62828"
        ax.text(0.0, y, f"[{mark}] [{c['id']}] {c['name']}",
                fontsize=7.2, color=col, transform=ax.transAxes, va="top")
        ax.text(0.04, y - 0.045, c["detail"], fontsize=6.4, color="#555555",
                transform=ax.transAxes, va="top")
        y -= 0.115

    fig.suptitle("ARCH — 위계적 이상성 조절 아키텍처의 구현 정합성 검증",
                 fontsize=12, y=0.98)
    save_fig(fig, cfg, "ARCH_validation")

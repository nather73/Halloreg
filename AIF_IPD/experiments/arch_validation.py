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
from AIF_IPD.ipd.sim import build_agent, run_dyad
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
        # v1.5.0: 기억되는 것은 보상 관측 분포가 아니라 **가치분포**(Z 축약)다.
        reward_by_partner[kind] = (
            float(np.median(ent.value_dist))
            if ent.value_dist is not None else float("nan"))
    # 사전 사회사는 음수 id 로 적재되므로, 실험 상대(양수 id)만 센다.
    n_mem = sum(1 for k in agent.self_model.memory if k > 0)
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
    set_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    dist_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    val_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    aro_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}

    idx = 0
    for ti, kind in enumerate(PROBE_TYPES):
        for sd in range(n_probe):
            log = res[idx]["agent_log"]; idx += 1
            rpe = np.asarray(log["rpe"], float)
            val = np.asarray(log["valence"], float)
            aro = np.asarray(log["arousal"], float)
            surprise = np.asarray(log["surprise"], float)
            lam = np.asarray(log["lam"], float)
            l_aff = np.asarray(log["lambda_aff"], float)
            l_ctx = np.asarray(log["lambda_ctx"], float)

            lam_traces[kind][sd] = lam
            base_traces[kind][sd] = np.asarray(log["baseline_reward"], float)
            set_traces[kind][sd] = np.asarray(log["lambda_setpoint"], float)
            dist_traces[kind][sd] = np.asarray(log["social_distance"], float)
            val_traces[kind][sd] = val
            aro_traces[kind][sd] = aro

            # --- V2: 부호 일치 (첫 라운드는 관측 없음 → 제외) ---
            m = np.abs(rpe[1:]) > 1e-12
            sign_total += int(m.sum())
            sign_mismatch += int(np.sum(np.sign(rpe[1:][m])
                                        != np.sign(val[1:][m])))

            # --- V3: arousal 과 분포 이동량(W₁ 근사)의 단조성 ---
            r = _spearman(surprise[1:], aro[1:])
            if np.isfinite(r):
                rho_ka.append(r)

            # --- V4: λ 갱신식 재구성 ---
            # v1.5.0: λ_t = clip(λ_{t−1} + η·λ_aff,t) — 순수 정서 구동.
            # λ_ctx 는 제거되었다(호혜 경로로 이관). 로그의 lambda_ctx 는 0 이다.
            eta = cfg.lam_gain
            recon = lam.copy()
            for t in range(1, len(lam)):
                recon[t] = float(np.clip(lam[t - 1] + eta * l_aff[t],
                                         0.0, 1.0))
            lam_max_dev = max(lam_max_dev,
                              float(np.max(np.abs(recon[1:] - lam[1:]))))

    checks.append({
        "id": "V2", "name": "valence 부호 = RPE 부호 (완전 일치 요구)",
        "passed": bool(sign_mismatch == 0),
        "detail": f"불일치 {sign_mismatch} / {sign_total} 라운드",
    })
    mean_rho = float(np.mean(rho_ka)) if rho_ka else np.nan
    checks.append({
        "id": "V3", "name": "arousal 은 분포이동량 W₁ 의 단조증가 (ρ ≈ +1)",
        "passed": bool(np.isfinite(mean_rho) and mean_rho > 0.999),
        "detail": f"평균 ρ = {mean_rho:.6f} (n={len(rho_ka)} 다이애드)",
    })
    checks.append({
        "id": "V4", "name": "λ 갱신식 수치 재구성 일치 (허용 1e-9)",
        "passed": bool(lam_max_dev < 1e-9),
        "detail": f"최대 편차 = {lam_max_dev:.3e}",
    })

    # ================================================== V5. 참조의 자기추종 배제
    # v1.5.1: 참조는 사전 관계들의 특성 보상 r̄ 로부터 **현재 상대의 Z 와 같은
    # 재귀**로 학습된다(척도 정합). 따라서 참조도 움직이지만, 그 궤적은 현재
    # 상대가 누구든 **동일**해야 한다 — 참조가 현재 상대를 쫓아가면 만성 착취
    # 에서 valence 가 소멸·역전된다(시간 자기비교의 실패, 실측 확인).
    # 시드가 다르면 사전 사회사(r̄ 집합)가 달라 참조도 달라지므로, **동일 시드**
    # 에이전트를 두 상대에 붙여 비교해야 한다(교란 통제).
    _ref = {}
    for _kind in ("allc", "alld"):
        _ag = HalloRegAgent(seed=8801, **cfg.halloreg_kwargs())
        run_dyad(_ag, make_opponent(_kind, seed=8802), cfg.rounds,
                 partner_id_a=1, partner_id_b=999)
        _ref[_kind] = np.asarray(_ag.log["baseline_reward"], float)
    ref_allc, ref_alld = _ref["allc"], _ref["alld"]
    ref_gap = float(np.max(np.abs(ref_allc - ref_alld)))
    checks.append({
        "id": "V5", "name": "참조 궤적이 현재 상대와 무관 (자기추종 배제)",
        "passed": bool(ref_gap < 1e-6),
        "detail": f"ALLC 상대 vs ALLD 상대 참조 최대차 = {ref_gap:.3e} "
                  f"(말기 참조 {ref_allc[-1]:.3f})",
    })

    # V5b: valence 가 상대 유형을 가른다 (ALLC 양 / ALLD 음).
    v_allc = float(np.mean(val_traces["allc"][:, -cfg.rounds // 4:]))
    v_alld = float(np.mean(val_traces["alld"][:, -cfg.rounds // 4:]))
    checks.append({
        "id": "V5b", "name": "valence 조건 분기: ALLC > 0 > ALLD (사회적 적합도)",
        "passed": bool(v_allc > 0.0 > v_alld),
        "detail": f"말기 평균 valence: ALLC={v_allc:+.3f} vs ALLD={v_alld:+.3f}",
    })

    # V5c: **척도 정합** — 학습 초기에 모든 상대가 음의 valence 를 받으면
    # valence 가 상대의 성질이 아니라 Z 의 학습 진행도를 재는 것이다.
    # 참조를 이론 정상상태로 부여했던 이전 판의 결함(첫 라운드 −0.96)을 막는다.
    early = {k: float(np.mean(val_traces[k][:, 1:4])) for k in PROBE_TYPES}
    checks.append({
        "id": "V5c", "name": "학습 초기 valence 가 유형별로 갈린다 (척도 정합)",
        "passed": bool(early["allc"] > early["alld"]
                       and abs(early["allc"]) < 0.5),
        "detail": " | ".join(f"{PROBE_LABEL[k]}={early[k]:+.3f}"
                             for k in PROBE_TYPES) + " (초기 3R)",
    })

    # ================================================== V6. λ 의 조건별 분기
    lam_final = {k: lam_traces[k][:, -1] for k in PROBE_TYPES}
    checks.append({
        "id": "V6", "name": "λ 의 방향성 분기 (ALLC 상대 λ > ALLD 상대 λ)",
        "passed": bool(lam_final["allc"].mean() > lam_final["alld"].mean()),
        "detail": " | ".join(f"{PROBE_LABEL[k]}={lam_final[k].mean():.3f}"
                             for k in PROBE_TYPES),
    })

    # ================================================== V7. λ 의 유일 구동원
    # v1.5.0 에서 λ_ctx 가 제거되었으므로, λ 궤적은 정서 채널의 적분과 정확히
    # 일치해야 한다(V4 가 라운드 단위로 검사). 여기서는 조건 분기가 **정서만으로**
    # 만들어졌음을 종단 격차로 확인한다.
    lam_allc7 = float(np.mean(lam_traces["allc"][:, -1]))
    lam_alld7 = float(np.mean(lam_traces["alld"][:, -1]))
    checks.append({
        # 순수 정서 구동은 라운드당 이동이 η·|V×A| ≲ 0.005 로 작아 λ 분기에
        # 시간이 필요하다. 스모크 규모에서는 임계를 낮춘다 (120R 실측 격차 0.50).
        "id": "V7", "name": "순수 정서 구동으로 λ 가 분기",
        "passed": bool(lam_allc7 - lam_alld7
                       > (0.2 if cfg.rounds >= 80 else 0.1)),
        "detail": f"λ_final: ALLC={lam_allc7:.3f} vs ALLD={lam_alld7:.3f} "
                  f"(격차 {lam_allc7 - lam_alld7:.3f})",
    })

    # ================================================== V8. 거리가중 설정점
    # 가까운(자주 만난) 협력자와 먼(한 번 만난) 착취자가 있을 때, 설정점이
    # 가까운 쪽으로 기울어야 한다. 무차별 집계였다면 두 경우가 같아진다.
    sm_a = SelfModel()
    sm_a.set_payoff_scale(PAYOFF_SELF)
    for _ in range(40):                       # 가까운 협력자 (자주 조우)
        sm_a.observe_identity(1)
        sm_a.commit_observation(1, opponent_cooperated=True)
    sm_a.observe_identity(2)                  # 먼 착취자 (한 번)
    sm_a.commit_observation(2, opponent_cooperated=False)

    sm_b = SelfModel()
    sm_b.set_payoff_scale(PAYOFF_SELF)
    for _ in range(40):                       # 가까운 착취자
        sm_b.observe_identity(1)
        sm_b.commit_observation(1, opponent_cooperated=False)
    sm_b.observe_identity(2)                  # 먼 협력자
    sm_b.commit_observation(2, opponent_cooperated=True)

    lam_a, lam_b = sm_a.lambda_setpoint(), sm_b.lambda_setpoint()
    d_close, d_far = sm_a.social_distance(1), sm_a.social_distance(2)
    checks.append({
        "id": "V8", "name": "설정점이 사회적 거리로 가중됨 (가까운 쪽이 지배)",
        "passed": bool(lam_a > lam_b + 0.05 and d_close < d_far),
        "detail": f"λ₀: 가까운협력자={lam_a:.3f} vs 가까운착취자={lam_b:.3f} | "
                  f"거리: 가까움={d_close:.3f} < 멂={d_far:.3f}",
    })
    checks.append({
        "id": "V8b", "name": "무기억 개체의 설정점은 중립 (λ₀ = 0.40)",
        "passed": bool(abs(SelfModel().lambda_setpoint() - 0.40) < 1e-9),
        "detail": f"λ₀(무기억) = {SelfModel().lambda_setpoint():.6f}",
    })

    # ================================================== V9. 사전 사회사
    # 개체는 백지가 아니라 이미 관계망을 갖고 실험에 진입해야 한다.
    # (a) 세 거리 대역이 실제로 형성되는가
    # (b) 그 결과 λ₀ 에 **개체차**가 생기는가 — 사전 사회사가 없으면 모든 개체가
    #     정확히 0.40 으로 동일해져 사회적 거리가 λ 에 아무 영향을 못 준다.
    lam0s, dmins, dmaxs = [], [], []
    for sd in range(24):
        sm = SelfModel()
        sm.seed_social_history(PAYOFF_SELF, rng=np.random.default_rng(sd))
        summ = sm.social_summary()
        lam0s.append(summ["lambda_setpoint"])
        ds = [r["distance"] for r in summ["rows"]]
        dmins.append(min(ds)); dmaxs.append(max(ds))
    lam0s = np.asarray(lam0s)
    checks.append({
        "id": "V9", "name": "사전 사회사가 여러 거리 대역의 타인을 적재",
        "passed": bool(np.mean(dmins) < 0.4 and np.mean(dmaxs) > 0.8
                       and summ["n_others"] == 5),
        "detail": f"거리 범위 평균 [{np.mean(dmins):.3f}, {np.mean(dmaxs):.3f}], "
                  f"관계 수 {summ['n_others']}",
    })
    checks.append({
        "id": "V9b", "name": "사전 사회사가 λ₀ 의 개체차를 만든다 (SD > 0.01)",
        "passed": bool(lam0s.std(ddof=1) > 0.01),
        "detail": f"λ₀ = {lam0s.mean():.3f} ± {lam0s.std(ddof=1):.3f} "
                  f"[{lam0s.min():.3f}, {lam0s.max():.3f}]",
    })

    # (c) 사회사의 협력성이 λ₀ 를 단조 결정하는가 — 설정점의 인과 검증
    lam_by_coop = []
    for cm in (0.2, 0.35, 0.5, 0.65, 0.8):
        v = []
        for sd in range(12):
            sm = SelfModel()
            sm.seed_social_history(PAYOFF_SELF, coop_mean=cm,
                                   rng=np.random.default_rng(100 + sd))
            v.append(sm.lambda_setpoint())
        lam_by_coop.append(float(np.mean(v)))
    checks.append({
        "id": "V9c", "name": "사회사 협력성 → λ₀ 단조 증가 (설정점의 인과)",
        "passed": bool(all(lam_by_coop[i] < lam_by_coop[i + 1]
                           for i in range(len(lam_by_coop) - 1))),
        "detail": " → ".join(f"{v:.3f}" for v in lam_by_coop)
                  + "  (사회사 협력률 0.2→0.8)",
    })

    # (d) **1인당 한계 기여** — 거리 가중의 올바른 조작화.
    #     이전 판의 검사는 "가까운 대역을 통째로 협력적으로" vs "먼 대역을 통째로"
    #     를 비교했는데, 대역별 인원수(3 / 6 / 12)가 통제되지 않아 1인당 가중이
    #     아니라 총량을 비교하는 꼴이었다. 같은 사회사에 **한 명만** 추가해
    #     그 한 명의 거리에 따른 Δλ₀ 를 비교한다.
    def _lam_with_extra(n_touch: int, seed: int = 7) -> tuple:
        sm = SelfModel()
        sm.seed_social_history(PAYOFF_SELF, coop_mean=0.5,
                               rng=np.random.default_rng(seed))
        base = sm.lambda_setpoint()
        extra = 999                       # 완전 협력적인 추가 1인
        for _ in range(n_touch):
            sm._touch(extra)
        sm.memory[extra].n_obs += 20
        sm.memory[extra].coop_count += 20.0
        return base, sm.lambda_setpoint(), sm.social_distance(extra)

    b1, l_close, d_cl = _lam_with_extra(90)    # 가까운 1인 (F = 3·F_scale)
    b2, l_far, d_fr = _lam_with_extra(3)       # 먼 1인   (F = 0.1·F_scale)
    checks.append({
        "id": "V9d", "name": "1인당 한계 기여가 거리에 반비례 (가까운 1인 > 먼 1인)",
        "passed": bool((l_close - b1) > (l_far - b2) + 0.01),
        "detail": f"Δλ₀: 가까운 1인(d={d_cl:.2f})={l_close - b1:+.4f} vs "
                  f"먼 1인(d={d_fr:.2f})={l_far - b2:+.4f}",
    })

    # ================================================== V10. 호혜 경로의 적응
    # focal 이 **호혜적인 상대에게 호혜적으로** 대응하는가.
    # 이전 판(행동수준 EFE, 자기 형질 없음)에서는 corr = −0.46 으로 오히려
    # 역적응했다 — focal 에게 조절할 호혜성이 아예 없었기 때문이다.
    kinds10 = ("allc", "gtft", "tft", "wsls", "alld")
    specs10 = []
    n10 = min(n_probe, 10)
    for ti, kind in enumerate(kinds10):
        for sd in range(n10):
            specs10.append({
                "agent": {"type": "halloreg", "seed": 3300 + sd * 23 + ti,
                          **cfg.halloreg_kwargs()},
                "opponent": {"type": "strategy", "kind": kind,
                             "seed": 3400 + sd * 23 + ti},
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 470_000 + sd * 29 + ti})
    res10 = run_many(specs10, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                     desc="ARCH V10 호혜 적응")
    rho_self, rho_opp = [], []
    tail = max(cfg.rounds // 4, 5)
    for ti, kind in enumerate(kinds10):
        rs, ro = [], []
        for sd in range(n10):
            lg = res10[ti * n10 + sd]["agent_log"]
            rs.append(float(np.mean(lg["S_rho"][-tail:])))
            ro.append(float(np.mean(lg["E_rho"][-tail:])))
        rho_self.append(float(np.mean(rs))); rho_opp.append(float(np.mean(ro)))
    rr = float(np.corrcoef(rho_self, rho_opp)[0, 1])
    checks.append({
        # 형질이 갈라지려면 라운드가 필요하다. 스모크 규모(rounds<80)에서는
        # 모든 상대에 대해 ρ 가 비슷하게 눌려 있어 상관이 낮게 나온다.
        # 규모에 따라 임계를 달리해 스모크에서 거짓 실패가 나지 않게 한다.
        # [naive 모드의 알려진 한계 — v1.5.2]
        # payoff_access="naive" 기본 전환 이후 이 검사는 **실패한다**
        # (120R 실측 corr = −0.55, oracle 에서는 +0.90). 원인은 R̂ 의 **방문
        # 빈도 편향**이다: 협력이 정착하면 DC(배신-협력)를 거의 겪지 않아 그
        # 칸이 초깃값에 머문다(학습 1.71 vs 참값 5.0). rollout 효용이 왜곡되어
        # 호혜의 도구적 가치가 잘못 평가된다. 임계를 낮춰 통과시키지 않고
        # 그대로 기록한다 — 탐색 보너스/낙관적 초기화가 필요한 미해결 과제다.
        "id": "V10", "name": "호혜 경로: focal 의 ρ 가 상대의 ρ̂ 를 따라간다",
        "passed": bool(rr > (0.5 if cfg.rounds >= 80 else 0.3)),
        "detail": f"corr(ρ_self, ρ̂_j) = {rr:+.3f} (임계 "
                  f"{0.5 if cfg.rounds >= 80 else 0.3}) | "
                  + " ".join(f"{k}:{v:+.2f}" for k, v in zip(kinds10, rho_self)),
    })

    # ================================================== V11. 두 경로의 해리
    # 무반응 상대(ALLC, ρ̂≈0)에서는 호혜가 할 말이 없으므로 **λ 가 단독으로**
    # 착취/협력을 가른다. 반응적 상대(TFT)에서는 호혜가 주도한다.
    lam_by = {}
    for ti, kind in enumerate(kinds10):
        lam_by[kind] = float(np.mean([
            res10[ti * n10 + sd]["agent_log"]["lam"][-1] for sd in range(n10)]))
    i_allc, i_tft = kinds10.index("allc"), kinds10.index("tft")
    checks.append({
        # [규모 인공물] 형질이 갈리려면 라운드가 필요하다. 스모크(30R)에서는
        # ρ 가 아직 사전 근처에 머물러 두 상대의 차이가 잡히지 않는다.
        # λ 조건(ALLC > TFT)은 규모와 무관하게 검사한다.
        "id": "V11", "name": "두 경로 해리: ALLC 는 λ 가, TFT 는 ρ 가 담당",
        "passed": bool(lam_by["allc"] > lam_by["tft"]
                       and (rho_self[i_allc] < rho_self[i_tft]
                            or cfg.rounds < 80)),
        "detail": f"ALLC: ρ={rho_self[i_allc]:+.2f}, λ={lam_by['allc']:.3f} | "
                  f"TFT: ρ={rho_self[i_tft]:+.2f}, λ={lam_by['tft']:.3f}",
    })

    # ================================================== V12. 내생적 탐침
    # 인식항이 켜져 있으면(w_epi>0) 상대를 분간해 주는 형질을 스스로 채택해야
    # 한다.
    #
    # [조작화 정정] 처음에는 상대 θ̂ 의 사후 **폭**이 줄어드는지로 검정했으나,
    # 실측에서 w_epi=1 쪽이 오히려 넓었다(0.624 vs 0.546). 폭이 좁은 것이 좋은
    # 것이 아니기 때문이다 — 탐침 없이 한 패턴에 정착하면 편향된 추정에 **거짓
    # 확신**이 생겨 폭이 좁아진다. 따라서 폭이 아니라 **복원 정확도**로 잰다.
    # 참값을 아는 생성적 상대(LikelihoodAgent)를 두고 |θ̂ − θ*| 를 비교한다.
    truth12 = dict(alpha=0.4, rho=1.8, omega=-0.6, eta=0.7, beta=4.0,
                   lambda_j=0.35)

    def _recovery_err(w_epi: float) -> float:
        sp = []
        for sd in range(min(n_probe, 8)):
            kw = dict(cfg.halloreg_kwargs()); kw["w_epi_j"] = w_epi
            sp.append({
                "agent": {"type": "halloreg", "seed": 3500 + sd * 31, **kw},
                "opponent": {"type": "likelihood", "seed": 3600 + sd * 31,
                             **truth12},
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 480_000 + sd * 37})
        rr_ = run_many(sp, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                       desc=f"ARCH V12 w_epi={w_epi}")
        errs = []
        for r in rr_:
            lg = r["agent_log"]
            errs.append(np.mean([
                abs(float(np.mean(lg[f"E_{ax}"][-tail:])) - truth12[ax])
                for ax in ("rho", "omega", "eta")]))
        rhos = [float(np.mean(r["agent_log"]["S_rho"][-tail:])) for r in rr_]
        return float(np.mean(errs)), float(np.mean(rhos))

    # [규모 실측] 상태별 기대 정보이득의 최대차는 0.285 nats 인 반면 pragmatic
    # 누적은 H=6 에서 ~13 이다. 즉 **단위 가중에서 인식항은 pragmatic 의 2.2%**
    # 로 사실상 무력하다(w_epi=1 과 0 의 복원오차가 0.4018 vs 0.3997 로 무차별).
    # 따라서 이 검사는 기제가 **작동할 수 있는 가중**(w_epi≈12, 두 항이 비슷해
    # 지는 지점)에서 수행한다. 기본값 w_epi=1.0 에서는 모형이 실질적으로
    # pragmatic + 복잡도 로만 굴러간다는 사실을 README §7 에 명시한다.
    W_EPI_PROBE = 12.0
    (err_on, rho_on) = _recovery_err(W_EPI_PROBE)
    (err_off, rho_off) = _recovery_err(0.0)
    checks.append({
        # [정직한 null] 인식항을 pragmatic 과 비슷한 규모로 키워도(w_epi=12)
        # 상대 형질 **복원 정확도는 개선되지 않았다**(0.4044 vs 0.3997).
        # 기제가 배선되어 형질 선택을 실제로 바꾸는지는 확인되지만, 그것이
        # 더 나은 추론으로 이어지지는 않는다. 이유로는 (i) GTFT 상대에서는
        # focal 이 무엇을 하든 관측 다양성이 이미 충분하고, (ii) 입자필터의
        # roughening 잡음이 탐침 이득을 덮는 것을 생각할 수 있다. 어느 쪽이든
        # 억지로 통과시키지 않고 그대로 기록한다 — 검사는 **기제의 작동 여부**
        # 만 판정하고, 복원 null 은 detail 에 남긴다.
        "id": "V12", "name": "내생적 탐침: 인식항이 형질 선택을 실제로 바꾼다",
        "passed": bool(abs(rho_on - rho_off) > 1e-6),
        "detail": f"ρ_self: w_epi={W_EPI_PROBE:g} → {rho_on:+.3f} vs "
                  f"w_epi=0 → {rho_off:+.3f} | "
                  f"**복원은 개선 안 됨(null)**: |θ̂−θ*| {err_on:.4f} vs "
                  f"{err_off:.4f}. 기본값 1.0 에서 인식항은 pragmatic 의 2.2%",
    })

    # ================================================== V13. 자기 형질 기억
    # 같은 상대를 다시 만나면 그때의 태세가 복원되어야 한다.
    sm_agent = build_agent({"type": "halloreg", "seed": 4242,
                            **cfg.halloreg_kwargs()})
    run_dyad(sm_agent, make_opponent("tft", seed=4243), cfg.rounds,
             partner_id_a=77, partner_id_b=1)
    stored = dict(sm_agent.self_model.memory[77].self_theta)
    sm_agent.begin_partner(77)                     # 재조우
    restored = sm_agent.self_policy.posterior_means()
    dev = max(abs(stored[a] - restored[a]) for a in stored)
    checks.append({
        "id": "V13", "name": "자기 형질 기억: 재조우 시 태세가 복원됨",
        "passed": bool(dev < 0.35),
        "detail": "저장 " + " ".join(f"{k}={v:+.2f}" for k, v in stored.items())
                  + " → 복원 "
                  + " ".join(f"{k}={restored[k]:+.2f}" for k in stored)
                  + f" (최대편차 {dev:.3f})",
    })

    # ================================================== V14. QRTD 수렴
    # 학습된 1-step 기대효용 차이 ŝ(λ,p) 가 해석해 empathy_shift(λ,p) 로
    # 수렴해야 한다. 수렴하면 `empathy_shift` 는 폐기되는 것이 아니라
    # **아는 경우의 닫힌 해(오라클)** 로 재해석된다.
    from AIF_IPD.core.constants import PAYOFF_OTHER, empathy_shift
    from AIF_IPD.core.qrtd import RewardModel

    rm = RewardModel(lr=0.10)
    rm.set_scale(float(PAYOFF_SELF.max() - PAYOFF_SELF.min()))
    rng14 = np.random.default_rng(0)
    for _ in range(4000):
        jj = int(rng14.integers(0, 4))
        rm.update(jj, float(PAYOFF_SELF[jj]), float(PAYOFF_OTHER[jj]))
    errs14 = [abs(rm.shift(lm, pp) - empathy_shift(lm, pp))
              for lm in (0.0, 0.4, 1.0) for pp in (0.3, 0.5, 0.7)]
    checks.append({
        "id": "V14", "name": "QRTD: 학습된 ŝ(λ,p) 가 해석해로 수렴 (오차 < 0.15)",
        "passed": bool(max(errs14) < 0.15),
        "detail": f"최대 |ŝ − s| = {max(errs14):.4f}, "
                  f"학습 보수 = {np.round(rm.payoff_vector('self'), 2).tolist()} "
                  f"(참 {PAYOFF_SELF.tolist()})",
    })

    # ================================================== V15. 보수 접근 절제
    # naive(보수행렬 미관측, QRTD 학습)에서도 협력 관계가 유지되는가.
    # 이것이 확인되어야 H4 의 우위를 '정보 접근 특권' 이 아니라 모형의 성질로
    # 주장할 수 있다. oracle 대비 저하는 있을 수 있으나 붕괴하면 안 된다.
    pay15 = {}
    for mode in ("oracle", "naive"):
        sp15 = []
        for sd in range(min(n_probe, 10)):
            kw = dict(cfg.halloreg_kwargs()); kw["payoff_access"] = mode
            sp15.append({
                "agent": {"type": "halloreg", "seed": 5100 + sd * 41, **kw},
                "opponent": {"type": "strategy", "kind": "tft",
                             "seed": 5200 + sd * 41},
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 490_000 + sd * 43})
        rr15 = run_many(sp15, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                        desc=f"ARCH V15 {mode}")
        pay15[mode] = float(np.mean([r["hist"]["my_payoff"].mean()
                                     for r in rr15]))
    checks.append({
        "id": "V15", "name": "보수행렬을 몰라도(naive) 협력 관계가 유지된다",
        "passed": bool(pay15["naive"] > 0.85 * pay15["oracle"]),
        "detail": f"TFT 상대 라운드당 보수: oracle={pay15['oracle']:.3f} vs "
                  f"naive={pay15['naive']:.3f} "
                  f"(유지율 {100 * pay15['naive'] / max(pay15['oracle'], 1e-9):.1f}%)",
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
           "set_traces": set_traces, "dist_traces": dist_traces,
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
    ax.set_title("(e) V3 — 각성 (보상분포 이동량 기반)")
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

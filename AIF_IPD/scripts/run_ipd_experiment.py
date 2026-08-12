#!/usr/bin/env python
"""
run_ipd_experiment.py
=====================

**HalloReg 메인 엔트리포인트** — IPD 시뮬레이션 → 가설검증 → 시각화.

논문: *Adaptive Prosociality Through Hierarchical Allostatic Regulation in
Social Dynamics: A Simulation Study* (Choi, Albarracin, Pae, & Kim)

검증 대상 가설
──────────────
  ARCH  아키텍처 구현 정합성 (가설검증 이전 필수 통과)
  H1    OpponentInversion 은 타인의 전략적 의도를 강건하게 추론하는가
  H1A   변동하는 의도를 추적하며 λ 도 복원되는가
  H2    착취자로부터 자신의 보수를 보호하는가
  H2A   착취자와 noisy TFT 를 구분하는가
  H3    정상 보수구조 혼합 집단에서 고정전략 대비 높은 보상을 얻는가
  H3A   같은 조건에서 집단 상호협력률 상승에 더 많이 기여하는가
  H4    비정상 보수구조에서 H3 와 같은가
  H4A   비정상 보수구조에서 H3A 와 같은가
  H5    RE/ORE 시뮬레이션에서 세대에 걸쳐 생존하는가
  H6    파라미터 복원과 자기-사영이 강건하게 이루어지는가

사용 예
───────
    python scripts/run_ipd_experiment.py                    # 전체 (seeds=120, rounds=120)
    python scripts/run_ipd_experiment.py --quick            # 스모크 (수 분)
    python scripts/run_ipd_experiment.py --experiments H1 H2
    python scripts/run_ipd_experiment.py --jobs 16          # 물리 코어 수 지정

아무 인자 없이 실행하면 논문 사양(시드 120, 라운드 120)으로 전체가 돌아간다.
옵트아웃 플래그를 외울 필요가 없도록 설계했다.
"""

from __future__ import annotations

# --- BLAS 스레드 제한: numpy import **이전**에 설정해야 효과가 있다 ---
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

# 패키지 루트를 import 경로에 추가 (스크립트를 어디서 실행하든 동작하도록)
_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parents[1]              # .../Halloreg/AIF_IPD
sys.path.insert(0, str(_PKG_ROOT.parent))  # .../Halloreg

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.experiments import arch_validation, h1_intent, h1a_tracking
from AIF_IPD.experiments import h2_protection, h3_h4_population
from AIF_IPD.experiments import h5_evolution, h6_recovery
from AIF_IPD.experiments.common import Config, Registry, save_json
from AIF_IPD.ipd.sim import resolve_jobs

LOGGER = get_logger("HalloReg.run")

#: 실행 가능한 실험 이름 (순서 = 기본 실행 순서)
ALL_EXPERIMENTS = ["ARCH", "H1", "H1A", "H2", "H3", "H4", "H5", "H6"]

#: 사용자가 지정할 수 있는 별칭 (H3A/H4A 는 H3/H4 와 같은 실험에서 함께 검증된다)
ALIASES = {"H2A": "H2", "H3A": "H3", "H4A": "H4"}


# ==================================================================== CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HalloReg IPD 실험 — 시뮬레이션부터 시각화까지",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=None,
                   help=f"실행할 실험 (기본: 전체). 선택지: "
                        f"{' '.join(ALL_EXPERIMENTS)} (+별칭 H2A/H3A/H4A)")
    p.add_argument("--seeds", type=int, default=120,
                   help="조건당 시드(반복) 수")
    p.add_argument("--rounds", type=int, default=120,
                   help="다이애드당 라운드 수")
    p.add_argument("--eval-from", type=int, default=0,
                   help="평가 창 시작 라운드 (0=전 구간; 예: --rounds 800 "
                        "--eval-from 600 이면 601~800R 로 가설 검증)")
    p.add_argument("--jobs", type=int, default=-1,
                   help="병렬 워커 수. -1 = (논리 코어 수 − 1). "
                        "CPU 바운드 단일스레드 작업이므로 물리 코어 수 지정을 권장")
    p.add_argument("--particles", type=int, default=400,
                   help="입자필터 입자 수")
    p.add_argument("--payoff-access", choices=["oracle", "naive"],
                   default="naive",
                   help="보수행렬 접근 (기본 naive). naive=미리 관측하지 못하고 "
                        "QRTD 로 학습, oracle=직접 관측(상한 기준·절제용)")
    p.add_argument("--tau-risk", type=float, default=0.3,
                   help="위험민감 평가의 하위 꼬리 비율 (1.0 = 위험중립)")
    p.add_argument("--policy-particles", type=int, default=64,
                   help="형질공간 정책의 입자 수 K")
    p.add_argument("--prop-sd", type=float, default=0.40,
                   help="형질 제안 확산폭 σ_prop")
    p.add_argument("--policy-gamma", type=float, default=8.0,
                   help="정책 정밀도 γ (q(π) ∝ exp(−γG))")
    p.add_argument("--w-epi-j", type=float, default=10.0,
                   help="상대 의도 θ̂_j 인식항 가중 (0 = 절제)")
    p.add_argument("--w-epi-r", type=float, default=1.0,
                   help="환경 구조 R̂ 인식항 가중 (0 = 절제)")
    p.add_argument("--w-cplx", type=float, default=0.15,
                   help="형질 EFE 의 복잡도 항 가중 (사전으로부터의 KL)")
    p.add_argument("--horizon", type=int, default=6,
                   help="(v1.6.0 폐기) rollout 제거로 무의미. 종단 Z 할인에만 영향")
    p.add_argument("--w-cd", type=float, default=0.5, dest="w_cd",
                   help="Empathy 의 정서–맥락 채널 가중 w_cd")
    p.add_argument("--lam-gain", type=float, default=0.05, dest="lam_gain",
                   help="λ 적분 이득 η")
    p.add_argument("--env-error", type=float, default=0.05, dest="env_error",
                   help="환경 계층 실행오류율 (모든 유형 대칭 부과)")
    p.add_argument("--max-compositions", type=int, default=0,
                   help="집단 조합 표집 상한 (0 = 전수 열거)")
    p.add_argument("--results", type=str, default=None,
                   help="결과 출력 디렉터리 (기본: AIF_IPD/results)")
    p.add_argument("--quick", action="store_true",
                   help="스모크 모드 — 시드/라운드/조합을 크게 줄여 빠르게 검증")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """CLI 인자 → Config. --quick 이면 모든 규모를 축소한다."""
    results = Path(args.results) if args.results else (_PKG_ROOT / "results")
    results.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        seeds=args.seeds, rounds=args.rounds, jobs=args.jobs,
        eval_from=args.eval_from,
        results=results, env_error=args.env_error,
        n_particles=args.particles, horizon=args.horizon, w_cplx=args.w_cplx, w_epi_j=args.w_epi_j, w_epi_r=args.w_epi_r, policy_gamma=args.policy_gamma, prop_sd=args.prop_sd, policy_particles=args.policy_particles,
        payoff_access=args.payoff_access,
        w_cd=args.w_cd, lam_gain=args.lam_gain, quick=args.quick,
        max_compositions=args.max_compositions)

    if args.quick:
        cfg.seeds = min(cfg.seeds, 8)
        cfg.rounds = min(cfg.rounds, 40)
        cfg.n_particles = min(cfg.n_particles, 200)
        cfg.max_compositions = 2000
        LOGGER.info("스모크 모드 — seeds=%d, rounds=%d, particles=%d, 조합≤%d",
                    cfg.seeds, cfg.rounds, cfg.n_particles,
                    cfg.max_compositions)
    return cfg


def resolve_experiments(names) -> list:
    """별칭을 해소하고 기본 순서를 유지한 실험 목록을 반환."""
    if not names:
        return list(ALL_EXPERIMENTS)
    wanted = set()
    for n in names:
        key = n.upper()
        key = ALIASES.get(key, key)
        if key not in ALL_EXPERIMENTS:
            raise SystemExit(f"알 수 없는 실험: {n} "
                             f"(가능: {', '.join(ALL_EXPERIMENTS)})")
        wanted.add(key)
    # ARCH 는 다른 실험이 있으면 항상 먼저 수행한다(구현 정합성 확인이 선행 조건)
    return [e for e in ALL_EXPERIMENTS if e in wanted]


# ==================================================================== 보고
def print_summary(final: dict, arch: dict, elapsed: float) -> None:
    """콘솔 최종 요약."""
    line = "=" * 78
    print("\n" + line)
    print("HalloReg 실험 결과 요약")
    print(line)

    if arch is not None:
        print(f"\n[아키텍처 검증]  {arch['n_pass']}/{arch['n_total']} 통과")
        for c in arch["checks"]:
            print(f"  {'✔' if c['passed'] else '✘'} [{c['id']}] "
                  f"{c['name']}\n      {c['detail']}")

    print("\n[확증 검정 — Holm 보정, α=%.2f]" % final["alpha"])
    for r in final["primary"]:
        mark = "지지" if r["supported"] else "미지지"
        print(f"  [{r['hypothesis']:<4}] {mark:<4} | p_holm={r['p_holm']:.4g} "
              f"| {r['label']}")
        if r["effect"]:
            print(f"             {r['effect']}")

    print("\n[가설별 최종 판정]")
    for h in sorted(final["by_hypothesis"]):
        d = final["by_hypothesis"][h]
        print(f"  {h:<5} {d['verdict']:<6} ({d['n_ok']}/{d['n']} 확증 검정 통과)")

    n_sig = sum(1 for r in final["exploratory"] if r.get("significant"))
    print(f"\n[탐색 검정 — BH-FDR] {n_sig}/{len(final['exploratory'])} 유의")
    print(f"\n총 실행시간: {elapsed / 60:.1f} 분")
    print(line + "\n")


# ==================================================================== main
def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    experiments = resolve_experiments(args.experiments)

    n_jobs = resolve_jobs(cfg.jobs)
    LOGGER.info("=" * 70)
    LOGGER.info("HalloReg IPD 실험 시작")
    LOGGER.info("  실험     : %s", ", ".join(experiments))
    LOGGER.info("  시드     : %d | 라운드: %d | 입자: %d | 지평: %d",
                cfg.seeds, cfg.rounds, cfg.n_particles, cfg.horizon)
    LOGGER.info("  병렬     : %d 워커 (논리 코어 %d)", n_jobs, os.cpu_count() or 1)
    LOGGER.info("  λ 조절   : w_cd=%.2f, η=%.3f", cfg.w_cd, cfg.lam_gain)
    LOGGER.info("  결과 경로: %s", cfg.results)
    LOGGER.info("=" * 70)

    t0 = time.time()
    reg = Registry()
    arch = None
    h34 = None
    comps = None

    for name in experiments:
        t1 = time.time()
        LOGGER.info("")
        LOGGER.info("─" * 70)
        LOGGER.info("▶ %s 시작", name)
        LOGGER.info("─" * 70)

        if name == "ARCH":
            arch = arch_validation.run(cfg)
            if arch["n_pass"] < arch["n_total"]:
                LOGGER.warning("아키텍처 검증에 실패 항목이 있습니다 — "
                               "이후 가설 결과 해석에 주의하십시오.")
        elif name == "H1":
            h1_intent.run(cfg, reg)
        elif name == "H1A":
            h1a_tracking.run(cfg, reg)
        elif name == "H2":
            h2_protection.run(cfg, reg)
        elif name in ("H3", "H4"):
            # H3 와 H4 는 조합 공간과 유형쌍 행렬을 공유하므로 한 번에 수행한다.
            if h34 is None:
                comps = h3_h4_population._prepare_comps(cfg)
                h34 = h3_h4_population.run(cfg, reg)
        elif name == "H5":
            if h34 is None:
                # H5 는 H4 의 보수행렬이 필요하다 — 없으면 먼저 계산한다.
                LOGGER.info("H5 는 H4 의 보수행렬이 필요 — H3/H4 를 먼저 수행")
                comps = h3_h4_population._prepare_comps(cfg)
                h34 = h3_h4_population.run(cfg, reg)
            h5_evolution.run(cfg, reg, h34, comps)
        elif name == "H6":
            h6_recovery.run(cfg, reg)

        LOGGER.info("◀ %s 완료 (%.1f 분)", name, (time.time() - t1) / 60)

    final = reg.finalize()
    elapsed = time.time() - t0

    save_json({"config": {"seeds": cfg.seeds, "rounds": cfg.rounds,
                          "particles": cfg.n_particles, "horizon": cfg.horizon,
                          "w_cd": cfg.w_cd, "lam_gain": cfg.lam_gain,
                          "env_error": cfg.env_error,
                          "experiments": experiments,
                          "elapsed_min": elapsed / 60},
               "architecture": ({"n_pass": arch["n_pass"],
                                 "n_total": arch["n_total"],
                                 "checks": arch["checks"]} if arch else None),
               "verdicts": final},
              cfg.results / "SUMMARY.json")

    print_summary(final, arch, elapsed)
    LOGGER.info("결과 저장 완료: %s", cfg.results)
    LOGGER.info("그림 저장 완료: %s", cfg.figdir)
    return 0


if __name__ == "__main__":
    # Windows/spawn 호환: 자식 프로세스가 이 모듈을 재import 할 때
    # main() 이 재실행되지 않도록 __main__ 가드 안에서만 호출한다.
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())

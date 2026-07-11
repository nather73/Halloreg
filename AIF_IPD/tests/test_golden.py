#!/usr/bin/env python
"""
test_golden.py — 황금 시드 회귀 테스트 (보완안 엔지니어링 안전망).

고정 시드 다이애드/집단의 요약 통계를 스냅숏(tests/golden_snapshot.json)으로
저장하고, 이후 변경(특히 잡음 계층 이동·DD-충전 분리·게임구조 매개화)이
의도치 않은 동작 변화를 일으키지 않았는지 허용오차 내에서 검증한다.

pytest 가 있으면 `pytest tests/` 로, 없으면 직접 실행:
    python tests/test_golden.py            # 검증 (스냅숏 없으면 생성)
    python tests/test_golden.py --update   # 스냅숏 재생성
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT.parent))

from AIF_IPD.core.constants import set_coop_index
from AIF_IPD.ipd.agent import AdaptiveAgent
from AIF_IPD.ipd.env import make_opponent
from AIF_IPD.ipd.sim import run_dyad, run_population_spec
from AIF_IPD.ipd.baselines import make_baseline

SNAP = Path(__file__).resolve().parent / "golden_snapshot.json"
TOL = 1e-6


def _dyad_summary():
    set_coop_index(None)                     # 기본 게임 복원
    ad = AdaptiveAgent(seed=7, kappa=0.9, sophisticated=True)
    opp = make_opponent("tit_for_tat", seed=99, error=0.0)
    h = run_dyad(ad, opp, 60)
    return {"my_coop": float(np.mean(h["my_act"] == 0)),
            "cc_rate": float(np.mean(h["state"] == 0)),
            "my_payoff_sum": float(np.sum(h["my_payoff"])),
            "lam_final": float(np.asarray(ad.log["lam"])[-5:].mean())}


def _dyad_env_noise():
    set_coop_index(None)
    a = make_opponent("tit_for_tat", seed=1, error=0.0)
    b = make_opponent("generous_tft", seed=2, error=0.0)
    h = run_dyad(a, b, 40, env_err_a=0.1, env_err_b=0.1, noise_seed=42)
    return {"cc_rate": float(np.mean(h["state"] == 0)),
            "payoff_sum": float(np.sum(h["my_payoff"]))}


def _baseline_summary():
    set_coop_index(None)
    q = make_baseline("bayes_br", seed=3)
    opp = make_opponent("tit_for_tat", seed=5, error=0.0)
    h = run_dyad(q, opp, 50)
    return {"coop": float(np.mean(h["my_act"] == 0)),
            "payoff_sum": float(np.sum(h["my_payoff"]))}


def _population_summary():
    set_coop_index(None)
    members = ([{"type": "strategy", "kind": "tit_for_tat", "seed": 0}] * 3
               + [{"type": "strategy", "kind": "alld", "seed": 0}] * 2
               + [{"type": "adaptive", "seed": 0, "kappa": 0.9}] * 2)
    labels = ["tit_for_tat"] * 3 + ["alld"] * 2 + ["adaptive"] * 2
    out = run_population_spec({"members": members, "labels": labels,
                              "n_rounds": 40, "seed": 11,
                              "partners_per_agent": None})
    return {"cc_rate": out["cc_rate"], "mean_payoff": out["mean_payoff"],
            "nonexploiter_mean_payoff": out["nonexploiter_mean_payoff"],
            "alld_exploit_gain_total": out["alld_exploit_gain_total"]}


def _game_ci_summary():
    set_coop_index(0.6)
    ad = AdaptiveAgent(seed=7, kappa=0.9)
    opp = make_opponent("tit_for_tat", seed=99)
    h = run_dyad(ad, opp, 40)
    set_coop_index(None)                     # 반드시 복원
    return {"payoff_sum": float(np.sum(h["my_payoff"]))}


def build_snapshot():
    return {"dyad": _dyad_summary(), "dyad_env_noise": _dyad_env_noise(),
            "baseline": _baseline_summary(), "population": _population_summary(),
            "game_ci_0.6": _game_ci_summary()}


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def test_golden():
    """pytest 진입점: 스냅숏과 현재 요약이 허용오차 내에서 일치."""
    assert SNAP.exists(), "스냅숏이 없습니다 — `python tests/test_golden.py --update` 실행"
    ref = _flatten(json.loads(SNAP.read_text()))
    cur = _flatten(build_snapshot())
    diffs = {k: (ref.get(k), cur[k]) for k in cur
             if k not in ref or abs(cur[k] - ref[k]) > TOL}
    assert not diffs, f"황금 시드 회귀 실패 (허용오차 {TOL}): {diffs}"


def main():
    update = "--update" in sys.argv
    if update or not SNAP.exists():
        SNAP.write_text(json.dumps(build_snapshot(), indent=2, ensure_ascii=False))
        print(f"[golden] 스냅숏 {'재생성' if update else '생성'}: {SNAP}")
        if update:
            return
    try:
        test_golden()
        print("[golden] PASS — 모든 요약 통계가 스냅숏과 일치")
    except AssertionError as e:
        print(f"[golden] FAIL — {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

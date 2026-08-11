"""
experiments.common
==================

모든 가설 실험이 공유하는 실행 설정·결과 등록·시각화 유틸.

[확증(confirmatory) 대 탐색(exploratory)]
가설마다 **사전에 지정한 주 검정**만 확증으로 등록하고, 나머지는 탐색으로 둔다.
확증 가족에는 Holm 보정(FWER 통제), 탐색에는 BH-FDR 을 적용한다. 이렇게 해야
"여러 지표를 돌려보고 유의한 것만 보고" 하는 것을 구조적으로 막을 수 있다.

[검증 문화]
`verdict` 는 p 값과 **방향성**을 모두 충족해야 '지지' 로 기록된다. 지지되지 않은
경우 억지로 긍정 서술을 만들지 않고 그대로 '미지지' 로 남긴다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")                   # 헤드리스 환경에서 파일로만 출력
import matplotlib.pyplot as plt
from matplotlib import font_manager

from AIF_IPD.core.logging_utils import get_logger, set_korean_font
from AIF_IPD.ipd.metrics import bh_fdr, holm

LOGGER = get_logger("HalloReg.exp")

set_korean_font(plt, font_manager)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
})

#: 유형별 고정 색상 — 모든 그림에서 일관되게 쓴다.
TYPE_COLORS = {
    "tft": "#4C72B0", "gtft": "#55A868", "wsls": "#C44E52",
    "allc": "#8172B2", "alld": "#937860", "halloreg": "#DA8BC3",
    "empathic_lo": "#8C8C8C", "empathic_hi": "#CCB974",
}


# ==================================================================== 설정
@dataclass
class Config:
    """실험 전역 설정 (CLI 가 채운다)."""
    seeds: int = 120                 # 기본 시드 수 (사양 고정값)
    rounds: int = 120                # 기본 라운드 수 (사양 고정값)
    jobs: int = -1                   # 병렬 워커 (-1 → 코어수 − 1)
    results: Path = Path("results")
    env_error: float = 0.05          # 환경 계층 실행오류 (모든 유형 대칭)
    n_particles: int = 400
    horizon: int = 6                     # 형질공간 rollout 지평
    policy_particles: int = 64
    prop_sd: float = 0.40
    policy_gamma: float = 8.0
    w_epi_j: float = 10.0
    w_epi_r: float = 1.0
    w_cplx: float = 0.15
    payoff_access: str = "naive"
    qrtd_gamma: float = 0.9
    qrtd_lr: float = 0.20
    w_cd: float = 0.5                # Empathy 의 정서–맥락 가중
    lam_gain: float = 0.05           # λ 적분 이득 η
    w_tonic: float = 0.10            # (v1.8.0 폐기 — 서명 호환)
    lam_gain_down: float = 0.45      # 위협 방향 이완률 (비대칭)
    aff_gain: float = 0.30           # 정서 이득 (라운드 단위 급변 허용)
    policy_mode: str = "lambda_only" # 행위 선택: λ 단독 (기존 "traits" 보존)
    lam_mode: str = "allostatic"     # λ 조절: 알로스테시스 직접 사상
    group_bias: float = 1.00         # 집단 적합성 편향 g (λ 상한)
    w_ig_r: float = 0.15             # 절편의 인식항 가중 — 보상 구조 IG
    w_ig_j: float = 0.15             # 절편의 인식항 가중 — 상대 의도 IG
    e_source: str = "z"              # E_t 원천: 'z' (Z̃ 장기가치) | 'reward'
    r_surv_fixed: object = None      # 생존 기준점 고정값 (None=학습 maximin)
    allo_aff_gain: float = 0.0       # 알로스테시스 모드 정서 미세조절 이득
    beta_es: float = 70.0            # es(λ,s) 로짓 정밀도 (정규화 절편 기준)
    plan_sweeps: int = 1             # 라운드당 모형 기반 계획 스윕 횟수
    reanchor_at: int = 8             # Z̃ 재기준화 시점 (R̂ 관측 수)
    bootstrap: str = "sarsa"         # 부트스트랩: 'sarsa' | 'greedy'
    alpha_kappa: float = 0.0         # 사회사 협력편향 α ~ N(0, κ²)
    sp_disposition: float = 0.50     # 보상적 λ_sp 의 성향 성분 m
    quick: bool = False              # 스모크 모드
    max_compositions: int = 0        # 0 이면 전수 열거

    def halloreg_kwargs(self) -> dict:
        """HalloRegAgent 생성 인자 (실험 전역에서 동일하게 쓴다)."""
        return {"n_particles": self.n_particles, "planning_horizon": self.horizon,
                "policy_particles": self.policy_particles,
                "prop_sd": self.prop_sd, "policy_gamma": self.policy_gamma,
                "w_epi_j": self.w_epi_j, "w_epi_r": self.w_epi_r,
                "w_cplx": self.w_cplx,
                "payoff_access": self.payoff_access,
                "qrtd_gamma": self.qrtd_gamma, "qrtd_lr": self.qrtd_lr,
                "w_cd": self.w_cd, "lam_gain": self.lam_gain,
                "w_tonic": self.w_tonic,
                "aff_gain": self.aff_gain,
                "lam_gain_down": self.lam_gain_down,
                "policy_mode": self.policy_mode,
                "lam_mode": self.lam_mode,
                "group_bias": self.group_bias,
                "w_ig_r": self.w_ig_r,
                "w_ig_j": self.w_ig_j,
                "e_source": self.e_source,
                "r_surv_fixed": self.r_surv_fixed,
                "allo_aff_gain": self.allo_aff_gain,
                "beta_es": self.beta_es,
                "plan_sweeps": self.plan_sweeps,
                "reanchor_at": self.reanchor_at,
                "bootstrap": self.bootstrap,
                "alpha_kappa": self.alpha_kappa,
                "sp_disposition": self.sp_disposition}

    def empathic_kwargs(self, lam: float) -> dict:
        """고정 λ 대조군 생성 인자."""
        return {"lam": lam, "n_particles": self.n_particles,
                "planning_horizon": self.horizon}

    @property
    def figdir(self) -> Path:
        d = self.results / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ==================================================================== 결과등록
@dataclass
class Registry:
    """확증/탐색 검정 결과 누적기."""
    primary: List[dict] = field(default_factory=list)
    exploratory: List[dict] = field(default_factory=list)

    def confirm(self, hyp: str, label: str, p: float, direction_ok: bool,
                effect: str = "", detail: Optional[dict] = None) -> None:
        """확증 검정 등록. Holm 보정은 전 가설 종료 후 일괄 적용한다."""
        self.primary.append({"hypothesis": hyp, "label": label,
                             "p_raw": float(p),
                             "direction_ok": bool(direction_ok),
                             "effect": effect, "detail": detail or {}})
        LOGGER.info("  [확증][%s] %s — %s, 방향성립=%s (p_raw=%.4g)",
                    hyp, label, effect, direction_ok, p)

    def explore(self, hyp: str, label: str, p: float,
                effect: str = "", detail: Optional[dict] = None) -> None:
        """탐색 검정 등록. BH-FDR 은 전 가설 종료 후 일괄 적용한다."""
        self.exploratory.append({"hypothesis": hyp, "label": label,
                                 "p_raw": float(p), "effect": effect,
                                 "detail": detail or {}})
        LOGGER.info("  [탐색][%s] %s — %s (p_raw=%.4g)", hyp, label, effect, p)

    def finalize(self, alpha: float = 0.05) -> dict:
        """
        다중비교 보정 후 최종 판정표를 만든다.

        확증: Holm 보정 p < alpha **그리고** 방향성립 → 지지.
        탐색: BH-FDR q < alpha → 유의(방향 판정은 별도로 하지 않는다).
        """
        if self.primary:
            adj = holm([r["p_raw"] for r in self.primary])
            for r, a in zip(self.primary, adj):
                r["p_holm"] = a
                r["supported"] = bool(a < alpha and r["direction_ok"])
        if self.exploratory:
            adj = bh_fdr([r["p_raw"] for r in self.exploratory])
            for r, a in zip(self.exploratory, adj):
                r["q_bh"] = a
                r["significant"] = bool(a < alpha)

        # 가설 단위 판정: 그 가설의 확증 검정이 **모두** 지지되면 지지.
        by_hyp: Dict[str, dict] = {}
        for r in self.primary:
            h = by_hyp.setdefault(r["hypothesis"], {"n": 0, "n_ok": 0})
            h["n"] += 1
            h["n_ok"] += int(r["supported"])
        for h, d in by_hyp.items():
            d["verdict"] = ("지지" if d["n_ok"] == d["n"]
                            else ("부분지지" if d["n_ok"] > 0 else "미지지"))
        return {"alpha": alpha, "primary": self.primary,
                "exploratory": self.exploratory, "by_hypothesis": by_hyp}


# ==================================================================== 직렬화
def _jsonable(o):
    """numpy 타입을 포함한 객체를 JSON 직렬화 가능하게 변환."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return [_jsonable(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(obj, path: Path) -> None:
    """결과 dict 를 UTF-8 JSON 으로 저장 (한글 그대로)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, ensure_ascii=False, indent=2)


# ==================================================================== 시각화
def save_fig(fig, cfg: Config, name: str) -> Path:
    """그림을 PNG 로 저장하고 닫는다."""
    path = cfg.figdir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("  그림 저장: %s", path)
    return path


def bar_with_ci(ax, labels, means, cis, colors=None, ylabel: str = "",
                title: str = "", rotate: int = 0) -> None:
    """부트스트랩 95% CI 오차막대가 있는 막대그림."""
    x = np.arange(len(labels))
    lo = np.array([m - c[0] for m, c in zip(means, cis)])
    hi = np.array([c[1] - m for m, c in zip(means, cis)])
    ax.bar(x, means, yerr=[lo, hi], capsize=3,
           color=colors if colors else "#4C72B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate,
                       ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def band_plot(ax, traces: np.ndarray, color: str = "#4C72B0",
              label: str = "", q: tuple = (25, 75)) -> None:
    """
    시드 궤적 집합의 중앙값 선 + 사분위 밴드.

    평균 ± SD 대신 백분위 밴드를 쓰는 이유: λ·협력률처럼 유계이고 분포가
    비대칭인 지표에서 SD 밴드는 정의역을 벗어나 오해를 부른다.
    """
    T = traces.shape[1]
    t = np.arange(T)
    med = np.nanmedian(traces, axis=0)
    lo = np.nanpercentile(traces, q[0], axis=0)
    hi = np.nanpercentile(traces, q[1], axis=0)
    ax.plot(t, med, color=color, lw=1.6, label=label)
    ax.fill_between(t, lo, hi, color=color, alpha=0.18, lw=0)


def annotate_n(ax, n: int) -> None:
    """표본 수를 그림 안에 명시 (재현성·해석 보조)."""
    ax.text(0.99, 0.01, f"n={n}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#555555")

#!/usr/bin/env python
"""
compare_versions.py — 두 `summary.json` 대조 시각화 (예: v0.6.6 vs v0.8.0).

개정이 **무엇을 바꿨고 무엇을 보존했는가**를 한눈에 본다. 확증 지표는 라벨의
지평 태그를 떼고 (hyp, tag) 로 짝지어 대조하며, 효과크기 문자열
("dz=-35.46 [-39.78, -32.16]", "β=0.468 [0.017, 0.856]") 을 파싱해 수치 비교한다.

정직 보고 원칙:
  · 한쪽에만 있는 지표(신규 RR/FG, 삭제된 지표)는 **숨기지 않고** 별도 표기한다.
  · 판정 반전(지지↔미지지)은 색으로 강조하되, p 가 아니라 효과크기+CI 로 읽도록
    배치한다.
  · 효과크기 척도가 다른 지표(dz vs β vs Δ)는 **섞어서 정렬하지 않는다**.

사용:
    python AIF_IPD/scripts/compare_versions.py \
        --old /path/v066/summary.json --new AIF_IPD/results/summary.json \
        --old-label v0.6.6 --new-label v0.8.0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT.parent))

from AIF_IPD.core.logging_utils import get_logger, set_korean_font

LOGGER = get_logger("HalloReg.compare")
RESULTS = _PKG_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# "dz=-35.46 [-39.78, -32.16]" / "β=0.468 [0.017, 0.856]" / "기울기=-0.0026 [...]"
_ES_RE = re.compile(
    r"([A-Za-zα-ωΑ-Ω_가-힣]+)\s*=\s*(-?[\d.]+(?:e[+-]?\d+)?)"
    r"(?:\s*\[\s*(-?[\d.]+(?:e[+-]?\d+)?)\s*,\s*(-?[\d.]+(?:e[+-]?\d+)?)\s*\])?")


def parse_es(es: str):
    """효과크기 문자열 → [(척도, 값, lo, hi), ...]. 결합가설은 항이 여러 개."""
    out = []
    for m in _ES_RE.finditer(es or ""):
        name, val, lo, hi = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            v = float(val)
        except ValueError:
            continue
        if not np.isfinite(v):
            continue
        try:
            lo_f = float(lo) if lo is not None else np.nan
            hi_f = float(hi) if hi is not None else np.nan
        except ValueError:
            lo_f = hi_f = np.nan
        out.append((name, v, lo_f, hi_f))
    return out


def strip_tag(label: str) -> str:
    """라벨에서 ' [T60]' 같은 지평 태그 제거 — 버전 간 짝짓기 키."""
    return re.sub(r"\s*\[T\d+\]\s*$", "", label or "").strip()


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_primary(d: dict) -> dict:
    """(hyp, tag, 라벨core) → 레코드."""
    out = {}
    for c in d.get("confirmatory", []):
        key = (c.get("hyp"), c.get("tag") or "", strip_tag(c.get("label", "")))
        out[key] = c
    return out


def build_rows(old: dict, new: dict):
    """공통/신규/제거 분류. 공통은 첫 효과크기 항을 대표값으로 쓴다."""
    io, inw = index_primary(old), index_primary(new)
    common, only_new, only_old = [], [], []
    for k in sorted(set(io) | set(inw), key=lambda x: (x[0], x[1])):
        a, b = io.get(k), inw.get(k)
        if a and b:
            ea, eb = parse_es(a.get("es", "")), parse_es(b.get("es", ""))
            # 척도가 같은 항만 대조 (dz vs β 혼합 방지)
            pair = None
            for na, va, la, ha in ea:
                for nb, vb, lb, hb in eb:
                    if na == nb:
                        pair = (na, va, la, ha, vb, lb, hb)
                        break
                if pair:
                    break
            common.append({"key": k, "old": a, "new": b, "pair": pair})
        elif b:
            only_new.append({"key": k, "rec": b})
        else:
            only_old.append({"key": k, "rec": a})
    return common, only_new, only_old


def verdict(rec: dict) -> bool:
    """지지 = 방향 성립 ∧ Holm 보정 p < .05."""
    p = rec.get("p_holm", rec.get("p", 1.0))
    return bool(rec.get("direction_met")) and (p is not None) and p < 0.05


def make_figure(common, only_new, only_old, old_lab, new_lab, out_stem):
    try:
        set_korean_font(plt, font_manager)
    except Exception:
        pass
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.32,
                          wspace=0.28)

    # ---------- (a) 판정 대조 매트릭스 ----------
    a = fig.add_subplot(gs[0, :2])
    labs, states = [], []
    for r in common:
        vo, vn = verdict(r["old"]), verdict(r["new"])
        labs.append(f"[{r['key'][0]}·{r['key'][1] or '—'}] {r['key'][2][:38]}")
        states.append((vo, vn))
    order = sorted(range(len(labs)), key=lambda i: (states[i][0] != states[i][1],
                                                    labs[i]))
    labs = [labs[i] for i in order]; states = [states[i] for i in order]
    y = np.arange(len(labs))
    CMAP = {(True, True): ("C2", "유지: 지지"),
            (False, False): ("0.75", "유지: 미지지"),
            (False, True): ("C0", "반전: 미지지→지지"),
            (True, False): ("C3", "반전: 지지→미지지")}
    for i, st in enumerate(states):
        col, _ = CMAP[st]
        a.barh(i, 1, left=0, color=col, alpha=.55, height=.72)
        a.text(0.5, i, ("지지" if st[0] else "미지지") + " → "
               + ("지지" if st[1] else "미지지"), ha="center", va="center",
               fontsize=7)
    a.set_yticks(y); a.set_yticklabels(labs, fontsize=6.5)
    a.set_xticks([]); a.set_xlim(0, 1)
    a.invert_yaxis()
    a.set_title(f"(a) 확증 지표 판정 대조 — {old_lab} → {new_lab}\n"
                "(방향 성립 ∧ Holm p<.05; 반전은 색 강조)", fontsize=11)
    a.legend(handles=[Patch(facecolor=v[0], alpha=.55, label=v[1])
                      for v in CMAP.values()], fontsize=7, loc="lower right")

    # ---------- (b) 효과크기 척도별 산점 ----------
    a = fig.add_subplot(gs[0, 2])
    scales = {}
    for r in common:
        if not r["pair"]:
            continue
        nm, vo, _, _, vn, _, _ = r["pair"]
        scales.setdefault(nm, []).append((vo, vn, r["key"]))
    cols = plt.cm.tab10(np.linspace(0, 1, max(len(scales), 1)))
    for (nm, pts), c in zip(sorted(scales.items()), cols):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        a.scatter(xs, ys, s=30, color=c, alpha=.8, label=f"{nm} (n={len(pts)})")
    if scales:
        allv = [v for pts in scales.values() for p in pts for v in p[:2]]
        lo, hi = float(min(allv)), float(max(allv))
        pad = 0.06 * max(hi - lo, 1e-9)
        a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2,
               label="y=x (불변)")
        a.axhline(0, color="0.6", lw=.7); a.axvline(0, color="0.6", lw=.7)
    a.set_xlabel(f"{old_lab} 효과크기"); a.set_ylabel(f"{new_lab} 효과크기")
    a.legend(fontsize=6.5)
    a.set_title("(b) 효과크기 이동\n(척도별 분리 — dz·β 혼합 정렬 금지)",
                fontsize=11)

    # ---------- (c) 척도별 Δ 막대 ----------
    a = fig.add_subplot(gs[1, 0])
    rows = [(r["key"], r["pair"]) for r in common if r["pair"]]
    rows = [(k, p) for k, p in rows if p[0] == "dz"]
    rows.sort(key=lambda t: abs(t[1][4] - t[1][1]), reverse=True)
    rows = rows[:14]
    if rows:
        names = [f"{k[0]}·{k[1] or '—'}" for k, _ in rows]
        d = [p[4] - p[1] for _, p in rows]
        a.barh(range(len(d)), d, color=["C0" if x >= 0 else "C3" for x in d],
               alpha=.8)
        a.set_yticks(range(len(d))); a.set_yticklabels(names, fontsize=7)
        a.axvline(0, color="k", lw=.9); a.invert_yaxis()
    a.set_xlabel(f"Δdz = {new_lab} − {old_lab}")
    a.set_title("(c) dz 변화 상위 (절대값 기준)", fontsize=11)

    # ---------- (d) 신규/제거 지표 ----------
    a = fig.add_subplot(gs[1, 1]); a.axis("off")
    txt = [f"신규 지표 ({new_lab} 에만 존재) — {len(only_new)}개", ""]
    for r in only_new[:9]:
        k = r["key"]; v = "지지" if verdict(r["rec"]) else "미지지"
        txt.append(f"  + [{k[0]}·{k[1] or '—'}] {k[2][:34]}")
        txt.append(f"      → {v} | {r['rec'].get('es','')[:40]}")
    txt += ["", f"제거/미실행 지표 ({old_lab} 에만) — {len(only_old)}개", ""]
    for r in only_old[:6]:
        k = r["key"]
        txt.append(f"  − [{k[0]}·{k[1] or '—'}] {k[2][:34]}")
    a.text(0.0, 1.0, "\n".join(txt), va="top", fontsize=7.2,
           family="monospace")
    a.set_title("(d) 짝지어지지 않는 지표 [정직 보고]", fontsize=11, loc="left")

    # ---------- (e) 요약 카운트 ----------
    a = fig.add_subplot(gs[1, 2])
    cnt = {k: 0 for k in CMAP}
    for r in common:
        cnt[(verdict(r["old"]), verdict(r["new"]))] += 1
    ks = list(CMAP)
    a.bar(range(len(ks)), [cnt[k] for k in ks],
          color=[CMAP[k][0] for k in ks], alpha=.75)
    for i, k in enumerate(ks):
        a.text(i, cnt[k], str(cnt[k]), ha="center", va="bottom", fontsize=10)
    a.set_xticks(range(len(ks)))
    a.set_xticklabels([CMAP[k][1].replace(": ", ":\n") for k in ks], fontsize=7)
    a.set_ylabel("확증 지표 수")
    a.set_title(f"(e) 요약 — 공통 {len(common)}개\n"
                f"신규 {len(only_new)} / 제거 {len(only_old)}", fontsize=11)

    fig.suptitle(f"HalloReg 버전 대조: {old_lab} → {new_lab}", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("그림 저장: %s", RESULTS / f"{out_stem}.png")
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="기준 summary.json (예: v0.6.6)")
    ap.add_argument("--new", required=True, help="대조 summary.json (예: v0.8.0)")
    ap.add_argument("--old-label", default="v0.6.6")
    ap.add_argument("--new-label", default="v0.8.1")
    ap.add_argument("--out", default="version_compare")
    args = ap.parse_args()

    old, new = load(Path(args.old)), load(Path(args.new))
    LOGGER.info("%s config: %s", args.old_label,
                json.dumps(old.get("config", {}), ensure_ascii=False))
    LOGGER.info("%s config: %s", args.new_label,
                json.dumps(new.get("config", {}), ensure_ascii=False))
    # 조건 불일치 경고 — 검정력 교락 방지 (정직 보고)
    for f in ("seeds", "rounds", "backend"):
        a, b = old.get("config", {}).get(f), new.get("config", {}).get(f)
        if a != b:
            LOGGER.warning("⚠ config 불일치: %s = %s vs %s — 효과크기 차이가 개정 "
                           "효과인지 조건 차이인지 **교락**됩니다", f, a, b)

    common, only_new, only_old = build_rows(old, new)
    cnt = make_figure(common, only_new, only_old, args.old_label,
                      args.new_label, args.out)

    LOGGER.info("공통 %d | 신규 %d | 제거 %d", len(common), len(only_new),
                len(only_old))
    for k, v in cnt.items():
        LOGGER.info("  %s: %d", {(True, True): "유지(지지)",
                                 (False, False): "유지(미지지)",
                                 (False, True): "반전(미지지→지지)",
                                 (True, False): "반전(지지→미지지)"}[k], v)
    for r in common:
        vo, vn = verdict(r["old"]), verdict(r["new"])
        if vo != vn:
            LOGGER.info("[반전] %s·%s %s: %s → %s | es %s → %s", r["key"][0],
                        r["key"][1] or "—", r["key"][2][:40],
                        "지지" if vo else "미지지", "지지" if vn else "미지지",
                        r["old"].get("es", ""), r["new"].get("es", ""))

    rep = {"old_label": args.old_label, "new_label": args.new_label,
           "old_config": old.get("config", {}), "new_config": new.get("config", {}),
           "counts": {f"{k[0]}->{k[1]}": v for k, v in cnt.items()},
           "common": [{"hyp": r["key"][0], "tag": r["key"][1],
                       "label": r["key"][2],
                       "old_es": r["old"].get("es"), "new_es": r["new"].get("es"),
                       "old_supported": verdict(r["old"]),
                       "new_supported": verdict(r["new"])} for r in common],
           "only_new": [{"hyp": r["key"][0], "tag": r["key"][1],
                         "label": r["key"][2], "es": r["rec"].get("es"),
                         "supported": verdict(r["rec"])} for r in only_new],
           "only_old": [{"hyp": r["key"][0], "tag": r["key"][1],
                         "label": r["key"][2]} for r in only_old]}
    with open(RESULTS / f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    LOGGER.info("결과 저장: %s", RESULTS / f"{args.out}.json")


if __name__ == "__main__":
    main()

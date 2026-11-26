#!/usr/bin/env python3
"""
04_eval.py — 세 방법의 정확도와 '확신의 질'을 같은 자로 비교한다

이 파일이 답하는 질문
    "01(무학습)·02(헤드 학습)·03(LoRA) 중 무엇이 더 잘 맞히고, 무엇이 더 정직하게 확신하는가?"

실행하면 무엇을 보게 되는지
    (1) 방법별 정확도·평균 확신·Brier·ECE·NLL 비교표(화면 + outputs/eval_report.md),
    (2) 신뢰도 다이어그램용 구간 표(outputs/reliability_*.csv),
    (3) 확신이 높은데 틀린 항목(위험한 오답) 목록.

개념 정리
    정확도(accuracy) : 맞힌 비율. 이걸 올리는 것만으로는 부족하다.
    보정(calibration) : "확률 0.9라고 말한 것 중 실제로 90%가 맞는가".
    Brier score      : 확률 예측의 평균 제곱오차(낮을수록 좋다, 0~2).
    ECE              : 구간별 |정확도 - 평균확신| 을 표본 수로 가중 평균한 값(낮을수록 좋다).
    NLL              : 정답에 준 확률의 -log 평균. 확신이 틀렸을 때 크게 벌점을 준다.

    판단 모델을 실제 시스템에 붙이면 "0.6이면 사람에게 넘긴다" 같은 문턱을 쓰게 된다.
    그 문턱이 의미를 가지려면 확률이 보정돼 있어야 하고, 그래서 이 파일이 필요하다.

사용 예
    python 04_eval.py                                  # outputs/preds_*.csv 전부
    python 04_eval.py outputs/preds_01_logprob.csv outputs/preds_03_lora.csv
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from common import OUT_DIR, ensure_out_dir, read_preds_csv


def multiclass_brier(probs: Sequence[float], answer_index: int) -> float:
    """Brier score(다중 분류): 각 클래스에서 (예측확률 - 정답여부)² 을 모두 더한 값.

    왜 정답 확률만 보지 않는가: "정답에 0.4를 줬지만 나머지 0.6을 한 곳에 몰아준" 경우와
    "0.4를 여러 곳에 흩뿌린" 경우는 위험도가 다르다. Brier는 그 차이를 반영한다.
    """
    total = 0.0
    for i, p in enumerate(probs):
        y = 1.0 if i == answer_index else 0.0
        total += (p - y) ** 2
    return total


def reliability_table(pairs: Sequence[Tuple[float, int]], bins: int = 10) -> Tuple[List[Dict[str, Any]], float]:
    """확신 구간별 표와 ECE를 계산한다.

    왜 구간을 나누는가: 확률 하나하나는 맞고 틀림만 알 수 없어서, 비슷한 확신끼리
    묶어 "이 구간에서 평균 확신 0.85, 실제 정답률 0.60"을 봐야 과신/과소신이 드러난다.

    왜 마지막 구간만 1.0을 포함하는가: 구간을 [0.0,0.1), [0.1,0.2) ... 로 나누면
    확률 1.0이 어디에도 안 들어간다. 그래서 마지막 구간은 1.0까지 포함시킨다.
    """
    n = len(pairs)
    rows: List[Dict[str, Any]] = []
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [(c, y) for c, y in pairs if (lo <= c < hi) or (b == bins - 1 and c >= hi - 1e-12)]
        count = len(in_bin)
        if count == 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "avg_confidence": None,
                         "accuracy": None, "gap": None})
            continue
        avg_conf = sum(c for c, _ in in_bin) / count
        acc = sum(y for _, y in in_bin) / count
        gap = acc - avg_conf          # 음수면 과신(확신보다 정답률이 낮다)
        ece += (count / n) * abs(gap)
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": count,
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(acc, 4),
            "gap": round(gap, 4),
        })
    return rows, ece


def analyze(path: Path, bins: int) -> Dict[str, Any]:
    """예측 CSV 하나를 읽어 지표를 계산한다."""
    df = read_preds_csv(path)
    if df.empty:
        return {"path": path, "error": "빈 파일"}

    n = len(df)
    correct = df["correct"].astype(int).to_numpy()
    conf = df["confidence"].astype(float).to_numpy()

    briers: List[float] = []
    nlls: List[float] = []
    pairs: List[Tuple[float, int]] = []
    for _, row in df.iterrows():
        probs = row["choice_probs"]
        if isinstance(probs, str):
            import json
            probs = json.loads(probs)
        if not probs:
            continue
        ans = int(row["answer_index"])
        briers.append(multiclass_brier(probs, ans))
        p_true = max(min(float(probs[ans]), 1.0), 1e-12)
        nlls.append(-math.log(p_true))
        pairs.append((float(row["confidence"]), int(row["correct"])))

    table, ece = reliability_table(pairs, bins)
    accuracy = float(correct.mean())
    mean_conf = float(conf.mean())

    return {
        "path": path,
        "method": str(df["method"].iloc[0]) if "method" in df.columns else path.stem,
        "model": str(df["model"].iloc[0]) if "model" in df.columns else "",
        "n": n,
        "accuracy": accuracy,
        "mean_confidence": mean_conf,
        "overconfidence": mean_conf - accuracy,   # 양수면 과신
        "brier": sum(briers) / max(len(briers), 1),
        "ece": ece,
        "nll": sum(nlls) / max(len(nlls), 1),
        "reliability": table,
        "risky": df[(df["confidence"] >= 0.9) & (df["correct"] == 0)][["id", "task", "answer_index", "predicted"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="보정 평가: 정확도·Brier·ECE")
    parser.add_argument("files", nargs="*", help="예측 CSV 경로들(없으면 outputs/preds_*.csv)")
    parser.add_argument("--bins", type=int, default=10, help="신뢰도 구간 수(기본 10)")
    parser.add_argument("--out-md", default="", help="보고서 경로(기본: outputs/eval_report.md)")
    args = parser.parse_args()

    out_dir = ensure_out_dir()
    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        paths = [Path(p) for p in sorted(glob.glob(str(OUT_DIR / "preds_*.csv")))]
    if not paths:
        print("예측 CSV가 없습니다. 먼저 01/02/03 중 하나를 실행하세요.")
        return

    results = [analyze(p, args.bins) for p in paths]
    results = [r for r in results if "error" not in r]
    if not results:
        print("읽을 수 있는 예측 CSV가 없습니다.")
        return

    # ------------------------------------------------------------------
    # 1. 비교표
    # ------------------------------------------------------------------
    header = f"{'방법':<18}{'n':>5}{'정확도':>9}{'평균확신':>10}{'과신':>8}{'Brier':>9}{'ECE':>8}{'NLL':>8}"
    print("=" * len(header))
    print(" 방법별 비교 (Brier·ECE·NLL은 낮을수록 좋다)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['method']:<18}{r['n']:>5}{r['accuracy']:>9.3f}{r['mean_confidence']:>10.3f}"
              f"{r['overconfidence']:>8.3f}{r['brier']:>9.3f}{r['ece']:>8.3f}{r['nll']:>8.3f}")
    print("-" * len(header))
    print("· 과신 = 평균확신 - 정확도. 0보다 크면 '자신 있게 틀리는' 쪽이다.")
    if min(r["n"] for r in results) < 50:
        print("· n이 50보다 작아 ECE는 흔들림이 큽니다. 경향 비교용으로만 보세요.")

    # ------------------------------------------------------------------
    # 2. 구간별 신뢰도 표(방법마다 CSV로 저장 → 엑셀/노트북에서 그림)
    # ------------------------------------------------------------------
    for r in results:
        out_csv = out_dir / f"reliability_{r['path'].stem}.csv"
        pd.DataFrame(r["reliability"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[{r['method']}] 신뢰도 구간 (저장: {out_csv.name})")
        print(f"{'구간':<12}{'n':>5}{'평균확신':>10}{'정확도':>9}{'차이':>8}")
        for row in r["reliability"]:
            if row["n"] == 0:
                continue
            print(f"{row['bin']:<12}{row['n']:>5}{row['avg_confidence']:>10.3f}"
                  f"{row['accuracy']:>9.3f}{row['gap']:>8.3f}")

    # ------------------------------------------------------------------
    # 3. 위험한 오답(확신 0.9 이상인데 틀림)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 확신 0.9 이상인 오답 (실서비스에서 가장 위험한 유형)")
    print("=" * 60)
    any_risky = False
    for r in results:
        risky = r["risky"]
        if len(risky) == 0:
            continue
        any_risky = True
        print(f"[{r['method']}] {len(risky)}건")
        for _, row in risky.head(5).iterrows():
            print(f"  - {row['id']} ({row['task']}): 예측={row['predicted']}")
    if not any_risky:
        print("없음 (좋은 신호다)")

    # ------------------------------------------------------------------
    # 4. Markdown 보고서
    # ------------------------------------------------------------------
    md_path = Path(args.out_md) if args.out_md else out_dir / "eval_report.md"
    lines = ["# 판단 모델 보정 평가 보고서", ""]
    lines.append(f"평가 항목 수: {', '.join(str(r['n']) for r in results)} / 구간 수: {args.bins}")
    lines.append("")
    lines.append("| 방법 | n | 정확도 | 평균 확신 | 과신 | Brier | ECE | NLL |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['method']} | {r['n']} | {r['accuracy']:.3f} | {r['mean_confidence']:.3f} "
            f"| {r['overconfidence']:+.3f} | {r['brier']:.3f} | {r['ece']:.3f} | {r['nll']:.3f} |"
        )
    lines.append("")
    lines.append("## 읽는 법")
    lines.append("- **정확도**는 맞힌 비율, **Brier/ECE/NLL**은 확률이 얼마나 정직한지를 잰다.")
    lines.append("- **과신**이 +0.1 이상이면, 확률을 문턱값으로 쓰는 판단(예: 0.8 이상이면 자동 처리)이 위험하다.")
    lines.append("- **ECE**는 구간 표에서 '평균 확신'과 '정확도'의 차이를 표본 수로 가중 평균한 값이다.")
    lines.append("- 항목 수가 적으면(수십 개) 우연에 따라 값이 크게 흔들린다. 데이터를 늘려 다시 볼 것.")
    lines.append("")
    for r in results:
        lines.append(f"## {r['method']} — 신뢰도 구간")
        lines.append("")
        lines.append("| 구간 | n | 평균 확신 | 정확도 | 차이 |")
        lines.append("|---|---|---|---|---|")
        for row in r["reliability"]:
            if row["n"] == 0:
                continue
            lines.append(f"| {row['bin']} | {row['n']} | {row['avg_confidence']:.3f} "
                         f"| {row['accuracy']:.3f} | {row['gap']:+.3f} |")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] 보고서: {md_path}")


if __name__ == "__main__":
    main()

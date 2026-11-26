#!/usr/bin/env python3
"""
01_logprob_baseline.py — 학습 없이, 선택지 로그확률만으로 채점한다

이 파일이 답하는 질문
    "아무것도 학습하지 않은 Qwen3.5-Instruct는 이 판단 문제를 얼마나 잘 맞히고, 얼마나 확신하는가?"

실행하면 무엇을 보게 되는지
    (1) 세 가지 채점 방식(전체 로그확률 합 / 길이 정규화 / 첫 토큰)의 정확도 비교표,
    (2) 평균 확신도와 실제 정확도의 차이(과신 정도),
    (3) 항목별 예측을 담은 outputs/preds_01_logprob.csv — 04의 보정 평가 입력.

왜 이 단계가 먼저인가
    Jev 같은 '판단 전용 모델'의 바닥값을 재는 단계다. 학습을 붙이기 전에
    "원래 모델이 이 문제를 이해는 하는가"를 알아야, 이후 학습이 얼마나 기여했는지
    말할 수 있다. 기준선 없는 개선은 개선이 아니다.

사용 예
    python 01_logprob_baseline.py --limit 5          # 먼저 5개로 빠르게
    python 01_logprob_baseline.py                    # 전체(28개)
    python 01_logprob_baseline.py --score sum        # 다른 채점 방식을 대표값으로
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DATA_DIR,
    DEFAULT_MODEL_ID,
    build_prompt,
    ensure_out_dir,
    load_model,
    load_tokenizer,
    make_pred_row,
    read_jsonl,
    score_choices,
    set_seed,
    softmax_over,
    write_preds_csv,
)

# 비교할 세 가지 채점 방식. 키 이름은 score_choices가 돌려주는 값의 이름과 같다.
SCORE_KEYS = ("sum", "mean", "first")
KEY_LABEL = {
    "sum": "전체 합 (log P)",
    "mean": "길이 정규화 (평균)",
    "first": "첫 토큰만",
}


def pick(probs_by_key, key):
    """확률 목록에서 가장 큰 값의 위치(선택지 인덱스)를 고른다."""
    seq = probs_by_key[key]
    return max(range(len(seq)), key=lambda i: seq[i])


def main() -> None:
    parser = argparse.ArgumentParser(description="무학습 로그확률 기준선")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data", default=str(DATA_DIR / "sample.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개만(빠른 확인용)")
    parser.add_argument(
        "--score",
        choices=SCORE_KEYS,
        default="mean",
        help="CSV에 대표 확률로 저장할 채점 방식(기본: 길이 정규화)",
    )
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--out", default="", help="결과 CSV 경로(기본: outputs/preds_01_logprob.csv)")
    args = parser.parse_args()

    set_seed(42)
    rows = read_jsonl(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {args.data}: {len(rows)}개 항목")

    tokenizer, model_id = load_tokenizer(args.model)
    model, model_id = load_model(model_id, load_in_4bit=args.four_bit)

    preds_csv_rows = []
    verdicts = {k: [] for k in SCORE_KEYS}     # 맞았는지(1/0)
    confidences = {k: [] for k in SCORE_KEYS}
    details = []                                # 화면에 보여 줄 요약

    for idx, item in enumerate(rows, start=1):
        choices = item["choices"]
        prompt = build_prompt(tokenizer, item["instruction"], choices)
        scores = score_choices(model, tokenizer, prompt, choices, args.max_prompt_tokens)

        if len(scores) != len(choices):
            # 선택지 문자열이 토큰화되지 않는 특수한 경우(빈 문자열 등) — 건너뛴다.
            print(f"  ! {item.get('id')}: 일부 선택지를 채점하지 못했습니다. 건너뜀")
            continue

        # 후보 집합 안에서의 확률. 세 방식 각각 따로 정규화한다.
        probs_by_key = {k: softmax_over(scores, k)[1] for k in SCORE_KEYS}
        answer_index = choices.index(item["answer"])

        for k in SCORE_KEYS:
            predicted = pick(probs_by_key, k)
            verdicts[k].append(int(predicted == answer_index))
            confidences[k].append(max(probs_by_key[k]))

        n_prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        preds_csv_rows.append(
            make_pred_row(
                item,
                choices,
                probs_by_key[args.score],
                scores=scores,
                model_id=model_id,
                method=f"logprob_{args.score}",
                n_prompt_tokens=n_prompt_tokens,
            )
        )

        if idx <= 5 or idx % 10 == 0:
            print(f"  ({idx}/{len(rows)}) {item.get('id')} — {item['task']}")

        details.append(
            {
                "id": item.get("id"),
                "task": item.get("task"),
                "answer": item["answer"],
                "pred": {k: choices[pick(probs_by_key, k)] for k in SCORE_KEYS},
                "p_answer": probs_by_key[args.score][answer_index],
                "scores": scores,
            }
        )

    # ------------------------------------------------------------------
    # 정확도 비교표
    # ------------------------------------------------------------------
    print("\n" + "=" * 66)
    print(" 무학습 기준선 — 채점 방식별 비교")
    print("=" * 66)
    print(f"{'방식':<22}{'정확도':>8}{'평균 확신':>12}{'과신(확신-정확도)':>20}")
    for k in SCORE_KEYS:
        if not verdicts[k]:
            continue
        acc = sum(verdicts[k]) / len(verdicts[k])
        conf = sum(confidences[k]) / len(confidences[k])
        print(f"{KEY_LABEL[k]:<22}{acc:>8.3f}{conf:>12.3f}{conf - acc:>20.3f}")
    print("-" * 66)
    print("· 과신 값이 크면 '자신 있게 틀리는' 상태다. 04에서 ECE로 다시 정량화한다.")
    print("· 항목이 28개뿐이라 정확도는 ±0.1 정도 흔들린다. 경향만 볼 것.")

    # 틀린 항목 몇 개를 눈으로 보여 준다(원인 파악용).
    wrong = [d for d in details if d["pred"][args.score] != d["answer"]]
    print(f"\n오답 {len(wrong)}개 중 앞의 5개:")
    for d in wrong[:5]:
        print(f"  - {d['id']} ({d['task']}): 정답={d['answer']!r}, 예측={d['pred'][args.score]!r}, "
              f"정답 확률={d['p_answer']:.3f}")

    # ------------------------------------------------------------------
    # CSV 저장 — 04가 이 파일을 읽어 Brier/ECE를 계산한다.
    # ------------------------------------------------------------------
    out = Path(args.out) if args.out else ensure_out_dir() / "preds_01_logprob.csv"
    write_preds_csv(preds_csv_rows, out)
    print(f"\n[saved] {out}")
    print("        다음: python 04_eval.py  (또는 02/03을 먼저 돌리고 한꺼번에 비교)")


if __name__ == "__main__":
    main()

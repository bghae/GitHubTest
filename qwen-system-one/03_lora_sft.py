#!/usr/bin/env python3
"""
03_lora_sft.py — LoRA로 소량 SFT를 하고, 그 뒤 같은 방식으로 확률을 재 본다

이 파일이 답하는 질문
    "백본을 조금만(1% 미만) 학습시키면 판단 정확도와 '확신의 질'이 좋아지는가?"

실행하면 무엇을 보게 되는지
    (1) LoRA가 붙은 모듈 목록과 학습 파라미터 비율,
    (2) epoch별 loss 감소(학습이 실제로 되고 있는지),
    (3) 저장한 어댑터를 디스크에서 다시 읽은 모델로 매긴 채점 결과 → outputs/preds_03_lora.csv,
    (4) outputs/lora_adapter/ (어댑터, 수십 MB). --merge 를 주면 합친 모델도 저장.

개념: LoRA
    원래 가중치 W는 그대로 두고, 옆에 작은 행렬 A(입력→r), B(r→출력)를 붙여
    W + BA 만 학습한다. r=16이면 한 층당 학습량이 원래의 1% 미만이라
    12GB 카드에서도 4B를 학습할 수 있고, 결과물(어댑터)도 수십 MB로 작다.

사용 예
    python 03_lora_sft.py                      # 기본 설정(4B, batch1, accum8, 3 epoch)
    python 03_lora_sft.py --epochs 1 --limit 10  # 5분 안에 되는지 먼저 확인
    python 03_lora_sft.py --merge              # 학습 후 합친 모델까지 저장(디스크 8GB 필요)
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from common import (
    DATA_DIR,
    DEFAULT_MODEL_ID,
    build_prompt,
    device_of,
    ensure_out_dir,
    load_model,
    load_tokenizer,
    make_pred_row,
    read_jsonl,
    score_choices,
    set_seed,
    softmax_over,
    vram_report,
    write_preds_csv,
)

# LoRA를 붙일 층 이름 후보.
# 왜 이렇게 많은가: Qwen3.5는 한 층 안에 두 종류의 어텐션을 섞어 쓴다.
#   - full_attention: q_proj/k_proj/v_proj/o_proj
#   - linear_attention(Gated DeltaNet): in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj
#   - MLP: gate_proj / up_proj / down_proj
# 이름을 하나로 못 박으면 절반이 빠져 학습 효과가 반감되므로, 실제 모델을 훑어
# 존재하는 이름만 골라 쓴다(target_modules 자동 탐색).
CANDIDATE_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)


def find_lora_targets(model: nn.Module) -> List[str]:
    """모델을 훑어 LoRA를 붙일 Linear 층 이름을 찾는다.

    왜 자동 탐색인가: 모델 크기(2B/4B/9B)나 transformers 버전에 따라 층 이름이
    조금씩 다르다. 이름을 하드코딩하면 "학습은 되는데 아무 효과 없음"이 되기 쉽다.
    여기서는 (1) 이름 접미사가 후보에 있고 (2) 비전 인코더가 아닌 Linear만 고른다.
    lm_head는 출력층이라 LoRA 대상에서 제외한다(판단 능력과 무관하고 메모리만 먹는다).
    """
    found = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "lm_head":
            continue
        if any(tag in name for tag in ("visual", "vision", "merger", "multi_modal")):
            continue
        if leaf in CANDIDATE_SUFFIXES:
            found.add(leaf)

    if found:
        return sorted(found)

    # 후보 이름이 하나도 없으면(구조가 다른 모델) lm_head·비전을 뺀 모든 Linear를 쓴다.
    # 이 경우 학습량이 커져 VRAM이 더 필요하니, 출력되는 개수를 꼭 확인할 것.
    fallback = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name and not any(
            tag in name for tag in ("visual", "vision", "merger")
        ):
            fallback.append(name)
    return fallback


def build_record(
    tokenizer,
    item: Dict[str, Any],
    with_rationale: bool = False,
    max_len: int = 512,
) -> Tuple[List[int], List[int]]:
    """학습 예제 하나를 (입력 토큰, 학습할 라벨)로 만든다.

    왜 라벨에 -100을 쓰는가: CrossEntropyLoss는 -100을 '무시'로 약속돼 있다.
    프롬프트 부분을 -100으로 덮으면 "질문을 외우기"가 아니라
    "질문을 읽고 정답 라벨을 말하기"만 학습한다. 이게 SFT의 핵심이다.

    왜 답 끝에 eos를 붙이는가: 모델이 답을 마치고 멈추는 법을 배워야 한다.
    eos를 안 붙이면 생성할 때 라벨 뒤로 계속 말을 이어 붙인다.
    """
    choices = item["choices"]
    prompt = build_prompt(tokenizer, item["instruction"], choices)

    target = item["answer"]
    if with_rationale and item.get("rationale"):
        # 이유까지 말하게 하면 설명력은 오르지만, 라벨 추출이 어려워진다.
        # 확률 채점(01/03)과 같이 쓰려면 기본값(라벨만)을 권한다.
        target = f"{target}\n이유: {item['rationale']}"

    eos = tokenizer.eos_token or "<|im_end|>"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # 답 뒤에 eos를 붙여 한 덩어리로 토큰화한다(경계에서 토큰이 합쳐지지 않게).
    target_ids = tokenizer(target + eos, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + target_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
    return input_ids, labels


def collate_lm(batch: Sequence[Tuple[List[int], List[int]]], pad_token_id: int):
    """학습 배치를 만든다. 라벨의 패딩은 -100(무시)으로 채운다."""
    max_len = max(len(x[0]) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids, labels_i) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(labels_i)] = torch.tensor(labels_i, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT 후 확률 채점")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data", default=str(DATA_DIR / "train.jsonl"), help="학습 데이터(JSONL)")
    parser.add_argument("--eval-data", default=str(DATA_DIR / "sample.jsonl"), help="채점용 데이터")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="12GB에서는 1을 권장")
    parser.add_argument("--grad-accum", type=int, default=8, help="실효 배치 = batch × accum")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="학습 데이터 앞 N개만(빠른 확인)")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--with-rationale", action="store_true", help="rationale 필드까지 학습")
    parser.add_argument("--merge", action="store_true", help="학습 후 어댑터를 본체에 합쳐 저장")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="4bit LoRA(9B용)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_out_dir()

    train_rows = read_jsonl(Path(args.data))
    if args.limit:
        train_rows = train_rows[: args.limit]
    eval_rows = read_jsonl(Path(args.eval_data))
    if args.eval_limit:
        eval_rows = eval_rows[: args.eval_limit]
    print(f"[data] 학습 {len(train_rows)}개 / 채점 {len(eval_rows)}개")

    tokenizer, model_id = load_tokenizer(args.model)
    model, model_id = load_model(model_id, load_in_4bit=args.four_bit, for_training=True)
    device = device_of(model)

    # ------------------------------------------------------------------
    # 1. LoRA 붙이기
    # ------------------------------------------------------------------
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict

    targets = find_lora_targets(model)
    print(f"[lora] 대상 모듈 {len(targets)}종: {', '.join(targets[:12])}"
          + (" ..." if len(targets) > 12 else ""))

    lora_config = LoraConfig(
        r=args.lora_r,                 # 저랭크 차원. 16이면 대부분의 판단 과제에 충분하다
        lora_alpha=args.lora_alpha,    # 스케일. 관례적으로 r의 2배를 준다(alpha/r = 2)
        lora_dropout=args.lora_dropout,
        bias="none",                   # bias는 학습하지 않는다(효과 대비 메모리 손해)
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] 학습 파라미터 {trainable/1e6:.2f}M / 전체 {total/1e9:.2f}B "
          f"({100 * trainable / max(total,1):.2f}%)")

    # gradient checkpointing: 12GB에서 긴 시퀀스를 학습하기 위한 필수 옵션.
    # 활성값을 저장하지 않고 역전파 때 다시 계산한다(느려지지만 메모리가 크게 준다).
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # 임베딩이 얼어 있어도 체크포인트가 기울기를 만들 수 있게 해 주는 옵션.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    except Exception as e:
        print(f"[warn] gradient checkpointing 설정 실패(무시 가능): {e}")

    # ------------------------------------------------------------------
    # 2. 학습 데이터 준비
    # ------------------------------------------------------------------
    records = [build_record(tokenizer, row, args.with_rationale, args.max_len) for row in train_rows]
    lengths = [len(r[0]) for r in records]
    print(f"[data] 토큰 길이 최소/평균/최대 = {min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}")

    # 첫 예제의 학습 대상 부분만 사람이 읽을 수 있게 보여 준다.
    sample_ids, sample_labels = records[0]
    shown = [t for t, l in zip(sample_ids, sample_labels) if l != -100]
    print("[data] 첫 예제에서 학습하는 부분:", repr(tokenizer.decode(shown)[:80]))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )

    # ------------------------------------------------------------------
    # 3. 학습 루프
    # ------------------------------------------------------------------
    model.train()
    pad_id = tokenizer.pad_token_id
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(records)))
        random.shuffle(order)
        batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]

        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.perf_counter()
        for bi, batch_idx in enumerate(batches):
            batch = [records[i] for i in batch_idx]
            input_ids, attention_mask, labels = collate_lm(batch, pad_id)
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            loss = outputs.loss
            # 배치를 여러 번 나눠 기울기를 모은다(실효 배치 = batch × accum).
            (loss / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(batches):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            epoch_loss += loss.item()

            if (bi + 1) % 5 == 0 or bi + 1 == len(batches):
                print(f"  epoch {epoch} [{bi+1}/{len(batches)}] loss {loss.item():.4f} | {vram_report()}")

        print(f"[epoch {epoch}] 평균 loss {epoch_loss / max(len(batches),1):.4f} "
              f"| {time.perf_counter()-t0:.1f}초 | step {global_step}")

    # ------------------------------------------------------------------
    # 4. 어댑터 저장 (+ 선택적 병합)
    # ------------------------------------------------------------------
    adapter_dir = out_dir / "lora_adapter"
    # dropout을 꺼 둔 추론 모드로 저장한다(다음에 읽었을 때와 조건을 같게 맞춰 두는 것).
    model.eval()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[saved] LoRA 어댑터: {adapter_dir}")

    merged = None
    if args.merge:
        if args.four_bit:
            # 4bit 가중치에 병합하면 오차가 누적된다. 정확한 병합은 bf16에서 할 것.
            print("[warn] 4bit 모델 병합은 정확도가 떨어질 수 있습니다.")
        merged = model.merge_and_unload()
        merged_dir = out_dir / "lora_merged"
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[saved] 병합 모델: {merged_dir} (약 8GB, 디스크 확인)")

    # ------------------------------------------------------------------
    # 5. 저장한 어댑터를 다시 읽어서 채점 — 01과 똑같은 함수로 확률을 낸다(비교 가능하게)
    #
    #    왜 저장만 하고 메모리에 있는 모델로 채점하지 않는가:
    #    "저장했다"는 말과 "복원된다"는 것은 다르다. 방금 좋아진 것은 메모리 안의 객체이고,
    #    내일 다시 쓰는 것은 디스크의 파일이다. 경로가 어긋나거나 peft 버전이 바뀌면
    #    학습은 끝까지 돌고도 그 결과를 다시 쓸 수 없다. 02가 헤드 파일을 저장 직후 되읽어
    #    가중치를 되돌려 넣어 보는 것과 같은 이유로, 03도 디스크에서 새로 읽은 어댑터로 채점한다.
    # ------------------------------------------------------------------
    eval_model: nn.Module = model
    eval_label = "학습한 모델(메모리)"

    if merged is not None:
        # 병합본에는 어댑터 가중치가 본체에 섞여 들어가 있다. 그 위에 어댑터를 또 얹으면
        # 같은 학습분이 두 번 적용되므로, 병합 경로에서는 어댑터를 다시 읽지 않고 병합본으로 채점한다.
        eval_model = merged
        eval_label = f"병합 모델({merged_dir.name}, 어댑터 병합 완료)"
        print(f"[eval] 채점 모델: {eval_label} — 재로드하지 않고 병합본을 그대로 쓴다")
    else:
        # 재로드가 실패해도 학습 결과를 잃지 않도록, 방금 학습한 LoRA 가중치를 메모리에 한 벌 복사해 둔다(수십 MB).
        lora_now = {k: v.detach().clone() for k, v in model.named_parameters() if "lora_" in k}
        base_model = None
        try:
            # model.get_base_model()을 그대로 넘기지 않는다: get_peft_model이 본체의 Linear 층을
            # LoRA 층으로 바꿔 놓아서, 그 위에 어댑터를 또 얹으면 층이 겹쳐 같은 학습분이 두 번 적용된다.
            # unload()로 LoRA 층을 떼어낸 깨끗한 본체를 만든 뒤, 디스크의 어댑터를 새 모델 객체로 읽는다.
            base_model = model.unload()
            reloaded = PeftModel.from_pretrained(base_model, str(adapter_dir))
            reloaded.eval()

            # "읽었다"를 값으로 확인한다: 디스크에서 올린 LoRA 가중치가 방금 학습한 것과 같은가.
            # (학습분이 빠진 채로 채점하는 상황을 여기서 잡는다. 02의 재로드 검증과 같은 취지)
            lora_loaded = {k: v.detach().cpu() for k, v in reloaded.named_parameters() if "lora_" in k}
            if lora_loaded.keys() != lora_now.keys():
                raise RuntimeError(
                    f"어댑터 키가 어긋납니다(디스크 {len(lora_loaded)}개 / 학습 {len(lora_now)}개)"
                )
            max_diff = max(
                (lora_loaded[k].float() - lora_now[k].detach().cpu().float()).abs().max().item()
                for k in lora_loaded
            )
            print(f"[reload] 성공: {adapter_dir} → 새 모델 객체"
                  f" (LoRA 텐서 {len(lora_loaded)}개, 학습본과 최대 차이 {max_diff:.2e})")
            eval_model = reloaded
            eval_label = f"재로드한 어댑터({adapter_dir.name})"
        except Exception as e:
            # 여기서 조용히 넘어가면 '학습이 빠진 모델'의 점수가 정상 결과처럼 보인다. 크게 남긴다.
            print(f"[warn] 어댑터 재로드 실패 — {type(e).__name__}: {e}")
            print(f"[warn] 학습 결과는 {adapter_dir} 에 저장돼 있습니다. 이번 채점은 학습 가중치로 계속합니다.")
            if base_model is None:
                # unload 전에 실패했다면 학습한 모델이 그대로 메모리에 있으므로 그것을 쓴다.
                eval_model, eval_label = model, "학습한 모델(메모리, 재로드 실패)"
            else:
                # unload 뒤 읽기만 실패한 경우: 메모리 스냅샷으로 학습분을 다시 붙인다.
                try:
                    restored = get_peft_model(base_model, lora_config)
                    set_peft_model_state_dict(restored, lora_now)
                    restored.eval()
                    eval_model = restored
                    eval_label = "메모리 스냅샷으로 되붙인 학습 모델(재로드 실패)"
                    print("[reload] 메모리 스냅샷으로 학습 가중치를 되붙였습니다.")
                except Exception as e2:
                    print(f"[warn] 복원도 실패 — {type(e2).__name__}: {e2}")
                    raise SystemExit("학습 가중치를 되살리지 못해 채점을 중단합니다.")

    eval_model.eval()
    pred_rows = []
    print(f"\n[eval] 채점 모델: {eval_label} / 항목 {len(eval_rows)}개")
    for idx, item in enumerate(eval_rows, start=1):
        choices = item["choices"]
        prompt = build_prompt(tokenizer, item["instruction"], choices)
        scores = score_choices(eval_model, tokenizer, prompt, choices)
        if len(scores) != len(choices):
            continue
        _, probs = softmax_over(scores, "mean")
        pred_rows.append(
            make_pred_row(
                item,
                choices,
                probs,
                scores=scores,
                model_id=model_id,
                method="lora_mean",
                n_prompt_tokens=len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
            )
        )
        if idx % 10 == 0:
            print(f"  ({idx}/{len(eval_rows)})")

    acc = sum(r["correct"] for r in pred_rows) / max(len(pred_rows), 1)
    print(f"[eval] 학습 후 정확도 {acc:.3f} (n={len(pred_rows)})")
    csv_path = write_preds_csv(pred_rows, out_dir / "preds_03_lora.csv")
    print(f"[saved] {csv_path}")
    print("        다음: python 05_eval.py  — 01/02/03/04를 같은 표에서 비교")


if __name__ == "__main__":
    main()

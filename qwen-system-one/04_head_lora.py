#!/usr/bin/env python3
"""
04_head_lora.py — 헤드 스왑 + LoRA: 판단 헤드와 어댑터를 '함께' 학습한다

이 파일이 답하는 질문
    "02는 헤드만 학습했고 03은 백본에 LoRA를 붙였다. 둘을 같이 하면?
     즉, 판단 쪽으로 조금 움직인 백본 위에 전용 헤드를 얹으면 정확도와 확신이 더 좋아지는가?"

실행하면 무엇을 보게 되는지
    (1) LoRA가 붙은 모듈 목록과 학습 파라미터 수(헤드 + LoRA) / 전체 대비 비율,
    (2) epoch별 학습 loss와 과제 그룹별 검증 정확도,
    (3) outputs/head_lora_cls.pt(헤드) + outputs/lora_adapter_04/(어댑터) — 둘 다 저장한 뒤
        파일을 다시 읽어 복원되는지 확인하고, 그 '디스크에서 읽은 가중치'로 채점한다,
    (4) data/sample.jsonl 28건 채점 결과 → outputs/preds_04_head_lora.csv.
        파일 이름이 preds_ 로 시작하므로 05_eval.py가 자동으로 같은 표에 넣는다.

개념: 헤드 스왑 + LoRA (02와 03의 결합)
    02는 lm_head(어휘 24만 차원)를 떼고 선택지 개수만 출력하는 헤드를 그 자리에 얹어
    '그 층만' 학습했다. 백본이 완전히 얼어 있어서 판단에 유리한 표현을 새로 만들 수는 없다.
    03은 LoRA로 백본의 1% 미만을 움직였지만, 출력은 여전히 lm_head를 거쳐
    '선택지 문구의 로그확률'을 재는 방식이다.
    여기서는 두 장치를 합친다.
      · 출력 쪽: lm_head → 판단 헤드 (02)
      · 표현 쪽: 백본의 LoRA 어댑터 (03)
    헤드와 LoRA를 한 옵티마이저에 넣고 함께 학습하므로, 백본이 '판단하기 좋은 방향'으로
    조금 움직이는 동시에 헤드가 그 표현을 읽는 법을 배운다. 어느 한쪽만 학습할 때보다
    같은 데이터로 더 멀리 갈 수 있고, 학습량은 여전히 전체의 1% 안쪽이다.

사용 예
    python 04_head_lora.py                                 # 기본: train.jsonl 학습 → sample.jsonl 채점
    python 04_head_lora.py --epochs 1 --limit 16           # 5분 안에 도는지 먼저 확인
    python 04_head_lora.py --model Qwen/Qwen3.5-2B         # 12GB에서 가장 편한 크기
    python 04_head_lora.py --eval-data ""                  # sample 대신 학습 검증 분할로 채점
"""

from __future__ import annotations

import argparse
import random
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from common import (
    DATA_DIR,
    DEFAULT_MODEL_ID,
    device_of,
    drop_lm_head,
    ensure_out_dir,
    group_by_task,
    hidden_size_of,
    load_model,
    load_tokenizer,
    make_pred_row,
    read_jsonl,
    resolve_text_backbone,
    set_seed,
    vram_report,
    write_preds_csv,
)

# 번호로 시작하는 파일명은 `import 02_head_swap` 처럼 import 문으로 쓸 수 없다
# (모듈 이름이 식별자 규칙을 따르지 않는다). 그래서 importlib로 읽는다.
# 02/03의 코드를 복사해 오지 않는 이유: 같은 로직이 두 파일에 생기면 한쪽만 고쳐져서
# 결과가 조용히 어긋난다. 헤드 구조·데이터 준비·재로드 검증은 02 한 곳만 고치면 되게 둔다.
_s02 = import_module("02_head_swap")
_s03 = import_module("03_lora_sft")

BackboneWithHead = _s02.BackboneWithHead      # 백본 + 판단 헤드 (02와 같은 구조)
JudgmentHead = _s02.JudgmentHead
collate = _s02.collate                        # (토큰 id, 어텐션 마스크, 라벨) 배치 만들기
evaluate = _s02.evaluate                      # 검증 정확도/MSE 계산
load_heads = _s02.load_heads
prepare_examples = _s02.prepare_examples
save_heads = _s02.save_heads
find_lora_targets = _s03.find_lora_targets    # lm_head·비전 인코더를 뺀 대상 층 탐색

# 05_eval.py가 방법 이름으로 구분한다. CSV 한 파일 안에서는 한 값으로 통일해야
# 05_eval이 이 파일을 '한 방법'으로 집계한다(그래서 과제 그룹과 무관하게 고정).
METHOD = "head_lora"

# 산출물 이름. 02(head_*.pt)·03(lora_adapter/)와 겹치지 않게 따로 둔다.
# 겹치면 나중에 실행한 쪽이 앞의 결과를 덮어써 비교가 사라진다.
ADAPTER_DIR_NAME = "lora_adapter_04"

# LoRA 학습률 = 헤드 학습률 × 이 값.
# 왜 나누는가: 헤드는 무작위로 초기화된 새 층이라 큰 보폭(1e-3 근처)이 필요하고,
# LoRA는 이미 학습된 백본에 얹히는 작은 보정이라 2e-4 근처가 안전하다(03의 기본값).
# 같은 학습률을 주면 LoRA 쪽이 너무 크게 흔들려 백본의 기존 능력이 깨진다.
LORA_LR_SCALE = 0.2


# ---------------------------------------------------------------------------
# 1. 잡다한 도구
# ---------------------------------------------------------------------------


def _param_count(module: nn.Module) -> int:
    """모듈이 가진 파라미터 개수(학습 여부와 무관)."""
    return sum(p.numel() for p in module.parameters())


def split_train_val(
    examples: Sequence[Dict[str, Any]], eval_ratio: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """과제 그룹 하나의 항목을 학습/검증으로 나눈다(02와 같은 방식).

    왜 나누는가: 같은 데이터로 학습하고 평가하면 '외운 것'을 실력으로 착각한다.
    검증 항목은 학습에 한 번도 쓰지 않고, epoch마다 여기서만 정확도를 잰다.
    """
    shuffled = list(examples)
    random.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_ratio))
    eval_set = shuffled[:n_eval]
    train_set = shuffled[n_eval:] or shuffled[:1]   # 데이터가 적을 때 최소 1개는 남긴다
    return train_set, eval_set


def enable_input_grads(backbone: nn.Module) -> str:
    """임베딩 출력에 기울기 연결을 붙인다(gradient checkpointing의 필수 조건).

    왜 필요한가: LoRA를 붙이면 백본의 원래 가중치는 전부 requires_grad=False가 된다.
    그러면 첫 층(임베딩)의 출력도 기울기를 갖지 않는데, gradient checkpointing은
    역전파 때 '입력을 다시 넣어' 중간값을 재계산하므로 입력 쪽에 기울기 연결이 있어야 한다.
    없으면 loss.backward()가 "None of the inputs have requires_grad=True"로 멈춘다.

    왜 백본에 직접 하는가: 우리는 모델 전체가 아니라 백본을 직접 호출한다(02와 같은 이유 —
    lm_head를 거치지 않는다). 그래서 훅도 실제로 호출되는 그 모듈의 임베딩에 걸어야 한다.
    """
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
        return "임베딩 출력에 기울기 연결(enable_input_require_grads)"

    embed = backbone.get_input_embeddings() if hasattr(backbone, "get_input_embeddings") else None
    if embed is None:
        return "임베딩을 찾지 못해 건너뜀(체크포인팅에서 경고가 날 수 있음)"

    def _hook(_module, _inputs, output):
        if torch.is_tensor(output):
            output.requires_grad_(True)

    embed.register_forward_hook(_hook)
    return "임베딩 출력에 기울기 연결(직접 훅 등록)"


# ---------------------------------------------------------------------------
# 2. 학습 — 헤드와 LoRA를 한 루프에서
# ---------------------------------------------------------------------------


def train_heads_with_lora(
    train_sets: Dict[str, List[Dict[str, Any]]],
    val_sets: Dict[str, List[Dict[str, Any]]],
    wrappers: Dict[str, "BackboneWithHead"],
    lora_params: List[torch.nn.Parameter],
    head_kind: str,
    pad_token_id: int,
    args,
    device,
) -> List[Tuple[str, Dict[str, float]]]:
    """과제 그룹별 헤드를, 하나의 LoRA 어댑터 위에서 함께 학습한다.

    왜 그룹마다 차례로 학습하지 않는가(02와 다른 점):
        02는 백본이 얼어 있어서 그룹별로 차례로 학습해도 서로 영향이 없었다.
        여기서는 LoRA 어댑터를 하나만 두고 모든 그룹이 공유한다. '예/아니오' 헤드를
        학습한 뒤 '긍정/부정/중립' 헤드를 이어서 학습하면 어댑터가 두 번째 과제 쪽으로
        움직여 버려서, 첫 헤드는 이미 바뀐 어댑터 위에서 채점된다(학습할 때의 조건과 다름).
        그래서 모든 그룹의 항목을 한 목록에 섞어 두고, 같은 epoch 안에서 함께 갱신한다.

    왜 옵티마이저를 하나로 두는가: 헤드와 LoRA가 같은 방향으로 움직여야 '백본이 판단하기
    좋은 표현으로 변하면서 헤드가 그것을 읽는' 관계가 만들어진다. 학습률만 group별로 나눈다.
    """
    head_params: List[torch.nn.Parameter] = []
    for wrapper in wrappers.values():
        head_params.extend(wrapper.head.parameters())

    param_groups = [
        {"params": head_params, "lr": args.lr},                        # 새로 만든 헤드
        {"params": lora_params, "lr": args.lr * LORA_LR_SCALE},         # 백본 위의 작은 보정
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)

    loss_fns: Dict[str, nn.Module] = {
        name: (nn.MSELoss() if head_kind == "reg" else nn.CrossEntropyLoss()) for name in wrappers
    }

    # (과제 이름, 항목)을 한 줄로 세워 두고 epoch마다 섞는다. 이렇게 해야 한 배치 안에
    # 여러 과제가 섞여도 각 항목이 자기 과제의 헤드로 계산된다.
    pool: List[Tuple[str, Dict[str, Any]]] = [
        (name, example) for name, examples in train_sets.items() for example in examples
    ]
    bs = max(1, args.batch_size)
    last_metrics: List[Tuple[str, Dict[str, float]]] = []

    for epoch in range(1, args.epochs + 1):
        # wrapper.train()은 공유 백본까지 train 모드로 바꾼다(드롭아웃이 켜져야 학습이 된다).
        for wrapper in wrappers.values():
            wrapper.train()

        order = list(range(len(pool)))
        random.shuffle(order)
        batches = [order[i : i + bs] for i in range(0, len(order), bs)]

        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.perf_counter()
        for bi, batch_idx in enumerate(batches):
            # 배치 안에서 과제별로 묶는다. 헤드는 과제마다 출력 차원이 달라 섞어 넣을 수 없다.
            by_group: Dict[str, List[Dict[str, Any]]] = {}
            for i in batch_idx:
                name, example = pool[i]
                by_group.setdefault(name, []).append(example)

            batch_loss = 0.0
            for name, batch in by_group.items():
                input_ids, attention_mask, labels = collate(batch, pad_token_id)
                # freeze_backbone=False인 wrapper라 이 호출로 LoRA까지 기울기가 흐른다.
                logits = wrappers[name](input_ids.to(device), attention_mask.to(device))
                target = labels.to(device)
                loss = loss_fns[name](
                    logits.float(), target.float() if head_kind == "reg" else target
                )
                # 과제마다 항목 수가 다르므로 '한 항목의 무게'가 같아지도록 나눠 더한다.
                # (fidelity 35개, sentiment 21개라면 그냥 더할 때 35쪽이 1.7배 무거워진다)
                share = len(batch) / len(batch_idx)
                (loss * share / args.grad_accum).backward()
                batch_loss += loss.item() * share

            # gradient accumulation: 배치 1로 여러 번 계산해 기울기를 모은 뒤 한 번 갱신한다
            # (실효 배치 = batch_size × grad_accum). 12GB에서 큰 배치를 흉내 내는 표준 방법.
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(batches):
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += batch_loss

        # epoch마다 검증: 학습에 쓰지 않은 항목으로만 잰다.
        last_metrics = []
        parts: List[str] = []
        for name, wrapper in wrappers.items():
            if not val_sets.get(name):
                continue
            metrics = evaluate(wrapper, val_sets[name], device, pad_token_id, head_kind)[2]
            last_metrics.append((name, metrics))
            if head_kind == "reg":
                parts.append(f"{name} 검증 MSE {metrics['mse']:.4f}")
            else:
                parts.append(f"{name} 검증 정확도 {metrics['acc']:.3f}")

        print(f"  epoch {epoch}: 평균 loss {epoch_loss / max(len(batches), 1):.4f} | "
              + " | ".join(parts) + f" | {time.perf_counter() - t0:.1f}초 | {vram_report()}")

    return last_metrics


# ---------------------------------------------------------------------------
# 3. 채점 — 05_eval이 읽을 CSV 행 만들기
# ---------------------------------------------------------------------------


def predict_rows(
    score_items: Sequence[Dict[str, Any]],
    wrappers: Dict[str, "BackboneWithHead"],
    tokenizer,
    pad_token_id: int,
    device,
    model_id: str,
    head_kind: str,
    max_len: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """채점 항목을 헤드로 예측해 05_eval용 pred 행을 만든다.

    왜 선택지별 로그확률이 아니라 헤드 출력인가: 02와 같은 방식이기 때문이다.
    헤드는 '선택지 개수'만큼 로짓을 내고, 그 안에서 softmax를 취하면
    "이 항목에서 각 선택지일 확률"이 된다. 05_eval은 이 확률로 Brier/ECE를 계산하므로
    로짓을 반드시 확률로 정규화해서 넘긴다(정규화하지 않으면 보정 지표가 무의미해진다).

    항목 → 행 형식은 common.make_pred_row 하나로 통일한다. 02도 같은 함수를 쓰므로
    CSV 열이 정확히 같아지고, 05_eval이 두 파일을 같은 표에 놓을 수 있다.
    """
    rows: List[Dict[str, Any]] = []
    skipped = 0
    total = len(score_items)

    for idx, item in enumerate(score_items, start=1):
        group = item.get("task") or "|".join(item["choices"])
        wrapper = wrappers.get(group)
        # 학습한 헤드가 없는 과제이거나, 선택지 수가 헤드 출력 차원과 다르면 건너뛴다.
        # (억지로 넣으면 엉뚱한 차원의 확률이 CSV에 섞여 평가가 조용히 틀어진다)
        if wrapper is None or len(item["choices"]) != wrapper.head.linear.out_features:
            skipped += 1
            continue

        example = prepare_examples([item], tokenizer, head_kind, max_len)[0]
        input_ids, attention_mask, _ = collate([example], pad_token_id)
        with torch.no_grad():
            logits = wrapper(input_ids.to(device), attention_mask.to(device))
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()

        rows.append(
            make_pred_row(
                item,
                item["choices"],
                probs,
                scores={},          # 헤드는 로그확률을 내지 않는다(02와 같은 형식 유지)
                model_id=model_id,
                method=METHOD,
                n_prompt_tokens=len(example["input_ids"]),
            )
        )
        if idx % 10 == 0 or idx == total:
            print(f"  ({idx}/{total})")

    return rows, skipped


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자 정의. 02(헤드 학습)와 03(LoRA)의 기본값을 그대로 이어받는다."""
    parser = argparse.ArgumentParser(description="헤드 교체 + LoRA 결합 학습 데모")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data", default=str(DATA_DIR / "train.jsonl"), help="학습 데이터(JSONL)")
    parser.add_argument(
        "--eval-data",
        default=str(DATA_DIR / "sample.jsonl"),
        help="채점용 데이터(기본: sample.jsonl 28건). 빈 값('')이면 학습 검증 분할로 채점",
    )
    parser.add_argument("--head", choices=["cls", "reg"], default="cls")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1, help="12GB에서는 1을 권장")
    parser.add_argument("--grad-accum", type=int, default=8, help="실효 배치 = batch × accum")
    parser.add_argument("--lr", type=float, default=1e-3, help="헤드 학습률. LoRA는 이 값의 0.2배")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1, help="판단 헤드의 dropout")
    parser.add_argument("--eval-ratio", type=float, default=0.3, help="학습 데이터에서 검증으로 뺄 비율")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="학습 데이터 앞 N개만(빠른 확인)")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="4bit LoRA(9B용)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    set_seed(args.seed)
    out_dir = ensure_out_dir()

    train_rows = read_jsonl(Path(args.data))
    if args.limit:
        train_rows = train_rows[: args.limit]
    eval_rows = read_jsonl(Path(args.eval_data)) if args.eval_data else []
    score_source = Path(args.eval_data).name if eval_rows else "학습 데이터의 검증 분할"
    print(f"[data] 학습 {len(train_rows)}개({Path(args.data).name}) / 채점 {score_source}")

    tokenizer, model_id = load_tokenizer(args.model)
    model, model_id = load_model(model_id, load_in_4bit=args.four_bit, for_training=True)

    # ------------------------------------------------------------------
    # 1. LoRA 붙이기(03의 순서) → lm_head 떼기(02의 순서)
    # ------------------------------------------------------------------
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict

    targets = find_lora_targets(model)
    print(f"[lora] 대상 모듈 {len(targets)}종: {', '.join(targets[:12])}"
          + (" ..." if len(targets) > 12 else ""))

    lora_config = LoraConfig(
        r=args.lora_r,                 # 저랭크 차원. 16이면 대부분의 판단 과제에 충분하다
        lora_alpha=args.lora_alpha,    # 스케일. 관례대로 r의 2배(alpha/r = 2)
        lora_dropout=args.lora_dropout,
        bias="none",                   # bias는 학습하지 않는다(효과 대비 메모리 손해)
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)

    # 왜 LoRA를 먼저 붙이는가: peft는 모델 구조를 훑어 대상 층을 바꿔 끼운다. 모델이
    # 원래 모양 그대로일 때가 가장 안전하다. lm_head는 대상에서 빠져 있으므로,
    # 붙인 다음에 02처럼 떼어 내면 메모리도 그대로 돌아온다.
    base_model = model.get_base_model()
    backbone, lm_head_ref = resolve_text_backbone(base_model)
    # 참조를 남겨 두면 lm_head 가중치가 메모리에 살아남는다(9B는 약 2GB).
    del lm_head_ref
    freed = drop_lm_head(base_model)
    if freed:
        print(f"[head-swap] lm_head 제거: {freed/1e6:.1f}M 파라미터 확보")

    lora_params = [
        p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad
    ]
    print(f"[lora] LoRA 학습 파라미터: 텐서 {len(lora_params)}개 / "
          f"{sum(p.numel() for p in lora_params)/1e6:.2f}M")

    hidden_size = hidden_size_of(backbone)
    pad_token_id = tokenizer.pad_token_id
    device = device_of(model)
    print(f"[head-swap] 백본={type(backbone).__name__}, hidden={hidden_size}, device={device}")

    # gradient checkpointing: 층마다 활성값을 저장하지 않고 역전파 때 다시 계산한다.
    # 02의 last_block 학습과 같은 이유로 여기서도 필수다(헤드+LoRA를 함께 학습하므로
    # 활성값까지 들고 있으면 12GB에서 4B는 버티지 못한다).
    try:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print(f"[ckpt] gradient checkpointing 켬 + {enable_input_grads(backbone)}")
    except Exception as e:
        print(f"[warn] gradient checkpointing 설정 실패(무시 가능, 대신 VRAM을 더 씀): {e}")

    # ------------------------------------------------------------------
    # 2. 과제 그룹별 데이터·헤드 준비 (02와 같은 단위)
    # ------------------------------------------------------------------
    groups = group_by_task(train_rows)
    train_sets: Dict[str, List[Dict[str, Any]]] = {}
    val_sets: Dict[str, List[Dict[str, Any]]] = {}
    wrappers: Dict[str, "BackboneWithHead"] = {}
    heads: Dict[str, nn.Module] = {}

    meta: Dict[str, Any] = {
        "model_id": model_id,
        "hidden_size": hidden_size,
        "head_kind": args.head,
        "method": METHOD,
        "num_classes": {},
        "choices": {},
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": targets,
            "adapter_dir": ADAPTER_DIR_NAME,
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for group_name, group_rows in groups.items():
        examples = prepare_examples(group_rows, tokenizer, args.head, args.max_len)
        if args.head == "reg":
            num_classes = None
        else:
            num_classes = len(group_rows[0]["choices"])

        train_set, eval_set = split_train_val(examples, args.eval_ratio)

        # wrapper를 그룹마다 만들지만 backbone은 같은 객체를 나눠 쓴다.
        # 헤드만 그룹별로 따로 있고(LoRA는 하나), freeze_backbone=False라 기울기가 백본까지 흐른다.
        wrapper = BackboneWithHead(
            backbone, hidden_size, num_classes, dropout=args.dropout, freeze_backbone=False
        )
        # 02와 같은 이유로 헤드만 지정 장치로 옮긴다(device_map/4bit 모델은 통째로 못 옮긴다).
        wrapper.head.to(device)

        train_sets[group_name] = train_set
        val_sets[group_name] = eval_set
        wrappers[group_name] = wrapper
        heads[group_name] = wrapper.head
        meta["num_classes"][group_name] = num_classes
        meta["choices"][group_name] = list(group_rows[0]["choices"])
        print(f"[{group_name}] 학습/검증 = {len(train_set)}/{len(eval_set)}개, "
              f"출력 차원={num_classes if num_classes else 1}")

    head_n = sum(_param_count(w.head) for w in wrappers.values())
    lora_n = sum(p.numel() for p in lora_params)
    trainable_n = head_n + lora_n
    total_n = _param_count(model) + head_n     # model에는 LoRA가 포함되고 헤드는 별도 모듈이다
    print(f"[params] 헤드 {head_n/1e6:.3f}M + LoRA {lora_n/1e6:.2f}M = 학습 {trainable_n/1e6:.2f}M / "
          f"전체 {total_n/1e9:.2f}B ({100 * trainable_n / max(total_n, 1):.2f}%)")

    # ------------------------------------------------------------------
    # 3. 학습
    # ------------------------------------------------------------------
    print(f"\n[train] 헤드 lr {args.lr} / LoRA lr {args.lr * LORA_LR_SCALE:.1e}"
          f" (LoRA는 사전학습된 백본 위라 보폭을 작게)")
    last_metrics = train_heads_with_lora(
        train_sets, val_sets, wrappers, lora_params, args.head, pad_token_id, args, device
    )

    print("\n[요약] 최종 검증 성적 (학습에서 떼어 둔 항목. sample.jsonl이 아니다)")
    for name, metrics in last_metrics:
        n_val = len(val_sets.get(name, []))
        if args.head == "reg":
            print(f"  - {name}: MSE {metrics['mse']:.4f} (n={n_val})")
        else:
            print(f"  - {name}: 정확도 {metrics['acc']:.3f} (n={n_val})")

    # ------------------------------------------------------------------
    # 4. 산출물 저장 (헤드 + 어댑터) — 02/03과 겹치지 않는 이름
    # ------------------------------------------------------------------
    head_path = out_dir / f"head_lora_{args.head}.pt"
    save_heads(head_path, heads, meta)
    print(f"\n[saved] 헤드 가중치: {head_path}")

    # 저장한 헤드를 실제로 다시 읽어 새 헤드에 되돌려 넣어 본다.
    # 왜: 저장이 '됐다'는 말과 복원이 '된다'는 것은 다르다(02와 같은 검증).
    payload = load_heads(head_path)
    heads_from_disk: Dict[str, nn.Module] = {}
    for name in heads:
        probe = JudgmentHead(meta["hidden_size"], meta["num_classes"][name]).to(device)
        try:
            probe.load_state_dict(payload["heads"][name])
            heads_from_disk[name] = probe
        except Exception as e:
            print(f"  ! {name} 재로드 실패: {e}")
    print(f"[reload] {head_path.name} 재로드 "
          f"{'성공' if len(heads_from_disk) == len(heads) else '실패'} "
          f"(헤드 {len(heads_from_disk)}/{len(heads)}개)")

    model.eval()
    adapter_dir = out_dir / ADAPTER_DIR_NAME
    model.save_pretrained(adapter_dir)
    print(f"[saved] LoRA 어댑터: {adapter_dir}")

    # 어댑터 재로드: 03과 같은 방식. unload()로 LoRA를 떼어 낸 깨끗한 본체를 만든 뒤,
    # 디스크의 어댑터를 '새 모델 객체'로 읽는다. 읽은 뒤 백본 객체는 같은 것이므로
    # 이어서 그 가중치로 채점할 수 있다(= 채점이 디스크 왕복을 실제로 거친다).
    lora_now = {k: v.detach().clone() for k, v in model.named_parameters() if "lora_" in k}
    adapter_label = "학습한 모델(메모리)"
    base_for_reload = None
    try:
        base_for_reload = model.unload()
        reloaded = PeftModel.from_pretrained(base_for_reload, str(adapter_dir))
        reloaded.eval()

        # "읽었다"를 값으로 확인한다: 디스크에서 올린 LoRA가 방금 학습한 것과 같은가.
        # (학습분이 빠진 채로 채점하는 상황을 여기서 잡는다)
        lora_loaded = {k: v.detach().cpu() for k, v in reloaded.named_parameters() if "lora_" in k}
        if lora_loaded.keys() != lora_now.keys():
            raise RuntimeError(
                f"어댑터 키가 어긋납니다(디스크 {len(lora_loaded)}개 / 학습 {len(lora_now)}개)"
            )
        max_diff = max(
            (lora_loaded[k].float() - lora_now[k].detach().cpu().float()).abs().max().item()
            for k in lora_loaded
        )
        adapter_label = f"재로드한 어댑터({ADAPTER_DIR_NAME})"
        print(f"[reload] LoRA 어댑터 재로드 성공: 텐서 {len(lora_loaded)}개, "
              f"학습본과 최대 차이 {max_diff:.2e}")
    except Exception as e:
        # 여기서 조용히 넘어가면 'LoRA가 빠진 모델'의 점수가 정상 결과처럼 보인다. 크게 남긴다.
        print(f"[warn] 어댑터 재로드 실패 — {type(e).__name__}: {e}")
        print(f"[warn] 학습 결과는 {adapter_dir} 에 저장돼 있습니다.")
        if base_for_reload is None:
            adapter_label = "학습한 모델(메모리, 재로드 실패)"
        else:
            try:
                restored = get_peft_model(base_for_reload, lora_config)
                set_peft_model_state_dict(restored, lora_now)
                restored.eval()
                adapter_label = "메모리 스냅샷으로 되붙인 모델(재로드 실패)"
                print("[reload] 메모리 스냅샷으로 학습 가중치를 되붙였습니다.")
            except Exception as e2:
                raise SystemExit(f"학습 가중치를 되살리지 못해 채점을 중단합니다: {e2}")

    # ------------------------------------------------------------------
    # 5. 채점 (05_eval이 읽을 CSV)
    # ------------------------------------------------------------------
    if args.head == "reg":
        print("[note] 회귀 헤드는 선택지 확률을 내지 않으므로 CSV를 만들지 않습니다"
              "(05_eval의 보정 평가 대상이 아님).")
        return

    # 헤드도 디스크에서 읽은 것으로 바꿔 끼워, 채점이 '저장된 산출물'만으로 이뤄지게 한다.
    if len(heads_from_disk) == len(heads):
        for name, probe in heads_from_disk.items():
            wrappers[name].head = probe
    else:
        print("[warn] 헤드 재로드가 불완전해 학습 중이던 헤드로 채점합니다.")

    for wrapper in wrappers.values():
        wrapper.eval()

    # 채점 항목: 기본은 --eval-data(sample.jsonl 28건). 없으면 학습 검증 분할.
    if eval_rows:
        score_items: List[Dict[str, Any]] = list(eval_rows)
    else:
        score_items = [example for name in val_sets for example in val_sets[name]]

    print(f"\n[eval] 채점 항목 {len(score_items)}개 / 가중치: {adapter_label}")
    pred_rows, skipped = predict_rows(
        score_items, wrappers, tokenizer, pad_token_id, device, model_id, args.head, args.max_len
    )

    if not pred_rows:
        print("[warn] 채점할 항목이 없습니다. --eval-data 경로와 과제 이름을 확인하세요.")
        return

    accuracy = sum(row["correct"] for row in pred_rows) / len(pred_rows)
    print(f"[eval] 채점 정확도 {accuracy:.3f} (n={len(pred_rows)})")
    for name in wrappers:
        subset = [row for row in pred_rows if row["task"] == name]
        if subset:
            print(f"  → {name}: 정확도 {sum(r['correct'] for r in subset)/len(subset):.3f} "
                  f"(n={len(subset)})")
    if skipped:
        print(f"[warn] 학습한 헤드가 없거나 선택지 수가 다른 {skipped}건은 건너뛰었습니다.")

    # 파일 이름이 preds_*.csv 이므로 05_eval.py가 폴더를 훑을 때 자동으로 집어 간다.
    csv_path = write_preds_csv(pred_rows, out_dir / "preds_04_head_lora.csv")
    print(f"[saved] 예측 CSV: {csv_path}")
    print("        다음: python 05_eval.py  — 01/02/03/04 네 방법을 한 표에서 비교")


if __name__ == "__main__":
    main()

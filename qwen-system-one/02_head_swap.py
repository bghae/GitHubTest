#!/usr/bin/env python3
"""
02_head_swap.py — LM 헤드를 떼고 판단용 헤드를 얹어 '헤드만' 학습한다

이 파일이 답하는 질문
    "24만 개 단어를 예측하던 출력층을 '선택지 개수'만 출력하는 작은 층으로 바꾸고,
     그 층만 학습하면 판단 정확도가 얼마나 오르는가?"

실행하면 무엇을 보게 되는지
    (1) 백본은 얼려 두고 헤드만 학습할 때의 epoch별 loss 감소,
    (2) 학습에 쓰지 않은 검증 항목의 정확도,
    (3) outputs/head.pt (헤드 가중치)와 outputs/preds_02_head.csv (04용 예측).

개념: linear probe
    Qwen3.5의 마지막 층 hidden state에는 "이 문장이 어떤 판단을 요구하는지"에 대한
    정보가 이미 압축돼 있다. 그래서 그 벡터 위에 선형층 하나만 얹고 학습해도
    상당한 정확도가 나온다. 백본을 건드리지 않으므로 12GB에서도 4B를 쓸 수 있다.

사용 예
    python 02_head_swap.py --mode linear_probe          # 기본: 헤드만 학습(가볍다)
    python 02_head_swap.py --mode last_block --unfreeze-last-blocks 1   # 마지막 블록까지
    python 02_head_swap.py --head reg --data data/reg_sample.jsonl      # 회귀 헤드(별도 데이터)
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


# ---------------------------------------------------------------------------
# 1. 백본 + 헤드 구조
# ---------------------------------------------------------------------------


def pool_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """시퀀스의 '마지막 실제 토큰' 위치의 hidden state를 뽑는다.

    왜 마지막 토큰인가: decoder-only 모델은 다음 토큰을 예측하도록 학습됐기 때문에,
    마지막 위치의 벡터가 "지금까지의 문장 전체를 읽고 다음에 무엇이 와야 하는지"를
    가장 많이 담고 있다. 분류에서 이 위치를 쓰는 것이 표준이다.
    (평균 pooling은 정보가 흐려져 성능이 조금 떨어지는 경우가 많다)

    왜 attention_mask로 위치를 세는가: 오른쪽 패딩이면 마지막 실제 토큰은
    mask 합 - 1 이고, 왼쪽 패딩이면 위치가 달라진다. mask 합으로 세면 두 경우 모두 맞는다.
    """
    last_index = attention_mask.sum(dim=1) - 1                       # (batch,)
    index = last_index.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))  # gather용 형태
    return hidden.gather(1, index).squeeze(1)                        # (batch, hidden)


class JudgmentHead(nn.Module):
    """선택지 개수(분류) 또는 1개 값(회귀)을 출력하는 작은 층.

    왜 hidden을 float32로 올리는가: 백본은 bf16으로 돌지만, 작은 헤드를 bf16으로
    학습하면 기울기가 거칠어 loss가 잘 안 내려간다. 헤드만 fp32로 두면 정확도가 안정된다.
    """

    def __init__(self, hidden_size: int, num_classes: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # num_classes가 None이면 회귀(출력 1개)
        self.linear = nn.Linear(hidden_size, num_classes if num_classes else 1)
        self.is_regression = num_classes is None

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.dropout(pooled.float()))
        # 회귀는 (batch,) 형태로 눌러서 MSE와 바로 비교할 수 있게 한다.
        return out.squeeze(-1) if self.is_regression else out


class BackboneWithHead(nn.Module):
    """LM 헤드를 제거한 백본 + 우리가 만든 헤드.

    왜 백본을 직접 호출하는가: 원래 모델의 forward는 lm_head까지 계산해
    어휘 24만 차원 로짓을 만든다(4B에서 약 1GB 메모리, 계산도 낭비).
    백본만 호출하면 hidden state까지만 나오고 그 뒤는 우리 헤드가 처리한다.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_classes: Optional[int],
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.head = JudgmentHead(hidden_size, num_classes, dropout)

    def _run_backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """백본 호출. use_cache 인자를 모르는 구현이면 한 번 더 시도한다."""
        try:
            return self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except TypeError:
            return self.backbone(input_ids=input_ids, attention_mask=attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            # 얼린 백본은 기울기를 계산할 필요가 없다. no_grad로 감싸면
            # 중간 활성값을 저장하지 않아 VRAM이 크게 줄어든다(4B를 12GB에서 돌리는 비결).
            with torch.no_grad():
                out = self._run_backbone(input_ids, attention_mask)
        else:
            out = self._run_backbone(input_ids, attention_mask)
        pooled = pool_last_token(out.last_hidden_state, attention_mask)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# 2. 데이터 준비
# ---------------------------------------------------------------------------


def prepare_examples(
    rows: Sequence[Dict[str, Any]],
    tokenizer,
    head_kind: str,
    max_len: int = 512,
) -> List[Dict[str, Any]]:
    """JSONL 항목을 (토큰 id, 라벨) 형태로 바꾼다.

    왜 truncation=True: Qwen3.5는 26만 토큰까지 받지만, 우리 예제는 짧다.
    길이를 512로 자르면 배치 메모리를 예측할 수 있고 OOM 위험이 줄어든다.
    """
    examples: List[Dict[str, Any]] = []
    for item in rows:
        choices = item["choices"]
        prompt = build_prompt(tokenizer, item["instruction"], choices)
        # add_special_tokens=False: chat template이 이미 특수 토큰을 넣었다.
        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
        )
        if head_kind == "reg":
            if "target" not in item:
                raise KeyError(
                    f"{item.get('id')}: 회귀 헤드는 각 항목에 숫자 필드 'target'이 필요합니다."
                )
            label: Any = float(item["target"])
        else:
            label = choices.index(item["answer"])

        examples.append(
            {
                "id": item.get("id", ""),
                "task": item.get("task", ""),
                "instruction": item.get("instruction", ""),
                "choices": choices,
                "answer": item["answer"],
                "input_ids": enc["input_ids"],
                "label": label,
            }
        )
    return examples


def collate(batch: Sequence[Dict[str, Any]], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """길이가 다른 문장들을 한 배치로 묶는다(오른쪽 패딩).

    왜 오른쪽 패딩인가: 학습에서는 마지막 실제 토큰을 mask 합으로 찾으므로
    어느 쪽이든 되지만, 오른쪽 패딩이 토큰 인덱스를 읽기 쉽다.
    (생성할 때는 반대로 왼쪽 패딩을 써야 한다 — 00/03 참고)
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.float if isinstance(batch[0]["label"], float) else torch.long)

    for i, ex in enumerate(batch):
        ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1        # 패딩은 0 → 풀링/어텐션에서 무시된다
        labels[i] = ex["label"]
    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# 3. 학습 / 평가
# ---------------------------------------------------------------------------


def evaluate(wrapper: BackboneWithHead, examples: Sequence[Dict[str, Any]], device, pad_token_id: int, head_kind: str):
    """검증 항목의 예측 확률과 정확도를 계산한다."""
    wrapper.eval()
    all_probs: List[List[float]] = []
    all_labels: List[Any] = []
    with torch.no_grad():
        for start in range(0, len(examples), 16):
            batch = examples[start : start + 16]
            input_ids, attention_mask, labels = collate(batch, pad_token_id)
            logits = wrapper(input_ids.to(device), attention_mask.to(device))
            if head_kind == "reg":
                probs = logits.detach().cpu().tolist()      # 회귀는 확률이 아니라 예측값
                all_probs.extend([[p] for p in probs])
            else:
                probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
                all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    if head_kind == "reg":
        mse = sum((p[0] - y) ** 2 for p, y in zip(all_probs, all_labels)) / max(len(all_labels), 1)
        return all_probs, all_labels, {"mse": mse}
    correct = sum(int(max(range(len(p)), key=lambda i: p[i]) == int(y)) for p, y in zip(all_probs, all_labels))
    return all_probs, all_labels, {"acc": correct / max(len(all_labels), 1)}


def train_one_group(
    group_name: str,
    examples: List[Dict[str, Any]],
    backbone: nn.Module,
    hidden_size: int,
    pad_token_id: int,
    args,
    device,
):
    """선택지 집합 하나에 대한 헤드를 학습한다.

    왜 집합별로 따로 만드는가: '예/아니오'는 출력 2개, '긍정/부정/중립'은 3개다.
    출력 차원이 다른 문제를 한 헤드로 학습할 수 없으므로 문제 종류마다 헤드를 하나씩 둔다.
    (실무에서는 보통 한 가지 판단 문제만 다루므로 헤드는 하나면 충분하다)
    """
    head_kind = args.head
    num_classes = None if head_kind == "reg" else len(examples[0]["choices"])

    # 검증 분리: 같은 데이터로 학습하고 평가하면 '외운 것'을 실력으로 착각한다.
    shuffled = examples[:]
    random.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * args.eval_ratio))
    eval_set = shuffled[:n_eval]
    train_set = shuffled[n_eval:] or shuffled[:1]   # 데이터가 적을 때 최소 1개는 남긴다

    # --- 백본 동결 범위 정하기 ---
    # (1) linear_probe: 전부 동결. 헤드만 학습한다.
    # (2) last_block: 마지막 N개 블록 + 마지막 정규화층만 학습한다.
    #     왜 마지막 블록인가: 판단에 필요한 상위 개념이 마지막 층에 모여 있어서,
    #     거기를 조금 움직이면 적은 데이터로도 큰 효과를 본다. 앞쪽은 그대로 둬서
    #     언어 능력이 망가지는 것을 막는다.
    for p in backbone.parameters():
        p.requires_grad_(False)

    wrapper = BackboneWithHead(
        backbone,
        hidden_size,
        num_classes,
        dropout=args.dropout,
        freeze_backbone=(args.mode == "linear_probe"),
    )

    backbone_params = []
    if args.mode == "last_block":
        layers = backbone.layers
        for layer in layers[-args.unfreeze_last_blocks :]:
            for p in layer.parameters():
                p.requires_grad_(True)
                backbone_params.append(p)
        if hasattr(backbone, "norm"):
            for p in backbone.norm.parameters():
                p.requires_grad_(True)
                backbone_params.append(p)
        # gradient checkpointing: 층마다 활성값을 저장하지 않고 다시 계산한다.
        # 속도는 20~30% 느려지지만 VRAM은 크게 줄어 12GB에서 마지막 블록 학습이 가능해진다.
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 헤드만 지정 장치로 옮긴다.
    # 왜 wrapper 전체를 .to(device) 하지 않는가: device_map이나 4bit로 불러온 모델은
    # 통째로 옮길 수 없다(transformers가 거부한다). 백본은 이미 제자리에 있다.
    wrapper.head.to(device)

    # 헤드는 lr 그대로, 백본은 아주 작게(1/100). 백본을 크게 흔들면 기존 능력이 깨진다.
    param_groups = [{"params": list(wrapper.head.parameters()), "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr * 0.01})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)

    if head_kind == "reg":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    bs = args.batch_size
    print(f"\n[{group_name}] 학습/검증 = {len(train_set)}/{len(eval_set)}개, "
          f"출력 차원={num_classes if num_classes else 1}, mode={args.mode}")

    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        if args.mode == "linear_probe":
            wrapper.freeze_backbone = True

        order = list(range(len(train_set)))
        random.shuffle(order)
        batches = [order[i : i + bs] for i in range(0, len(order), bs)]

        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.perf_counter()
        for bi, batch_idx in enumerate(batches):
            batch = [train_set[i] for i in batch_idx]
            input_ids, attention_mask, labels = collate(batch, pad_token_id)
            logits = wrapper(input_ids.to(device), attention_mask.to(device))
            loss = loss_fn(logits.float(), labels.to(device).float() if head_kind == "reg" else labels.to(device))

            # gradient accumulation: 배치 1로 여러 번 계산해 기울기를 모은 뒤 한 번 갱신한다.
            # 12GB에서 큰 배치를 흉내 내는 표준 방법이다(실효 배치 = batch_size × grad_accum).
            (loss / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(batches):
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()

        metrics = evaluate(wrapper, eval_set, device, pad_token_id, head_kind)[2]
        metric_text = f"검증 MSE {metrics['mse']:.4f}" if head_kind == "reg" else f"검증 정확도 {metrics['acc']:.3f}"
        print(f"  epoch {epoch}: 평균 loss {epoch_loss / max(len(batches),1):.4f} | {metric_text} "
              f"| {time.perf_counter() - t0:.1f}초 | {vram_report()}")

    return wrapper, eval_set


# ---------------------------------------------------------------------------
# 4. 저장 / 재로드
# ---------------------------------------------------------------------------


def save_heads(path: Path, heads: Dict[str, nn.Module], meta: Dict[str, Any]) -> None:
    """헤드 가중치만 저장한다(수 MB).

    왜 백본을 저장하지 않는가: 백본은 허깅페이스에서 언제든 다시 받을 수 있고
    8GB나 된다. 우리가 학습한 것은 헤드뿐이므로, 헤드 + 재현에 필요한 메타정보만 남긴다.
    """
    payload = {
        "meta": meta,
        "heads": {name: {k: v.detach().cpu() for k, v in m.state_dict().items()} for name, m in heads.items()},
    }
    torch.save(payload, path)


def load_heads(path: Path) -> Dict[str, Any]:
    """저장한 헤드를 읽는다.

    쓰는 법(README에도 있음):
        payload = load_heads("outputs/head.pt")
        head = JudgmentHead(payload["meta"]["hidden_size"], payload["meta"]["num_classes"][group])
        head.load_state_dict(payload["heads"][group])
    """
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LM 헤드 교체 + 헤드 학습 데모")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data", default=str(DATA_DIR / "sample.jsonl"))
    parser.add_argument("--head", choices=["cls", "reg"], default="cls")
    parser.add_argument("--mode", choices=["linear_probe", "last_block"], default="linear_probe")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-ratio", type=float, default=0.3)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = read_jsonl(Path(args.data))
    print(f"[data] {args.data}: {len(rows)}개 항목")

    tokenizer, model_id = load_tokenizer(args.model)
    model, model_id = load_model(model_id, load_in_4bit=args.four_bit, for_training=True)

    backbone, lm_head_ref = resolve_text_backbone(model)
    # 참조를 남겨 두면 lm_head 가중치가 메모리에 살아남는다(9B는 약 2GB).
    # 지우기 전에 참조를 끊어야 실제로 VRAM이 돌아온다.
    del lm_head_ref
    freed = drop_lm_head(model)
    if freed:
        print(f"[head-swap] lm_head 제거: {freed/1e6:.1f}M 파라미터 확보")

    hidden_size = hidden_size_of(backbone)
    pad_token_id = tokenizer.pad_token_id
    device = device_of(model)
    print(f"[head-swap] 백본={type(backbone).__name__}, hidden={hidden_size}, device={device}")

    groups = group_by_task(rows)
    heads: Dict[str, nn.Module] = {}
    meta: Dict[str, Any] = {
        "model_id": model_id,
        "hidden_size": hidden_size,
        "head_kind": args.head,
        "mode": args.mode,
        "num_classes": {},
        "choices": {},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    pred_rows: List[Dict[str, Any]] = []
    for group_name, group_rows in groups.items():
        examples = prepare_examples(group_rows, tokenizer, args.head, args.max_len)
        wrapper, eval_set = train_one_group(
            group_name, examples, backbone, hidden_size, pad_token_id, args, device
        )
        heads[group_name] = wrapper.head
        meta["num_classes"][group_name] = None if args.head == "reg" else len(group_rows[0]["choices"])
        meta["choices"][group_name] = list(group_rows[0]["choices"])

        probs, _labels, metrics = evaluate(wrapper, eval_set, device, pad_token_id, args.head)
        if args.head == "cls":
            for ex, p in zip(eval_set, probs):
                pred_rows.append(
                    make_pred_row(
                        ex,
                        ex["choices"],
                        p,
                        scores={},
                        model_id=model_id,
                        method=f"head_{args.mode}",
                        n_prompt_tokens=len(ex["input_ids"]),
                    )
                )
            print(f"  → {group_name} 최종 검증 정확도 {metrics['acc']:.3f} (n={len(eval_set)})")
        else:
            print(f"  → {group_name} 최종 검증 MSE {metrics['mse']:.4f} (n={len(eval_set)})")

    out_dir = ensure_out_dir()
    head_path = out_dir / f"head_{args.head}_{args.mode}.pt"
    save_heads(head_path, heads, meta)
    print(f"\n[saved] 헤드 가중치: {head_path}")

    # 저장한 파일을 실제로 다시 읽어 가중치를 되돌려 넣어 본다.
    # 왜: 저장이 '됐다'고 말하는 것과 복원이 되는 것은 다르다. 여기서 확인해야 신뢰할 수 있다.
    payload = load_heads(head_path)
    reloaded_ok = True
    for name, head in heads.items():
        probe = JudgmentHead(meta["hidden_size"], meta["num_classes"][name]).to(device)
        try:
            probe.load_state_dict(payload["heads"][name])
        except Exception as e:
            reloaded_ok = False
            print(f"  ! {name} 재로드 실패: {e}")
    print(f"[reload] {head_path.name} 재로드 {'성공' if reloaded_ok else '실패'} "
          f"(헤드 {len(payload['heads'])}개)")

    if args.head == "cls" and pred_rows:
        csv_path = write_preds_csv(pred_rows, out_dir / "preds_02_head.csv")
        print(f"[saved] 예측 CSV(검증 항목만): {csv_path}")
        print("        다음: python 04_eval.py  — 01과 02를 같은 표에서 비교")
    else:
        print("[note] 회귀 헤드는 확률을 내지 않으므로 04의 보정 평가 대상이 아닙니다.")


if __name__ == "__main__":
    main()

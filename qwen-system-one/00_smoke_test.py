#!/usr/bin/env python3
"""
00_smoke_test.py — 모델이 이 PC에서 실제로 뜨는지 5분 안에 확인한다

이 파일이 답하는 질문
    "내 3060 Ti 12GB에 Qwen3.5-4B가 올라가고, 생성까지 되는가?"

실행하면 무엇을 보게 되는지
    (1) 모델 로드에 걸린 시간, (2) 로드 직후 VRAM 사용량,
    (3) 짧은 프롬프트에 대한 생성 결과 1개, (4) 초당 생성 토큰 수.
    여기서 OOM이 나면 01 이후는 시도하지 말고 README의 'VRAM 가이드'를 먼저 볼 것.

라이선스 메모
    Qwen 시리즈는 Apache 2.0이다(상업적 사용 가능).
    Gemma로 바꿔 실험할 경우 라이선스는 구글 자체 약관이므로 반드시 원문을 확인할 것.

사용 예
    python 00_smoke_test.py
    python 00_smoke_test.py --model Qwen/Qwen3.5-2B
    python 00_smoke_test.py --model Qwen/Qwen3.5-9B --4bit
"""

from __future__ import annotations

import argparse
import time

import torch

from common import (
    DEFAULT_MODEL_ID,
    hidden_size_of,
    load_model,
    load_tokenizer,
    resolve_text_backbone,
    set_seed,
    vram_report,
)


def build_simple_chat(tokenizer, user_text: str) -> str:
    """생성 확인용 짧은 프롬프트를 만든다.

    왜 chat template을 쓰는가: Qwen3.5-Instruct는 대화 형식으로 학습됐다.
    원문 문자열을 그대로 넣으면 "<|im_start|>user" 같은 특수 토큰이 빠져
    엉뚱한 답이 나오고, 그러면 모델이 고장 난 것처럼 보인다.
    """
    messages = [
        {"role": "system", "content": "너는 간결하게 답하는 한국어 비서다."},
        {"role": "user", "content": user_text},
    ]
    try:
        # 사고(thinking) 토큰을 끄고 바로 답하게 한다. 끄지 않으면 수백 토큰을
        # 생각하는 데 쓰느라 첫 실행이 훨씬 오래 걸린다.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:      # 구형 템플릿: enable_thinking 인자를 모름
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 로드·생성 스모크 테스트")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="허깅페이스 모델 id")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="bitsandbytes 4bit 로드(9B용)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="삼척에서 오늘 입기 좋은 옷차림을 두 문장으로 알려줘.")
    args = parser.parse_args()

    set_seed(42)

    print(f"[1/4] 토크나이저 로드: {args.model}")
    t0 = time.perf_counter()
    tokenizer, model_id = load_tokenizer(args.model)

    print(f"[2/4] 모델 로드: {model_id} (4bit={args.four_bit})")
    t1 = time.perf_counter()
    model, model_id = load_model(model_id, load_in_4bit=args.four_bit)
    load_seconds = time.perf_counter() - t1

    # 모델 구조를 눈으로 확인한다. 02에서 헤드를 갈아 끼울 때 어디를 만지는지
    # 미리 보여 주는 것이 이 단계의 숨은 목적이다.
    backbone, lm_head = resolve_text_backbone(model)
    n_params = sum(p.numel() for p in model.parameters())
    print("   - 백본 클래스:", type(backbone).__name__)
    print(f"   - decoder layers: {len(backbone.layers)} / hidden: {hidden_size_of(backbone)}")
    print(f"   - lm_head: {'있음' if lm_head is not None else '없음'}")
    print(f"   - 전체 파라미터: {n_params/1e9:.2f}B / dtype: {next(model.parameters()).dtype}")
    print(f"   - 토크나이저 로드까지 {t1 - t0:.1f}초, 모델 로드 {load_seconds:.1f}초")

    if not torch.cuda.is_available():
        print("   ! CUDA가 없어 CPU로 실행 중입니다. 4B는 CPU에서 매우 느립니다(수 분/문장).")

    print("[3/4] 짧은 생성 1회")
    prompt = build_simple_chat(tokenizer, args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    gen_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,          # 재현 가능한 확인을 위해 greedy
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_seconds = time.perf_counter() - gen_start

    # 새로 생성된 토큰만 잘라서 디코딩한다(프롬프트는 빼고).
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("[4/4] 결과")
    print("   프롬프트 토큰 수:", inputs["input_ids"].shape[1])
    print("   생성 토큰 수:", len(new_tokens))
    print("   생성 시간: %.2f초 (%.1f 토큰/초)" % (gen_seconds, len(new_tokens) / max(gen_seconds, 1e-6)))
    print("-" * 60)
    print(text)
    print("-" * 60)
    print(vram_report("최종"))

    if not text:
        print("! 생성 결과가 비었습니다. chat template 적용 여부와 max_new_tokens를 확인하세요.")


if __name__ == "__main__":
    main()

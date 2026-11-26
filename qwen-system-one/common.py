#!/usr/bin/env python3
"""
common.py — 실습 전체가 함께 쓰는 공통 도구

이 파일이 답하는 질문
    "모델 불러오기, chat template 적용, 선택지 로그확률 계산, 결과 CSV 저장을 어떻게 한 곳에 모을까?"

실행하면 무엇을 보게 되는지
    직접 실행하는 파일은 아니다. 00~05 스크립트가 `from common import ...` 로 가져다 쓴다.
    01(무학습 로그확률), 02(헤드 학습), 03(LoRA), 04(헤드+LoRA)가 **같은 함수**로 점수를 내고
    같은 CSV 형식으로 남기 때문에, 네 방법의 확률을 05에서 같은 자로 비교할 수 있다. 이게 이 파일의 존재 이유다.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import transformers
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# 경로와 기본값
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent          # qwen-system-one/
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"                      # 실행 결과 CSV/보고서가 쌓이는 곳

# 기본 모델. CLI에서 --model 로 바꿀 수 있고, 환경변수로도 바꿀 수 있다.
# 왜 상수로 빼 두는가: 00~04가 모두 같은 모델을 봐야 결과 비교가 의미를 갖는다.
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B-Instruct"

# 공식 저장소 이름이 -Instruct 없이 배포되는 경우가 있어, 기본 모델을 못 찾으면
# 아래 후보를 차례로 시도한다. (--model 로 직접 지정한 경우에는 대체하지 않는다)
MODEL_FALLBACKS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B")

# 판단 모델에게 주는 시스템 프롬프트.
# 왜 필요한가: 모델이 "정답은 예입니다"처럼 말을 덧붙이면, 선택지 문자열의
# 로그확률만 재는 01/03 채점이 흔들린다. "선택지 문구만 출력"을 미리 못박아 둔다.
SYSTEM_PROMPT = (
    "너는 판단 모델이다. 주어진 질문에 대해 제시된 선택지 중 하나의 문구만 그대로 출력한다. "
    "설명, 인사, 문장 부호를 덧붙이지 않는다."
)

# 01/02/03/04가 모두 이 열 순서로 CSV를 쓴다. 05는 이 형식만 읽는다.
PRED_COLUMNS = [
    "id",            # 데이터 항목 id
    "task",          # 선택지 집합 이름 (예: fidelity, sentiment)
    "model",         # 사용한 모델 id
    "method",        # 채점 방법 (logprob_mean / head_probe / lora_mean ...)
    "num_choices",   # 이 항목의 선택지 수
    "answer_index",  # 정답 선택지의 위치 (0부터)
    "predicted_index",
    "predicted",     # 예측한 선택지 문자열
    "correct",       # 1 또는 0
    "confidence",    # 고른 선택지의 확률 (0~1)
    "choice_probs",  # 선택지 순서대로의 확률 JSON 배열
    "scores",        # 선택지별 원점수(로그확률 등) JSON 객체 — 디버깅용
    "n_prompt_tokens",
]

# ---------------------------------------------------------------------------
# 잡다한 유틸
# ---------------------------------------------------------------------------


def ensure_out_dir() -> Path:
    """outputs/ 폴더를 만들고 경로를 돌려준다.

    왜: 스크립트를 처음 돌릴 때 폴더가 없어서 저장 단계에서 죽는 일을 막는다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def set_seed(seed: int = 42) -> None:
    """난수 시드를 고정한다.

    왜: 데이터 분할, 헤드 초기화, LoRA 초기화가 매번 달라지면
    "어제는 정확도 0.8, 오늘은 0.6" 같은 결과가 나와 비교가 불가능해진다.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_dtype() -> torch.dtype:
    """이 PC에서 쓸 수 있는 가장 빠른 부동소수점 형식을 고른다.

    왜: 30~40 시리즈 NVIDIA GPU는 bf16을 지원해서 float16보다 안전하다
    (float16은 큰 값에서 넘쳐 NaN이 되기 쉽다). 지원하지 않는 구형 GPU나
    CPU에서는 float32로 떨어뜨린다.
    """
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def device_of(model: torch.nn.Module) -> torch.device:
    """모델 가중치가 실제로 올라가 있는 장치를 찾는다.

    왜: device_map="auto"로 불러오면 모델이 GPU/CPU에 나뉘어 올라간다.
    입력 텐서는 첫 번째 파라미터가 있는 장치로 보내면 된다.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def vram_report(tag: str = "") -> str:
    """현재 GPU 메모리 사용량을 사람이 읽는 문자열로 만든다.

    왜: OOM(메모리 부족)은 원인을 모르면 손쓸 수 없다. 단계마다 숫자를 찍어 두면
    "모델 로드만으로 이미 7.9GB를 썼다" 같은 사실이 바로 보인다.
    """
    if not torch.cuda.is_available():
        return "CUDA 없음(CPU 모드)"
    allocated = torch.cuda.memory_allocated() / 1024**3      # 지금 텐서가 쓰는 양
    reserved = torch.cuda.memory_reserved() / 1024**3        # 캐시까지 포함해 잡아 둔 양
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    prefix = f"[{tag}] " if tag else ""
    return f"{prefix}VRAM {allocated:.2f}GB 사용 / {reserved:.2f}GB 예약 / {total:.1f}GB 전체"


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    """JSONL(한 줄에 JSON 하나) 파일을 읽는다.

    왜 JSONL인가: 학습 데이터는 한 줄이 한 예제라, 파일 전체를 한 번에
    파싱하지 않아도 되고 사람이 열어서 고치기도 쉽다.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):   # 빈 줄과 주석 줄은 건너뛴다
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 파싱 실패 — {e}") from e
    return rows


# ---------------------------------------------------------------------------
# 모델 / 토크나이저 불러오기
# ---------------------------------------------------------------------------


def _candidate_ids(model_id: Optional[str]) -> List[str]:
    """시도할 모델 id 목록을 만든다.

    왜: 기본 모델(Qwen3.5-4B-Instruct)의 공식 저장소 이름이 바뀌었을 때
    첫 실행이 그대로 실패하지 않도록, 기본값일 때만 대체 후보를 붙인다.
    사용자가 --model 로 직접 지정한 경우에는 임의로 다른 모델을 쓰면 안 되므로
    그 id 하나만 시도한다.
    """
    if model_id and model_id != DEFAULT_MODEL_ID:
        return [model_id]
    ids = [model_id or DEFAULT_MODEL_ID]
    for m in MODEL_FALLBACKS:
        if m not in ids:
            ids.append(m)
    return ids


def _from_pretrained(auto_cls, model_id: str, dtype: torch.dtype, device_map, extra: Dict[str, Any]):
    """transformers 버전 차이를 흡수해서 모델을 로드한다.

    왜: transformers 5.x는 `dtype=` 을 쓰고 4.x는 `torch_dtype=` 을 쓴다.
    둘 다 시도해 두면 설치된 transformers 버전이 어느 쪽이어도 첫 실행이 된다.
    """
    try:
        return auto_cls.from_pretrained(model_id, dtype=dtype, device_map=device_map, **extra)
    except TypeError:
        # 구버전: dtype 이라는 인자를 모르면 TypeError가 난다.
        return auto_cls.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map, **extra)


def load_tokenizer(model_id: Optional[str] = None, padding_side: str = "right"):
    """토크나이저를 불러오고 pad 토큰을 보정한다.

    왜 pad 토큰을 채우는가: Qwen 계열은 pad_token이 비어 있는 경우가 있다.
    그대로 두면 (1) 배치에서 시퀀스 길이를 맞출 수 없고 (2) generate()가
    eos를 pad로 대신 쓰겠다는 경고를 내며, 그 경고를 무시하면 경고가 반복된다.
    pad=eos로 두면 학습 시 loss에서 -100으로 가리므로 결과에도 영향이 없다.

    padding_side: 학습은 오른쪽(right) 패딩이 자연스럽고, 생성은 왼쪽(left)이 편하다.
    """
    last_error = None
    for cand in _candidate_ids(model_id):
        try:
            tok = AutoTokenizer.from_pretrained(cand)
        except Exception as e:      # 저장소 없음, 네트워크 오류 등
            last_error = e
            continue

        if tok.pad_token is None:
            # eos를 pad로 재사용한다. (Qwen은 <|im_end|>)
            tok.pad_token = tok.eos_token
        tok.padding_side = padding_side
        return tok, cand
    raise RuntimeError(f"토크나이저를 불러오지 못했습니다. 마지막 오류: {last_error}")


def load_model(
    model_id: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = False,
    device_map: Optional[str] = "auto",
    for_training: bool = False,
):
    """모델을 불러와 (model, 실제로 쓴 model_id)를 돌려준다.

    왜 두 개의 자동 클래스를 시도하는가: Qwen3.5는 이미지도 받는 모델이라
    체크포인트에 따라 AutoModelForCausalLM으로는 안 열리고
    AutoModelForImageTextToText(구 Vision2Seq)로 열리는 경우가 있다.
    텍스트만 쓸 때도 이 클래스로 열 수 있으므로, 실패하면 다음 클래스로 넘어간다.

    load_in_4bit: 12GB에서 9B를 쓰려면 필요하다(bitsandbytes). 4bit는 추론·LoRA 전용이며,
    헤드 학습(02)에서 백본을 얼리면 4bit로도 충분하다.
    """
    model_id = model_id or DEFAULT_MODEL_ID
    dtype = dtype or pick_dtype()

    extra: Dict[str, Any] = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        # nf4: 4bit 양자화 방식 중 품질이 좋은 축. compute_dtype은 계산할 때만 bf16으로 올린다.
        extra["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"   # 4bit는 가중치가 장치별로 나뉘므로 auto가 안전하다

    last_error: Optional[Exception] = None
    for cand in _candidate_ids(model_id):
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            auto_cls = getattr(transformers, cls_name, None)
            if auto_cls is None:       # 구버전 transformers에는 없는 이름
                continue
            try:
                model = _from_pretrained(auto_cls, cand, dtype, device_map, extra)
            except Exception as e:
                last_error = e
                continue

            if for_training:
                # 학습 중에는 KV 캐시를 끈다. 캐시는 추론용이라 학습에서는 메모리만 먹는다.
                if hasattr(model, "config"):
                    model.config.use_cache = False
                model.train()
            else:
                model.eval()
            print(f"[model] {cand} 로드 완료 (클래스={cls_name}) — {vram_report('load')}")
            return model, cand

    raise RuntimeError(
        "모델을 불러오지 못했습니다. 마지막 오류: "
        f"{type(last_error).__name__}: {last_error}\n"
        "확인할 것: (1) transformers 버전이 Qwen3.5를 지원하는지(5.x 이상 권장), "
        "(2) --model 이름이 정확한지, (3) 인터넷/디스크 여유 공간."
    )


# ---------------------------------------------------------------------------
# 모델 해부: 백본(디코더 스택)과 lm_head 찾기
# ---------------------------------------------------------------------------


def resolve_text_backbone(model: torch.nn.Module) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    """모델에서 텍스트 디코더 스택과 lm_head를 찾아 돌려준다.

    왜 이렇게 하는가: Qwen3.5는 이미지도 받는 구조라 한 겹이 더 들어가 있다.

        Qwen3_5ForConditionalGeneration
          .model : Qwen3_5Model
            .visual          : 이미지 인코더 (우리는 쓰지 않는다)
            .language_model  : Qwen3_5TextModel  ← 텍스트 백본
          .lm_head           : 어휘 크기로 사상하는 출력층

    반면 텍스트 전용 체크포인트(Qwen3_5ForCausalLM)는

        .model : Qwen3_5TextModel   ← 여기가 바로 백본
        .lm_head

    처럼 한 겹이 적다. 그래서 후보 경로를 위에서부터 확인하며
    "layers 와 embed_tokens 를 가진 모듈"을 찾는다. 이 두 속성은
    디코더 스택을 구분하는 확실한 표지다(비전 인코더에는 없다).
    """
    for path in ("model.language_model", "language_model", "model", "model.model"):
        obj: Any = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and all(hasattr(obj, attr) for attr in ("layers", "embed_tokens")):
            lm_head = None
            if hasattr(model, "get_output_embeddings"):
                lm_head = model.get_output_embeddings()
            return obj, lm_head

    raise RuntimeError(
        "텍스트 백본을 찾지 못했습니다. `print(model)` 결과를 보고 "
        "common.resolve_text_backbone 의 후보 경로를 추가해 주세요."
    )


def drop_lm_head(model: torch.nn.Module) -> int:
    """lm_head를 모델에서 떼어내고, 없앤 가중치 개수를 돌려준다.

    왜 지우는가: 헤드를 우리 것으로 갈아 끼우면 원래 lm_head는 쓸 일이 없다.
    2B/4B는 lm_head가 임베딩과 가중치를 공유(tied)해서 지워도 이득이 없지만,
    9B처럼 따로 가진 모델은 248320×4096 ≈ 10억 개(bf16 약 2GB)를 그대로
    들고 있게 된다. 12GB 카드에서는 그 2GB가 곧 OOM이다.
    """
    if not hasattr(model, "lm_head"):
        return 0
    head = model.lm_head
    n_params = sum(p.numel() for p in head.parameters()) if head is not None else 0
    # nn.Module은 __delattr__를 오버라이드해서 _modules 등록까지 함께 지운다.
    del model.lm_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n_params


def hidden_size_of(backbone: torch.nn.Module) -> int:
    """백본의 hidden 차원을 읽는다(헤드의 입력 크기).

    왜 config를 두 군데서 보는가: VLM은 바깥 config가 vision 설정이라
    hidden_size가 텍스트와 다를 수 있다. text_config가 있으면 그쪽을 우선한다.
    """
    cfg = getattr(backbone, "config", None)
    if cfg is not None:
        # VLM 바깥 config는 vision 쪽 값일 수 있으므로 text_config를 먼저 본다.
        text_cfg = getattr(cfg, "text_config", None) or cfg
        val = getattr(text_cfg, "hidden_size", None)
        if isinstance(val, int):
            return val
    embed = backbone.get_input_embeddings() if hasattr(backbone, "get_input_embeddings") else None
    if embed is not None and hasattr(embed, "embedding_dim"):
        return int(embed.embedding_dim)
    raise RuntimeError("hidden_size를 알 수 없습니다. config를 확인해 주세요.")


# ---------------------------------------------------------------------------
# 프롬프트 만들기
# ---------------------------------------------------------------------------


def build_prompt(tokenizer, instruction: str, choices: Sequence[str]) -> str:
    """판단용 chat 프롬프트 문자열을 만든다(토큰화 전).

    왜 chat template을 쓰는가: Qwen3.5는 `<|im_start|>user ... <|im_end|>` 형식을
    학습 때 봤다. 이 형식을 손으로 흉내 내면 특수 토큰이 어긋나 성능이 뚝 떨어진다.

    왜 enable_thinking=False 인가: Qwen3.5는 기본이 '생각 모드'라, 그대로 두면
    선택지 앞에 긴 사고 과정이 먼저 나온다. 우리는 "선택지 문구가 바로 다음 토큰"이길
    원하므로 사고를 끈다. (구버전 템플릿은 이 인자를 모르므로 TypeError를 잡아 넘긴다)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                "[선택지]\n" + "\n".join(f"- {c}" for c in choices) + "\n\n"
                "위 선택지 중 하나의 문구만 그대로 출력하세요."
            ),
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # chat template이 enable_thinking을 모르는 경우(구형 체크포인트)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# 선택지 로그확률 채점 (01과 03이 공유)
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_choices(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    choices: Sequence[str],
    max_prompt_tokens: int = 1024,
) -> Dict[str, Dict[str, float]]:
    """프롬프트 뒤에 각 선택지가 이어질 로그확률을 계산한다.

    반환: { 선택지 문자열: {"n_tokens", "sum", "mean", "first"} }
      - sum   : 선택지 전체 토큰의 로그확률 합 = log P(선택지 전체 | 프롬프트)
      - mean  : 합을 토큰 수로 나눈 값 = 길이 정규화 로그확률
      - first : 첫 토큰 하나의 로그확률 (후보를 1토큰으로 강제할 때 쓰는 방식)

    왜 세 가지를 모두 내는가:
      * sum은 확률 이론상 맞는 값이지만, 선택지가 길수록(토큰이 많을수록) 무조건 작아져
        "짧은 선택지가 유리"해진다. 예: "예"(1토큰) vs "아니오"(3토큰).
      * mean은 그 편향을 없애 대체로 정확도가 높다.
      * first는 후보를 한 토큰으로 제한하는 실제 서빙(예: Jev식 판독)에서 쓰는 방식이다.
      세 값의 정확도가 어떻게 다른지 01에서 직접 비교해 보는 것이 이 실습의 첫 번째 관찰이다.

    왜 후보마다 따로 forward 하는가: 프롬프트+후보를 각각 한 번씩 넣으면
    "그 후보가 이어졌을 때"의 확률을 정확히 잴 수 있다. 배치로 묶으면 더 빠르지만
    패딩 위치 계산이 복잡해져 초보가 읽기 어려우므로, 28개 예제 규모에서는 단순함을 택했다.
    """
    device = device_of(model)

    # chat template 문자열은 이미 특수 토큰을 포함하므로 add_special_tokens=False.
    # (True로 두면 <bos>가 또 붙어 학습 때와 다른 분포가 된다)
    prompt_ids: List[int] = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    # 너무 긴 프롬프트는 앞을 잘라 길이를 맞춘다. 뒤(질문+선택지)를 남기는 이유는
    # 판단에 필요한 정보가 끝쪽에 몰려 있기 때문이다. (앞의 시스템 프롬프트가 잘리면
    # 성능이 떨어질 수 있으니 max_prompt_tokens를 넉넉히 두는 편이 좋다)
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]

    results: Dict[str, Dict[str, float]] = {}
    for choice in choices:
        cand_ids: List[int] = tokenizer(choice, add_special_tokens=False)["input_ids"]
        if not cand_ids:
            continue

        input_ids = torch.tensor([prompt_ids + cand_ids], device=device)
        logits = model(input_ids=input_ids).logits[0].float()   # (전체 길이, 어휘 크기)

        # 로짓 → 로그확률. softmax를 거치지 않고 log_softmax를 쓰는 이유는
        # 확률이 1e-30처럼 작아져 0으로 붕괴하는 것을 막기 위해서다.
        logprobs = torch.log_softmax(logits, dim=-1)

        # 핵심 인덱스 계산: logprobs[t]는 "t+1번째 토큰"의 분포다.
        # 따라서 첫 후보 토큰(위치 P = len(prompt_ids))을 예측하는 행은 P-1이다.
        start = len(prompt_ids) - 1
        rows = logprobs[start : start + len(cand_ids)]                     # (후보 토큰 수, 어휘)
        target = torch.tensor(cand_ids, device=logprobs.device).unsqueeze(1)  # (후보 토큰 수, 1)
        token_logprobs = rows.gather(1, target).squeeze(1)                # 각 후보 토큰의 로그확률

        results[choice] = {
            "n_tokens": len(cand_ids),
            "sum": float(token_logprobs.sum().item()),
            "mean": float(token_logprobs.mean().item()),
            "first": float(token_logprobs[0].item()),
        }
    return results


def softmax_over(scores: Dict[str, Dict[str, float]], key: str) -> Tuple[List[str], List[float]]:
    """선택지별 로그확률을 '후보 집합 안에서의 확률'로 정규화한다.

    왜 필요한가: 모델의 로그확률은 어휘 전체(24만 개)에 대한 값이라 그대로는
    "얼마나 확신하는가"를 알 수 없다. 후보들만 놓고 다시 softmax를 취하면
    "이 4개 중 A일 확률 0.87"처럼 해석할 수 있고, 05의 보정 평가(Brier/ECE)가 가능해진다.

    왜 최댓값을 빼는가: exp는 -1000 같은 값에서 0이 되어 버린다. 최댓값을 빼면
    계산 결과는 같으면서 수치적으로 안전하다(표준적인 softmax 트릭).
    """
    items = [(c, s[key]) for c, s in scores.items()]
    if not items:
        return [], []
    m = max(v for _, v in items)
    exps = [(c, math.exp(v - m)) for c, v in items]
    z = sum(e for _, e in exps)
    if z == 0:
        # 모든 후보가 극단적으로 작은 경우(수치 한계). 균등 분포로 대체한다.
        n = len(exps)
        return [c for c, _ in exps], [1.0 / n] * n
    return [c for c, _ in exps], [e / z for _, e in exps]


# ---------------------------------------------------------------------------
# 결과 CSV (01/02/03/04 → 05)
# ---------------------------------------------------------------------------


def write_preds_csv(rows: Iterable[Dict[str, Any]], path: Path | str) -> Path:
    """예측 결과를 공통 형식 CSV로 저장한다.

    왜 CSV인가: 05가 pandas 한 줄로 읽어 지표를 계산할 수 있고,
    엑셀로 열어 오답을 눈으로 확인할 수도 있다.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        # utf-8-sig: 엑셀에서 한글이 깨지지 않게 BOM을 붙인다.
        writer = csv.DictWriter(f, fieldnames=PRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in PRED_COLUMNS}
            # 리스트/딕셔너리는 CSV에 그대로 못 들어가므로 JSON 문자열로 바꾼다.
            for key in ("choice_probs", "scores"):
                if not isinstance(out[key], str):
                    out[key] = json.dumps(out[key], ensure_ascii=False)
            writer.writerow(out)
    return path


def read_preds_csv(path: Path | str):
    """05가 쓰는 역방향 함수. JSON 문자열 열을 다시 파이썬 객체로 되돌린다."""
    import pandas as pd

    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("choice_probs", "scores"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) and v else None)
    return df


def make_pred_row(
    item: Dict[str, Any],
    choices: Sequence[str],
    probs: Sequence[float],
    scores: Optional[Dict[str, Dict[str, float]]] = None,
    model_id: str = "",
    method: str = "",
    n_prompt_tokens: int = 0,
) -> Dict[str, Any]:
    """한 항목의 예측 행을 공통 형식으로 만든다.

    왜 따로 함수로 빼는가: 01(로그확률)·02(헤드)·03(LoRA)이 같은 형식을 써야
    05가 네 방법을 한 표에서 비교할 수 있다. 형식이 어긋나는 사고를 한 곳에서 막는다.
    """
    answer_index = choices.index(item["answer"])
    predicted_index = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "id": item.get("id", ""),
        "task": item.get("task", ""),
        "model": model_id,
        "method": method,
        "num_choices": len(choices),
        "answer_index": answer_index,
        "predicted_index": predicted_index,
        "predicted": choices[predicted_index],
        "correct": int(predicted_index == answer_index),
        "confidence": float(probs[predicted_index]),
        "choice_probs": list(map(float, probs)),
        "scores": scores or {},
        "n_prompt_tokens": n_prompt_tokens,
    }


def group_by_task(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """선택지 집합이 같은 항목끼리 묶는다.

    왜 필요한가: "예/아니오"(2지선다)와 "긍정/부정/중립"(3지선다)은 출력 차원이 다르다.
    하나의 분류 헤드로 둘을 함께 학습할 수 없으므로, 02는 이 묶음마다 헤드를 따로 만든다.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = row.get("task") or "|".join(row["choices"])
        groups.setdefault(key, []).append(row)
    return groups


if __name__ == "__main__":
    # 직접 실행하면 지금 설정만 확인해 준다(모델을 내려받지 않는다).
    print("qwen-system-one 공통 모듈")
    print("기본 모델:", DEFAULT_MODEL_ID)
    print("대체 후보:", ", ".join(MODEL_FALLBACKS))
    print("데이터 폴더:", DATA_DIR)
    print("결과 폴더:", OUT_DIR)
    print("CUDA:", torch.cuda.is_available(), "|", vram_report())

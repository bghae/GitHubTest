# Qwen3.5 기반 전용 판단 모델 만들기 — 실습 코드

RTX 3060 Ti 12GB / RAM 16GB / Windows 환경을 기준으로 작성한, **직접 읽고 돌려 보는** 실습 묶음입니다.
"Jev 같은 판단 전용 모델"이 어떻게 만들어지는지, 그 최소 골격을 5단계로 따라갑니다.

---

## 1. 이 실습이 하는 일 (3줄 요약)

1. **학습 없이** — Qwen3.5에게 선택지를 주고 "각 선택지가 이어질 로그확률"만 재서 채점합니다(기준선).
2. **헤드만 학습** — 단어 24만 개를 뽑던 출력층(`lm_head`)을 떼고 선택지 개수만 뽑는 작은 층으로 바꿔, 그 층만 학습합니다.
3. **LoRA SFT** — 백본의 1% 미만만 학습해 판단을 가르치고, 세 방법의 **정확도와 보정(calibration)** 을 같은 자로 비교합니다.

핵심 관찰 포인트: **정확도가 올라가는 것과 "확신이 정직해지는 것"은 다른 문제**라는 점입니다. 04가 그 차이를 숫자로 보여 줍니다.

---

## 2. 용어 미니 사전

| 용어 | 뜻 | 이 실습에서 |
|---|---|---|
| **로그확률 (logprob)** | 모델이 어떤 토큰에 준 확률의 자연로그. 곱을 덧셈으로 바꿔 다루기 쉽게 한 값 | 선택지 문자열 전체의 로그확률 합 = 그 선택지가 나올 확률 |
| **소프트맥스 정규화** | 후보들의 점수를 "합이 1인 확률"로 바꾸는 것 | 어휘 24만 개에 흩어진 확률을 **선택지 안에서만** 다시 나눔 |
| **linear probe** | 얼린 모델의 마지막 벡터 위에 선형층 하나만 얹어 학습하는 방법 | 02의 기본 모드. 4B를 12GB에서 돌리는 비결 |
| **보정 (calibration)** | "확률 0.9라고 말한 것 중 실제로 90%가 맞는가" | 04의 주제 |
| **Brier score** | 확률 예측의 평균 제곱오차(0~2, 낮을수록 좋음) | 정답에 낮은 확률을 준 경우 크게 벌점 |
| **ECE (기대 보정 오차)** | 확신 구간별 `|정확도 − 평균확신|` 을 표본 수로 가중 평균 | 0.05 이하면 양호, 0.15 넘으면 위험 |
| **LoRA** | 원래 가중치는 두고 작은 저랭크 행렬만 학습하는 기법 | 03. 어댑터가 수십 MB |
| **MoE / 활성 파라미터** | 전문가 층 여러 개 중 일부만 쓰는 구조. "A17B"는 총 397B 중 토큰당 17B만 사용 | 이 실습의 4B/9B는 **dense**(전부 사용)라 해당 없음. 큰 Qwen3.5를 볼 때 필요한 개념 |
| **chat template** | `<|im_start|>user …` 같은 대화 형식을 만들어 주는 규칙 | 손으로 쓰면 성능이 떨어짐. 반드시 `apply_chat_template` 사용 |
| **thinking 모드** | 답 전에 속으로 생각하는 모드. Qwen3.5는 기본이 켜짐 | 판단 채점에는 방해가 되므로 `enable_thinking=False` |

---

## 3. 폴더 구조

```
qwen-system-one/
├── README.md               ← 지금 이 문서
├── requirements.txt
├── common.py               공통: 모델 로드, 백본 찾기, 선택지 로그확률 채점, CSV 형식
├── 00_smoke_test.py        모델이 뜨는지 + VRAM/속도 확인
├── 01_logprob_baseline.py  학습 없는 기준선 (세 가지 채점 방식 비교)
├── 02_head_swap.py         lm_head 제거 → 판단 헤드 학습 (linear probe / last block)
├── 03_lora_sft.py          LoRA SFT + 학습 후 확률 재측정
├── 05_eval.py              정확도·Brier·ECE·신뢰도 표
├── data/
│   ├── sample.jsonl        채점/평가용 28건 (01·02 검증, 03 평가)
│   └── train.jsonl         학습용 80건 (03 학습 전용, sample과 겹치지 않음)
└── outputs/                실행하면 자동 생성 (예측 CSV, 헤드, 어댑터, 보고서)
```

---

## 4. 설치

### 4-1. 가상환경 (PowerShell)

```powershell
cd qwen-system-one
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
```

### 4-2. PyTorch (CUDA 12.1 이상 빌드)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> 드라이버가 CUDA 12.4 이상이면 `cu124` 로 바꿔도 됩니다. 설치 후 확인:
> `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

**이미 CPU 빌드 torch가 깔려 있다면** 위 명령만으로는 바뀌지 않습니다.
pip이 "Requirement already satisfied: torch"라고 하며 CPU 빌드를 그대로 두기 때문입니다.
`torch.cuda.is_available()`이 `False`로 나오면 먼저 지우고 다시 설치하세요.

```powershell
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

같은 이유로 `pip install -r requirements.txt` 는 torch를 새로 깔지 않습니다(이미 설치된 것으로 판단).
단, CUDA 빌드가 아닌 torch가 깔려 있으면 그것을 바꿔 주지도 않습니다.

### 4-3. 나머지

```powershell
pip install -r requirements.txt
```

**Qwen3.5는 transformers 5.x(또는 최신 main)가 필요합니다.** 4.x에서는 `qwen3_5` 모델 클래스가 없어 로드 단계에서 실패합니다.
`pip show transformers` 로 버전을 확인하고, 낮으면 `pip install -U --pre transformers` 로 올리세요.
`common.py`는 `dtype=`(5.x)과 `torch_dtype=`(4.x)을 모두 시도하므로 인자 차이로 죽지는 않습니다.

4bit(9B용)를 쓸 때만 추가:

```powershell
pip install "bitsandbytes>=0.45"
```

---

## 5. 실행 순서

| 단계 | 명령 | 무엇을 확인하는가 | 3060 Ti 기준 대략 |
|---|---|---|---|
| 00 | `python 00_smoke_test.py` | 모델이 뜨는가 / 로드 시간 / VRAM / 생성 속도 | 첫 실행 5~20분(다운로드 8GB 포함), 이후 1~2분 |
| 01 | `python 01_logprob_baseline.py --limit 5` → 전체 | 학습 없이 얼마나 맞히는가 / 채점 방식 3종 차이 | 5건 30초, 28건 2~4분 |
| 02 | `python 02_head_swap.py` | 헤드만 학습해도 정확도가 오르는가 / loss가 내려가는가 | 3~6분 |
| 03 | `python 03_lora_sft.py --epochs 1 --limit 8` (예열) → `python 03_lora_sft.py` | LoRA가 확률을 얼마나 바꾸는가 | 예열 3~5분, 전체 15~30분 |
| 04 | `python 05_eval.py` | 세 방법의 정확도·Brier·ECE 비교 | 즉시 |

전체를 한 번에:

```powershell
python 00_smoke_test.py
python 01_logprob_baseline.py
python 02_head_swap.py
python 03_lora_sft.py
python 05_eval.py
```

**먼저 00에서 OOM이 나면 01 이후는 의미가 없습니다.** README 6절(VRAM)을 보고 모델을 낮추세요.

---

## 6. VRAM 가이드 (RTX 3060 Ti 12GB 기준)

Qwen3.5-4B는 bf16 가중치만 약 8.0GB입니다(어휘 248,320 × hidden 2,560 임베딩이 1.3GB를 차지하고, lm_head와 공유됩니다).

| 모델 | bf16 가중치 | 00 스모크 | 01 로그확률 | 02 linear probe | 02 last block | 03 LoRA SFT |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 약 1.6GB | 여유 | 여유 | 여유 | 여유 | 여유 |
| Qwen3.5-2B | 약 4.0GB | 여유 | 여유 | 여유 | 가능 | **권장** |
| Qwen3.5-4B | 약 8.0GB | 가능 | 가능 | 가능(빠듯) | 조건부(1블록+체크포인팅) | 빠듯(기본값으로 시도) |
| Qwen3.5-9B | 약 17.5GB | 4bit 필요(약 5~6GB) | 4bit | 4bit | 비권장 | 4bit LoRA만 |
| 27B 이상 | 50GB+ | 불가 | 불가 | 불가 | 불가 | **비권장** |

원칙:
- **추론(00/01)** 은 4B bf16이 그대로 들어갑니다.
- **학습(02/03)** 은 가중치 외에 활성값·기울기가 더 필요합니다. 4B는 batch 1 + `grad_accum` 크게 + gradient checkpointing으로 겨우 들어가는 수준이고, **처음에는 2B로 전체 흐름을 익힌 뒤 4B로 올리는 것**을 권합니다.
- 12GB에서 12B급 이상은 권장하지 않습니다(4bit로도 학습은 사실상 불가).

```powershell
python 02_head_swap.py --model Qwen/Qwen3.5-2B
python 03_lora_sft.py --model Qwen/Qwen3.5-2B --batch-size 1 --grad-accum 8
python 03_lora_sft.py --model Qwen/Qwen3.5-9B --4bit
```

---

## 7. 데이터 넣는 법

한 줄에 JSON 하나(JSONL). 필드:

```json
{"id": "f01", "task": "fidelity", "instruction": "질문과 답변...", "choices": ["예", "아니오"], "answer": "예", "rationale": "선택"}
```

| 필드 | 필수 | 설명 |
|---|---|---|
| `id` | 권장 | 항목 식별자 |
| `task` | 선택 | 선택지 집합 이름. 같은 `task`는 선택지가 같아야 함(02가 이 단위로 헤드를 만든다) |
| `instruction` | 필수 | 판단할 내용(질문+답변, 분류할 문장 등) |
| `choices` | 필수 | 선택지 문자열 배열. 2개 이상 |
| `answer` | 필수 | `choices` 중 하나와 **정확히 같은 문자열** |
| `rationale` | 선택 | 판단 이유. `03 --with-rationale` 로 함께 학습 가능 |

- `data/sample.jsonl`(28건)은 **평가용**, `data/train.jsonl`(80건)은 **학습용**으로 내용이 겹치지 않습니다.
  같은 데이터로 학습하고 평가하면 "외운 것"을 실력으로 착각하므로, 실제로 쓸 때도 반드시 나눠 주세요.
- 실제 데이터로 바꿀 때는 `--data` / `--eval-data` 로 경로만 지정하면 됩니다.

---

## 8. 결과 파일 형식 (01·02·03이 모두 같은 형식)

`outputs/preds_*.csv` — `05_eval.py`가 이 파일들을 읽습니다.

| 열 | 의미 |
|---|---|
| `method` | `logprob_mean` / `head_linear_probe` / `lora_mean` 등 |
| `answer_index`, `predicted_index` | 정답·예측의 선택지 위치 |
| `correct` | 1/0 |
| `confidence` | 고른 선택지의 확률 |
| `choice_probs` | 선택지 순서대로의 확률 JSON 배열 |
| `scores` | 선택지별 원점수(로그확률) JSON — 디버깅용 |

`04`가 추가로 만드는 것: `eval_report.md`, `reliability_*.csv`(신뢰도 다이어그램용, 엑셀에서 바로 그림).

---

## 9. 흔한 오류와 대처

| 증상 | 원인 | 대처 |
|---|---|---|
| `ModuleNotFoundError: transformers.models.qwen3_5` 또는 `KeyError: 'qwen3_5'` | transformers가 Qwen3.5보다 오래됨 | `pip install -U --pre transformers` (5.x 이상) |
| `Setting pad_token_id to eos_token_id` 경고가 반복 | 토크나이저에 pad 토큰이 없음 | `common.load_tokenizer`가 pad=eos로 자동 처리. 직접 코드를 쓸 때도 같은 처리를 할 것 |
| `TypeError: apply_chat_template() got an unexpected keyword argument 'enable_thinking'` | 구형 chat template | 코드가 자동으로 인자 없이 재시도함. 그래도 실패하면 transformers 업그레이드 |
| `torch.cuda.OutOfMemoryError` | 학습 시 활성값까지 메모리 초과 | `--batch-size 1`, `--grad-accum` ↑, `--max-len 256`, 4bit, 또는 2B로 변경. 03은 gradient checkpointing이 기본 on |
| 결과가 `!!!!` 처럼 반복 | bf16 미지원 GPU에서 float16 넘침 | `common.pick_dtype()`이 자동으로 float16을 고르지만, 그래도 깨지면 `load_model(dtype=torch.float32)` 로 테스트 |
| `You can't move a model that has been loaded with device_map` | device_map 로드 후 `.to()` 호출 | 02는 헤드만 옮기도록 작성돼 있음. 직접 고칠 때 주의 |
| `None of the inputs have requires_grad=True` 경고 | gradient checkpointing + 얼린 임베딩 | 03은 `enable_input_require_grads()`를 호출함 |
| bitsandbytes 설치 실패(Windows) | CUDA 버전 불일치 | `pip install bitsandbytes --upgrade` 후 재시도. 실패하면 4B bf16으로 진행 |
| 로그확률이 다 비슷해서 argmax가 무의미 | 선택지가 길어 `sum` 채점이 눌림 | `--score mean`(길이 정규화) 사용. 01이 세 방식의 정확도를 모두 출력해 비교해 줌 |
| 정확도가 1.0이거나 0.0으로 튐 | 항목이 28개뿐 | 정상. 데이터를 늘려야 신뢰 가능(최소 수백 건) |

---

## 10. 이 코드가 "Jev 같은 모델"의 축소판인 이유 — 그리고 실제와의 거리

Jev 같은 판단 모델의 핵심은 **"생성"이 아니라 "확률 판독"** 입니다. 문장을 만들어 내는 대신, 정해진 선택지 각각에 얼마의 확률을 주는지 읽어 가장 높은 것을 고릅니다. 이 실습의 01이 바로 그 원형이고, 02는 그 판독을 **작은 전용 헤드**로 옮기는 단계, 03은 그 판독 자체를 **데이터로 교정**하는 단계입니다.

```
[원형]  질문+선택지 → 모델 → 선택지별 로그확률 → softmax → argmax/문턱값
                ↑
        01: 그대로 재기
        02: lm_head(24만 차원) → 판단 헤드(2~5차원)
        03: LoRA로 확률 분포 자체를 우리 업무에 맞게 조정
```

다만 실제 제품과의 차이는 분명히 알아 두셔야 합니다.

- **데이터 규모**: 여기서는 28건입니다. 실제 판단 모델은 수만~수십만 건의 라벨로 학습합니다. 28건 결과로 "이 모델은 판단을 잘한다"고 결론 내리면 안 됩니다.
- **보정 학습**: 실제로는 정확도보다 **보정**이 중요합니다. 확률을 문턱값(0.8 이상이면 자동 처리, 그 아래는 사람에게)으로 쓰기 때문입니다. 이를 위해 온도 스케일링, 라벨 스무딩, 사람 피드백(RLHF/RLCD류)을 추가로 씁니다. 이 실습의 04는 그 필요성을 **측정**하는 도구까지입니다.
- **프롬프트·헤드 설계**: 선택지를 어떻게 제시하고(레이블 토큰을 1개로 강제할지), 헤드를 어느 층에 얹을지는 성능을 크게 바꿉니다. 02의 `--mode last_block`, `01 --score` 옵션이 그 감각을 잡기 위한 장치입니다.
- **평가의 함정**: 정확도만 보면 "자신 있게 틀리는" 모델이 좋아 보입니다. 04의 `과신` 열과 "확신 0.9 이상 오답" 목록을 꼭 확인하세요.

---

## 11. 라이선스

- **Qwen 시리즈: Apache 2.0** (상업적 사용 가능). 모델 카드의 최신 고지를 확인할 것.
- **Gemma 등 다른 모델로 바꿔 실험할 경우**: 구글 자체 약관이 적용되므로 코드의 `--model`만 바꾸지 말고 **라이선스를 먼저 확인**해야 합니다. (`common.py`, 각 스크립트 상단에도 같은 주석이 있습니다)
- 데이터(`data/*.jsonl`)는 이 실습을 위해 직접 작성한 것이며, 자유롭게 늘리거나 바꿔 쓰면 됩니다.

---

## 12. 검증 범위에 대한 정직한 안내

이 폴더의 코드는 **GPU가 없는 환경에서 작성**되었습니다. 따라서:

- ✅ `python -m py_compile` 문법 검증 통과, 데이터 JSON/라벨 일관성 검사 통과, 04의 지표 계산 로직은 가짜 예측 데이터로 실제 실행 검증.
- ⚠️ **모델을 실제로 내려받아 돌린 것은 아닙니다.** Qwen3.5-4B 가중치(8GB)와 GPU가 필요한 부분은 로컬 PC에서의 첫 실행이 곧 검증입니다.
- 첫 실행에서 막히면 9절(흔한 오류)을 보고, 그래도 안 되면 오류 메시지 전체를 알려 주세요. `common.py`의 백본 탐색 경로나 로더 부분은 바로 손볼 수 있습니다.

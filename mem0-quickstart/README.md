# mem0 빠른 실습 (mem0-quickstart)

[mem0](https://github.com/mem0ai/mem0)는 AI 에이전트에 **장기기억**을 붙이는 라이브러리다.
여기 있는 파일들은 설치 → 설정 → 샘플 대화 적재 → 검색까지 한 번에 해 보는 실습 묶음이다.

- 코드 기준: **mem0ai 2.2.1** (2026-11에 실제 설치해 API 시그니처를 확인했다)
- 샘플 데이터는 전부 **가상의 대화**다. 실제 개인정보는 들어 있지 않다.
- 네 가지 실행 경로를 지원한다: **OpenAI / 완전 로컬(Ollama) / OpenAI 호환 서버 / NVIDIA NIM**

---

## 1. mem0가 뭘 하는지 (3계층으로 보기)

mem0를 "대화 로그 저장소"로 오해하면 결과가 이상해 보인다. 실제로는 세 층으로 나뉜다.

| 계층 | 내용 | 이 실습에서 어디에 있나 |
| --- | --- | --- |
| ① 원본 로그 | 사용자와 AI가 주고받은 대화 그 자체 | **mem0 밖** — 우리 앱/DB가 보관 (`seed_data.json`이 그 역할) |
| ② 인덱스 | ①에서 뽑아낸 **핵심 사실**을 임베딩한 벡터 | `data/qdrant/` (내장 Qdrant) |
| ③ 핵심 요약 | LLM이 문장으로 정리한 기억 (예: "사용자는 RTX 3060 Ti 12GB를 쓴다") | ②와 같은 곳에 문장+벡터로 저장, 변경 이력은 `data/history.db` |

`memory.add(messages, user_id=...)`를 부르면 내부에서 이렇게 흐른다.

1. LLM이 대화에서 **기억할 만한 사실**을 뽑는다 (`infer=True` 기본).
2. 기존 기억과 비교해 **ADD / UPDATE / DELETE / NOOP** 중 하나를 고른다.
   → 이 판단을 JSON 구조화 출력으로 받기 때문에, 이 단계에 쓰는 LLM이 성능을 좌우한다.
3. 새 기억을 임베딩해 벡터스토어에 넣고, 무엇이 바뀌었는지 `history.db`에 남긴다.

`memory.search("질문", filters={"user_id": ...})`는 ③의 **문장**끼리 유사도를 비교한다.
그래서 "대화 원문을 그대로 검색"하는 것과 결과가 다르다.

---

## 2. 폴더 구성

| 파일 | 역할 |
| --- | --- |
| `README.md` | 지금 보고 있는 문서 |
| `requirements.txt` | mem0ai, python-dotenv, pytest (그 외 없음) |
| `.env.example` | 환경변수 예시. **실제 값은 없음.** `.env`로 복사해서 채운다 |
| `.gitignore` | `.env`, `data/`, `__pycache__/`, `*.db`, `.venv/` 차단 |
| `config.py` | 설정 dict 4종: `openai_config()`, `local_ollama_config()`, `compatible_config()`, `nvidia_nim_config()` |
| `nim_embedder.py` | NVIDIA QA 임베딩용 `input_type` 어댑터 (경로 D 전용) |
| `seed_data.json` | 가상 대화 14개 메시지 (7쌍) |
| `demo.py` | 적재 → 검색 → 전체 조회. `--config`, `--reset`, `--query`, `--retry` 지원 |
| `test_mem0.py` | 오프라인 pytest (API 키 없이 통과) |
| `install-windows.ps1` | Windows 설치 스크립트 (관리자 권한 불필요) |
| `install.sh` | Linux/macOS/WSL용 같은 스크립트 |

데이터(`data/`)는 저장소에 올라가지 않는다. 지우고 싶으면 `--reset` 또는 폴더 삭제.

---

## 3. 준비물과 예상 소요 시간

| 경로 | 필요한 것 | 첫 실행까지 |
| --- | --- | --- |
| A. OpenAI | Python 3.10+, OpenAI API 키(유료) | **5분** (적재 1~2분) |
| B. 완전 로컬 | Python 3.10+, Ollama, GPU 권장 | 20~40분 (모델 다운로드가 대부분) |
| C. OpenAI 호환 | Python 3.10+, vLLM/LM Studio/OpenRouter/Groq 중 하나 | 10분 |
| D. NVIDIA NIM | Python 3.10+, NVIDIA API 키(무료) | **5분** (첫 적재는 요청 제한 때문에 느릴 수 있음) |

Python은 3.10 이상이면 된다. 이 폴더 기준으로 3.12에서 확인했다.

---

## 4. 설치 (한 번만)

### Windows

```powershell
cd mem0-quickstart
powershell -ExecutionPolicy Bypass -File .\install-windows.ps1
Copy-Item .env.example .env
```

### Linux / macOS / WSL

```bash
cd mem0-quickstart
bash install.sh
cp .env.example .env
```

스크립트를 쓰지 않고 직접 해도 같다.

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

> 이후 명령은 `.\.venv\Scripts\python.exe`(Windows) 또는 `.venv/bin/python`(Linux/macOS)을
> `python`으로 줄여 쓴다. 가상환경을 활성화했다면 그냥 `python`을 쓰면 된다.

---

## 5. 경로 A — OpenAI로 5분 안에

1. `.env`에 키를 넣는다.

```dotenv
OPENAI_API_KEY=sk-...
```

2. 실행한다.

```bash
python demo.py --config openai
```

기본 모델은 `gpt-4o-mini`(기억 추출) + `text-embedding-3-small`(1536차원)이다.
바꾸려면 `.env`에 `MEM0_LLM_MODEL`, `MEM0_EMBED_MODEL`을 넣는다.

**임베딩 모델을 바꾸면 차원이 달라져 기존 벡터를 못 쓴다.** 그때는 `--reset`으로 `data/`를 비우고 다시 적재한다.

---

## 6. 경로 B — API 키 없이 완전 로컬 (Ollama + 내장 Qdrant)

1. Ollama를 설치한다.

```powershell
winget install Ollama.Ollama        # Windows
```

```bash
curl -fsSL https://ollama.com/install.sh | sh    # Linux
brew install ollama                              # macOS
```

2. 모델 두 개를 받는다. (합계 9GB 안팎, 다운로드가 오래 걸린다)

```bash
ollama pull qwen2.5:14b          # 기억 추출용 LLM
ollama pull nomic-embed-text     # 임베딩 (768차원)
```

3. 실행한다.

```bash
python demo.py --config local
```

- RTX 3060 Ti 12GB라면 `qwen2.5:14b`(4비트 양자화, 9GB 안팎)가 현실적이다. 32B는 메모리에 안 들어가거나 매우 느려진다.
- Ollama가 꺼져 있으면 `ollama serve`를 먼저 실행한다.
- 모델명을 바꾸려면 `.env`에 `MEM0_LLM_MODEL`, `MEM0_EMBED_MODEL`, `MEM0_EMBED_DIMS`를 넣는다.

---

## 7. 경로 C — OpenAI 호환 엔드포인트 (vLLM · LM Studio · Ollama `/v1` · OpenRouter · Groq)

OpenAI 규격(`/v1/chat/completions`, `/v1/embeddings`)을 말하는 서버면 무엇이든 붙는다.
`config.py`의 `compatible_config(base_url, llm_model, embed_model=None, ...)`를 쓰고,
`demo.py`에서는 `--config compatible`로 실행한다.

```dotenv
OPENAI_BASE_URL=http://localhost:1234/v1
MEM0_LLM_MODEL=qwen2.5-14b-instruct
MEM0_EMBED_MODEL=nomic-embed-text
MEM0_EMBED_DIMS=768
# 임베딩 서버가 LLM과 다르면
# MEM0_EMBED_BASE_URL=https://api.openai.com/v1
```

```bash
python demo.py --config compatible
```

서버별 예시:

| 서버 | `OPENAI_BASE_URL` | 비고 |
| --- | --- | --- |
| Ollama | `http://localhost:11434/v1` | 모델명은 `qwen2.5:14b`처럼 Ollama 표기 |
| LM Studio | `http://localhost:1234/v1` | 서버 탭에서 Local Server 켜기 |
| vLLM | `http://localhost:8000/v1` | `--served-model-name` 값과 모델명을 맞출 것 |
| OpenRouter | `https://openrouter.ai/api/v1` | 임베딩은 제공하지 않으므로 `MEM0_EMBED_BASE_URL` 필수 |
| Groq | `https://api.groq.com/openai/v1` | 임베딩 모델이 없어 `MEM0_EMBED_BASE_URL` 필요 |

### 이 경로의 주의사항 3가지

1. **7B급 소형 모델은 자주 깨진다.**
   mem0는 ADD/UPDATE/DELETE 판단을 **JSON 구조화 출력**으로 받는다
   (코드 확인: `generate_response(..., response_format={"type": "json_object"})`).
   소형 모델은 JSON을 흘리거나 형식이 어긋나 적재가 실패한다. 실습에는 **Qwen2.5 14B~32B급**을 권한다.
   12GB VRAM이면 14B(4비트)가 현실적이고, 32B는 양자화해도 느리다.
2. **임베딩 차원은 처음에 하나로 고정한다.**
   벡터 컬렉션은 차원이 고정이라, 임베딩 모델을 바꾸면 기존 데이터와 섞이지 않는다.
   바꿀 때는 `python demo.py --config compatible --reset` 으로 `data/`를 비우고 다시 적재한다.
3. **한국어 지시를 따르는 모델을 고른다.**
   샘플 대화가 한국어라, 추출 프롬프트를 한국어 문장에서 제대로 따르는 모델이어야 기억이 쓸 만하게 나온다.

---

## 8. 경로 D — NVIDIA NIM으로 실행하기

NVIDIA API Catalog(build.nvidia.com)는 `https://integrate.api.nvidia.com/v1`에서 **완전한 OpenAI 호환**이라
설정만 바꾸면 같은 코드가 그대로 돈다. 무료 등급으로 실습하기에 충분하다.

### 1) 키 발급

1. [build.nvidia.com](https://build.nvidia.com)에 가입한다(NVIDIA Developer Program, 무료, 카드 정보 요구 없음).
2. 아무 모델 페이지에서 **Get API Key**를 누르거나 [설정 화면](https://build.nvidia.com/settings)에서 발급한다.
3. 키는 `nvapi-` 로 시작하고 **카탈로그 전체에 공용**이다. 한 번만 보여 주므로 바로 `.env`에 넣는다.

### 2) `.env`에 세 줄

```dotenv
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
# 모델을 바꿀 때만 (비워 두면 아래 기본값)
MEM0_LLM_MODEL=meta/llama-3.3-70b-instruct
MEM0_EMBED_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v1
MEM0_EMBED_DIMS=2048
```

### 3) 실행

```bash
python demo.py --config nim
```

기본 조합은 `meta/llama-3.3-70b-instruct`(기억 추출) + `nvidia/llama-3.2-nv-embedqa-1b-v1`(2048차원, 한국어 포함 다국어 임베딩)이다.
2026-11에 카탈로그와 엔드포인트에서 이름을 확인한 값이다.

### 4) 임베딩에서 삐끗하기 쉬운 지점

- **NV 임베딩은 `input_type`을 요구한다.** `nvidia/…-nv-embedqa-…`, `nemotron-…-embed-…` 계열은
  요청에 `input_type="passage"`(적재) 또는 `"query"`(검색)를 넣어야 하고, 없으면
  400 `'input_type' parameter is required for asymmetric models` 가 난다.
  mem0의 OpenAI 임베더는 이 필드를 보내지 않으므로, 이 폴더의 `nim_embedder.py`가
  `Memory.from_config()` 뒤에 임베더를 감싸 `memory_action`에 따라 적재는 `passage`, 검색은 `query`로 보낸다.
  **직접 코드를 쓸 때도 `attach_nim_embedder(memory)`를 호출해야 한다.**

  ```python
  from mem0 import Memory
  from config import nvidia_nim_config
  from nim_embedder import attach_nim_embedder

  memory = Memory.from_config(nvidia_nim_config())
  attach_nim_embedder(memory)     # 비대칭 모델일 때만 실제로 바뀐다
  ```

- `baai/bge-m3`는 **2026-08-25부로 서비스 종료(EOL)** 되어 호출하면 410이 온다.
  예전 글에 이 모델이 자주 보이지만 지금은 쓸 수 없다.
- 대칭형이라 `input_type`이 필요 없는 `snowflake/arctic-embed-l`(1024차원)도 있지만 영어 중심이라 한국어에는 권하지 않는다.

### 5) 무료 등급은 분당 요청 수가 낮다

- NVIDIA는 정확한 수치를 공개하지 않지만, 무료 계정은 대개 **분당 40회 수준**으로 알려져 있다
  (계정 전체 합산, 모델별 아님). 실제 한도는 build.nvidia.com 오른쪽 위에서 확인한다.
- 적재 한 번에 "기존 기억 검색 + 추출 + 임베딩" 호출이 붙어서, 대화 7묶음이면 **수십 회 호출**이 된다.
  첫 적재가 1~3분 걸릴 수 있고 중간에 429(Too Many Requests)가 날 수 있다.
- `demo.py`는 429를 만나면 **20초 → 40초 간격으로 자동 재시도**한다(`--retry` 횟수, 기본 3회).
  그래도 계속 걸리면 창을 그대로 두고 몇 분 뒤 같은 명령을 다시 실행하면 된다.
  이미 들어간 기억은 그대로 남으므로 `--reset` 없이 다시 돌려도 된다.

### 6) 모델 이름 확인

```bash
curl https://integrate.api.nvidia.com/v1/models -H "Authorization: Bearer $NVIDIA_API_KEY"
```

카탈로그에 있는 다른 LLM 예: `nvidia/nemotron-3.5-lightning-30b-a3b`(빠름),
`nvidia/nemotron-3-ultra-550b-a55b`(품질 우선), `moonshotai/kimi-k3.1-pro`.

---

## 9. 실행하면 이렇게 나온다

```text
[1/5] 환경 확인
  Python: 3.12.16
  작업 폴더: ...\mem0-quickstart
  .env 읽음: ...\mem0-quickstart\.env

[2/5] 데이터 폴더 준비
  데이터 위치: ...\mem0-quickstart\data

[3/5] mem0 초기화
  경로: nim
  LLM: openai / meta/llama-3.3-70b-instruct
  LLM base_url: https://integrate.api.nvidia.com/v1
  임베딩: openai / nvidia/llama-3.2-nv-embedqa-1b-v1
  벡터스토어: qdrant → ...\data\qdrant
  이력 DB: ...\data\history.db
  임베딩 어댑터: NIM QA 모델이라 input_type을 함께 보낸다 (적재 passage / 검색 query)
  초기화 완료

[4/5] 샘플 대화 적재
  메시지 14개 → 대화 7묶음
  (1) add 2개 메시지: 요즘 하천 수위 예측 모델 마무리 작업 중이야. LSTM이랑 GRU로…
      · ADD: 하천 수위 예측 모델을 LSTM과 GRU로 4년째 개발 중
  ...

[5/5] 검색과 전체 조회

  검색: '내 수위 예측 모델은 어떤 구조였지?'
    - LSTM과 GRU로 최대 7일 앞 수위를 예측하는 모델을 개발 중  score=0.71  2026-11-26T04:12:03
  ...
  전체 조회 (user_id=demo_user)
    총 18건
```

LLM이 문장을 요약해서 넣기 때문에 처음 돌릴 때마다 기억 문장과 개수가 조금씩 달라진다. 정상이다.

---

## 10. 테스트 (API 키 없이)

```bash
python -m pytest -q
```

이 폴더 기준 결과: **35 passed, 1 skipped**
(마지막 1개는 `OPENAI_API_KEY`가 있을 때만 도는 실제 호출 테스트다. mem0ai가 설치돼 있지 않으면
스키마 검증 테스트 4개가 추가로 skip된다.)

테스트가 확인하는 것: 설정 dict 구조 4종, mem0 스키마 통과 여부, 데이터 경로와 `--reset` 안전장치,
샘플 데이터 스키마와 개인정보 패턴 부재, 메시지 변환, 구버전/신버전 API 차이 흡수,
429 재시도 동작, NIM 임베딩이 `input_type`을 제대로 붙이는지.

---

## 11. mem0 API 요약 (2.2.1 기준)

```python
from mem0 import Memory
from config import openai_config

memory = Memory.from_config(openai_config())

# 1) 적재 — 메시지 목록을 넣는다
messages = [
    {"role": "user", "content": "나는 LSTM으로 수위를 예측한다."},
    {"role": "assistant", "content": "기억해 둘게요."},
]
result = memory.add(messages, user_id="demo_user")
# → {"results": [{"id": "...", "memory": "LSTM으로 수위를 예측한다", "event": "ADD"}, ...]}

# 2) 검색 — user_id는 filters 안에 넣는다 (2.x 방식)
hits = memory.search("무슨 모델을 쓰지?", filters={"user_id": "demo_user"}, top_k=5)
# → {"results": [{"id": "...", "memory": "...", "score": 0.82, ...}]}

# 3) 전체 조회
all_items = memory.get_all(filters={"user_id": "demo_user"}, top_k=50)

# 4) 초기화 (메모리 객체 기준 — data/ 폴더는 그대로)
memory.reset()
```

주의할 점:

- **`memory.search(..., user_id="x")`는 2.x에서 오류가 난다.**
  `filters={"user_id": "x"}` 를 써야 한다. 구버전(0.x)은 `user_id=`를 받았으므로,
  `demo.py`의 `search_memories()` / `get_all_memories()`가 두 방식을 모두 처리한다.
- 적재 결과의 `event`가 `ADD`/`UPDATE`/`DELETE`/`NOOP` 중 무엇인지 보면 LLM이 어떻게 판단했는지 알 수 있다.
- `user_id` 말고 `agent_id`, `run_id` 단위로도 기억을 나눌 수 있다.

---

## 12. `--reset` 경고

```bash
python demo.py --config local --reset
```

**`data/` 폴더를 묻지 않고 통째로 지운다.** 벡터스토어와 이력 DB가 모두 사라지고 되돌릴 수 없다.
다른 실습 데이터가 들어 있는 폴더라면 먼저 백업할 것. (안전장치로 프로젝트 폴더 자체나 이름이 `data`가 아닌
외부 경로는 지우지 않고 오류를 낸다. 테스트로 확인한다.)

실습을 처음부터 다시 하는 게 아니라면 `--reset` 없이 그냥 실행하면 된다. 기존 기억에 이어서 쌓인다.

---

## 13. 흔한 오류와 해결

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `mem0ai가 설치되지 않았다` | venv 밖에서 실행 | `.\.venv\Scripts\python.exe demo.py ...` 처럼 venv 파이썬으로 실행 |
| `OPENAI_API_KEY가 없다` / `NVIDIA_API_KEY가 없다` | `.env` 미작성 | `.env.example`을 복사해 키를 채운다. `.env`는 저장소에 올리지 않는다 |
| 401 / 403 | 키 오류·오타 | 키를 다시 복사. NVIDIA 키는 `nvapi-` 로 시작 |
| 404 model not found | 모델명 오타, 카탈로그에서 내려간 모델 | `/v1/models` 목록 또는 카탈로그에서 정확한 이름 확인 |
| 400 `'input_type' parameter is required` | NVIDIA 비대칭 임베딩 | `attach_nim_embedder(memory)` 호출(경로 D 참고) |
| 410 Gone (예: `baai/bge-m3`) | 모델 서비스 종료 | 다른 임베딩 모델로 바꾸고 `--reset` |
| 429 Too Many Requests | 분당 요청 제한 | 그대로 두면 `demo.py`가 자동 재시도. 계속되면 몇 분 뒤 재실행 |
| `Collection ... dimension mismatch` | 임베딩 차원 변경 | `--reset` 후 재적재 |
| Ollama 연결 실패 | 서버 꺼짐 | `ollama serve` 실행 |
| 적재는 되는데 검색 결과가 0건 | `user_id` 불일치 또는 차원 불일치 | `--user-id`를 맞추고 로그 확인 |
| LLM이 JSON을 깨서 적재 실패 | 소형 모델 한계 | 14B 이상 모델로 바꾼다(경로 C 주의사항 1) |

---

## 14. 버전에 따라 다를 수 있음

아래는 2.2.1에서 확인한 내용이다. 다른 버전에서는 달라질 수 있다.

- `search` / `get_all`의 `filters=` 방식 (1.0 이전은 `user_id=` 직접 전달)
- `Memory()` 기본값: OpenAI `gpt-5-mini` + `text-embedding-3-small`, Qdrant `/tmp/qdrant`, `~/.mem0/history.db`
  (이 실습은 `config.py`에서 모두 `data/` 아래로 바꿔 잡는다)
- Ollama 임베더의 기본 차원이 512로 잡혀 있다 → `nomic-embed-text`는 실제 768이라 설정에서 맞춰 줘야 한다
- 확인하지 못한 부분: NVIDIA NIM 쪽 모델별 세부 동작(모델별 분당 한도, 임베딩 최대 토큰)은 NVIDIA 문서 기준이며
  실제 계정 한도는 build.nvidia.com 화면에서 확인해야 한다

---

## 15. 다음에 해 볼 것

1. `seed_data.json`을 자기 대화로 바꿔 보기 (개인정보는 넣지 말고 가상으로)
2. `memory.search(..., filters={"user_id": "x", "category": ...})` 대신 메타데이터 필터 실험
3. `agent_id`를 써서 에이전트별로 기억 분리하기
4. `memory.history(memory_id)`로 한 기억이 어떻게 바뀌었는지 추적하기

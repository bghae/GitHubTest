#!/usr/bin/env python
"""mem0-quickstart 데모.

seed_data.json의 가상 대화를 mem0에 적재하고, 검색과 전체 조회까지 해 본다.

사용 예 (mem0-quickstart 폴더 안에서):
    python demo.py --config openai
    python demo.py --config local
    python demo.py --config compatible
    python demo.py --config openai --reset          # data/ 를 지우고 처음부터
    python demo.py --config openai --query "내 GPU 사양은?" --query "도서관 책은?"

--reset 은 묻지 않고 mem0-quickstart/data/ 를 통째로 지운다. 되돌릴 수 없다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as appconfig  # noqa: E402  (같은 폴더의 config.py)
import nim_embedder  # noqa: E402

SEED_PATH = HERE / "seed_data.json"
DEFAULT_USER_ID = "demo_user"
DEFAULT_QUERIES = [
    "내 수위 예측 모델은 어떤 구조였지?",
    "내가 쓰는 그래픽카드 사양은?",
    "도서관에서 빌린 책 반납은 어떻게 됐지?",
]
VALID_ROLES = {"user", "assistant", "system"}

TOTAL_STEPS = 5
CONFIG_KINDS = ["openai", "local", "compatible", "nim"]


# ---------------------------------------------------------------- 도우미


def load_seed_data(path: Optional[str | Path] = None) -> list[dict]:
    """seed_data.json을 읽어 role/content 메시지 목록으로 돌려준다."""
    path = Path(path) if path else SEED_PATH
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"{path}: 최상위는 메시지 목록(list)이어야 한다")
    return data


def to_messages(records: Iterable[dict]) -> list[dict]:
    """seed 레코드를 mem0 add()가 받는 메시지 형식으로 변환하고 검증한다.

    mem0는 [{"role": "user", "content": "..."}] 형태를 받는다.
    """
    messages = []
    for i, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{i}번째 항목이 dict가 아니다: {record!r}")
        role = str(record.get("role", "")).strip().lower()
        content = record.get("content")
        if role not in VALID_ROLES:
            raise ValueError(f"{i}번째 항목의 role이 잘못됐다: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{i}번째 항목의 content가 비어 있다")
        messages.append({"role": role, "content": content.strip()})
    if not messages:
        raise ValueError("메시지가 하나도 없다")
    return messages


def to_conversations(messages: list[dict], size: int = 2) -> list[list[dict]]:
    """메시지를 size개씩 묶어 대화 단위로 만든다. (기본: 질문/답변 한 쌍)"""
    if size < 1:
        raise ValueError("size는 1 이상이어야 한다")
    return [messages[i : i + size] for i in range(0, len(messages), size)]


def build_filters(user_id: str) -> dict:
    """mem0 2.x의 검색 필터. user_id는 반드시 filters 안에 넣는다."""
    user_id = (user_id or "").strip()
    if not user_id:
        raise ValueError("user_id가 비어 있다")
    return {"user_id": user_id}


def result_items(payload: Any) -> list[dict]:
    """mem0 응답을 항상 list로 맞춘다.

    2.x는 {"results": [...]}, 일부 구버전은 [...]를 돌려준다.
    """
    if isinstance(payload, dict):
        items = payload.get("results") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def search_memories(memory, query: str, user_id: str, top_k: int = 5) -> list[dict]:
    """버전 차이를 흡수한 검색.

    mem0 2.x: memory.search(query, filters={"user_id": ...})
    mem0 0.x: memory.search(query, user_id=...)
    """
    try:
        payload = memory.search(query, filters=build_filters(user_id), top_k=top_k)
    except TypeError:
        payload = memory.search(query, user_id=user_id, limit=top_k)
    except ValueError as exc:
        if "filters" not in str(exc):
            raise
        payload = memory.search(query, user_id=user_id, limit=top_k)
    return result_items(payload)


def get_all_memories(memory, user_id: str, top_k: int = 50) -> list[dict]:
    """버전 차이를 흡수한 전체 조회."""
    try:
        payload = memory.get_all(filters=build_filters(user_id), top_k=top_k)
    except TypeError:
        payload = memory.get_all(user_id=user_id)
    except ValueError as exc:
        if "filters" not in str(exc):
            raise
        payload = memory.get_all(user_id=user_id)
    return result_items(payload)


def memory_text(item: dict) -> str:
    """저장된 기억 문장을 꺼낸다. (키 이름이 버전마다 조금 다르다)"""
    return str(item.get("memory") or item.get("text") or item.get("data") or "").strip()


def format_memory_line(item: dict) -> str:
    """한 줄 출력용 문자열."""
    text = memory_text(item) or "(빈 기억)"
    score = item.get("score")
    score_part = f"  score={score:.3f}" if isinstance(score, (int, float)) else ""
    created = item.get("created_at") or ""
    created_part = f"  {created[:19]}" if created else ""
    return f"- {text}{score_part}{created_part}"


def step(n: int, message: str) -> None:
    print(f"\n[{n}/{TOTAL_STEPS}] {message}")


# ---------------------------------------------------------------- 실행


def retry_call(func, *args, attempts: int = 3, base_wait: float = 20.0, label: str = "", **kwargs):
    """분당 요청 제한(429)을 만나면 쉬었다가 다시 시도한다.

    NVIDIA NIM 무료 구간이나 무료 API는 분당 요청 수가 낮아서
    첫 적재 중에 429가 나기 쉽다.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 종류를 가리지 않고 429만 골라 본다
            message = str(exc)
            retryable = "429" in message or "rate limit" in message.lower() or "too many requests" in message.lower()
            last_error = exc
            if not retryable or attempt == attempts:
                raise
            wait = base_wait * attempt
            print(f"      ! 요청 제한에 걸렸다({label or func.__name__}). {wait:.0f}초 쉬고 재시도 {attempt}/{attempts - 1}")
            time.sleep(wait)
    raise last_error  # pragma: no cover


def explain_error(exc: Exception) -> list[str]:
    """자주 나는 API 오류를 사람이 읽을 수 있는 안내로 바꾼다."""
    msg = str(exc).lower()
    name = type(exc).__name__
    hints: list[str] = []

    if "input_type" in msg:
        hints.append("이 임베딩 모델은 NIM이 input_type(query/passage)을 요구한다.")
        hints.append("README의 'NVIDIA NIM으로 실행하기'에서 임베딩 모델 설명을 볼 것.")
        hints.append("baai/bge-m3 같은 대칭형 모델을 쓰면 이 오류가 없다.")
    if isinstance(exc, ImportError) or "no module named" in msg or "modulenotfound" in msg:
        hints.append("파이썬 패키지가 빠졌다. pip install -r requirements.txt 를 다시 실행할 것.")
        hints.append("(경로 B는 ollama 패키지가, 경로 D는 mem0ai 2.x가 필요하다)")
    if (
        "401" in msg
        or "403" in msg
        or "unauthorized" in msg
        or "forbidden" in msg
        or "permission" in msg
        or "authentication" in msg
        or "invalid jwt" in msg
        or "api key" in msg
    ):
        hints.append("키가 거부됐다(401/403). .env의 키 값을 다시 확인할 것. (NVIDIA 키는 nvapi- 로 시작)")
        hints.append("키를 바꿔 .env에 넣은 뒤 다시 실행하면 된다. data/ 는 지우지 않아도 된다.")
    if "404" in msg or ("model" in msg and "not found" in msg):
        hints.append("모델 이름이 서버에 없다. 카탈로그의 정확한 model id를 확인할 것.")
    if "dimension" in msg or "vector" in msg:
        hints.append("임베딩 차원이 기존 컬렉션과 다르다. --reset 으로 data/ 를 지우고 다시 적재할 것.")
    if "connection" in msg or "connect" in msg or "refused" in msg or "timed out" in msg:
        hints.append("서버에 연결되지 않는다. 인터넷, 방화벽, (로컬이면) ollama 실행 상태를 볼 것.")

    if not hints:
        hints.append("예상하지 못한 오류다. 아래 메시지와 README의 '흔한 오류' 표를 볼 것.")
    hints.append(f"원본: {name}: {exc}")
    return hints


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("  (python-dotenv 미설치 → .env는 읽지 않고 환경변수만 쓴다)")
        return
    env_path = HERE / ".env"
    if env_path.exists():
        # override=True: Windows 시스템 환경변수가 .env보다 먼저 읽히면
        # 옛 OPENAI_BASE_URL(예: OpenRouter)·MEM0_EMBED_BASE_URL(예: NVIDIA)이
        # 남아 있을 때 어디로 요청이 가는지 알 수 없어진다. .env를 우선한다.
        before = {
            k: os.environ[k]
            for k in ("OPENAI_BASE_URL", "MEM0_EMBED_BASE_URL", "OPENAI_API_KEY")
            if k in os.environ
        }
        load_dotenv(env_path, override=True)
        if before:
            print("  (시스템 환경변수를 .env 값으로 덮어씀: " + ", ".join(before) + ")")
        print(f"  .env 읽음: {env_path}")
    else:
        print("  .env 없음 → 환경변수만 사용 (.env.example을 복사해 만들 것)")
    neutralize_openrouter_autoswitch()


def neutralize_openrouter_autoswitch() -> None:
    """mem0의 OpenAILLM은 OPENROUTER_API_KEY가 환경에 있으면 config를 무시하고
    모든 요청을 OpenRouter로 보낸다(mem0ai 2.2.1 mem0/llms/openai.py 42행).

    .env에 적은 base_url을 쓰는 것이 이 프로젝트의 전제이므로, OpenRouter를
    쓰겠다고 명시(MEM0_USE_OPENROUTER=1)하지 않는 한 그 변수를 지운다.
    """
    if os.environ.get("MEM0_USE_OPENROUTER") == "1":
        print("  OPENROUTER_API_KEY 사용 (MEM0_USE_OPENROUTER=1)")
        return
    if os.environ.pop("OPENROUTER_API_KEY", None):
        print("  OPENROUTER_API_KEY 무시: mem0가 OpenRouter로 새지 않도록 지움 "
              "(OpenRouter를 쓰려면 MEM0_USE_OPENROUTER=1)")


def build_memory(kind: str):
    """mem0 Memory 인스턴스를 만든다. mem0는 여기서 처음 import한다."""
    try:
        from mem0 import Memory
    except ImportError:
        raise SystemExit(
            "mem0ai가 설치되지 않았다. 다음을 먼저 실행할 것:\n"
            "    pip install -r requirements.txt"
        )

    if kind == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY가 없다. mem0-quickstart/.env에 넣거나\n"
                "API 키 없이 쓰려면 --config local (Ollama) 또는\n"
                "--config compatible, --config nim (OpenAI 호환/NVIDIA)을 쓸 것."
            )
        conf = appconfig.openai_config()
    elif kind == "local":
        conf = appconfig.local_ollama_config()
    elif kind == "nim":
        if not (os.getenv("NVIDIA_API_KEY") or os.getenv("MEM0_COMPAT_API_KEY")):
            raise SystemExit(
                "NVIDIA_API_KEY가 없다. https://build.nvidia.com 에서 키(nvapi-...)를 받아\n"
                "mem0-quickstart/.env 에 NVIDIA_API_KEY=nvapi-... 로 넣을 것.\n"
                "기본 모델: meta/llama-3.3-70b-instruct + baai/bge-m3 (바꾸려면 MEM0_LLM_MODEL, MEM0_EMBED_MODEL)"
            )
        conf = appconfig.nvidia_nim_config()
    elif kind == "compatible":
        if not os.getenv("OPENAI_BASE_URL") and not os.getenv("MEM0_LLM_MODEL"):
            raise SystemExit(
                "--config compatible 에는 최소한 OPENAI_BASE_URL, MEM0_LLM_MODEL 이 필요하다.\n"
                "예) .env 에\n"
                "    OPENAI_BASE_URL=http://localhost:11434/v1\n"
                "    MEM0_LLM_MODEL=qwen2.5:14b\n"
                "    MEM0_EMBED_MODEL=nomic-embed-text"
            )
        conf = appconfig.compatible_config()
    else:
        raise SystemExit(f"알 수 없는 --config 값: {kind}")

    return Memory.from_config(conf), conf


def describe_config(kind: str, conf: dict) -> None:
    llm = conf.get("llm", {})
    embedder = conf.get("embedder", {})
    vs = conf.get("vector_store", {}).get("config", {})
    print(f"  경로: {kind}")
    print(f"  LLM: {llm.get('provider')} / {llm.get('config', {}).get('model')}")
    if llm.get("config", {}).get("openai_base_url"):
        print(f"  LLM base_url: {llm['config']['openai_base_url']}")
    if llm.get("config", {}).get("ollama_base_url"):
        print(f"  Ollama: {llm['config']['ollama_base_url']}")
    print(f"  임베딩: {embedder.get('provider')} / {embedder.get('config', {}).get('model')}")
    if embedder.get("config", {}).get("openai_base_url"):
        print(f"  임베딩 base_url: {embedder['config']['openai_base_url']}")
    print(f"  벡터스토어: {conf.get('vector_store', {}).get('provider')} → {vs.get('path')}")
    print(f"  이력 DB: {conf.get('history_db_path')}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="mem0 빠른 실습: 샘플 대화 적재 → 검색 → 전체 조회",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python demo.py --config openai\n"
            "  python demo.py --config local --reset\n"
            "  python demo.py --config nim\n"
            "  python demo.py --config compatible --query '내 GPU 사양은?'\n"
        ),
    )
    parser.add_argument(
        "--config",
        choices=CONFIG_KINDS,
        default="openai",
        help=(
            "openai: OpenAI API 키 / local: Ollama 완전 로컬 / "
            "compatible: OpenAI 호환 서버 / nim: NVIDIA NIM"
        ),
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"기억을 붙일 사용자 ID (기본: {DEFAULT_USER_ID})",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="검색 질문. 여러 번 쓸 수 있다. 생략하면 예시 질문 3개",
    )
    parser.add_argument("--top-k", type=int, default=5, help="검색 결과 개수 (기본: 5)")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="mem0-quickstart/data/ 를 지우고 처음부터 적재한다 (되돌릴 수 없음)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="대화 묶음 크기: 메시지 몇 개를 한 번에 add 할지 (기본: 2 = 질문/답변 한 쌍)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="분당 요청 제한(429)을 만났을 때 재시도 횟수 (기본: 3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="오류 시 안내 문구 대신 전체 traceback을 보여 준다",
    )
    args = parser.parse_args(argv)

    started = time.time()
    user_id = args.user_id
    queries = args.queries or DEFAULT_QUERIES

    step(1, "환경 확인")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  작업 폴더: {HERE}")
    load_dotenv_if_available()

    step(2, "데이터 폴더 준비")
    if args.reset:
        print("  --reset: data/ 를 지운다. (되돌릴 수 없음)")
        print("  " + appconfig.reset_storage())
    root = appconfig.ensure_data_dir()
    print(f"  데이터 위치: {root}")

    step(3, "mem0 초기화")
    try:
        memory, conf = build_memory(args.config)
    except Exception as exc:  # noqa: BLE001 - 설정/패키지 문제를 짚어 주려고 잡는다
        if args.debug:
            raise
        print("\nmem0 초기화에 실패했다.")
        for line in explain_error(exc):
            print("  " + line)
        return 1
    describe_config(args.config, conf)
    if nim_embedder.attach_nim_embedder(memory):
        print("  임베딩 어댑터: NIM QA 모델이라 input_type을 함께 보낸다 (적재 passage / 검색 query)")
    print("  초기화 완료")

    step(4, "샘플 대화 적재")
    records = load_seed_data()
    messages = to_messages(records)
    print(f"  메시지 {len(messages)}개 → 대화 {len(to_conversations(messages, args.batch))}묶음")
    added = 0
    try:
        for idx, conversation in enumerate(to_conversations(messages, args.batch), start=1):
            first = conversation[0]["content"]
            preview = first[:38] + ("…" if len(first) > 38 else "")
            print(f"  ({idx}) add {len(conversation)}개 메시지: {preview}")
            result = retry_call(
                memory.add, conversation, user_id=user_id, attempts=args.retry, label=f"add {idx}"
            )
            for item in result_items(result):
                event = item.get("event", "").upper()
                text = memory_text(item)
                if event:
                    print(f"      · {event}: {text[:70]}")
                added += 1
        print(f"  적재 응답 이벤트 {added}건")

        step(5, "검색과 전체 조회")
        for query in queries:
            print(f"\n  검색: {query!r}")
            hits = retry_call(
                search_memories, memory, query, user_id, top_k=args.top_k,
                attempts=args.retry, label="search",
            )
            if not hits:
                print("    (결과 없음) 임베딩 차원이나 적재 로그를 확인할 것")
            for item in hits:
                print("    " + format_memory_line(item))

        print(f"\n  전체 조회 (user_id={user_id})")
        all_items = get_all_memories(memory, user_id)
        print(f"    총 {len(all_items)}건")
        for item in all_items:
            print("    " + format_memory_line(item))
    except KeyboardInterrupt:
        print("\n중단됨 (Ctrl+C)")
        return 130
    except Exception as exc:  # noqa: BLE001 - 사용자에게 원인을 짚어 주려고 잡는다
        if args.debug:
            raise
        print("\n적재/검색 중 오류가 났다.")
        for line in explain_error(exc):
            print("  " + line)
        print(f"  다시 실행: python demo.py --config {args.config} --reset")
        return 1

    elapsed = time.time() - started
    print("\n완료.")
    print(f"  걸린 시간: {elapsed:.1f}초")
    print(f"  벡터 데이터: {appconfig.data_paths()['qdrant_path']}")
    print(f"  이력 DB: {appconfig.data_paths()['history_db_path']}")
    print(f"  다시 처음부터: python demo.py --config {args.config} --reset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

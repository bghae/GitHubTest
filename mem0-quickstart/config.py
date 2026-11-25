"""mem0-quickstart 설정 모듈.

세 가지 실행 경로를 지원한다. 각 함수가 돌려주는 dict는
`mem0.Memory.from_config(...)`에 그대로 넘길 수 있다.

    1) openai_config()          — OpenAI API 키로 5분 안에 시작
    2) local_ollama_config()    — API 키 없이 완전 로컬 (Ollama + 내장 Qdrant)
    3) compatible_config()      — OpenAI 호환 엔드포인트
                                  (vLLM, LM Studio, Ollama 의 /v1, OpenRouter, Groq ...)
    4) nvidia_nim_config()      — NVIDIA NIM(NVIDIA API Catalog) 전용 편의 함수

기준 버전: mem0ai 2.2.1 (2026-11에 실제 설치해 시그니처와 설정 키를 확인했다).
mem0는 버전마다 설정 키 이름이 조금씩 달라질 수 있으므로,
다른 버전을 쓰면 README의 "버전에 따라 다를 수 있음" 항목을 확인할 것.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- 경로

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"

# 컬렉션 이름. 임베딩 모델/차원을 바꿀 때는 이 이름도 바꾸면
# 기존 컬렉션과 섞이지 않는다.
DEFAULT_COLLECTION = "mem0_quickstart"

# 자주 쓰는 임베딩 모델의 차원 수
EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "baai/bge-m3": 1024,
    "nvidia/nv-embedqa-e5-v5": 1024,
    "nvidia/llama-3.2-nv-embedqa-1b-v1": 2048,
}


def data_root(data_dir: Optional[str | Path] = None) -> Path:
    """데이터 루트 경로를 돌려준다. (만들지는 않는다)

    우선순위: 인자 > 환경변수 MEM0_DATA_DIR > ./data
    """
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    env_dir = os.getenv("MEM0_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return DEFAULT_DATA_DIR.resolve()


def data_paths(data_dir: Optional[str | Path] = None) -> dict:
    """벡터스토어(Qdrant) 경로와 이력 DB(SQLite) 경로를 돌려준다. (만들지는 않는다)"""
    root = data_root(data_dir)
    return {
        "data_dir": str(root),
        "qdrant_path": str(root / "qdrant"),
        "history_db_path": str(root / "history.db"),
    }


def ensure_data_dir(data_dir: Optional[str | Path] = None) -> Path:
    """데이터 디렉터리를 만든다. (이미 있으면 그대로 둔다)"""
    root = data_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def reset_storage(data_dir: Optional[str | Path] = None) -> str:
    """데이터 디렉터리를 통째로 지운다. 되돌릴 수 없다.

    안전장치: 프로젝트 폴더 바깥이나, 이름이 'data'가 아닌 기존 디렉터리는
    지우지 않고 예외를 던진다. (실수로 홈 디렉터리나 저장소 루트를 지우는 사고 방지)
    """
    root = data_root(data_dir)
    project = PROJECT_DIR.resolve()

    if root == project:
        raise ValueError(f"프로젝트 폴더 자체는 지울 수 없다: {root}")
    if root.parent == root:
        raise ValueError(f"파일 시스템 루트는 지울 수 없다: {root}")

    if env_dir := os.getenv("MEM0_DATA_DIR"):
        # 외부 경로를 지정한 경우: 이름이 data로 끝나야만 지운다.
        if Path(env_dir).expanduser().resolve() == root and root.name != "data":
            raise ValueError(
                f"MEM0_DATA_DIR로 지정한 경로({root})의 이름이 'data'가 아니라 지우지 않는다. "
                "직접 확인 후 지울 것."
            )

    if root.exists():
        shutil.rmtree(root)
        return f"삭제됨: {root}"
    return f"이미 없음: {root}"


# ---------------------------------------------------------------- 공통 조각


def _storage_block(collection_name: str, embed_dims: int, data_dir) -> dict:
    """Qdrant(로컬 디스크) + SQLite 이력 저장소 설정."""
    paths = data_paths(data_dir)
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": paths["qdrant_path"],
                "collection_name": collection_name,
                "embedding_model_dims": embed_dims,
                "on_disk": True,
            },
        },
        "history_db_path": paths["history_db_path"],
        "version": "v1.1",
    }


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


# ---------------------------------------------------------------- 경로 1: OpenAI


def openai_config(
    llm_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION,
    data_dir: Optional[str | Path] = None,
) -> dict:
    """OpenAI API 키만 있으면 되는 가장 짧은 경로.

    필요 환경변수: OPENAI_API_KEY
    선택 환경변수: MEM0_LLM_MODEL, MEM0_EMBED_MODEL
    """
    llm_model = llm_model or _env("MEM0_LLM_MODEL", "gpt-4o-mini")
    embed_model = embed_model or _env("MEM0_EMBED_MODEL", "text-embedding-3-small")

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embed_model,
            },
        },
    }
    config.update(
        _storage_block(
            collection_name,
            EMBED_DIMS.get(embed_model, 1536),
            data_dir,
        )
    )
    return config


# ---------------------------------------------------------------- 경로 2: 완전 로컬


def local_ollama_config(
    llm_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    embed_dims: Optional[int] = None,
    ollama_base_url: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION,
    data_dir: Optional[str | Path] = None,
) -> dict:
    """Ollama + 내장 Qdrant. API 키도, 인터넷도 필요 없다.

    필요 환경변수: 없음 (Ollama가 11434 포트에서 돌고 있어야 한다)
    선택 환경변수: MEM0_LLM_MODEL, MEM0_EMBED_MODEL, MEM0_EMBED_DIMS, OLLAMA_BASE_URL

    주의: mem0의 Ollama 임베더 기본 차원은 512로 잡혀 있는데
    nomic-embed-text의 실제 차원은 768이다. 그래서 embed_dims를 반드시 맞춰 준다.
    """
    llm_model = llm_model or _env("MEM0_LLM_MODEL", "qwen2.5:14b")
    embed_model = embed_model or _env("MEM0_EMBED_MODEL", "nomic-embed-text")
    if embed_dims is None and _env("MEM0_EMBED_DIMS"):
        embed_dims = int(os.environ["MEM0_EMBED_DIMS"])
    if embed_dims is None:
        embed_dims = EMBED_DIMS.get(embed_model, 768)
    ollama_base_url = ollama_base_url or _env("OLLAMA_BASE_URL", "http://localhost:11434")

    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "temperature": 0.1,
                "max_tokens": 2000,
                "ollama_base_url": ollama_base_url,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": embed_model,
                "embedding_dims": embed_dims,
                "ollama_base_url": ollama_base_url,
            },
        },
    }
    config.update(_storage_block(collection_name, embed_dims, data_dir))
    return config


# ---------------------------------------------------------------- 경로 3: OpenAI 호환


def compatible_config(
    base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    embed_base_url: Optional[str] = None,
    embed_dims: Optional[int] = None,
    api_key: Optional[str] = None,
    embed_api_key: Optional[str] = None,
    pass_dimensions_to_api: bool = False,
    collection_name: str = DEFAULT_COLLECTION,
    data_dir: Optional[str | Path] = None,
) -> dict:
    """OpenAI 호환 엔드포인트용 설정.

    대상: vLLM, LM Studio(http://localhost:1234/v1), Ollama의 OpenAI 호환
    (/v1), OpenRouter(https://openrouter.ai/api/v1), Groq, Together 등
    `/v1/chat/completions`와 `/v1/embeddings`를 말하는 서버.

    인자 대신 환경변수로도 넘길 수 있다:
        OPENAI_BASE_URL, MEM0_LLM_MODEL, MEM0_EMBED_MODEL,
        MEM0_EMBED_BASE_URL, MEM0_EMBED_DIMS

    api_key: 대부분의 로컬 서버는 아무 문자열이나 받는다. 기본값은
        MEM0_COMPAT_API_KEY > OPENAI_API_KEY > "not-needed".
    embed_dims: 임베딩 벡터 차원(벡터 컬렉션 차원과 같아야 한다).
        지정하지 않으면 1536으로 잡는다. 모델 실제 차원과 다르면 적재가 실패한다.
    pass_dimensions_to_api: True면 임베딩 요청에 `dimensions` 파라미터를 함께 보낸다.
        OpenAI의 matryoshka 모델은 이 파라미터를 지원하지만, vLLM·Groq 같은
        서버는 모르는 필드라 거부하는 경우가 많다. 그래서 기본은 False다.
        (True로 두면 embed_dims 값이 `dimensions`로도 전달된다)
    """
    base_url = base_url or _env("OPENAI_BASE_URL", "http://localhost:11434/v1")
    llm_model = llm_model or _env("MEM0_LLM_MODEL", "qwen2.5:14b")
    embed_model = embed_model or _env("MEM0_EMBED_MODEL", "nomic-embed-text")
    embed_base_url = embed_base_url or _env("MEM0_EMBED_BASE_URL", base_url)
    if embed_dims is None and _env("MEM0_EMBED_DIMS"):
        embed_dims = int(os.environ["MEM0_EMBED_DIMS"])
    if embed_dims is None:
        embed_dims = EMBED_DIMS.get(embed_model, 1536)
    api_key = api_key if api_key is not None else (
        _env("MEM0_COMPAT_API_KEY") or _env("OPENAI_API_KEY") or "not-needed"
    )
    embed_api_key = embed_api_key if embed_api_key is not None else api_key

    embedder_conf = {
        "model": embed_model,
        "openai_base_url": embed_base_url,
        "api_key": embed_api_key,
    }
    if pass_dimensions_to_api:
        embedder_conf["embedding_dims"] = embed_dims

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0.1,
                "max_tokens": 2000,
                "openai_base_url": base_url,
                "api_key": api_key,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": embedder_conf,
        },
    }
    config.update(_storage_block(collection_name, embed_dims, data_dir))
    return config


# ---------------------------------------------------------------- 경로 4: NVIDIA NIM

# NVIDIA API Catalog(build.nvidia.com)는 https://integrate.api.nvidia.com/v1 에서
# OpenAI 호환이므로, 위의 compatible_config()에 주소와 키만 바꿔 넣으면 된다.
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_DEFAULT_LLM = "meta/llama-3.3-70b-instruct"
# 한국어를 포함한 다국어 QA 임베딩(2048차원).
# input_type을 요구하는 비대칭 모델이라 demo.py가 nim_embedder.py로 감싸서 쓴다.
NIM_DEFAULT_EMBED = "nvidia/llama-3.2-nv-embedqa-1b-v1"


def nvidia_nim_config(
    api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    embed_dims: Optional[int] = None,
    llm_base_url: Optional[str] = None,
    embed_base_url: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION,
    data_dir: Optional[str | Path] = None,
) -> dict:
    """NVIDIA NIM(NVIDIA API Catalog) 설정.

    필요 환경변수: NVIDIA_API_KEY  (nvapi- 로 시작하는 키, https://build.nvidia.com)
    선택 환경변수: NVIDIA_BASE_URL, MEM0_LLM_MODEL, MEM0_EMBED_MODEL,
                  MEM0_EMBED_DIMS, MEM0_EMBED_BASE_URL

    기본 모델
        LLM   : meta/llama-3.3-70b-instruct
        임베딩: nvidia/llama-3.2-nv-embedqa-1b-v1 (2048차원, 한국어 포함 다국어)

    임베딩 모델은 QA(asymmetric) 계열이라 NIM이 input_type(``query``/``passage``)을
    요구한다. demo.py가 nim_embedder.attach_nim_embedder()로 감싸서 적재는 passage,
    검색은 query로 보내 준다. 직접 코드를 쓸 때도 같은 처리가 필요하다.
    """
    api_key = (
        api_key
        if api_key is not None
        else (_env("NVIDIA_API_KEY") or _env("MEM0_COMPAT_API_KEY") or "")
    )
    llm_model = llm_model or _env("MEM0_LLM_MODEL", NIM_DEFAULT_LLM)
    embed_model = embed_model or _env("MEM0_EMBED_MODEL", NIM_DEFAULT_EMBED)
    if embed_dims is None and _env("MEM0_EMBED_DIMS"):
        embed_dims = int(os.environ["MEM0_EMBED_DIMS"])
    if embed_dims is None:
        embed_dims = EMBED_DIMS.get(embed_model, 1024)
    llm_base_url = llm_base_url or _env("NVIDIA_BASE_URL", NIM_BASE_URL)
    embed_base_url = embed_base_url or _env("MEM0_EMBED_BASE_URL", llm_base_url)

    return compatible_config(
        base_url=llm_base_url,
        llm_model=llm_model,
        embed_model=embed_model,
        embed_base_url=embed_base_url,
        embed_dims=embed_dims,
        api_key=api_key,
        collection_name=collection_name,
        data_dir=data_dir,
    )


# ---------------------------------------------------------------- 선택


def build_config(kind: str, **kwargs) -> dict:
    """--config 인자 값을 설정 dict로 바꾼다.

    kind: "openai" | "local" | "compatible" | "nim"
    """
    kind = (kind or "").strip().lower()
    if kind == "openai":
        return openai_config(**kwargs)
    if kind == "local":
        return local_ollama_config(**kwargs)
    if kind == "compatible":
        return compatible_config(**kwargs)
    if kind == "nim":
        return nvidia_nim_config(**kwargs)
    raise ValueError(
        f"알 수 없는 설정 종류: {kind!r} (openai / local / compatible / nim 중 하나)"
    )

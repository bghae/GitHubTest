"""mem0-quickstart 오프라인 테스트.

API 키도, 네트워크도, Ollama도 없이 통과해야 한다.
mem0ai가 설치돼 있으면 설정 dict를 mem0의 스키마로 검증하는 테스트까지 돌고,
설치돼 있지 않으면 그 테스트만 건너뛴다.

    pytest -q
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as appconfig  # noqa: E402
import demo  # noqa: E402
import nim_embedder  # noqa: E402

HAS_MEM0 = True
try:
    from mem0.configs.base import MemoryConfig  # noqa: F401
except Exception:  # pragma: no cover - mem0ai 미설치 환경
    HAS_MEM0 = False

requires_mem0 = pytest.mark.skipif(not HAS_MEM0, reason="mem0ai 미설치")
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY 없음 (실제 호출 테스트)"
)


# ---------------------------------------------------------------- 설정 구조


def test_openai_config_structure(tmp_path):
    conf = appconfig.openai_config(data_dir=tmp_path)

    assert conf["llm"]["provider"] == "openai"
    assert conf["llm"]["config"]["model"]
    assert conf["embedder"]["provider"] == "openai"
    assert conf["embedder"]["config"]["model"] == "text-embedding-3-small"
    assert conf["vector_store"]["provider"] == "qdrant"
    assert conf["vector_store"]["config"]["embedding_model_dims"] == 1536
    assert conf["vector_store"]["config"]["path"] == str(tmp_path.resolve() / "qdrant")
    assert conf["history_db_path"] == str(tmp_path.resolve() / "history.db")


def test_openai_config_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MEM0_LLM_MODEL", "gpt-4.1-mini")
    conf = appconfig.openai_config(data_dir=tmp_path)
    assert conf["llm"]["config"]["model"] == "gpt-4.1-mini"


def test_local_ollama_config_structure(tmp_path):
    conf = appconfig.local_ollama_config(data_dir=tmp_path)

    assert conf["llm"]["provider"] == "ollama"
    assert conf["llm"]["config"]["model"]
    assert conf["llm"]["config"]["ollama_base_url"].startswith("http://")
    assert conf["embedder"]["provider"] == "ollama"
    assert conf["embedder"]["config"]["model"] == "nomic-embed-text"
    # nomic-embed-text는 768차원. mem0 기본 512와 맞지 않으므로 반드시 채워야 한다.
    assert conf["embedder"]["config"]["embedding_dims"] == 768
    assert conf["vector_store"]["config"]["embedding_model_dims"] == 768


def test_compatible_config_structure(tmp_path):
    """OpenAI 호환 경로: base_url이 LLM과 임베딩에 모두 들어가야 한다."""
    conf = appconfig.compatible_config(
        base_url="http://localhost:1234/v1",
        llm_model="qwen2.5-14b-instruct",
        embed_model="bge-m3",
        embed_dims=1024,
        api_key="sk-test",
        collection_name="compat_test",
        data_dir=tmp_path,
    )

    assert conf["llm"]["provider"] == "openai"
    assert conf["llm"]["config"]["model"] == "qwen2.5-14b-instruct"
    assert conf["llm"]["config"]["openai_base_url"] == "http://localhost:1234/v1"
    assert conf["llm"]["config"]["api_key"] == "sk-test"

    emb = conf["embedder"]["config"]
    assert conf["embedder"]["provider"] == "openai"
    assert emb["model"] == "bge-m3"
    assert emb["openai_base_url"] == "http://localhost:1234/v1"
    # 기본값에서는 dimensions 파라미터를 서버로 보내지 않는다 (vLLM/Groq 호환)
    assert "embedding_dims" not in emb
    assert conf["vector_store"]["config"]["embedding_model_dims"] == 1024
    assert conf["vector_store"]["config"]["collection_name"] == "compat_test"


def test_compatible_config_separate_embed_base_url(tmp_path):
    conf = appconfig.compatible_config(
        base_url="https://openrouter.ai/api/v1",
        llm_model="qwen/qwen-2.5-72b-instruct",
        embed_model="text-embedding-3-small",
        embed_base_url="https://api.openai.com/v1",
        api_key="router-key",
        embed_api_key="openai-key",
        data_dir=tmp_path,
    )
    assert conf["llm"]["config"]["openai_base_url"] == "https://openrouter.ai/api/v1"
    assert conf["embedder"]["config"]["openai_base_url"] == "https://api.openai.com/v1"
    assert conf["embedder"]["config"]["api_key"] == "openai-key"


def test_compatible_config_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("MEM0_LLM_MODEL", "local-model")
    monkeypatch.setenv("MEM0_EMBED_DIMS", "768")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MEM0_COMPAT_API_KEY", raising=False)

    conf = appconfig.compatible_config(data_dir=tmp_path)
    assert conf["llm"]["config"]["openai_base_url"] == "http://localhost:8080/v1"
    assert conf["llm"]["config"]["model"] == "local-model"
    assert conf["llm"]["config"]["api_key"] == "not-needed"
    assert conf["vector_store"]["config"]["embedding_model_dims"] == 768


def test_nvidia_nim_config_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("MEM0_LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    monkeypatch.setenv("MEM0_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")

    conf = appconfig.nvidia_nim_config(data_dir=tmp_path)
    assert conf["llm"]["config"]["api_key"] == "nvapi-test"
    assert conf["llm"]["config"]["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    # 임베딩 키도 같은 NIM 키를 쓴다
    assert conf["embedder"]["config"]["api_key"] == "nvapi-test"
    assert conf["embedder"]["config"]["model"] == "nvidia/nemotron-3.5-embed-1b"


def test_nvidia_nim_config_separate_embed_base_url(tmp_path):
    conf = appconfig.nvidia_nim_config(
        api_key="nvapi-test",
        llm_base_url="https://integrate.api.nvidia.com/v1",
        embed_base_url="http://localhost:8000/v1",
        embed_model="snowflake/arctic-embed-l",
        data_dir=tmp_path,
    )
    assert conf["llm"]["config"]["openai_base_url"] == "https://integrate.api.nvidia.com/v1"
    assert conf["embedder"]["config"]["openai_base_url"] == "http://localhost:8000/v1"


def test_build_config_dispatch(tmp_path):
    assert appconfig.build_config("openai", data_dir=tmp_path)["llm"]["provider"] == "openai"
    assert appconfig.build_config("local", data_dir=tmp_path)["llm"]["provider"] == "ollama"
    nim = appconfig.build_config("nim", api_key="nvapi-x", data_dir=tmp_path)
    assert nim["llm"]["config"]["openai_base_url"] == "https://integrate.api.nvidia.com/v1"
    compat = appconfig.build_config(
        "compatible", base_url="http://localhost:11434/v1", llm_model="qwen2.5:14b",
        data_dir=tmp_path,
    )
    assert compat["llm"]["config"]["openai_base_url"] == "http://localhost:11434/v1"

    with pytest.raises(ValueError):
        appconfig.build_config("gemini", data_dir=tmp_path)


def test_nvidia_nim_config_structure(tmp_path, monkeypatch):
    """NVIDIA NIM 경로: 키가 없어도 설정 dict는 만들어진다. (demo가 키를 확인한다)"""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MEM0_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.delenv("MEM0_LLM_MODEL", raising=False)
    monkeypatch.delenv("MEM0_EMBED_MODEL", raising=False)
    monkeypatch.delenv("MEM0_EMBED_DIMS", raising=False)

    conf = appconfig.nvidia_nim_config(data_dir=tmp_path)

    assert conf["llm"]["provider"] == "openai"
    assert conf["llm"]["config"]["openai_base_url"] == "https://integrate.api.nvidia.com/v1"
    assert conf["llm"]["config"]["model"] == "meta/llama-3.3-70b-instruct"
    assert conf["llm"]["config"]["api_key"] == ""  # 키 없음 — demo가 막는다
    assert conf["embedder"]["config"]["model"] == "nvidia/llama-3.2-nv-embedqa-1b-v1"
    assert conf["embedder"]["config"]["openai_base_url"] == "https://integrate.api.nvidia.com/v1"
    assert conf["vector_store"]["config"]["embedding_model_dims"] == 2048


def test_is_asymmetric_embedding():
    assert nim_embedder.is_asymmetric_embedding("nvidia/llama-3.2-nv-embedqa-1b-v1")
    assert nim_embedder.is_asymmetric_embedding("nvidia/nemotron-3.5-embed-8b")
    assert nim_embedder.is_asymmetric_embedding("nvidia/embed-qa-4")
    assert not nim_embedder.is_asymmetric_embedding("snowflake/arctic-embed-l")
    assert not nim_embedder.is_asymmetric_embedding("text-embedding-3-small")
    assert not nim_embedder.is_asymmetric_embedding(None)


@requires_mem0
def test_nim_embedding_sends_input_type():
    """적재는 passage, 검색은 query로 input_type이 붙는지 확인한다. (네트워크 없음)"""
    from mem0.configs.embeddings.base import BaseEmbedderConfig

    NimEmbedding = nim_embedder._build_class()
    cfg = BaseEmbedderConfig(
        model="nvidia/llama-3.2-nv-embedqa-1b-v1",
        api_key="test-key",
        openai_base_url="https://example.invalid/v1",
        embedding_dims=2048,
    )
    embedder = NimEmbedding(cfg)

    calls = []

    class _Data:
        def __init__(self, vector):
            self.embedding = vector

    class _Response:
        def __init__(self, n):
            self.data = [_Data([0.1, 0.2]) for _ in range(n)]

    class _Embeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _Response(len(kwargs["input"]))

    class _Client:
        embeddings = _Embeddings()

    embedder.client = _Client()

    vector = embedder.embed("수위 예측\n이야기", "add")
    assert vector == [0.1, 0.2]
    assert calls[0]["extra_body"] == {"input_type": "passage"}
    assert "\n" not in calls[0]["input"][0]

    embedder.embed("내 GPU는?", "search")
    assert calls[1]["extra_body"] == {"input_type": "query"}

    embedder.embed_batch(["a", "b"], "add")
    assert calls[2]["extra_body"] == {"input_type": "passage"}
    assert calls[2]["input"] == ["a", "b"]


@requires_mem0
@pytest.mark.parametrize("kind", ["openai", "local", "compatible", "nim"])
def test_configs_pass_mem0_schema(tmp_path, kind):
    """설정 dict가 mem0의 MemoryConfig 검증을 통과하는지 확인한다. (네트워크 없음)"""
    kwargs = {"data_dir": tmp_path}
    if kind == "compatible":
        kwargs.update(base_url="http://localhost:11434/v1", llm_model="qwen2.5:14b")
    conf = appconfig.build_config(kind, **kwargs)
    parsed = MemoryConfig(**conf)
    assert parsed.vector_store.provider == "qdrant"
    assert parsed.history_db_path == str(tmp_path.resolve() / "history.db")


# ---------------------------------------------------------------- 경로/안전장치


def test_data_paths_under_project_data(monkeypatch):
    monkeypatch.delenv("MEM0_DATA_DIR", raising=False)
    paths = appconfig.data_paths()
    data_dir = Path(paths["data_dir"])
    assert data_dir.name == "data"
    assert data_dir.parent == appconfig.PROJECT_DIR.resolve()
    assert paths["history_db_path"].endswith("history.db")


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MEM0_DATA_DIR", str(tmp_path))
    assert appconfig.data_paths()["data_dir"] == str(tmp_path.resolve())


def test_reset_guard_refuses_project_dir():
    with pytest.raises(ValueError):
        appconfig.reset_storage(appconfig.PROJECT_DIR)


def test_reset_deletes_only_data_dir(tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("남아야 함", encoding="utf-8")
    data = tmp_path / "data"
    (data / "qdrant").mkdir(parents=True)
    (data / "history.db").write_text("{}", encoding="utf-8")

    message = appconfig.reset_storage(data)

    assert not data.exists()
    assert keep.exists()
    assert "삭제됨" in message


def test_gitignore_covers_local_artifacts():
    text = (HERE / ".gitignore").read_text(encoding="utf-8")
    for pattern in [".env", "data/", "__pycache__/", "*.db", ".venv/"]:
        assert pattern in text, f".gitignore에 {pattern} 없음"


# ---------------------------------------------------------------- 샘플 데이터


def test_seed_data_schema():
    records = demo.load_seed_data()

    assert 10 <= len(records) <= 15, f"샘플 메시지는 10~15개: 현재 {len(records)}개"
    for i, record in enumerate(records, start=1):
        assert set(record) == {"role", "content"}, f"{i}번째 항목 키: {set(record)}"
        assert record["role"] in {"user", "assistant"}
        assert isinstance(record["content"], str) and record["content"].strip()

    roles = [r["role"] for r in records]
    assert roles[0] == "user" and "assistant" in roles


def test_seed_data_has_no_real_contact_info():
    """샘플에는 전화번호·이메일·주민번호 같은 실제 개인정보가 들어가면 안 된다."""
    raw = (HERE / "seed_data.json").read_text(encoding="utf-8")
    assert not re.search(r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}", raw), "전화번호 형태 발견"
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", raw), "이메일 형태 발견"
    assert not re.search(r"\d{6}[-\s]?[1-4]\d{6}", raw), "주민번호 형태 발견"


def test_to_messages_normalizes():
    messages = demo.to_messages(demo.load_seed_data())
    assert len(messages) == len(demo.load_seed_data())
    assert all(set(m) == {"role", "content"} for m in messages)
    assert all(m["role"] in demo.VALID_ROLES for m in messages)


def test_to_messages_rejects_bad_input():
    with pytest.raises(ValueError):
        demo.to_messages([{"role": "robot", "content": "안녕"}])
    with pytest.raises(ValueError):
        demo.to_messages([{"role": "user", "content": "   "}])
    with pytest.raises(ValueError):
        demo.to_messages([])


def test_to_conversations_batches():
    messages = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    batches = demo.to_conversations(messages, size=2)
    assert [len(b) for b in batches] == [2, 2, 1]
    with pytest.raises(ValueError):
        demo.to_conversations(messages, size=0)


# ---------------------------------------------------------------- 버전 차이 흡수


def test_build_filters():
    assert demo.build_filters("alice") == {"user_id": "alice"}
    with pytest.raises(ValueError):
        demo.build_filters("  ")


def test_result_items_handles_both_shapes():
    items = [{"id": "1", "memory": "a"}]
    assert demo.result_items({"results": items}) == items
    assert demo.result_items(items) == items
    assert demo.result_items(None) == []
    assert demo.result_items({"results": None}) == []


def test_format_memory_line():
    line = demo.format_memory_line(
        {"memory": "RTX 3060 Ti를 쓴다", "score": 0.8123, "created_at": "2026-11-26T04:00:00+09:00"}
    )
    assert "RTX 3060 Ti" in line
    assert "0.812" in line
    assert "2026-11-26T04:00:00" in line


class _FilterStyleMemory:
    """mem0 2.x 스타일: filters= 를 받는다."""

    def __init__(self):
        self.calls = []

    def search(self, query, *, filters=None, top_k=20, **kwargs):
        self.calls.append(("search", filters, top_k))
        return {"results": [{"memory": f"{query} 결과", "score": 0.9}]}

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        self.calls.append(("get_all", filters, top_k))
        return {"results": [{"memory": "기억1"}]}


class _LegacyMemory:
    """filters= 를 모르는 구버전 스타일: user_id= 만 받는다."""

    def __init__(self):
        self.calls = []

    def search(self, query, user_id=None, limit=100):
        self.calls.append(("search", user_id, limit))
        return [{"memory": f"{query} 결과"}]

    def get_all(self, user_id=None, limit=100):
        self.calls.append(("get_all", user_id, limit))
        return [{"memory": "기억1"}]


def test_search_memories_filter_style():
    memory = _FilterStyleMemory()
    hits = demo.search_memories(memory, "질문", "alice", top_k=3)
    assert hits[0]["memory"] == "질문 결과"
    assert memory.calls == [("search", {"user_id": "alice"}, 3)]


def test_search_memories_legacy_style():
    memory = _LegacyMemory()
    hits = demo.search_memories(memory, "질문", "alice", top_k=3)
    assert hits[0]["memory"] == "질문 결과"
    assert memory.calls == [("search", "alice", 3)]


def test_get_all_memories_both_styles():
    assert demo.get_all_memories(_FilterStyleMemory(), "alice")[0]["memory"] == "기억1"
    assert demo.get_all_memories(_LegacyMemory(), "alice")[0]["memory"] == "기억1"


def test_retry_call_gives_up_after_attempts():
    calls = {"n": 0}

    def always_429():
        calls["n"] += 1
        raise RuntimeError("Error code: 429 - rate limit exceeded")

    with pytest.raises(RuntimeError):
        demo.retry_call(always_429, attempts=2, base_wait=0, label="test")
    assert calls["n"] == 2


def test_retry_call_does_not_retry_other_errors():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise RuntimeError("400 Bad Request")

    with pytest.raises(RuntimeError):
        demo.retry_call(bad_request, attempts=3, base_wait=0)
    assert calls["n"] == 1


def test_retry_call_succeeds_after_retry():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    assert demo.retry_call(flaky, attempts=3, base_wait=0) == "ok"
    assert calls["n"] == 2


# ---------------------------------------------------------------- 실제 호출 (키 필요)


@requires_openai
def test_live_add_search_get_all(tmp_path):
    """OPENAI_API_KEY가 있을 때만 도는 실제 호출 테스트 (과금 주의: 아주 작게)."""
    from mem0 import Memory

    conf = appconfig.openai_config(data_dir=tmp_path, collection_name="live_test")
    memory = Memory.from_config(conf)
    memory.add(
        [{"role": "user", "content": "내 이름은 테스트사용자이고 취미는 등산이다."}],
        user_id="pytest_live",
    )
    hits = demo.search_memories(memory, "취미가 뭐지?", "pytest_live", top_k=3)
    assert isinstance(hits, list)
    assert demo.get_all_memories(memory, "pytest_live")

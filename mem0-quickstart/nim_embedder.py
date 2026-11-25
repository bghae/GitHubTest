"""NVIDIA NIM QA 임베딩용 어댑터.

왜 필요한가
    NIM의 QA 임베딩 모델(nvidia/llama-3.2-nv-embedqa-1b-v1 등)은 요청 본문에
    input_type="passage"(문서 적재용) 또는 "query"(검색용)를 요구한다.
    이 필드가 없으면 NIM이 400 "'input_type' parameter is required for asymmetric models"를 돌려준다.
    그런데 mem0의 OpenAI 임베더는 input/model/encoding_format만 보내므로 그대로는 쓸 수 없다.

    NVIDIA 문서도 이 점을 인정하고, OpenAI 호환 클라이언트에서는 모델명 뒤에
    -query/-passage를 붙이는 우회를 안내한다. 하지만 mem0는 같은 모델명으로
    적재와 검색을 모두 하므로, 모델명을 한쪽으로 고정하면 반대쪽 검색 품질이 떨어진다.
    그래서 여기서는 memory_action("add"/"update"/"search")에 따라
    input_type을 골라 보내는 얇은 서브클래스를 쓴다.

적용 방법
    memory = Memory.from_config(conf)
    attach_nim_embedder(memory)      # 비대칭 모델일 때만 끼워진다.

기준: mem0ai 2.2.1. mem0 내부에서 임베딩은 self.embedding_model.embed(text, action)로 호출된다.
"""

from __future__ import annotations

from typing import Optional

# input_type을 요구하는(비대칭) 모델 이름에 공통으로 들어가는 조각들
ASYMMETRIC_HINTS = ("embedqa", "embed-qa", "nv-embed", "embed-1b", "embed-8b", "nemoretriever")


def is_asymmetric_embedding(model_name: Optional[str]) -> bool:
    """이 임베딩 모델이 input_type(passage/query)을 요구하는 계열인지 이름으로 판단한다."""
    if not model_name:
        return False
    lowered = model_name.lower()
    return any(hint in lowered for hint in ASYMMETRIC_HINTS)


def _build_class():
    """mem0의 OpenAIEmbedding을 상속한 클래스를 만든다. (mem0 미설치면 ImportError)"""
    from mem0.embeddings.openai import OpenAIEmbedding

    class NimEmbedding(OpenAIEmbedding):
        """input_type을 memory_action에 맞춰 보내는 OpenAI 임베더."""

        def _create(self, texts: list[str], input_type: str) -> list[list[float]]:
            response = self.client.embeddings.create(
                input=texts,
                model=self.config.model,
                encoding_format="float",
                extra_body={"input_type": input_type},
            )
            return [item.embedding for item in response.data]

        def embed(self, text, memory_action=None):  # noqa: D401 - mem0 시그니처
            input_type = "query" if memory_action == "search" else "passage"
            cleaned = str(text).replace("\n", " ")
            return self._create([cleaned], input_type)[0]

        def embed_batch(self, texts, memory_action="add"):
            input_type = "query" if memory_action == "search" else "passage"
            cleaned = [str(t).replace("\n", " ") for t in texts]
            return self._create(cleaned, input_type) if cleaned else []

    return NimEmbedding


def attach_nim_embedder(memory, force: bool = False) -> bool:
    """mem0 Memory의 임베딩 객체를 NIM용으로 바꿔 끼운다.

    반환값: 실제로 바꿨으면 True, 필요 없거나 실패하면 False.
    실패해도 예외를 올리지 않는다. (동기 OpenAI 임베더로도 bge-m3가 아닌
    비대칭 모델이면 400이 나므로, README의 안내를 보게 된다)
    """
    try:
        embedder = memory.embedding_model
        model_name = getattr(getattr(embedder, "config", None), "model", None)
        if not (force or is_asymmetric_embedding(model_name)):
            return False
        NimEmbedding = _build_class()
        memory.embedding_model = NimEmbedding(embedder.config)
        return True
    except Exception as exc:  # noqa: BLE001 - 실패하면 원래 임베더로 그대로 진행
        print(f"  (NIM 임베딩 어댑터를 적용하지 못했다: {type(exc).__name__}: {exc})")
        return False

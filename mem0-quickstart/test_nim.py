"""NIM 엔드포인트 진단 스크립트.

어느 호출이 403/404로 막히는지 모델별로 찍어 본다.
    .\.venv\Scripts\python.exe test_nim.py > nim_test.log 2>&1
키는 .env에서 읽고, 화면에는 앞 6자만 남겨 출력한다.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


load_env(ROOT / ".env")

KEY = os.getenv("NVIDIA_API_KEY", "").strip()
BASE = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")

print(f"base_url : {BASE}")
print(f"key      : {KEY[:6] + '...' if KEY else '(없음)'}  len={len(KEY)}")
print(f".env LLM : {os.getenv('MEM0_LLM_MODEL', '(없음, 코드 기본값)')}")
print(f".env EMB : {os.getenv('MEM0_EMBED_MODEL', '(없음, 코드 기본값)')}")
print("-" * 60)


def post(path: str, payload: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read(300).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(300).decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


LLM_CANDIDATES = [
    os.getenv("MEM0_LLM_MODEL"),
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "z-ai/glm-5.3-flash",
    "minimaxai/minimax-m3.1-flash",
    "google/gemma-3-4b-it",
    "mistralai/mistral-large-4-1t-a52b",
]
EMBED_CANDIDATES = [
    os.getenv("MEM0_EMBED_MODEL"),
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
    "nvidia/nemotron-3.5-embed-8b",
    "nvidia/nv-embedqa-mistral-7b-v2",
    "snowflake/arctic-embed-l",
]

print("[LLM 호출]")
seen = set()
for m in LLM_CANDIDATES:
    if not m or m in seen:
        continue
    seen.add(m)
    code, body = post(
        "/chat/completions",
        {
            "model": m,
            "messages": [{"role": "user", "content": "한 단어로 답해: OK"}],
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    print(f"  {code:<4} {m}")
    print(f"       {body[:200]}")

print("[임베딩 호출]")
seen = set()
for m in EMBED_CANDIDATES:
    if not m or m in seen:
        continue
    seen.add(m)
    code, body = post(
        "/embeddings",
        {
            "model": m,
            "input": ["테스트 문장입니다."],
            "input_type": "passage",
            "encoding_format": "float",
        },
    )
    print(f"  {code:<4} {m}")
    print(f"       {body[:200]}")

print("-" * 60)
print("401=키 무효 / 403=키는 유효하나 그 모델 권한 없음 / 404=모델명 없음 / 200=정상")

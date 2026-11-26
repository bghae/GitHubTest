"""Gemini OpenAI 호환 경로 진단 스크립트.

용도: mem0에서 임베딩 호출이 404일 때, 원인이
  (a) 키/프로젝트 권한  (b) 모델 이름  (c) 엔드포인트 경로
중 무엇인지 한 번에 가려낸다.

실행 (프로젝트 폴더에서):
    python test_gemini.py

.env 의 MEM0_COMPAT_API_KEY 를 읽는다. (.env가 없으면 환경변수에서)
결과는 콘솔에만 찍고 아무것도 저장하지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

LLM_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
]
EMBED_CANDIDATES = [
    "gemini-embedding-001",
    "models/gemini-embedding-001",
    "gemini-embedding-2",
]


def load_env() -> str:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.getenv("MEM0_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not key or "붙여넣기" in key:
        raise SystemExit("MEM0_COMPAT_API_KEY 가 .env 에 없습니다. 키를 먼저 넣어 주세요.")
    return key


def short(e: Exception) -> str:
    s = str(e)
    for line in s.splitlines():
        if "Error code" in line or "not found" in line or "message" in line:
            return line.strip()[:300]
    return s.splitlines()[0][:300] if s else e.__class__.__name__


def main() -> None:
    key = load_env()
    print(f"key: {key[:6]}...{key[-4:]} (길이 {len(key)})")
    print(f"base: {BASE}\n")

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=BASE)

    print("== 1) 모델 목록 (키가 살아 있는지 확인)")
    try:
        models = client.models.list()
        names = [m.id for m in models.data]
        print(f"  OK — {len(names)}개")
        for want in ("embedding", "flash"):
            hits = [n for n in names if want in n]
            print(f"  {want}: {hits[:12]}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL — {short(e)}")

    print("\n== 2) 임베딩 (모델 이름별)")
    for m in EMBED_CANDIDATES:
        try:
            r = client.embeddings.create(model=m, input="테스트 문장입니다")
            print(f"  {m}: OK dim={len(r.data[0].embedding)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {m}: FAIL — {short(e)}")

    print("\n== 3) 채팅 (모델 이름별)")
    for m in LLM_CANDIDATES:
        try:
            r = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "OK 라고만 답해"}],
                max_tokens=16,
            )
            print(f"  {m}: OK — {r.choices[0].message.content!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  {m}: FAIL — {short(e)}")

    print("\n== 끝. 이 출력을 그대로 복사해 주면 됩니다.")


if __name__ == "__main__":
    main()

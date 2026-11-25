#!/usr/bin/env bash
# mem0-quickstart 설치 스크립트 (Linux / macOS / WSL)
#
# 실행:  bash install.sh
# Windows는 install-windows.ps1 을 쓴다.

set -euo pipefail

cd "$(dirname "$0")"

echo ""
echo "==== mem0-quickstart 설치 ===="
echo "폴더: $(pwd)"

# ---------------------------------------------------------------- 1. Python
echo ""
echo "[1/5] Python 확인"

PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY="$candidate"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "  Python 3.10 이상이 필요하다."
  echo "  Debian/Ubuntu:  sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  echo "  macOS:          brew install python@3.12"
  exit 1
fi
echo "  OK: $($PY -V 2>&1)"

# ---------------------------------------------------------------- 2. 가상환경
echo ""
echo "[2/5] 가상환경(.venv) 준비"
if [ -x ".venv/bin/python" ]; then
  echo "  이미 있음: .venv/bin/python"
else
  if ! "$PY" -m venv .venv; then
    echo "  venv 생성 실패. python3-venv 패키지가 필요할 수 있다."
    exit 1
  fi
  echo "  생성: .venv/bin/python"
fi

# ---------------------------------------------------------------- 3. 패키지
echo ""
echo "[3/5] 패키지 설치 (mem0ai, python-dotenv, pytest)"
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt
echo "  설치된 mem0ai: $(.venv/bin/python -c 'import importlib.metadata as m; print(m.version("mem0ai"))')"

# ---------------------------------------------------------------- 4. .env
echo ""
echo "[4/5] .env 준비"
if [ -f ".env" ]; then
  echo "  이미 있음: .env (덮어쓰지 않는다)"
else
  cp .env.example .env
  echo "  .env.example → .env 복사. OPENAI_API_KEY 또는 OPENAI_BASE_URL을 채울 것."
fi

# ---------------------------------------------------------------- 5. Ollama
echo ""
echo "[5/5] Ollama 확인 (API 키 없이 완전 로컬로 돌릴 때만 필요)"
if command -v ollama >/dev/null 2>&1; then
  echo "  ollama 명령 있음"
  if curl -fsS --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "  서버 응답 OK"
    echo "  필요하면:  ollama pull qwen2.5:14b   /   ollama pull nomic-embed-text"
  else
    echo "  서버가 꺼져 있는 것 같다. 'ollama serve' 를 실행할 것."
  fi
else
  echo "  ollama 없음. OpenAI 키로 시작하면 그대로 진행해도 된다."
  echo "  로컬로 돌리려면:  curl -fsSL https://ollama.com/install.sh | sh"
fi

# ---------------------------------------------------------------- 마무리
echo ""
echo "==== 설치 완료 ===="
echo ""
echo "실행:"
echo "  .venv/bin/python demo.py --config openai"
echo "  .venv/bin/python demo.py --config local"
echo "  .venv/bin/python -m pytest -q"
echo ""
echo "가상환경 활성화:  source .venv/bin/activate"
echo ""

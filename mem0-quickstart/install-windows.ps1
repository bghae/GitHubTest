# mem0-quickstart 설치 스크립트 (Windows / PowerShell)
#
# 실행 방법 (mem0-quickstart 폴더에서):
#   powershell -ExecutionPolicy Bypass -File .\install-windows.ps1
#
# 관리자 권한은 필요 없다. 모든 설치물은 이 폴더 안의 .venv 에만 들어간다.

#Requires -Version 5.1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host ""
Write-Host "==== mem0-quickstart 설치 ====" -ForegroundColor Cyan
Write-Host "폴더: $PSScriptRoot"

# ---------------------------------------------------------------- 1. Python
Write-Host ""
Write-Host "[1/5] Python 확인" -ForegroundColor Yellow

$pythonCmd = $null
$pythonArgs = @()
foreach ($candidate in @(@("python", @()), @("py", @("-3")))) {
    $name = $candidate[0]
    $args0 = $candidate[1]
    if (Get-Command $name -ErrorAction SilentlyContinue) {
        try {
            $ver = & $name @args0 -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver) {
                $parts = $ver.Split(".")
                if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10) {
                    $pythonCmd = $name
                    $pythonArgs = $args0
                    Write-Host "  OK: $name $ver" -ForegroundColor Green
                    break
                }
                Write-Host "  $name $ver — 3.10 이상이 필요하다." -ForegroundColor DarkYellow
            }
        } catch { }
    }
}

if (-not $pythonCmd) {
    Write-Host ""
    Write-Host "  Python 3.10 이상을 찾지 못했다." -ForegroundColor Red
    Write-Host "  1) https://www.python.org/downloads/windows/ 에서 설치하거나"
    Write-Host "  2) 관리자 PowerShell에서:  winget install Python.Python.3.12"
    Write-Host "  설치할 때 'Add python.exe to PATH'를 체크할 것."
    exit 1
}

# ---------------------------------------------------------------- 2. 가상환경
Write-Host ""
Write-Host "[2/5] 가상환경(.venv) 준비" -ForegroundColor Yellow

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Host "  이미 있음: $venvPython" -ForegroundColor Green
} else {
    & $pythonCmd @pythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "가상환경 생성 실패" }
    Write-Host "  생성: $venvPython" -ForegroundColor Green
}

# ---------------------------------------------------------------- 3. 패키지
Write-Host ""
Write-Host "[3/5] 패키지 설치 (mem0ai, python-dotenv, pytest)" -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "패키지 설치 실패. 위 오류 메시지를 확인할 것." }
$mem0Version = & $venvPython -c "import mem0, importlib.metadata as m; print(m.version('mem0ai'))"
Write-Host "  설치된 mem0ai: $mem0Version" -ForegroundColor Green

# ---------------------------------------------------------------- 4. .env
Write-Host ""
Write-Host "[4/5] .env 준비" -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  이미 있음: .env (덮어쓰지 않는다)" -ForegroundColor Green
} else {
    Copy-Item ".env.example" ".env"
    Write-Host "  .env.example → .env 복사. OPENAI_API_KEY 또는 OPENAI_BASE_URL을 채울 것." -ForegroundColor Green
}

# ---------------------------------------------------------------- 5. Ollama
Write-Host ""
Write-Host "[5/5] Ollama 확인 (API 키 없이 완전 로컬로 돌릴 때만 필요)" -ForegroundColor Yellow
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "  ollama 명령 있음" -ForegroundColor Green
    try {
        $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3
        $names = @($tags.models | ForEach-Object { $_.name })
        Write-Host "  서버 응답 OK. 받아 둔 모델: $($names -join ', ')"
        Write-Host "  필요하면:  ollama pull qwen2.5:14b   /   ollama pull nomic-embed-text"
    } catch {
        Write-Host "  서버가 꺼져 있는 것 같다. 다른 창에서 'ollama serve' 를 실행하거나 앱을 켤 것." -ForegroundColor DarkYellow
    }
} else {
    Write-Host "  ollama 없음. OpenAI 키로 시작할 거면 그대로 진행해도 된다." -ForegroundColor DarkYellow
    Write-Host "  로컬로 돌리려면:  winget install Ollama.Ollama   (또는 https://ollama.com/download)"
}

# ---------------------------------------------------------------- 마무리
Write-Host ""
Write-Host "==== 설치 완료 ====" -ForegroundColor Cyan
Write-Host ""
Write-Host "다음 명령으로 실행한다. (활성화 없이 venv 파이썬을 바로 쓰므로 실행 정책 문제가 없다)"
Write-Host "  .\.venv\Scripts\python.exe demo.py --config openai" -ForegroundColor White
Write-Host "  .\.venv\Scripts\python.exe demo.py --config local" -ForegroundColor White
Write-Host "  .\.venv\Scripts\python.exe -m pytest -q" -ForegroundColor White
Write-Host ""
Write-Host "가상환경을 활성화해서 쓰려면:"
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor DarkGray
Write-Host "(이 줄이 '실행 정책' 오류를 내면 위처럼 파이썬 경로를 직접 쓰면 된다.)"
Write-Host ""

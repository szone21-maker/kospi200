# ================================================
#  📈 코스피200 주식 추천 봇 v11 - PowerShell 실행기
# ================================================

# 한글 출력 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "코스피200 주식 추천 봇 v11"

# 실행 파일명 (변경 시 여기만 수정)
$TARGET = "streamlit_kospi200v11.py"
$PORT   = 8501

Write-Host ""
Write-Host " ╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host " ║   📈 코스피200 주식 추천 봇 v11 실행기   ║" -ForegroundColor Cyan
Write-Host " ╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Python 설치 확인 ──────────────────────────────
Write-Host " [확인] Python 설치 여부..." -ForegroundColor Yellow
try {
    $pyVer = python --version 2>&1
    Write-Host " ✅ $pyVer" -ForegroundColor Green
} catch {
    Write-Host " ❌ Python이 설치되어 있지 않습니다." -ForegroundColor Red
    Write-Host "    https://www.python.org 에서 설치 후 다시 실행하세요." -ForegroundColor Red
    Read-Host "`n 아무 키나 눌러 종료"
    exit 1
}

# ── 실행 파일 존재 확인 ───────────────────────────
Write-Host ""
Write-Host " [확인] $TARGET 파일 존재 여부..." -ForegroundColor Yellow
if (-Not (Test-Path $TARGET)) {
    Write-Host " ❌ $TARGET 파일을 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "    현재 폴더: $(Get-Location)" -ForegroundColor Red
    Write-Host "    이 스크립트와 같은 폴더에 py 파일이 있는지 확인하세요." -ForegroundColor Red
    Read-Host "`n 아무 키나 눌러 종료"
    exit 1
}
Write-Host " ✅ $TARGET 확인 완료" -ForegroundColor Green

# ── 필수 패키지 확인 및 설치 ─────────────────────
Write-Host ""
Write-Host " [1/2] 필수 패키지 확인 중..." -ForegroundColor Yellow
Write-Host ""

$packages = @("streamlit","requests","pandas","plotly","numpy","urllib3","openpyxl")

foreach ($pkg in $packages) {
    $check = pip show $pkg 2>$null
    if ($null -eq $check -or $check -eq "") {
        Write-Host "  ⚠️  $pkg 없음 → 설치 중..." -ForegroundColor Yellow
        pip install $pkg -q
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ❌ $pkg 설치 실패. 네트워크를 확인하세요." -ForegroundColor Red
            Read-Host "`n 아무 키나 눌러 종료"
            exit 1
        }
        Write-Host "  ✅ $pkg 설치 완료" -ForegroundColor Green
    } else {
        Write-Host "  ✅ $pkg 확인 완료" -ForegroundColor Green
    }
}

# ── 포트 사용 여부 확인 ───────────────────────────
Write-Host ""
Write-Host " [포트 확인] $PORT 번 포트 사용 여부..." -ForegroundColor Yellow
$portInUse = Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host " ⚠️  포트 $PORT 이 이미 사용 중입니다." -ForegroundColor Yellow
    $answer = Read-Host "    다른 포트($($PORT+1))로 실행하시겠습니까? (Y/N)"
    if ($answer -eq "Y" -or $answer -eq "y") {
        $PORT = $PORT + 1
        Write-Host " → 포트 $PORT 으로 변경합니다." -ForegroundColor Cyan
    }
}

# ── Streamlit 실행 ────────────────────────────────
Write-Host ""
Write-Host " [2/2] 앱 실행 중..." -ForegroundColor Yellow
Write-Host ""
Write-Host " ✅ 브라우저가 자동으로 열립니다." -ForegroundColor Green
Write-Host " 🌐 주소: http://localhost:$PORT" -ForegroundColor Cyan
Write-Host " 🛑 종료하려면 Ctrl + C 를 누르세요." -ForegroundColor Red
Write-Host ""

try {
    streamlit run $TARGET --server.port $PORT
} catch {
    Write-Host ""
    Write-Host " ❌ 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "`n 아무 키나 눌러 종료"
}
@echo off
chcp 65001 > nul
echo ========================================
echo   코스피200 주식 추천 시스템 실행
echo ========================================
echo.

echo [1/3] 필요한 패키지 확인 중...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Streamlit이 설치되어 있지 않습니다. 설치를 시작합니다...
    pip install streamlit requests pandas plotly numpy urllib3
) else (
    echo ✅ 패키지 확인 완료!
)

echo.
echo [2/3] Streamlit 서버 시작 중...
echo.

streamlit run streamlit_kospi200v10.py

if errorlevel 1 (
    echo.
    echo ❌ 오류가 발생했습니다!
    echo 파일 경로를 확인해주세요: streamlit_kospi200v10.py
    echo.
    pause
)
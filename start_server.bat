@echo off
cd /d "%~dp0"
echo [TKU SAM] Starting uvicorn server...
start "TKU SAM Server" python -m uvicorn app:app --host 0.0.0.0 --port 8000
echo [TKU SAM] Waiting for server to load models (60s)...
timeout /t 60 /nobreak
echo [TKU SAM] Starting ngrok tunnel...
start "TKU SAM ngrok" ngrok http 8000
echo [TKU SAM] Done. Check the ngrok window for the public URL.
pause

@echo off
cd /d "%~dp0"
echo Starting NBA MVP Predictor Web GUI...
start http://127.0.0.1:5000
python app.py
pause

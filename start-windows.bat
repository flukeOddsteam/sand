@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate.bat
python -c "import torch, tokenizers" 2>nul || pip install torch tokenizers
python app.py
pause

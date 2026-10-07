@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" SolomonPocketAI.py
) else (
  py SolomonPocketAI.py
)
if errorlevel 1 (
  echo.
  echo Solomon Pocket AI could not start. Run setup.bat first.
  pause
)

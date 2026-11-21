@echo off
title Drone Design Studio
cd /d "%~dp0"
echo Starting Drone Design Studio... (close this window to stop it)
".venv\Scripts\python.exe" drone_app.py
pause

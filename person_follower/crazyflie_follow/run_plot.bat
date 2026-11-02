@echo off
cd /d "%~dp0"

echo Activating environment...
call .venv\Scripts\activate.bat

echo.
echo ==============================================
echo Running 3D Rigid Body Plot
echo Client IP: 10.10.10.3
echo Server IP: 10.10.10.100
echo ==============================================
echo.

python plot_rigid_body.py 10.10.10.100 --client 10.10.10.3 %*

pause

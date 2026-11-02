@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    echo Python launcher "py" was not found.
    echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo and check "Add python.exe to PATH" in the installer.
    exit /b 1
)

py -3 -m venv .venv
if errorlevel 1 exit /b 1

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Ready. Activate the venv, then start a script:
echo   .venv\Scripts\activate.bat
echo   python plot_rigid_body.py MOTIVE_SERVER_IP
echo   python mocap_extpos_lab.py MOTIVE_SERVER_IP --client THIS_PC_IP --body-id 2 --address 0xE7E7E7E7E7 --hover
endlocal

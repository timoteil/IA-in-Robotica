@echo off
cd /d "%~dp0"

echo Activating environment...
call .venv\Scripts\activate.bat

set MOTIVE_IP=10.10.10.100
set CLIENT_IP=10.10.10.3
set BODY_ID=2
set RADIO_ADDRESS=0xE7E7E7E7E2

echo ========================================================
echo Starting Crazyflie Mocap External Position Flight
echo Motive Server : %MOTIVE_IP%
echo Client IP     : %CLIENT_IP%
echo Rigid Body ID : %BODY_ID%
echo Radio Address : %RADIO_ADDRESS%
echo ========================================================
echo.

python mocap_extpos_lab.py %MOTIVE_IP% --client %CLIENT_IP% --body-id %BODY_ID% --address %RADIO_ADDRESS% --hover %*

pause

@echo off
setlocal
cd /d "%~dp0Project" || exit /b 1
set "PYTHON=%CD%\.venv\Scripts\python.exe"
set "YOLO_CONFIG_DIR=%CD%\.venv\ultralytics"
set "MPLCONFIGDIR=%CD%\.venv\matplotlib"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

if exist "%PYTHON%" goto check_packages
python -m venv .venv
if errorlevel 1 goto failed

:check_packages
"%PYTHON%" -c "import cv2, lap, torchvision, ultralytics; from scipy.optimize import linear_sum_assignment; from pycocotools.cocoeval import COCOeval" >nul 2>&1
if not errorlevel 1 goto run_demo
echo Installing demo dependencies. Internet access is required for first-time setup.
"%PYTHON%" -m pip install "ultralytics==8.4.219" "lap==0.5.14" "scipy==1.18.1" "pycocotools==2.0.11"
if errorlevel 1 goto failed

:run_demo
if "%~1"=="" goto graphical
"%PYTHON%" run_project.py CAVIAR_Walk2_Reentry.mp4 --save-video --out demonstration %*
exit /b %errorlevel%

:graphical
"%PYTHON%" run_project.py CAVIAR_Walk2_Reentry.mp4 --save-video --out demonstration
if errorlevel 1 goto failed
echo.
echo Output saved under Project\demonstration.
pause
exit /b 0

:failed
echo.
echo The demo could not start. See the error above.
pause
exit /b 1

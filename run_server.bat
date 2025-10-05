@echo off
echo ===============================================================================
echo                     Google Cloud Image Generation Setup
echo ===============================================================================
echo.
echo IMPORTANT: Before running this application, you need to:
echo.
echo 1. Install Google Cloud SDK from:
echo    https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe
echo.
echo 2. Run these commands in a separate Command Prompt:
echo    gcloud auth application-default login
echo    gcloud auth application-default set-quota-project [YOUR_PROJECT_ID]
echo.
echo ===============================================================================
echo.
pause

echo Starting Image Generation Flask Server...
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo WARNING: Virtual environment not found. Running without venv.
)

REM Install required packages if needed
echo Installing required packages...
pip install flask flask-cors pillow google-cloud-aiplatform

REM Run the Flask server and open browser
echo.
echo Starting server and opening browser...
start "" http://127.0.0.1:5000/
python server.py

pause
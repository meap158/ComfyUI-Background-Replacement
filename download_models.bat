@echo off

:: Check if embedded Python is installed
if not exist ..\..\..\python_embeded\python.exe (
    echo Embedded python not found. Please install manually.
    pause
    exit /b 1
)

:: Download the models
python download_models.py

@pause

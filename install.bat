@echo off

:: Check if embedded Python is installed
if not exist ..\..\..\python_embeded\python.exe (
    echo Embedded python not found. Please install manually.
    pause
    exit /b 1
)

:: Remove the flag file if it exists
if exist requirements_installed.flag (
    del requirements_installed.flag
)

:: Install the package
echo Installing...
..\..\..\python_embeded\python.exe install.py
echo Done!

@pause

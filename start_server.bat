@echo off
setlocal EnableDelayedExpansion
title Model Quantization Server

rem This launcher remembers your python.exe and quant_server.py paths in
rem start_server_config.txt next to this file, so you only pick them once
rem per machine. Delete that file (or edit it) if you ever need to change
rem the paths again.

set "SCRIPT_DIR=%~dp0"
set "CONFIG_FILE=%SCRIPT_DIR%start_server_config.txt"
set "PYTHON_EXE="
set "SERVER_SCRIPT="

if exist "%CONFIG_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%CONFIG_FILE%") do (
        if /i "%%A"=="PYTHON_EXE" set "PYTHON_EXE=%%B"
        if /i "%%A"=="SERVER_SCRIPT" set "SERVER_SCRIPT=%%B"
    )
)

rem Try the common layout first (this .bat sitting next to python.exe and
rem quant_server.py, e.g. inside a ComfyUI "python_embeded" folder) before
rem asking the user to pick anything.
if not exist "%PYTHON_EXE%" if exist "%SCRIPT_DIR%python.exe" set "PYTHON_EXE=%SCRIPT_DIR%python.exe"
if not exist "%SERVER_SCRIPT%" if exist "%SCRIPT_DIR%quant_server.py" set "SERVER_SCRIPT=%SCRIPT_DIR%quant_server.py"

if not exist "%PYTHON_EXE%" (
    echo.
    echo Could not find python.exe automatically.
    echo A file picker window will open -- please select your ComfyUI
    echo python.exe ^(usually inside a "python_embeded" folder^).
    echo.
    call :PICK_FILE "Select python.exe" "Python executable|python.exe|All files|*.*" PYTHON_EXE
)
if not exist "%PYTHON_EXE%" (
    echo [ERROR] No valid python.exe selected. Aborting.
    pause
    exit /b 1
)

if not exist "%SERVER_SCRIPT%" (
    echo.
    echo Could not find quant_server.py automatically.
    echo A file picker window will open -- please select quant_server.py.
    echo.
    call :PICK_FILE "Select quant_server.py" "Python script|quant_server.py|Python files|*.py|All files|*.*" SERVER_SCRIPT
)
if not exist "%SERVER_SCRIPT%" (
    echo [ERROR] No valid quant_server.py selected. Aborting.
    pause
    exit /b 1
)

rem Since version 1.3 the app consists of several files: quant_server.py needs the
rem "templates" and "static" folders right next to it.
for %%I in ("%SERVER_SCRIPT%") do set "SERVER_DIR=%%~dpI"
if not exist "%SERVER_DIR%templates\index.html" goto :MISSING_FILES
if not exist "%SERVER_DIR%static\app.js" goto :MISSING_FILES

> "%CONFIG_FILE%" (
    echo PYTHON_EXE=%PYTHON_EXE%
    echo SERVER_SCRIPT=%SERVER_SCRIPT%
)

"%PYTHON_EXE%" -c "import flask" >nul 2>nul
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  Flask is required but not installed
    echo ============================================================
    echo  This tool runs a small local web server so you can control
    echo  the quantization from your browser at http://127.0.0.1:8877
    echo  Flask is the Python library that provides that local web
    echo  server -- without it, quant_server.py cannot start.
    echo.
    echo  It only listens on 127.0.0.1 ^(this PC only^) and does not
    echo  expose anything to the internet.
    echo ============================================================
    echo.
    set "INSTALL_FLASK="
    set /p "INSTALL_FLASK=Install Flask now? (Y/N): "
    if /i "!INSTALL_FLASK!"=="Y" (
        echo.
        "%PYTHON_EXE%" -m pip install flask
        if errorlevel 1 (
            echo [ERROR] Flask installation failed.
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo Aborted -- Flask is required to run the server.
        pause
        exit /b 0
    )
)

"%PYTHON_EXE%" -c "import convert_to_quant" >nul 2>nul
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  convert_to_quant ^(ctq^) is required but not installed
    echo ============================================================
    echo  This whole tool exists to quantize models, and it does that
    echo  by calling a separate program named "ctq" ^(convert_to_quant^)
    echo  for every conversion -- without it, nothing here can actually
    echo  run. It is NOT part of a normal ComfyUI installation, so it
    echo  needs to be installed once, just like Flask above.
    echo ============================================================
    echo.
    set "INSTALL_CTQ="
    set /p "INSTALL_CTQ=Install convert_to_quant now? (Y/N): "
    if /i "!INSTALL_CTQ!"=="Y" (
        echo.
        "%PYTHON_EXE%" -m pip install convert_to_quant
        if errorlevel 1 (
            echo [ERROR] convert_to_quant installation failed.
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo Aborted -- convert_to_quant is required to run the server.
        pause
        exit /b 0
    )
)

echo.
echo Starting Model Quantization Server ...
echo   Python: %PYTHON_EXE%
echo   Script: %SERVER_SCRIPT%
echo.
echo This window shows the server log. To stop: close this window or press Ctrl+C.
echo.

rem Opens the browser automatically after a short delay, while the server starts.
start "" cmd /c "timeout /t 2 >nul && start http://127.0.0.1:8877"

"%PYTHON_EXE%" "%SERVER_SCRIPT%"

echo.
echo Server has stopped.
pause
exit /b 0

:MISSING_FILES
echo.
echo [ERROR] The folders "templates" and "static" were not found next to quant_server.py:
echo   %SERVER_DIR%
echo GFlava-Quant needs them since version 1.3 -- please copy the complete folder.
echo.
pause
exit /b 1

:PICK_FILE
setlocal
set "DLG_TITLE=%~1"
set "DLG_FILTER=%~2"
set "PICKED="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Title = '%DLG_TITLE%'; $f.Filter = '%DLG_FILTER%'; $f.InitialDirectory = '%SCRIPT_DIR%'; if ($f.ShowDialog() -eq 'OK') { Write-Output $f.FileName }"`) do set "PICKED=%%F"
endlocal & set "%~3=%PICKED%"
goto :eof

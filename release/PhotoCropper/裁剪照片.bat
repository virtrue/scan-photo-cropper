@echo off
chcp 65001 >nul
title 扫描照片自动裁剪工具 v5
echo.
echo ==================================================
echo   扫描照片自动裁剪工具 v5
echo   基于 YOLO26n 照片检测模型
echo ==================================================
echo.

REM === 查找 Python 安装路径 ===
set PYTHON_EXE=

REM 方法1: 检查常见安装路径
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%ProgramFiles%\Python313\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles%\Python310\python.exe"
) do (
    if exist %%P (
        set PYTHON_EXE=%%~P
        goto :found
    )
)

REM 方法2: 尝试 where 命令（会跳过 App Execution Aliases 如果 Python 在 PATH 中）
for /f "delims=" %%i in ('where python 2^>nul') do (
    echo %%i | findstr /i "WindowsApps" >nul
    if errorlevel 1 (
        set PYTHON_EXE=%%i
        goto :found
    )
)

REM 方法3: 尝试 py launcher
where py >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_EXE=py
    goto :found
)

REM 未找到 Python
echo [错误] 未找到 Python 安装
echo.
echo 请安装 Python 3.10 或更高版本:
echo   下载地址: https://www.python.org/downloads/
echo   安装时务必勾选 "Add Python to PATH"
echo.
pause
exit /b 1

:found
echo [OK] 找到 Python: %PYTHON_EXE%

REM === 检查依赖 ===
"%PYTHON_EXE%" -c "import ultralytics" >nul 2>nul
if %errorlevel% neq 0 (
    echo [提示] 正在安装依赖（首次运行需要）...
    "%PYTHON_EXE%" -m pip install ultralytics opencv-python numpy
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败，请手动运行:
        echo   "%PYTHON_EXE%" -m pip install ultralytics opencv-python numpy
        pause
        exit /b 1
    )
    echo [OK] 依赖安装完成
    echo.
)

REM === 检查 input 目录 ===
set INPUT_DIR=%~dp0input
set HAS_FILES=0
for %%f in ("%INPUT_DIR%\*.jpg" "%INPUT_DIR%\*.jpeg" "%INPUT_DIR%\*.png" "%INPUT_DIR%\*.bmp") do (
    if exist "%%f" set HAS_FILES=1
)
if %HAS_FILES% equ 0 (
    echo [提示] input/ 目录为空
    echo 请将扫描图放入 input/ 文件夹，然后重新运行
    echo.
    explorer "%INPUT_DIR%"
    pause
    exit /b 0
)

REM === 运行裁剪 ===
echo.
echo 开始处理...
echo.
"%PYTHON_EXE%" "%~dp0crop_photos.py" --input "%INPUT_DIR%" --output "%~dp0output"

echo.
echo 处理完成! 正在打开输出目录...
explorer "%~dp0output"
pause

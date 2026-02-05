@echo off
REM 호스트 빌드 -> main.exe 실행 -> 골든 비교 (CMD에서 프로젝트 폴더로 cd 한 뒤 실행)
setlocal
cd /d "%~dp0"

REM Usage:
REM   build_run_compare.bat       -> FP32 빌드/실행/비교
REM   build_run_compare.bat w8    -> W8A32 빌드/실행/비교 (assets/weights_w8.bin 필요)

set "GCC=C:\msys64\ucrt64\bin\gcc.exe"
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
if not exist "%GCC%" (
    echo [ERROR] gcc not found at %GCC%
    echo Add MSYS2 ucrt64\bin to PATH, or edit GCC path in this script.
    exit /b 1
)

echo [1/3] Building main.exe ...
set "CFLAGS=-I. -Icsrc -std=c99 -O2 -lm"
if /i "%1"=="w8" (
  set "CFLAGS=%CFLAGS% -DUSE_WEIGHTS_W8"
)
"%GCC%" -o main.exe csrc/main.c csrc/blocks/conv_w8a32.c csrc/blocks/c3_w8a32.c csrc/blocks/decode.c csrc/blocks/detect_w8a32.c csrc/blocks/nms.c csrc/blocks/sppf_w8a32.c csrc/operations/bottleneck_w8a32.c csrc/operations/concat_w8a32.c csrc/operations/conv2d_w8a32.c csrc/operations/maxpool2d_w8a32.c csrc/operations/silu_w8a32.c csrc/operations/upsample_w8a32.c csrc/utils/feature_pool.c csrc/utils/image_loader.c csrc/utils/weights_loader.c csrc/utils/uart_dump.c %CFLAGS%
if errorlevel 1 (
    echo [ERROR] Build failed. Fix errors above, then run again.
    exit /b 1
)
echo Build OK.

echo.
echo [2/3] Running main.exe ...
main.exe
if errorlevel 1 (
    echo [ERROR] main.exe failed.
    exit /b 1
)

echo.
echo [3/3] Comparing with golden (decode_detections.py --compare) ...
py -3 tools/decode_detections.py --compare
echo.
echo Done.
exit /b 0

@echo off
REM Host build (no BARE_METAL). Requires gcc in PATH (MinGW/WSL).
REM Usage: build_host.bat        -> FP32 (assets/weights.bin)
REM        build_host.bat w8     -> W8A32 (assets/weights_w8.bin)
REM        build_host.bat w8a16  -> W8A16 (assets/weights_w8.bin)
setlocal

REM Try MSYS2 gcc if not in PATH
if exist "C:\msys64\ucrt64\bin\gcc.exe" (
  set "PATH=C:\msys64\ucrt64\bin;%PATH%"
) else if exist "C:\msys64\mingw64\bin\gcc.exe" (
  set "PATH=C:\msys64\mingw64\bin;%PATH%"
)

set CSRC=csrc
set INC=-I. -I%CSRC%
set CFLAGS=-std=c99 -O2 -lm

if /i "%1"=="w8a16" (
  set "CFLAGS=%CFLAGS% -DUSE_W8A16 -DUSE_WEIGHTS_W8"
  echo Building main.exe [W8A16] ...
  gcc -o main.exe %CSRC%\main.c ^
    %CSRC%\blocks\conv_w8a16.c %CSRC%\blocks\c3_w8a16.c %CSRC%\blocks\decode.c %CSRC%\blocks\detect_w8a16.c %CSRC%\blocks\nms.c %CSRC%\blocks\sppf_w8a16.c ^
    %CSRC%\operations\bottleneck_w8a16.c %CSRC%\operations\concat_w8a16.c %CSRC%\operations\conv2d_w8a16.c %CSRC%\operations\maxpool2d_w8a16.c %CSRC%\operations\silu_w8a16.c %CSRC%\operations\upsample_w8a16.c ^
    %CSRC%\utils\feature_pool.c %CSRC%\utils\image_loader.c %CSRC%\utils\weights_loader.c %CSRC%\utils\timing.c %CSRC%\utils\uart_dump.c ^
    %INC% %CFLAGS%
) else if /i "%1"=="w8" (
  set "CFLAGS=%CFLAGS% -DUSE_WEIGHTS_W8"
  echo Building main.exe [W8A32] ...
  gcc -o main.exe %CSRC%\main.c ^
    %CSRC%\blocks\conv_w8a32.c %CSRC%\blocks\c3_w8a32.c %CSRC%\blocks\decode.c %CSRC%\blocks\detect_w8a32.c %CSRC%\blocks\nms.c %CSRC%\blocks\sppf_w8a32.c ^
    %CSRC%\operations\bottleneck_w8a32.c %CSRC%\operations\concat_w8a32.c %CSRC%\operations\conv2d_w8a32.c %CSRC%\operations\maxpool2d_w8a32.c %CSRC%\operations\silu_w8a32.c %CSRC%\operations\upsample_w8a32.c ^
    %CSRC%\utils\feature_pool.c %CSRC%\utils\image_loader.c %CSRC%\utils\weights_loader.c %CSRC%\utils\timing.c %CSRC%\utils\uart_dump.c ^
    %INC% %CFLAGS%
) else (
  echo Building main.exe [FP32] ...
  gcc -o main.exe %CSRC%\main.c ^
    %CSRC%\blocks\conv_w8a32.c %CSRC%\blocks\c3_w8a32.c %CSRC%\blocks\decode.c %CSRC%\blocks\detect_w8a32.c %CSRC%\blocks\nms.c %CSRC%\blocks\sppf_w8a32.c ^
    %CSRC%\operations\bottleneck_w8a32.c %CSRC%\operations\concat_w8a32.c %CSRC%\operations\conv2d_w8a32.c %CSRC%\operations\maxpool2d_w8a32.c %CSRC%\operations\silu_w8a32.c %CSRC%\operations\upsample_w8a32.c ^
    %CSRC%\utils\feature_pool.c %CSRC%\utils\image_loader.c %CSRC%\utils\weights_loader.c %CSRC%\utils\timing.c %CSRC%\utils\uart_dump.c ^
    %INC% %CFLAGS%
)
if errorlevel 1 exit /b 1

echo Build OK. Run: main.exe
if /i "%1"=="w8" echo (W8A32 needs assets/weights_w8.bin - run: py -3 tools/quantize_weights.py)
if /i "%1"=="w8a16" echo (W8A16 needs assets/weights_w8.bin - run: py -3 tools/quantize_weights.py)
echo For unit tests see TESTING.md (per-test gcc commands).
exit /b 0

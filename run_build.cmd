@echo off
cd /d "c:\Users\lenovo\Desktop\YOLOv5n_in_C"
set GCC=C:\msys64\ucrt64\bin\gcc.exe
if not exist "%GCC%" (
  echo gcc not found: %GCC%
  exit /b 1
)

echo Building main.exe with %GCC% ...
call "%GCC%" -o main.exe ^
  csrc/main.c ^
  csrc/blocks/conv_w8a32.c csrc/blocks/c3_w8a32.c csrc/blocks/decode.c csrc/blocks/detect_w8a32.c csrc/blocks/nms.c csrc/blocks/sppf_w8a32.c ^
  csrc/operations/bottleneck_w8a32.c csrc/operations/concat_w8a32.c csrc/operations/conv2d_w8a32.c csrc/operations/maxpool2d_w8a32.c csrc/operations/silu_w8a32.c csrc/operations/upsample_w8a32.c ^
  csrc/utils/feature_pool.c csrc/utils/image_loader.c csrc/utils/weights_loader.c csrc/utils/timing.c csrc/utils/uart_dump.c ^
  -I. -Icsrc -std=c99 -O2 -lm ^
  1>gcc_out.txt 2>gcc_err.txt

set ERR=%ERRORLEVEL%
echo EXITCODE=%ERR%
echo EXITCODE=%ERR%>> gcc_err.txt
exit /b %ERR%

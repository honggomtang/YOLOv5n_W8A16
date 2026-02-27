@echo off
REM Conv acc TB: iverilog + vvp (optional). Prefer Vivado: Add vsrc + tb_conv_acc.v, Run Behavioral Simulation.
set VSRC=%~dp0
cd /d %VSRC%
where iverilog >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo iverilog not in PATH. Use Vivado for simulation.
  exit /b 1
)
iverilog -o tb_conv_acc.vvp ^
  -s tb_conv_acc ^
  pe_mac.v pe_cluster.v conv_acc_requant.v conv_acc_buffer.v conv_acc_compute.v conv_acc_top.v tb_conv_acc.v
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
vvp tb_conv_acc.vvp
exit /b %ERRORLEVEL%

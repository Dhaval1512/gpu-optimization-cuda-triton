@echo off
REM JLR GPU Optimization - Nsight Compute Profiling Script
REM Run this on Windows PowerShell or CMD

echo ============================================================================
echo   NSIGHT COMPUTE PROFILING - CUDA KERNELS
echo ============================================================================
echo.

REM Set Nsight Compute path (adjust if needed)
set NCU="C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe"

REM Create output directory
if not exist profiling\nsight mkdir profiling\nsight

echo [1/4] Profiling GELU kernel...
%NCU% --set full --export profiling\nsight\gelu_profile python profile_single_kernel.py gelu
echo.

echo [2/4] Profiling Swish kernel...
%NCU% --set full --export profiling\nsight\swish_profile python profile_single_kernel.py swish
echo.

echo [3/4] Profiling LayerNorm kernel...
%NCU% --set full --export profiling\nsight\layernorm_profile python profile_single_kernel.py layernorm
echo.

echo [4/4] Profiling Fused LN+GELU kernel...
%NCU% --set full --export profiling\nsight\fused_ln_gelu_profile python profile_single_kernel.py fused_ln_gelu
echo.

echo ============================================================================
echo   PROFILING COMPLETE!
echo ============================================================================
echo.
echo Generated files:
echo   - profiling\nsight\gelu_profile.ncu-rep
echo   - profiling\nsight\swish_profile.ncu-rep
echo   - profiling\nsight\layernorm_profile.ncu-rep
echo   - profiling\nsight\fused_ln_gelu_profile.ncu-rep
echo.
echo Next: Run extract_ncu_metrics.py to get CSV data
echo.
pause
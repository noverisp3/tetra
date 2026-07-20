@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 (
    echo vcvarsall.bat failed
    exit /b 1
)

set SRC=.\tetra.cpp

if /I "%1"=="avx2" (
    cl /EHsc /O2 /std:c++17 /arch:AVX2 /Fe:tetra_avx2.exe %SRC%
    echo Build: tetra_avx2.exe (AVX2+FMA)
) else if /I "%1"=="avx10" (
    cl /EHsc /O2 /std:c++17 /arch:AVX10 /Fe:tetra_avx10.exe %SRC%
    echo Build: tetra_avx10.exe (AVX10)
) else if /I "%1"=="avx512" (
    cl /EHsc /O2 /std:c++17 /arch:AVX512 /Fe:tetra_avx512.exe %SRC%
    echo Build: tetra_avx512.exe (AVX-512)
) else if "%1"=="" (
    cl /EHsc /O2 /std:c++17 /Fe:tetra.exe %SRC%
    echo Build: tetra.exe (scalar fallback)
) else (
    echo Usage: build.bat [avx2^|avx10^|avx512]
    echo   (no args)  - scalar fallback
    echo   avx2       - AVX2+FMA
    echo   avx10      - AVX10
    echo   avx512     - AVX-512
    exit /b 1
)

if %errorlevel% neq 0 (
    echo Compilation failed
    exit /b 1
)

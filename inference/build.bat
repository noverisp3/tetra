@echo off
set VC_VARS=
where vswhere >nul 2>nul
if %errorlevel% equ 0 (
    for /f "usebackq delims=" %%i in (`vswhere -latest -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" set "VC_VARS=%%i\VC\Auxiliary\Build\vcvarsall.bat"
    )
)
if "%VC_VARS%"=="" (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" set "VC_VARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
)
if "%VC_VARS%"=="" (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" set "VC_VARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
)
if "%VC_VARS%"=="" (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" set "VC_VARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
)
if "%VC_VARS%"=="" (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" set "VC_VARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
)
if "%VC_VARS%"=="" (
    if exist "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" set "VC_VARS=C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat"
)
if "%VC_VARS%"=="" (
    echo ERROR: Visual Studio not found. Install VS 2022/2019 or set VC_VARS manually.
    exit /b 1
)
call "%VC_VARS%" x64
if %errorlevel% neq 0 (
    echo vcvarsall.bat failed
    exit /b 1
)

set SRC=.\tetra.cpp

set OPENMP=/openmp

if /I "%1"=="avx2" (
    cl /EHsc /O2 /std:c++17 %OPENMP% /arch:AVX2 /Fe:tetra_avx2.exe %SRC%
    echo Build: tetra_avx2.exe (AVX2+FMA)
) else if /I "%1"=="avx10" (
    cl /EHsc /O2 /std:c++17 %OPENMP% /arch:AVX10 /Fe:tetra_avx10.exe %SRC%
    echo Build: tetra_avx10.exe (AVX10)
) else if /I "%1"=="avx512" (
    cl /EHsc /O2 /std:c++17 %OPENMP% /arch:AVX512 /Fe:tetra_avx512.exe %SRC%
    echo Build: tetra_avx512.exe (AVX-512)
) else if "%1"=="" (
    cl /EHsc /O2 /std:c++17 %OPENMP% /Fe:tetra.exe %SRC%
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

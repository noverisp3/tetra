@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 (
    echo vcvarsall.bat failed
    exit /b 1
)
cl /EHsc /O2 /std:c++17 /arch:AVX512 /Fe:tetra.exe tetra.cpp
if %errorlevel% neq 0 (
    echo Compilation failed
    exit /b 1
)
echo Build successful: tetra.exe

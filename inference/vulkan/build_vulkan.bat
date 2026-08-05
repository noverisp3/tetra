@echo off
setlocal
set VULKAN_SDK_DEFAULT=C:\VulkanSDK\1.4.357.0
if "%VULKAN_SDK%"=="" set "VULKAN_SDK=%VULKAN_SDK_DEFAULT%"
if not exist "%VULKAN_SDK%\Include\vulkan\vulkan.h" (
    echo ERROR: Vulkan SDK not found at %VULKAN_SDK%. Install LunarG Vulkan SDK or set VULKAN_SDK.
    exit /b 1
)

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
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" set "VC_VARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
)
if "%VC_VARS%"=="" (
    if exist "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" set "VC_VARS=C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat"
)
if "%VC_VARS%"=="" (
    echo ERROR: Visual Studio not found.
    exit /b 1
)
call "%VC_VARS%" x64
if %errorlevel% neq 0 (
    echo vcvarsall.bat failed
    exit /b 1
)

echo == Compiling shaders ==
for %%s in (shaders\embed shaders\rmsnorm shaders\mm_partial shaders\mm_reduce shaders\attention shaders\silu shaders\add_residual shaders\cache_store shaders\capture shaders\rulec shaders\embgrad) do (
    "%VULKAN_SDK%\Bin\glslc.exe" -O %%s.comp -o %%s.spv
    if errorlevel 1 (
        echo glslc failed on %%s.comp
        exit /b 1
    )
)
echo == Shaders OK ==

echo == Compiling host ==
cl /nologo /EHsc /O2 /std:c++17 /I .. /I "%VULKAN_SDK%\Include" vulkan_forward.cpp /Fe:vulkan_forward.exe /link /LIBPATH:"%VULKAN_SDK%\Lib" vulkan-1.lib
if %errorlevel% neq 0 (
    echo Compilation failed
    exit /b 1
)
echo Build OK: vulkan_forward.exe

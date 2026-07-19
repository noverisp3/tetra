cmake_minimum_required(VERSION 3.14)
project(tetra CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific flags
if(MSVC)
    # MSVC: enable AVX512 if available
    add_compile_options(/arch:AVX512 /O2 /fp:fast)
else()
    # GCC/Clang
    add_compile_options(-O3 -ffast-math)

    # Detect and enable SIMD
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" HAS_AVX512)
    if(HAS_AVX512)
        add_compile_options(-mavx512f -mavx512bw -mavx512dq)
        message(STATUS "AVX-512 enabled")
    else()
        check_cxx_compiler_flag("-mavx2" HAS_AVX2)
        if(HAS_AVX2)
            add_compile_options(-mavx2 -mfma)
            message(STATUS "AVX2 enabled")
        endif()
    endif()

    check_cxx_compiler_flag("-march=native" HAS_NATIVE)
    if(HAS_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

# Tetra executable
add_executable(tetra tetra.cpp)
target_include_directories(tetra PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Info
message(STATUS "Tetra build type: ${CMAKE_BUILD_TYPE}")

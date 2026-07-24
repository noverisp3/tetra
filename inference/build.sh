#!/usr/bin/env bash
set -euo pipefail

SRC="$(dirname "$0")/tetra.cpp"
OUTDIR="$(dirname "$0")"
CXX="${CXX:-g++}"
BASEFLAGS="-O3 -std=c++17 -fopenmp"

case "${1:-scalar}" in
    avx2)
        $CXX $BASEFLAGS -mavx2 -mfma -o "$OUTDIR/tetra_avx2" "$SRC"
        echo "Build: tetra_avx2 (AVX2+FMA)"
        ;;
    avx512)
        $CXX $BASEFLAGS -mavx512f -mavx512bw -o "$OUTDIR/tetra_avx512" "$SRC"
        echo "Build: tetra_avx512 (AVX-512)"
        ;;
    scalar|"")
        $CXX $BASEFLAGS -o "$OUTDIR/tetra" "$SRC"
        echo "Build: tetra (scalar fallback)"
        ;;
    clean)
        rm -f "$OUTDIR/tetra" "$OUTDIR/tetra_avx2" "$OUTDIR/tetra_avx512"
        echo "Cleaned"
        ;;
    *)
        echo "Usage: $0 [avx2|avx512|scalar|clean]"
        echo "  avx2    - AVX2+FMA (default)"
        echo "  avx512  - AVX-512"
        echo "  scalar  - no SIMD flags"
        echo "  clean   - remove built binaries"
        exit 1
        ;;
esac

#!/usr/bin/env bash
set -euo pipefail

SRC="$(dirname "$0")/tetra.cpp"
SRC2="$(dirname "$0")/selflearn.cpp"
OUTDIR="$(dirname "$0")"
CXX="${CXX:-g++}"
BASEFLAGS="-O3 -std=c++17 -fopenmp"

case "${1:-scalar}" in
    avx2)
        $CXX $BASEFLAGS -mavx2 -mfma -o "$OUTDIR/tetra_avx2" "$SRC"
        $CXX $BASEFLAGS -mavx2 -mfma -o "$OUTDIR/selflearn_avx2" "$SRC2"
        echo "Build: tetra_avx2 + selflearn_avx2 (AVX2+FMA)"
        ;;
    avx512)
        $CXX $BASEFLAGS -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma \
            -o "$OUTDIR/tetra_avx512" "$SRC"
        $CXX $BASEFLAGS -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma \
            -o "$OUTDIR/selflearn_avx512" "$SRC2"
        echo "Build: tetra_avx512 + selflearn_avx512 (AVX-512+VL-BW-DQ-FMA)"
        ;;
    scalar|"")
        $CXX $BASEFLAGS -o "$OUTDIR/tetra" "$SRC"
        $CXX $BASEFLAGS -o "$OUTDIR/selflearn" "$SRC2"
        echo "Build: tetra + selflearn (scalar fallback)"
        ;;
    clean)
        rm -f "$OUTDIR/tetra" "$OUTDIR/tetra_avx2" "$OUTDIR/tetra_avx512" \
              "$OUTDIR/selflearn" "$OUTDIR/selflearn_avx2" "$OUTDIR/selflearn_avx512"
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

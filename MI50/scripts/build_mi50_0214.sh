#!/bin/bash
# build_mi50_0214.sh -- HIP/gfx906 build of the upstream v0.2.14.1 checkout in ~/port/mi50-0214.
# Same build options as the original MI50 production binary (~/strata/build-906-0210): no MMQ prefill,
# no portable, Release -- so the A/B isolates the engine version, not the build flavour.
set -eu

SRC=${SRC:-/home/yanyq/port/mi50-0214}
BD=${BD:-$SRC/build-906-0214}
ROCM=${ROCM:-/opt/rocm715-gfx906}
JOBS=${JOBS:-24}
ARCH=${ARCH:-gfx906}
LOG=${LOG:-$BD/build.log}
MMQ=${MMQ:-0}          # 1 -> STRATA_MMQ + STRATA_PREFILL_MMQ on (a separate arm)

[ -d "$ROCM" ] || { echo "ROCM=$ROCM not found" >&2; exit 1; }
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr
export ROCM_PATH=$ROCM HIP_PATH=$ROCM
export PATH="$ROCM/bin:$PATH"

mkdir -p "$BD"
echo "=== $(date -Is) $(git -C "$SRC" describe --tags) $(git -C "$SRC" rev-parse --short HEAD) arch=$ARCH mmq=$MMQ rocm=$ROCM" | tee "$LOG"

if [ "$MMQ" = "1" ]; then
  MMQ_FLAGS="-DSTRATA_MMQ=ON -DSTRATA_PREFILL_MMQ=ON"
else
  MMQ_FLAGS="-DSTRATA_MMQ=OFF -DSTRATA_PREFILL_MMQ=OFF"
fi

cmake -G Ninja -S "$SRC" -B "$BD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_HIP_GFX906=ON \
  $MMQ_FLAGS \
  -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
  -DCMAKE_C_COMPILER="$ROCM/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER="$ROCM/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM" \
  "-DCMAKE_PREFIX_PATH=$ROCM" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$ROCM --rocm-device-lib-path=$ROCM/lib/llvm/amdgcn/bitcode" \
  >> "$LOG" 2>&1 || { echo "CONFIGURE FAILED"; tail -40 "$LOG"; exit 1; }
echo "=== configure ok" | tee -a "$LOG"

nice -n 10 ninja -C "$BD" strata -j "$JOBS" >> "$LOG" 2>&1 || { echo "BUILD FAILED"; grep -E "error:|FAILED:" "$LOG" | head -30; exit 1; }
echo "=== built: $(ls -la "$BD/strata")" | tee -a "$LOG"
"$BD/strata" --version 2>&1 | head -3 | tee -a "$LOG" || true
echo DONE

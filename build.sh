#!/bin/sh
# build.sh -- configure, build and run the fast tests.
#
# With no arguments the build configures itself: if hipcc is present and a card can be enumerated,
# device kernels are compiled for that card's architecture, and otherwise the host backend is
# built. The host backend is a real implementation, not a stub -- everything except device kernels
# is exercised through it -- so a machine with no ROCm is a supported development configuration.
#
# Pass -DRAD_GPU_TARGETS=<arch>[;<arch>...] to cmake to compile for cards this machine does not
# have. Which of them actually get kernels is decided by the kernel libraries present; each one
# declares the architectures it serves.
#
#   ./build.sh              configure + build + the fast tests (ctest -L fast)
#   ./build.sh -c           wipe the build directory first
#   ./build.sh --no-hip     force the host backend even where hipcc exists
#   ./build.sh --asan       AddressSanitizer + UBSan, host backend, in build-asan/
#   ./build.sh --tsan       ThreadSanitizer, host backend, in build-tsan/
#   ./build.sh --debug      CMAKE_BUILD_TYPE=Debug
#   ./build.sh -b DIR       build in DIR instead of build/ (or build-asan/, build-tsan/)
#
# The other tiers (docs/GUIDE.md §1.3): ctest --test-dir build -LE 'gpu|oot' for everything that
# runs without a card, plain ctest --test-dir build for everything.
set -e

BUILD=
CLEAN=0
SAN=
CMAKE_ARGS=""

while [ $# -gt 0 ]; do
    case "$1" in
        -c|--clean) CLEAN=1 ;;
        --no-hip)   CMAKE_ARGS="$CMAKE_ARGS -DRAD_WITH_HIP=OFF" ;;
        --asan)     SAN=asan ;;
        --tsan)     SAN=tsan ;;
        --debug)    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug" ;;
        -b)         shift; BUILD="$1" ;;
        *)          echo "build.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

# A sanitised build gets a directory of its own, so it never reconfigures the normal tree. Host
# code only: device kernels are not sanitised, and a sanitised engine is not one to serve from.
case "$SAN" in
    asan) CMAKE_ARGS="$CMAKE_ARGS -DRAD_ASAN=ON -DRAD_TSAN=OFF -DRAD_WITH_HIP=OFF"
          BUILD=${BUILD:-build-asan} ;;
    tsan) CMAKE_ARGS="$CMAKE_ARGS -DRAD_TSAN=ON -DRAD_ASAN=OFF -DRAD_WITH_HIP=OFF"
          BUILD=${BUILD:-build-tsan} ;;
    *)    BUILD=${BUILD:-build} ;;
esac
[ "$CLEAN" = 1 ] && rm -rf "$BUILD"

# The engine's compiler is ROCm's clang, the one hipcc drives -- not whatever `cc` is on PATH and
# not a system GCC. A build with another compiler is not comparable and does not verify a change,
# so this script names both compilers itself; a deliberate other compiler is a separate configure.
ENGINE_CC="${ROCM_PATH:-/opt/rocm}/llvm/bin/clang"
ENGINE_CXX="${ROCM_PATH:-/opt/rocm}/llvm/bin/clang++"
if [ ! -x "$ENGINE_CC" ] || [ ! -x "$ENGINE_CXX" ]; then
    echo "build.sh: the engine's compiler is missing: $ENGINE_CXX" >&2
    echo "build.sh: install ROCm's clang or set ROCM_PATH; a system compiler is not a substitute" >&2
    exit 2
fi
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER=$ENGINE_CC -DCMAKE_CXX_COMPILER=$ENGINE_CXX"

cmake -S . -B "$BUILD" -G Ninja $CMAKE_ARGS
cmake --build "$BUILD" -j"$(nproc)"
ctest --test-dir "$BUILD" -L '^fast$' -j8 --no-tests=error --output-on-failure

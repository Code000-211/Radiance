# shared-i8a8-down -- an out-of-tree kernel plugin: the shared expert's down projection at the
# speculative-verify width, as `gemm_nt_q` at dtype i8a8. See README.md.
#
# Build, from this directory:
#
#   cmake -S . -B build -G Ninja \
#         -DCMAKE_PREFIX_PATH=<radiance install prefix> \
#         -DRAD_LIBR4D_DIR=<radiance source>/libr4d \
#         -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
#   cmake --build build -j
#
# Two inputs beyond the SDK, because neither is installed:
#
#   * RAD_LIBR4D_DIR -- libr4d's own sources. The row must publish libr4d's int8 weight layout, and
#     the only honest way to publish the same tag and the same bytes is to compile that library's
#     implementation (r4d_fp8_layout.cpp) into this DSO. Re-implementing it here would drift from
#     the bytes libr4d's wider row reads for the same weight. The headers it includes (r4d_args.h,
#     r4d_common.h, r4d_plugin.h, r4d_fp8_frag.h, r4d_enc.h, r4d_fp8_lm.h, r4d_hc_fp8.h) are not
#     installed either, so the same path is on the include line.
#   * RAD_LIBR4D_DIR is the engine tree this plugin is built against. It is a build-time input,
#     not a link dependency: the DSO has no undefined symbol from libr4d and does not load it.
#
# The candidate itself lives on `radiance::abi` alone, like any third-party plugin.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(shared_i8a8_down LANGUAGES CXX HIP)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# The plugin is gfx1201 only: the body is written against gfx12 WMMA-free `v_dot4_i32_i8` (sudot4)
# and the fragment order that libr4d's gfx1201 kernels use.
rad_plugin_gpu_targets(SI8_GPU_TARGETS "^gfx1201$")
if(NOT SI8_GPU_TARGETS)
  message(STATUS
    "shared-i8a8-down: gfx1201 only, and this build targets ${RAD_GPU_TARGETS} -- not built")
  return()
endif()

# Default: the libr4d of the radiance tree this candidate sits in
# (candidates/shared-i8a8-down -> ../../libr4d). An out-of-tree build of a tree that has the
# candidate elsewhere passes the path explicitly.
set(RAD_LIBR4D_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../libr4d" CACHE PATH
    "libr4d (radiance source) directory: supplies the int8 layout implementation and its headers")

if(NOT RAD_LIBR4D_DIR OR NOT EXISTS "${RAD_LIBR4D_DIR}/r4d_fp8_layout.cpp")
  message(FATAL_ERROR
    "shared-i8a8-down needs -DRAD_LIBR4D_DIR=<radiance source>/libr4d. The row publishes "
    "libr4d's int8 weight layout, and the only honest implementation of that layout is "
    "libr4d/r4d_fp8_layout.cpp compiled into this same DSO (see README.md).")
endif()
if(NOT EXISTS "${RAD_LIBR4D_DIR}/r4d_common.h" OR NOT EXISTS "${RAD_LIBR4D_DIR}/r4d_args.h")
  message(FATAL_ERROR "RAD_LIBR4D_DIR=${RAD_LIBR4D_DIR} does not hold libr4d's headers")
endif()

rad_add_plugin(shared_i8a8_down kernels
  SOURCES
    rows.cpp
    # libr4d's weight layouts, compiled into this DSO with hidden visibility. Its own CMakeLists
    # also gives libr4d's host sources no HIP, and this file includes no HIP header.
    "${RAD_LIBR4D_DIR}/r4d_fp8_layout.cpp"
  HIP_SOURCES
    shared_i8a8_down.hip
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RAD_LIBR4D_DIR}
  GPU_TARGETS ${SI8_GPU_TARGETS})

# Nothing but the ABI's export set leaves the library: the four accessors and the two identity
# functions carry SI8_EXPORT (shared_i8a8_down.h). This is what keeps the borrowed layout hooks
# internal to the DSO instead of interposing on libr4d.so's copies, which the loader opens
# RTLD_LOCAL.
#
# Set per host source rather than as a target-wide -fvisibility=hidden: a device translation unit
# carries HIP's kernel stubs and its registration records, and this build leaves the HIP language's
# defaults alone. The two symbols the .hip defines (the launch and its operand description) are
# declared SI8_LOCAL in shared_i8a8_down.h, which hides them without disturbing the stubs.
set_source_files_properties(rows.cpp "${RAD_LIBR4D_DIR}/r4d_fp8_layout.cpp" PROPERTIES
  COMPILE_OPTIONS "-fvisibility=hidden;-fvisibility-inlines-hidden")

# -ffp-contract=off in both languages. The INT32 dot is exact, but the FP32 group fold and the BF16
# epilogue must not be contracted into an FMA: that is the declaration in gate.json, and it is a
# correctness flag in this project (docs/IMPLEMENTATION.md; rad_hipify passes it for HIP too).
target_compile_options(shared_i8a8_down PRIVATE -ffp-contract=off)
# C++17 for the device source, as libr4d/CMakeLists.txt sets for libr4d itself: at C++20 clang
# tightens the address-space check on the global_load_tr_b128 builtin in r4d_common.h, which the
# row includes for f32_to_bf16.
set_target_properties(shared_i8a8_down PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.21)
# VERSION is the server release that /version reports. The plugin ABI version is separate and is
# read from abi/rad_abi.h below.
project(radiance VERSION 1.10.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# A cache STRING, not option(): option() is boolean, so AUTO would collapse to OFF and skip
# detection.
set(RAD_WITH_HIP "AUTO" CACHE STRING "ON | OFF | AUTO -- build the HIP backend and device kernels")
set_property(CACHE RAD_WITH_HIP PROPERTY STRINGS ON OFF AUTO)
set(RAD_WITH_FFMPEG "AUTO" CACHE STRING
    "ON | OFF | AUTO -- decode image, video and audio parts with libavformat/libavcodec/libswscale/libswresample")
set_property(CACHE RAD_WITH_FFMPEG PROPERTY STRINGS ON OFF AUTO)
option(RAD_BUILD_TESTS "Build the test suite"                            ON)
option(RAD_ASAN        "Build host code with AddressSanitizer and UBSan"  OFF)
option(RAD_TSAN        "Build host code with ThreadSanitizer"             OFF)
# LLVM 24's VOPD pairing is not repeatable unless patched (docker/llvm-vopd-pairing.patch); when
# OFF, pairing is disabled on LLVM 24. See rad_hip_repeatable in cmake/RadAddPlugin.cmake.
option(RAD_HIP_VOPD_REPEATABLE "The HIP compiler's VOPD pairing is repeatable (the image's LLVM 24 is)" OFF)

# ---------------------------------------------------------------- HIP detection
# RAD_GPU_TARGETS lists the offload architectures device kernels are compiled for. Which cards the
# engine serves is decided by the installed kernel plugins, not here (spec §18). Unset means
# enumerate this machine's devices with the ROCm toolchain.
#
# A kernel library covering only some architectures narrows the list with rad_plugin_gpu_targets()
# and skips itself when the intersection is empty (libr4d is RDNA4-only).
#
# Without ROCm the host backend is built, a supported configuration that exercises everything
# except device kernels.
set(RAD_HIP_FOUND OFF)
if(NOT RAD_WITH_HIP STREQUAL "OFF")
  find_program(RAD_HIPCC hipcc HINTS /opt/rocm/bin /opt/rocm/llvm/bin ENV ROCM_PATH)
  if(RAD_HIPCC)
    execute_process(COMMAND ${RAD_HIPCC} --version OUTPUT_VARIABLE RAD_HIPCC_VER ERROR_QUIET)
    set(RAD_HIP_FOUND ON)
  elseif(RAD_WITH_HIP STREQUAL "ON")
    message(FATAL_ERROR "RAD_WITH_HIP=ON but hipcc was not found.")
  endif()
endif()

if(RAD_HIP_FOUND AND NOT DEFINED RAD_GPU_TARGETS)
  # amdgpu-arch / offload-arch print one architecture per visible device, integrated graphics
  # included. Plugins filter the list to what they serve.
  find_program(RAD_ARCH_TOOL NAMES amdgpu-arch offload-arch
               HINTS /opt/rocm/bin /opt/rocm/llvm/bin ENV ROCM_PATH)
  if(RAD_ARCH_TOOL)
    execute_process(COMMAND ${RAD_ARCH_TOOL} OUTPUT_VARIABLE RAD_ARCH_OUT
                    RESULT_VARIABLE RAD_ARCH_RC ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(RAD_ARCH_RC EQUAL 0)
      string(REPLACE "\n" ";" RAD_ARCH_OUT "${RAD_ARCH_OUT}")
      list(FILTER RAD_ARCH_OUT INCLUDE REGEX "^gfx")
      list(REMOVE_DUPLICATES RAD_ARCH_OUT)
      set(RAD_GPU_TARGETS ${RAD_ARCH_OUT})
    endif()
  endif()
  if(NOT RAD_GPU_TARGETS)
    # No card attached: do not guess an architecture. Decline the device build and say how to
    # request one.
    set(RAD_HIP_FOUND OFF)
    message(WARNING
      "radiance: hipcc was found but no AMD GPU could be enumerated, so this build has no device "
      "kernels. To cross-compile them anyway, name the architectures: -DRAD_GPU_TARGETS=gfx1201 "
      "(semicolon-separated for several).")
  endif()
endif()

if(RAD_HIP_FOUND)
  message(STATUS "radiance: HIP backend ON, targets ${RAD_GPU_TARGETS}")
elseif(RAD_WITH_HIP STREQUAL "OFF")
  message(STATUS "radiance: HIP backend OFF by request -- host backend only")
elseif(NOT RAD_HIPCC)
  message(WARNING
    "radiance: HIP backend OFF -- hipcc was not found, so this build has no device kernels.\n"
    "  Expected on a machine without ROCm, wrong on one with cards. Looked in /opt/rocm/bin, "
    "/opt/rocm/llvm/bin and $ROCM_PATH.")
endif()

# ASan + UBSan for host code only. AMD's amdgcn ASan needs xnack+ for its page-fault-based shadow,
# and RDNA has no xnack, so the generator expressions keep the flags off HIP. Device kernels rely on
# rad-kbench's red zones instead: a poisoned margin around every operand and scratch allocation,
# checked after the launch, which catches out-of-bounds writes.
#
# TSan, likewise host code only, for the threaded host paths (scheduler, server, stager, tokenizer
# cache). libgomp is not instrumented, so tests run it with one thread (tests/CMakeLists.txt).
if(RAD_ASAN AND RAD_TSAN)
  message(FATAL_ERROR "RAD_ASAN and RAD_TSAN cannot be combined; use a build directory for each.")
endif()
if(RAD_ASAN)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
                      $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
                      $<$<COMPILE_LANGUAGE:CXX>:-fno-sanitize-recover=all>)
  add_link_options(-fsanitize=address,undefined)
endif()
if(RAD_TSAN)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread>
                      $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
  add_link_options(-fsanitize=thread)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)

# Before the ABI: its INSTALL_INTERFACE path uses CMAKE_INSTALL_INCLUDEDIR, which would otherwise
# expand to nothing and silently export the wrong include directory.
include(GNUInstallDirs)

# ---------------------------------------------------------------- the ABI
# Header-only, and the only thing plugins compile against.
add_library(rad_abi INTERFACE)
# Plugins link radiance::abi: an ALIAS here, an imported target when installed, so rad_add_plugin
# (cmake/RadAddPlugin.cmake) needs one body for both.
add_library(radiance::abi ALIAS rad_abi)
set_target_properties(rad_abi PROPERTIES EXPORT_NAME abi)
# CMAKE_INSTALL_FULL_DATADIR, not a literal "share", so it follows the plugin install destination.
set(RAD_DEFAULT_HOME "${CMAKE_INSTALL_FULL_DATADIR}/radiance" CACHE STRING "default $RADIANCE_HOME")
target_include_directories(rad_abi INTERFACE
  # abi/, not the repo root, so a plugin sees only the ABI headers, never core/ or arch/.
  # Generator expressions because the target is exported, and install(EXPORT) rejects a source
  # path. Plugins write `#include "rad_abi.h"` both in-tree and installed.
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/abi>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/radiance>)
target_compile_definitions(rad_abi INTERFACE RAD_DEFAULT_HOME="${RAD_DEFAULT_HOME}")

if(RAD_HIP_FOUND)
  target_compile_definitions(rad_abi INTERFACE RAD_HAVE_HIP=1)
  # Host translation units such as core/device/hip.cpp use the ordinary C++ compiler and need the
  # HIP headers explicitly.
  #
  # SYSTEM: hipcc already has this directory as a system include. A plain -I would take precedence
  # and make ROCm headers warn in every plugin (rad_abi is an INTERFACE dependency of all of them).
  #
  # BUILD_INTERFACE only: the path is this machine's. The installed package config finds the
  # headers on the consumer's machine (cmake/radiance-config.cmake.in).
  get_filename_component(RAD_ROCM_BIN "${RAD_HIPCC}" DIRECTORY)
  get_filename_component(RAD_ROCM_ROOT "${RAD_ROCM_BIN}" DIRECTORY)
  if(EXISTS "${RAD_ROCM_ROOT}/include/hip/hip_runtime.h")
    set(RAD_ROCM_INCLUDE "${RAD_ROCM_ROOT}/include")
  elseif(EXISTS "/opt/rocm/include/hip/hip_runtime.h")
    set(RAD_ROCM_INCLUDE "/opt/rocm/include")
  endif()
  if(RAD_ROCM_INCLUDE)
    target_include_directories(rad_abi SYSTEM INTERFACE $<BUILD_INTERFACE:${RAD_ROCM_INCLUDE}>)
  else()
    message(FATAL_ERROR
      "hipcc is at ${RAD_HIPCC} but hip/hip_runtime.h is not under ${RAD_ROCM_ROOT}/include "
      "or /opt/rocm/include. Set ROCM_PATH.")
  endif()
  target_compile_definitions(rad_abi INTERFACE __HIP_PLATFORM_AMD__=1)
endif()

set(RAD_PLUGIN_OUT ${CMAKE_BINARY_DIR}/radiance_home)
file(MAKE_DIRECTORY ${RAD_PLUGIN_OUT}/kernels ${RAD_PLUGIN_OUT}/architectures)

# Recorded kernel fixtures, copied into the build's $RADIANCE_HOME (where rad-kbench looks; an
# installed prefix has no source tree). data/kernels.rkb lets rad-kbench run with no model: it holds
# geometries, input seeds and libref's answers reduced to a norm and a sample. Re-record with a 27B
# container:
#
#   rad-kbench -m <model.rad> --max-cases 1 --record data/kernels.rkb
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/kernels.rkb)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/kernels.rkb
                 ${RAD_PLUGIN_OUT}/kernels.rkb COPYONLY)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/kernels.rkb
          DESTINATION ${CMAKE_INSTALL_DATADIR}/radiance)
endif()

# A recording covers one container's geometries, and no single model declares the whole vocabulary.
# The 27B is dense with a DFlash2 drafter and gqa 6; the 35B-A3B is routed (moe_gemm_q,
# router_topk, moe_scatter, moe_gather, scale_rows) with an MTP head and gqa 8. Re-record with
#
#   rad-kbench -m <model.rad> --max-cases 1 --record data/kernels_moe.rkb
#
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/kernels_moe.rkb)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/kernels_moe.rkb
                 ${RAD_PLUGIN_OUT}/kernels_moe.rkb COPYONLY)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/kernels_moe.rkb
          DESTINATION ${CMAKE_INSTALL_DATADIR}/radiance)
endif()
# The published refusal-direction vector for qwen4exp, copied into the built home's vectors/ so a
# Flash-Next container offers the switch with nothing for an operator to install: core/cvec.cpp looks
# for <home>/vectors/<arch>-control-vector.gguf after the sidecar paths. The file is Cudecnik's,
# unchanged (huggingface.co/Cudecnik/Qwen3.8-Flash-Next-refusal-projection; the gguf is sha256
# ef0724c5b79297e4..., the json d4a3dd842506a6fb..., and that is what the measurements in
# docs/GUIDE.md 4.10 were made with). Refresh with:
#
#   huggingface-cli download Cudecnik/Qwen3.8-Flash-Next-refusal-projection \
#     Qwen3.8-Flash-Next-refusal-projection.{gguf,json}
#
# The directions are derived from Qwen3.8-Flash-Next's weights and stay under the Qwen Community
# License 1.0 (data/vectors/LICENSE.qwen-community-1.0.txt, NOTICE); radiance's Apache-2.0 grant
# covers the code that reads them, never the file. It ships loaded only when the deployment asks
# (--control-vector), and the request default is off, because what it removes is a safety behaviour.
file(MAKE_DIRECTORY ${RAD_PLUGIN_OUT}/vectors)
foreach(rrad_vector qwen4exp-control-vector.gguf qwen4exp-control-vector.json)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/vectors/${rrad_vector}
                 ${RAD_PLUGIN_OUT}/vectors/${rrad_vector} COPYONLY)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/vectors/${rrad_vector}
          DESTINATION ${CMAKE_INSTALL_DATADIR}/radiance/vectors)
endforeach()
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/vectors/LICENSE.qwen-community-1.0.txt
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party RENAME qwen-community.LICENSE)

# The downloader behind `radiance --model <owner/name>` (core/hf_model.cpp). The engine looks in
# <bin>/../libexec/radiance/, so the build tree mirrors that layout.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/hfget.sh
               ${CMAKE_BINARY_DIR}/libexec/radiance/hfget.sh COPYONLY)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/hfget.sh
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/radiance)

# rad-export (a finetune to a .rad with the stock model's recipe) runs on the host, beside the
# trainer. It is installed with the tools so the image hands it out
# (`docker run --entrypoint busybox <image> cat /opt/radiance/bin/rad-export`), though the image
# has no Python to run it.
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/tools/rad-export
        DESTINATION ${CMAKE_INSTALL_BINDIR})

# rad_add_plugin() and rad_hipify(), shared with the installed package config. The variables it
# expects are all set above.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RadAddPlugin.cmake)

if(RAD_HIP_FOUND)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" RAD_CFG_UPPER)
  # The HIP language (CMake >= 3.21): device sources are LANGUAGE HIP and everything else stays on
  # the C++ compiler. Do not set CMAKE_CXX_COMPILER to hipcc; after project() it is silently
  # ignored. CMake refuses hipcc as CMAKE_HIP_COMPILER, so this is ROCm's clang++. ROCm 7.2 /
  # LLVM 22 requires a global address space for __builtin_amdgcn_global_load_tr_b128_v8i16, hence
  # the cast in libr4d/r4d_common.h.
  enable_language(HIP)
  rad_hip_rule_fixes()

  # Device optimisation level for every config. Depending on its version, CMake leaves
  # CMAKE_HIP_FLAGS_<CONFIG> empty (device code at -O0 with asserts) or, from CMake 4.4, pre-fills
  # the host-style '-O2 -g -DNDEBUG'. At -O0 decode reads weights at ~7 GB/s on a ~640 GB/s card.
  # -O3 for kernels; no -g on the device, where it slows compiles and inhibits scheduling.
  #
  # The assignment is unconditional (FORCE) because a guard cannot tell a toolchain default from an
  # operator's choice. Override by name with RAD_HIP_FLAGS_<CONFIG>.
  foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    if(cfg STREQUAL "DEBUG")
      set(lvl "-O0 -g")
    elseif(cfg STREQUAL "MINSIZEREL")
      set(lvl "-Os -DNDEBUG")
    else()
      set(lvl "-O3 -DNDEBUG")
    endif()
    if(RAD_HIP_FLAGS_${cfg})
      set(lvl "${RAD_HIP_FLAGS_${cfg}}")
      message(STATUS "radiance: HIP ${cfg} flags overridden by RAD_HIP_FLAGS_${cfg}: '${lvl}'")
    endif()
    set(CMAKE_HIP_FLAGS_${cfg} "${lvl}" CACHE STRING "HIP flags for ${cfg}" FORCE)
  endforeach()
  message(STATUS "radiance: HIP ${CMAKE_BUILD_TYPE} flags '${CMAKE_HIP_FLAGS_${RAD_CFG_UPPER}}'")

  find_library(RAD_HIP_RUNTIME amdhip64 HINTS /opt/rocm/lib ENV ROCM_PATH)
  # The AQL backend writes packets into HSA queues itself (core/device/aql.cpp).
  find_library(RAD_HSA_RUNTIME hsa-runtime64 HINTS /opt/rocm/lib ENV ROCM_PATH)
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- media decoding
# Image, video and audio request parts are decoded by libavformat/libavcodec and converted by
# libswscale/libswresample (core/mm/media.cpp). Without them the engine serves text and refuses media
# parts. Tri-state like HIP, so a machine with the libraries does not silently build without them.
set(RAD_FFMPEG_FOUND OFF)
if(NOT RAD_WITH_FFMPEG STREQUAL "OFF")
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(RAD_FFMPEG QUIET IMPORTED_TARGET libavformat libavcodec libavutil libswscale
                      libswresample)
  endif()
  if(RAD_FFMPEG_FOUND)
    message(STATUS "radiance: media decoding via libavformat ${RAD_FFMPEG_libavformat_VERSION}")
  elseif(RAD_WITH_FFMPEG STREQUAL "ON")
    message(FATAL_ERROR "RAD_WITH_FFMPEG=ON but libavformat/libavcodec/libavutil/libswscale/libswresample were not found.")
  else()
    message(STATUS "radiance: no libavformat -- image, video and audio input will be refused")
  endif()
endif()

add_subdirectory(vendor)
add_subdirectory(core)
# Kernel plugins, one root directory each. Each holds its kernels, row table and layouts and links
# only rad_abi, exactly like a plugin written out of tree.
add_subdirectory(libref)
# libquant: the built-in quantisers, the third plugin kind (spec §4.5). Host C++, built everywhere:
# rad-convert drives it and the oracle decodes through it.
add_subdirectory(libquant)
if(RAD_HIP_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libr4d/CMakeLists.txt)
  add_subdirectory(libr4d)
endif()
# libavx: the production host kernels (x86 SIMD, dispatched by CPU feature), built on every x86-64.
# Not optional: host-placed ops need a kernel other than the oracle (rad_gate_reference_kernels
# refuses libref), and a table too large for the card, such as the n-gram embedding's gather, runs
# on the host every step. Also builds standalone without ROCm (`cmake -S libavx -B build-avx`).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  add_subdirectory(libavx)
endif()
add_subdirectory(arch)
# spechub: the linear-MTP coupling kernels (RADIANCE_MTP_COUPLED), the one in-tree plugin directory
# the experimental line owns. Built by default so the flag's kernels exist in the same tree that
# enables it; a deployment that never sets the flag loads a library whose rows it never resolves.
add_subdirectory(plugin/spechub)
# hc_up_wmma: an opt-in `hc_read` row for the M=3/4 band of Qwen Flash-Next's drafter, with the
# up-dot reduction on WMMA (ported from the ABI16 archive, plugin/hc_up_wmma/README.md). Installed
# but inert without RADIANCE_HC_UP_WMMA=1; with the switch set it wins the op by the default load
# order, where its file (hc_up_wmma.so) sorts before libavx.so and libr4d.so. Listing it first on
# RADIANCE_KERNELS pins that order; it is not needed for the default install.
add_subdirectory(plugin/hc_up_wmma)
add_subdirectory(tools)
if(RAD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------- the installed package
# Install-only rules for `find_package(radiance)`; they do not affect the build tree.
#
# (a) The ABI headers, which include only each other and C standard headers. arch/common/ is
# installed by arch/CMakeLists.txt. Under include/radiance/ to avoid clashes in shared prefixes such
# as /usr; the include spelling is unchanged because the target's include path is radiance/ itself.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/abi/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/radiance
        FILES_MATCHING PATTERN "*.h")

# (b) The ABI target as radiance::abi. It carries the include path, RAD_DEFAULT_HOME and RAD_HAVE_HIP,
# the flags a plugin must be compiled with to load in this engine.
install(TARGETS rad_abi EXPORT radianceTargets)
install(EXPORT radianceTargets
        NAMESPACE radiance::
        FILE radianceTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radiance)

# (c) The package config. Its version is the ABI version, read from the header: the loader requires
# an exact RAD_ABI_VERSION match (core/plugin/loader.cpp), which SameMajorVersion with the ABI as
# the major expresses to CMake.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/abi/rad_abi.h" RAD_ABI_VERSION_LINE
     REGEX "^#define[ \t]+RAD_ABI_VERSION[ \t]")
string(REGEX MATCH "[0-9]+" RAD_ABI_VERSION "${RAD_ABI_VERSION_LINE}")
if(NOT RAD_ABI_VERSION)
  message(FATAL_ERROR
    "could not read RAD_ABI_VERSION out of abi/rad_abi.h; the package config cannot state the "
    "plugin compatibility it is built for without it.")
endif()

include(CMakePackageConfigHelpers)

# PATH_VARS, so the config resolves these relative to where the package is found, which keeps
# staged (DESTDIR) and relocated installs working.
set(RAD_HOME_RELDIR "${CMAKE_INSTALL_DATADIR}/radiance")
set(RAD_INCLUDE_RELDIR "${CMAKE_INSTALL_INCLUDEDIR}/radiance")
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/radiance-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/radiance-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radiance
  PATH_VARS RAD_INCLUDE_RELDIR RAD_HOME_RELDIR)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/radiance-config-version.cmake
  VERSION ${RAD_ABI_VERSION}.0.0
  COMPATIBILITY SameMajorVersion)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/radiance-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/radiance-config-version.cmake
        # The config includes this exact file, so there is one copy for both builds.
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RadAddPlugin.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radiance)

# (d) License and notices. Apache-2.0 and the bundled components' licenses require them to travel
# with every copy, including images built from the install.
install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(FILES vendor/llama/LICENSE
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party RENAME llama.cpp.LICENSE)
install(FILES vendor/nlohmann/LICENSE.MIT
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party RENAME nlohmann-json.LICENSE)
install(FILES vendor/cpp-httplib/LICENSE
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party RENAME cpp-httplib.LICENSE)
install(FILES core/server/LICENSE.lfm
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party RENAME liquidai-d1.LICENSE)

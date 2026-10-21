cmake_minimum_required(VERSION 3.21)
# VERSION is what /version reports and what a client checks compatibility against. It is NOT the
# ABI version -- that lives in abi/rad_abi.h and is read out of the header below, because
# a plugin contract and a server release do not move together.
project(radiance VERSION 1.0.13 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# NOT option(): option() is boolean-only, so a default of AUTO collapses to OFF and the detection
# below never runs -- a machine with cards then silently builds the host backend. Tri-state wants a
# cache STRING.
set(RAD_WITH_HIP "AUTO" CACHE STRING "ON | OFF | AUTO -- build the HIP backend and device kernels")
set_property(CACHE RAD_WITH_HIP PROPERTY STRINGS ON OFF AUTO)
set(RAD_WITH_FFMPEG "AUTO" CACHE STRING "ON | OFF | AUTO -- decode image and video parts with libavformat/libavcodec/libswscale")
set_property(CACHE RAD_WITH_FFMPEG PROPERTY STRINGS ON OFF AUTO)
option(RAD_BUILD_TESTS "Build the test suite"                            ON)
option(RAD_ASAN        "Build with AddressSanitizer"                     OFF)


# ---------------------------------------------------------------- HIP detection
# RAD_GPU_TARGETS is the set of offload architectures device kernels are compiled for. The project
# does not fix it and holds no opinion about which cards are acceptable: a kernel library is a
# plugin, so which hardware the engine can serve is decided by which kernel plugins are installed,
# not here (spec §2.1). Leaving it unset means "ask this machine", through the enumerator the ROCm
# toolchain ships.
#
# A kernel library that covers only some architectures narrows the list for itself with
# rad_plugin_gpu_targets() and skips its own build when the intersection is empty. That is how
# libr4d, which is RDNA4-only, coexists with a target list that may name anything, and it is the
# same mechanism a library for another card uses from out of tree.
#
# With no ROCm at all the host backend is built. That is a supported configuration and not a
# degraded one: everything except device kernels is exercised by it.
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
  # amdgpu-arch and offload-arch each print one architecture per visible device. EVERY device is
  # listed, integrated graphics included, and that is the right input: a plugin filters the list
  # down to what it serves, so an architecture no library covers costs a skipped plugin rather
  # than a refused configure.
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
    # A toolchain with no card attached is an ordinary build machine. Guessing an architecture here
    # would emit kernels for hardware nobody named, so the device build is declined and the way to
    # ask for it anyway is spelled out.
    set(RAD_HIP_FOUND OFF)
    message(WARNING
      "radiance: hipcc was found but no AMD GPU could be enumerated, so this build has NO device "
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
    "radiance: HIP backend OFF -- hipcc was not found, so this build has NO device kernels.\n"
    "  That is correct on a machine without ROCm and WRONG on one with cards. Looked in "
    "/opt/rocm/bin, /opt/rocm/llvm/bin and \.")
endif()

# HOST CODE ONLY, AND THAT IS NOT A COMPROMISE -- IT IS THE WHOLE AVAILABLE SURFACE.
#
# AMD ships an amdgcn AddressSanitizer (/opt/rocm/amdgcn/bitcode/asanrtl.bc) and it cannot run
# on an RDNA card: its shadow is page-fault based, which needs xnack+, and RDNA does not implement
# xnack -- rocminfo reports `XNACK enabled: NO`. Handing -fsanitize=address to the HIP
# compilation therefore buys a device binary that cannot work, so the generator expression keeps
# it off the HIP language entirely. What device kernels get instead is rad-kbench's red zones: a
# poisoned margin either side of every operand and every scratch allocation, verified after the
# launch. That catches the out-of-bounds WRITE, which is the one that corrupts a neighbour.
#
# What this DOES cover is everything else, which is most of the project: the core, the tools,
# libref in full, and every host-side launch wrapper in libr4d. UBSan comes with it because the
# two share a build and the interesting reports here -- a signed shift, a misaligned load, an
# enum out of range -- are UBSan's, not ASan's.
if(RAD_ASAN)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
                      $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
                      $<$<COMPILE_LANGUAGE:CXX>:-fno-sanitize-recover=all>)
  add_link_options(-fsanitize=address,undefined)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)

# GNUInstallDirs before the ABI, not after it, because the ABI target's INSTALL_INTERFACE include
# path is spelled in CMAKE_INSTALL_INCLUDEDIR and an undefined variable there expands to nothing --
# which is not an error, it is an exported target whose include directory is the install prefix
# itself, so every consumer's `#include "rad_abi.h"` misses by one directory level.
include(GNUInstallDirs)

# ---------------------------------------------------------------- the ABI
# Header-only. The ONLY thing plugins compile against.
add_library(rad_abi INTERFACE)
# radiance::abi is the name plugins link, in-tree and out. It is an ALIAS here and an imported
# target in the installed package, and the two are spelled identically on purpose: rad_add_plugin
# has ONE body (cmake/RadAddPlugin.cmake) and it cannot branch on which world it is running in.
add_library(radiance::abi ALIAS rad_abi)
set_target_properties(rad_abi PROPERTIES EXPORT_NAME abi)
# CMAKE_INSTALL_FULL_DATADIR and not a literal "share": it is <prefix>/share under every default,
# and it moves together with the destination the plugins are actually installed to. Those two
# disagreeing is a prefix that looks complete and whose loader finds nothing.
set(RAD_DEFAULT_HOME "${CMAKE_INSTALL_FULL_DATADIR}/radiance" CACHE STRING "default $RADIANCE_HOME")
target_include_directories(rad_abi INTERFACE
  # abi/ AND NOT THE REPO ROOT, which is the whole reason the headers sit in a directory of their
  # own rather than being included as `abi/rad_abi.h` from the top. A plugin links this target and
  # gets exactly the headers in abi/ on its include path -- not core/, not arch/, not the tree. An
  # ABI a plugin can see past is not a boundary.
  #
  # BUILD_INTERFACE/INSTALL_INTERFACE rather than one bare path, because this target is exported:
  # a bare ${CMAKE_CURRENT_SOURCE_DIR}/abi is a hard error out of install(EXPORT) ("path which is
  # prefixed in the source directory"), and the export is the whole point -- an interface target
  # that cannot leave the tree is the defect, not the fix.
  #
  # The spelling is the same in both worlds: `#include "rad_abi.h"` in-tree and installed, because
  # rad_add_plugin has ONE body and cannot branch on which world it is running in.
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/abi>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/radiance>)
target_compile_definitions(rad_abi INTERFACE RAD_DEFAULT_HOME="${RAD_DEFAULT_HOME}")

if(RAD_HIP_FOUND)
  target_compile_definitions(rad_abi INTERFACE RAD_HAVE_HIP=1)
  # core/device/hip.cpp is compiled by the ORDINARY C++ compiler -- only plugins go through
  # hipcc -- so it needs the HIP headers on the include path explicitly. hipcc supplies them to
  # itself, which is why this is invisible until the first non-plugin translation unit includes
  # hip_runtime.h.
  #
  # SYSTEM, AND NOT AS A MATTER OF TASTE. hipcc has this same directory on its own path as a
  # SYSTEM include, where a warning inside it is suppressed. Naming it again as an ordinary -I
  # puts the user-include copy first and the headers stop being system headers -- for every
  # plugin, because this is an INTERFACE property of rad_abi and every plugin links it. One ROCm
  # header alone (amd_device_functions.h: `__fns`, `__fns64`, unused) then warns in every
  # translation unit that includes it, which is a large share of the build output and none of it
  # ours. A build whose warnings are mostly somebody else's is a build where nobody reads the
  # warnings.
  #
  # BUILD_INTERFACE only: the path is this machine's, and an exported target carrying it fails to
  # configure wherever ROCm sits elsewhere. The installed package config finds the headers again on
  # the consumer's machine (cmake/radiance-config.cmake.in).
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

# THE RECORDED KERNEL FIXTURE, COPIED INTO THE BUILD'S HOME.
#
# data/kernels.rkb is checked in and is what makes `rad-kbench` runnable with no model present: it
# holds the geometries, the seeds the inputs are drawn from and libref's answers reduced to a norm
# and a sample. Copied rather than pointed at, because $RADIANCE_HOME is what the tool looks in and
# an installed prefix has no source tree beside it. Recording a new one is
#
#   rad-kbench -m <model.rad> --max-cases 1 --record data/kernels.rkb
#
# and that is the only step that needs a 27B container on the machine.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/kernels.rkb)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/kernels.rkb
                 ${RAD_PLUGIN_OUT}/kernels.rkb COPYONLY)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/kernels.rkb
          DESTINATION ${CMAKE_INSTALL_DATADIR}/radiance)
endif()


# THE SECOND FIXTURE, AND WHY THERE ARE TWO. A recording is of ONE container -- it holds that
# model's declared geometries -- and no single container in scope declares the whole vocabulary.
# The 27B is dense, has a DFlash2 drafter and gqa 6; the 35B-A3B is routed (moe_gemm_q,
# router_topk, moe_scatter, moe_gather, scale_rows), has an MTP head and gqa 8. Recording only one
# leaves five ops and a whole attention dispatch with no replay coverage at all, so a kernel that
# no locally available model exercises would regress unseen. Recording a new one is
#
#   rad-kbench -m <model.rad> --max-cases 1 --record data/kernels_moe.rkb
#
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/kernels_moe.rkb)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/kernels_moe.rkb
                 ${RAD_PLUGIN_OUT}/kernels_moe.rkb COPYONLY)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/kernels_moe.rkb
          DESTINATION ${CMAKE_INSTALL_DATADIR}/radiance)
endif()
# rad_add_plugin() and rad_hipify() live in cmake/RadAddPlugin.cmake because the installed package
# config includes that same file: a third party building a plugin from their own repository needs
# the function, and a second copy of it in the package config would drift from this one. The file
# documents the variables it expects; every one of them is set above.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RadAddPlugin.cmake)

if(RAD_HIP_FOUND)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" RAD_CFG_UPPER)
  # NOT `set(CMAKE_CXX_COMPILER hipcc)`. The C++ compiler is fixed by project() and assigning it
  # afterwards is silently ignored -- device sources then reach the host C++ compiler with
  # --offload-arch, an error that reads like a bad flag rather than a bad compiler.
  #
  # CMake has had a HIP language since 3.21. Device sources are marked LANGUAGE HIP and the rest of
  # the tree stays on the ordinary C++ compiler, which is what we want: only kernels need hipcc.
  # CMake refuses hipcc as CMAKE_HIP_COMPILER by design ("use Clang directly"), so this is ROCm's
  # own clang++. That needs the address-space cast in libr4d/r4d_common.h: ROCm 7.2 / LLVM 22
  # tightened __builtin_amdgcn_global_load_tr_b128_v8i16 to require a global address space, and
  # the builtin fails identically under hipcc and clang++ without it.
  enable_language(HIP)

  # AND CMAKE EMITS NO DEPENDENCY FLAGS FOR HIP, which is the same shape of omission as the two
  # below and by far the most dangerous of the three, because it does not fail -- it silently
  # builds the wrong thing.
  #
  # `CMAKE_DEPFILE_FLAGS_<LANG>` is set by the compiler modules for C and CXX and is EMPTY for HIP.
  # Ninja's HIP rule still declares `depfile = $DEP_FILE` and `deps = gcc`, so it WAITS for a
  # depfile the command was never told to write, and records nothing:
  #
  #     $ ninja -t deps libr4d/CMakeFiles/libr4d.dir/r4d_model_bf16.hip.o
  #     ...hip.o: #deps 0, deps mtime ... (VALID)
  #
  # Zero. No header is a dependency of any .hip object, so editing a header libr4d includes --
  # r4d_emit.h, r4d_common.h, r4d_fp8_wmma.h, or the ABI itself -- recompiles nothing and RELINKS
  # the plugin around stale objects. The library then exports one signature and the engine calls
  # another -- a hook called with more arguments than it takes, which segfaults inside the plugin
  # with a stack that blames the kernel -- and every test after the edit runs objects that predate
  # it.
  #
  # It is a one-line omission with no upper bound on what it can hide, so it is fixed here rather
  # than worked around with `touch`.
  if(NOT CMAKE_DEPFILE_FLAGS_HIP)
    set(CMAKE_DEPFILE_FLAGS_HIP "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
  endif()

  # AND CMAKE DOES NOT SUPPLY A SYSTEM-INCLUDE FLAG FOR HIP EITHER, which is the same shape of
  # omission as the optimisation level below and just as quiet. `CMAKE_INCLUDE_SYSTEM_FLAG_<LANG>`
  # is set by the compiler modules for C and CXX and is EMPTY for HIP, and CMake's fallback when it
  # is empty is a plain `-I` -- so every SYSTEM include directory reaches a .hip translation unit as
  # an ordinary user include and every warning inside it becomes ours. Without this, ONE
  # target_include_directories(... SYSTEM ...) call reaches a CXX source as
  # `-isystem /opt/rocm/include` and a .hip source as `-I/opt/rocm/include` -- so this is not
  # something an author can get right at the call site. It is set once, here, for every HIP target
  # in the tree, vendor/ included.
  set(CMAKE_INCLUDE_SYSTEM_FLAG_HIP "-isystem ")

  # OPTIMISATION LEVEL, SET HERE FOR EVERY CONFIG BECAUSE CMAKE DOES NOT SUPPLY A USABLE ONE FOR
  # HIP. Depending on the CMake version `CMAKE_HIP_FLAGS_<CONFIG>` is either EMPTY -- every device
  # kernel compiles at -O0 with asserts live while the host half gets -O2 -g -DNDEBUG -- or, from
  # CMake 4.4's HIP toolchain on, pre-filled with the host-style '-O2 -g -DNDEBUG'. Nothing in the
  # build says so either way: an absent flag reads as the default, a pre-filled one as a choice.
  #
  # It is the whole performance story. At -O0 decode reads weights at about 7 GB/s against a card
  # that does ~640, and no kernel change can fix a build that does not optimise the kernel. -O3 and
  # not -O2 because these are the kernels; -g is deliberately NOT carried onto the device side,
  # where it costs compile time and inhibits scheduling for a symbol table no profiler here reads.
  #
  # THE ASSIGNMENT IS UNCONDITIONAL. A guard such as `if(NOT CMAKE_HIP_FLAGS_${cfg})` cannot tell a
  # toolchain default from an operator's choice: on a CMake that pre-fills the variable it sees a
  # value and leaves every device kernel at -O2 with debug info, and a build tree whose cache still
  # holds -O3 from an older CMake hides that until its next fresh configure drops it, with no
  # signal in the build output. An operator who genuinely wants other device flags says so BY NAME
  # through RAD_HIP_FLAGS_<CONFIG>. A deliberate override is a thing someone typed; a toolchain
  # default that happens to be non-empty is not, and the two must not be spelled the same way.
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
# An image or video part of a chat request is decoded by libavformat/libavcodec and converted by
# libswscale (core/mm/media.cpp). Without them the engine builds and serves text, and refuses a
# media part by name -- the same tri-state as HIP, because a machine that has the libraries should
# not silently build without them.
set(RAD_FFMPEG_FOUND OFF)
if(NOT RAD_WITH_FFMPEG STREQUAL "OFF")
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(RAD_FFMPEG QUIET IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
  endif()
  if(RAD_FFMPEG_FOUND)
    message(STATUS "radiance: media decoding via libavformat ${RAD_FFMPEG_libavformat_VERSION}")
  elseif(RAD_WITH_FFMPEG STREQUAL "ON")
    message(FATAL_ERROR "RAD_WITH_FFMPEG=ON but libavformat/libavcodec/libavutil/libswscale were not found.")
  else()
    message(STATUS "radiance: no libavformat -- image and video input will be refused")
  endif()
endif()

add_subdirectory(vendor)
add_subdirectory(core)
# THE KERNEL PLUGINS, one directory each, at the root, with no `kernels/` grouping: a plugin's
# kernels, row table, layouts and launch adapter all live in its own directory, so it reads as a
# plugin built on abi/ rather than something the engine assembles. Each links rad_abi and nothing
# else, from one directory, exactly like a plugin written out of tree. That is the acceptance test,
# and this is the layout that states it.
add_subdirectory(libref)
# libquant: the built-in quantisers, the third plugin kind (spec §4.5). Host C++, built everywhere:
# rad-convert drives it and the oracle decodes through it.
add_subdirectory(libquant)
if(RAD_HIP_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libr4d/CMakeLists.txt)
  add_subdirectory(libr4d)
endif()
add_subdirectory(mine)
# libavx: the conventional op vocabulary in hand-written x86 SIMD, runtime-dispatched by CPU
# feature -- the production HOST kernels, built wherever libref is, on x86-64. An op the placement
# puts on the host needs a host kernel that is not the oracle (rad_gate_reference_kernels refuses
# to serve on libref), and a model whose table is too large for the card -- the n-gram
# embedding's gather -- puts one there every step. It is not behind an option for that reason: a
# build without it cannot serve such a model. It also builds standalone (`cmake -S libavx -B
# build-avx`) on a box with no ROCm.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  add_subdirectory(libavx)
endif()
add_subdirectory(arch)
add_subdirectory(tools)
if(RAD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------- the installed package
# Everything below is ADDITIVE: it changes nothing about what gets built or where the build tree
# puts it. It is what makes `find_package(radiance)` work: the ABI headers -- the entire documented
# contract with a plugin -- and the package config are installed, so an out-of-tree plugin compiles
# against an installed prefix rather than a hand-copied abi/. Without it the plugin boundary is
# real inside the tree and imaginary outside it.
#
# (a) the ABI headers. They are self-contained -- they include each other and C standard headers,
# and nothing else -- so the directory installs as a unit with no filtering beyond the
# extension. arch/common/ is installed by arch/CMakeLists.txt, which owns those files.
#
# Under include/radiance/ rather than loose in include/: a prefix of /usr shares that directory with
# every other package, and rad_*.h and arch/ are names someone else can have. The include spelling
# does not change -- the target's include path is the radiance/ directory itself.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/abi/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/radiance
        FILES_MATCHING PATTERN "*.h")

# (b) the ABI target itself, as radiance::abi. An INTERFACE library has no artifact to install, but
# it carries the include path, RAD_DEFAULT_HOME, RAD_HAVE_HIP and the ROCm include directory -- and
# those are exactly the flags a plugin has to be compiled with to be loadable by this engine, so
# exporting the target is what stops an out-of-tree author reconstructing them by reading our
# CMakeLists and getting one of them wrong.
install(TARGETS rad_abi EXPORT radianceTargets)
install(EXPORT radianceTargets
        NAMESPACE radiance::
        FILE radianceTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radiance)

# (c) the package config. The version IS the ABI version, read out of the header rather than written
# down a second time here: the loader refuses any plugin that does not report exactly
# RAD_ABI_VERSION (core/plugin/loader.cpp), so a package version that could drift from the header
# would let find_package(radiance 3) accept an installation whose plugins the consumer cannot load.
# SameMajorVersion, with the ABI as the major, is that same exact-match rule expressed to CMake.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/abi/rad_abi.h" RAD_ABI_VERSION_LINE
     REGEX "^#define[ \t]+RAD_ABI_VERSION[ \t]")
string(REGEX MATCH "[0-9]+" RAD_ABI_VERSION "${RAD_ABI_VERSION_LINE}")
if(NOT RAD_ABI_VERSION)
  message(FATAL_ERROR
    "could not read RAD_ABI_VERSION out of abi/rad_abi.h. That number is the whole "
    "compatibility contract with a plugin; a build that cannot find it must not ship a package "
    "config that claims one.")
endif()

include(CMakePackageConfigHelpers)

# PATH_VARS, so the config resolves these against wherever the package was actually found rather
# than against the prefix it was configured with. A staged install (DESTDIR) or a moved tree would
# otherwise hand a consumer absolute paths into a directory that no longer exists, and the symptom
# is a plugin that compiles and installs into nothing.
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
        # The function itself, not a copy of it: the config includes this exact file, so the rule
        # that decides what a plugin links and where it lands is one file for both worlds.
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RadAddPlugin.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radiance)

# (d) the license and the notices. Apache-2.0 asks that both travel with every copy, and the MIT
# License asks the same of each bundled component's text, so they install beside the binaries and
# anything built from the install (the Docker image) carries them.
install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(FILES vendor/llama/LICENSE
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party RENAME llama.cpp.LICENSE)
install(FILES vendor/nlohmann/LICENSE.MIT
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party RENAME nlohmann-json.LICENSE)
install(FILES vendor/cpp-httplib/LICENSE
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party RENAME cpp-httplib.LICENSE)

# The gdn-query-reload candidate: one kernel plugin, built out of tree against an installed
# radiance exactly as any third party's would be. It needs nothing from this repository's
# CMakeLists: `find_package(radiance)` supplies the ABI target, the plugin shape and the device
# flags, and `rad_add_plugin` decides where the .so lands so the engine's loader finds it.
#
# ============================== BUILD ========================================================
#
#   cmake -S candidates/gdn-query-reload -B /tmp/gdn-query-reload-build \
#         -DCMAKE_PREFIX_PATH=<radiance install prefix> \
#         -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#         -DGQRELOAD_LIBR4D_INCLUDE=<tree>/libr4d
#   cmake --build /tmp/gdn-query-reload-build
#
# THREE THINGS ABOUT THAT LINE ARE NOT INTERCHANGEABLE, and each is read off this SDK rather than
# guessed:
#
# (1) CMAKE_HIP_COMPILER MUST BE ROCm's clang++, NOT hipcc. CMake refuses the hipcc driver as a
#     HIP compiler by design; the tree says so where it enables the language and its own cache
#     records the resolved compiler as a clang++ path. hipcc is still used by the tree, but only as
#     the FEATURE PROBE that decides whether the backend exists (RAD_HIP_FOUND), which is why the
#     SDK carries RAD_HIP_FOUND rather than a compiler path.
#
# (2) CMAKE_PREFIX_PATH MUST BE THE INSTALL PREFIX, NOT THE BUILD DIRECTORY. The build tree
#     contains a radiance-config.cmake, but it computes PACKAGE_PREFIX_DIR from its own location
#     and then set_and_check of include/radiance plus radianceTargets.cmake -- neither exists there.
#     Point at the prefix; find_package appends lib/cmake/radiance itself.
#
# (3) THE libr4d PRIVATE HEADERS ARE NOT INSTALLED. The SDK exports exactly one target,
#     radiance::abi, whose INTERFACE_INCLUDE_DIRECTORIES is <prefix>/include/radiance (abi/*.h and
#     arch/*.h only). No r4d*.h is installed anywhere. This row forks libr4d's arithmetic body and
#     includes r4d.h, r4d_args.h (through r4d_gdn_args.h), r4d_gdn_wmma.h, r4d_common.h,
#     r4d_gdn_conv_tap.h, r4d_q8.h and r4d_emit.h, so that directory is named explicitly.
#     GQRELOAD_LIBR4D_INCLUDE is that name, and the configure refuses by name when it is wrong
#     rather than letting the build fail inside a system include.
#
# WHERE THE .so GOES. rad_add_plugin writes it to ${RAD_PLUGIN_OUT}/kernels, and the SDK sets
# RAD_PLUGIN_OUT to ${CMAKE_BINARY_DIR}/radiance_home when it is not already defined. A -D on this
# project's configure line defines it, so the SDK keeps the given value. For an A/B against the
# frozen libr4d row, point it at the engine's own home so the loader finds it without a
# RADIANCE_HOME override:
#   -DRAD_PLUGIN_OUT=<engine build>/radiance_home
# Without that flag the .so is private to this build directory and nothing serves it, which is the
# right default for a candidate nobody has selected. (The INSTALL destination is a different
# variable: rad_add_plugin installs to RAD_PLUGIN_INSTALL_HOME, which the SDK points at the found
# installation's $RADIANCE_HOME.)
#
# THIS SOURCE IS WRITTEN FOR RAD_ABI_VERSION 26 (radiance 1.10.0). ABI 26 has no RadDescribeFn and
# RadKernelInfo has no `describe` field; the row therefore publishes one launch entry and relies on
# the engine capturing real dispatches. There is nothing here to adapt by changing a version macro.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(gdn_query_reload LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  if(NOT RAD_HIP_FOUND)
    # A host-only radiance cannot compile this file at all. rad_add_plugin does warn and then drop
    # the HIP source, which here would leave a plugin with zero kernels behind it: a .so that loads,
    # exports the six ABI entry points, and answers for nothing. Refuse at configure instead, naming
    # the fact that is missing.
    message(FATAL_ERROR
      "gdn_query_reload: the radiance found at ${RADIANCE_HOME} has no device build "
      "(RAD_HIP_FOUND is OFF), so this row cannot be compiled. Configure against a radiance built "
      "with ROCm and a named architecture (-DRAD_GPU_TARGETS=gfx1201).")
  endif()
  enable_language(HIP)
endif()

# The libr4d headers this unit includes, and where they come from. No install rule publishes them
# (see (3) above), so the path is a parameter with a same-tree guess and a refusal by name when it
# does not hold. r4d_emit.h is the file checked: it is the one header here that is not also a
# kernel's own, so its absence proves the directory is not the libr4d source directory rather than
# merely being an older one.
set(GQRELOAD_LIBR4D_INCLUDE "" CACHE PATH
    "libr4d source directory holding r4d.h / r4d_emit.h / r4d_gdn_wmma.h (not installed by radiance)")
if(NOT GQRELOAD_LIBR4D_INCLUDE)
  # The usual case: this candidate lives in <tree>/candidates/<name>, beside <tree>/libr4d.
  get_filename_component(_gqreload_guess
    "${CMAKE_CURRENT_SOURCE_DIR}/../../libr4d" ABSOLUTE)
  if(EXISTS "${_gqreload_guess}/r4d_emit.h")
    set(GQRELOAD_LIBR4D_INCLUDE "${_gqreload_guess}" CACHE PATH
        "libr4d source directory holding r4d.h / r4d_emit.h / r4d_gdn_wmma.h" FORCE)
  endif()
  unset(_gqreload_guess)
endif()
if(NOT EXISTS "${GQRELOAD_LIBR4D_INCLUDE}/r4d_emit.h" OR
   NOT EXISTS "${GQRELOAD_LIBR4D_INCLUDE}/r4d_common.h")
  message(FATAL_ERROR
    "gdn_query_reload: GQRELOAD_LIBR4D_INCLUDE='${GQRELOAD_LIBR4D_INCLUDE}' does not hold r4d.h / "
    "r4d_emit.h / r4d_common.h. This row forks libr4d's arithmetic body and includes its private "
    "headers, which radiance does not install, so name the libr4d source directory: "
    "-DGQRELOAD_LIBR4D_INCLUDE=<tree>/libr4d.")
endif()

# The row is written for one chip and refuses anything else at selection; narrowing here means a
# build on a machine whose radiance was configured for other architectures says so and skips,
# rather than compiling code objects that the kernel's wave32 assumptions do not hold on.
rad_plugin_gpu_targets(GQRELOAD_GPU_TARGETS "^gfx1201$")
if(NOT GQRELOAD_GPU_TARGETS)
  message(STATUS
    "gdn_query_reload: gfx1201 only, and this radiance targets ${RAD_GPU_TARGETS} -- not built")
  return()
endif()

# A candidate must never install itself into a real installation: rad_add_plugin would otherwise
# put this .so in the found radiance's own $RADIANCE_HOME/kernels on `cmake --install`, where it
# would be loaded by every later run. RAD_PLUGIN_INSTALL is rad_add_plugin's opt-out.
set(RAD_PLUGIN_INSTALL OFF)

rad_add_plugin(gdn_query_reload kernels
  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/gdn_query_reload.hip
  INCLUDES    ${GQRELOAD_LIBR4D_INCLUDE}
  GPU_TARGETS ${GQRELOAD_GPU_TARGETS})

# libr4d is built at C++17 and so is this: the body it forks reaches
# __builtin_amdgcn_global_load_tr_b128_v8i16 through r4d_common.h, where C++20 tightens the
# address-space check, and C++17 is the configuration every cost stated in the source was taken
# in. rad_plugin_defaults() would otherwise take the language to C++20.
set_target_properties(gdn_query_reload PROPERTIES
  HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# `-ffp-contract=off` is a CORRECTNESS flag here, not a tuning one: the query/key LDS reload is
# bit-identical to the in-place scale only while no multiply fuses into the add after it.
# rad_hipify already sets it for HIP sources; this states it for the target, and -fno-fast-math
# keeps a future toolchain default from weakening the shared-norm argument.
target_compile_options(gdn_query_reload PRIVATE -ffp-contract=off -fno-fast-math)

# The six ABI exports and nothing else. The version script is what makes the visible set a stated
# fact rather than a property of which helpers happen to be non-static; rad_add_plugin's MODULE
# target exports every default-visibility symbol otherwise. This is also what keeps this DSO's
# private copies of r4d_dispatch_log / r4d_gdn_dims from entering the dynamic table beside
# libr4d's.
set(GQRELOAD_EXPORTS ${CMAKE_CURRENT_BINARY_DIR}/exports.map)
file(WRITE ${GQRELOAD_EXPORTS}
"{
  global:
    rad_plugin_abi_version;
    rad_plugin_info;
    rad_kernel_schema_count;
    rad_kernel_schema_at;
    rad_kernel_count;
    rad_kernel_at;
  local: *;
};
")
target_link_options(gdn_query_reload PRIVATE
  "-Wl,--version-script=${GQRELOAD_EXPORTS}" -Wl,--gc-sections)
# The unit pulls in device templates it does not instantiate; discard them.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/gdn_query_reload.hip PROPERTIES
  COMPILE_OPTIONS "-ffunction-sections;-fdata-sections")

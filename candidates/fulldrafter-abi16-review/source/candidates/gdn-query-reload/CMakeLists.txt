# The gdn-query-reload candidate: one kernel plugin, built out of tree against an installed
# radiance exactly as any third party's would be. The whole point of the layout is that it needs
# nothing from this repository's CMakeLists: `find_package(radiance)` supplies the ABI target, the
# plugin shape and the device flags, and `rad_add_plugin` decides where the .so lands so the
# engine's loader finds it.
#
# ============================== BUILD (verified against the SDK as installed) ==============
#
#   cmake -S candidates/gdn-query-reload -B /tmp/gdn-query-reload-build \
#         -DCMAKE_PREFIX_PATH=/home/orca/FullDrafter/worktrees/dloop_dartree/build_hip/oot/prefix \
#         -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#         -DGQRELOAD_LIBR4D_INCLUDE=/home/orca/FullDrafter/worktrees/dloop_dartree/libr4d
#   cmake --build /tmp/gdn-query-reload-build
#
# THREE THINGS ABOUT THAT LINE ARE NOT INTERCHANGEABLE, and each was read off the SDK rather than
# guessed:
#
# (1) CMAKE_HIP_COMPILER MUST BE ROCm's clang++, NOT hipcc. CMake refuses the hipcc driver as a
#     HIP compiler by design ("use Clang directly") -- the tree says so where it enables the
#     language (CMakeLists.txt: "CMake refuses hipcc as CMAKE_HIP_COMPILER by design") and the
#     tree's own cache records the resolved compiler as
#     CMAKE_HIP_COMPILER:UNINITIALIZED=/opt/rocm/llvm/bin/clang++; that path is a symlink to
#     llvm/bin/clang, "AMD clang version 23.0.0git". hipcc is still used by the tree, but only as
#     the FEATURE PROBE that decides whether the backend exists at all (find_program(RAD_HIPCC
#     ...)), which is why the SDK's radiance-config.cmake carries RAD_HIP_FOUND rather than a
#     compiler path.
#
# (2) CMAKE_PREFIX_PATH MUST BE THE INSTALL PREFIX, NOT THE BUILD DIRECTORY. There is a
#     radiance-config.cmake in the build directory, and pointing find_package at it fails:
#     it computes PACKAGE_PREFIX_DIR from its own location and then
#     `set_and_check(RADIANCE_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include/radiance")` and
#     `include("${CMAKE_CURRENT_LIST_DIR}/radianceTargets.cmake")`. In the build tree neither the
#     include/radiance directory nor radianceTargets.cmake exists next to it -- the export file
#     lives under CMakeFiles/Export/... and the headers under abi/. The complete package is the
#     installed one: <prefix>/lib/cmake/radiance/{radiance-config,radiance-config-version,
#     radianceTargets,RadAddPlugin}.cmake, and the prefix this tree has installed to is
#     build_hip/oot/prefix (install_manifest.txt). Point at the prefix, not at lib/cmake/radiance:
#     find_package's default search appends the package directory itself.
#
# (3) THE libr4d PRIVATE HEADERS ARE NOT INSTALLED, SO THEY ARE NOT EXPORTED, AND THIS SOURCE
#     INCLUDES SEVEN OF THEM. The SDK's export carries exactly one target, radiance::abi, with
#     INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/radiance" and compile definitions
#     RAD_DEFAULT_HOME / RAD_HAVE_HIP / __HIP_PLATFORM_AMD__; there is no second target and no
#     libr4d include directory in it. The installed include tree holds abi/*.h and arch/*.h only
#     (the only header install rule is `install(DIRECTORY abi/ ... )`), and no r4d*.h is installed
#     anywhere under the prefix. This row forks the arithmetic body from libr4d and includes
#     r4d.h, r4d_args.h (through r4d_gdn_args.h), r4d_gdn_wmma.h, r4d_common.h,
#     r4d_gdn_conv_tap.h, r4d_q8.h and r4d_emit.h, so it needs that directory named explicitly.
#     GQRELOAD_LIBR4D_INCLUDE is that name, and the configure refuses by name when it is wrong
#     rather than letting the build fail somewhere inside a system include.
#
# WHERE THE .so GOES. rad_add_plugin writes it to ${RAD_PLUGIN_OUT}/kernels, and the SDK sets
# RAD_PLUGIN_OUT to ${CMAKE_BINARY_DIR}/radiance_home when it is not already defined
# (radiance-config.cmake, `if(NOT DEFINED RAD_PLUGIN_OUT)`). A -D on this project's configure line
# defines it, so the SDK keeps the given value and the plugin lands there instead. For an A/B
# against the frozen row the parent wants it in the engine's OWN build home:
#   -DRAD_PLUGIN_OUT=/home/orca/FullDrafter/worktrees/dloop_dartree/build_hip/radiance_home
# which puts it beside libr4d.so and mine.so, where RADIANCE_KERNELS=mine,gdn_query_reload finds it
# without a RADIANCE_HOME override. Without that flag the .so is private to this build directory
# and nothing serves it, which is the right default for a candidate nobody has selected. (The
# INSTALL destination is a different variable: rad_add_plugin installs to
# RAD_PLUGIN_INSTALL_HOME, which the SDK points at the found installation's $RADIANCE_HOME.)
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
# target exports every default-visibility symbol otherwise.
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

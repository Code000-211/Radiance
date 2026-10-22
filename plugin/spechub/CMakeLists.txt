# spechub -- the linear-MTP coupling kernels.
#
# A kernel plugin and nothing else: abi/ is the whole contract, so this builds in radiance's tree or
# standalone against an installed prefix
# (`cmake -S spechub -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`; see cmake/RadAddPlugin.cmake).
#
# It declares TWO ops, `mtp_couple_propose` and `mtp_couple_verify`, with one device row and one host
# reference row each. The coupling's arithmetic is NOT here: abi/rad_spechub.h defines it inline, host
# and device, so the reference, the kernel above it and the engine's own checks are one definition.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(spechub LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

set(SPECHUB_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/spechub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_spechub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exports.cpp)

if(NOT RAD_HIP_FOUND)
  message(STATUS "spechub: this radiance has no device build -- only the host reference is built")
  rad_add_plugin(spechub kernels SOURCES ${SPECHUB_COMMON})
else()
  # The owning project must enable HIP itself; enable_language() is project scope. The top-level tree
  # does it with -DRAD_WITH_HIP=ON, and a standalone build must do the same after project().
  get_property(spechub_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT "HIP" IN_LIST spechub_langs)
    message(FATAL_ERROR "spechub: HIP_SOURCES need `enable_language(HIP)` in this project (set "
                        "-DRAD_WITH_HIP=ON, or enable it after project()).")
  endif()
  rad_add_plugin(spechub kernels SOURCES ${SPECHUB_COMMON}
                  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/spechub.hip)
  target_compile_definitions(spechub PRIVATE RAD_SPECHUB_WITH_DEVICE=1)
endif()

# -ffp-contract=off is a correctness flag here, not a tuning choice: the coupling sums FP64 masses in
# ascending candidate order and compares them against a 53-bit uniform, so an FMA that fused a
# multiply into one of those sums would move the answer off the reference the tests check.
target_compile_options(spechub PRIVATE -ffp-contract=off)

# hc_up_wmma -- the opt-in M=3/4 HC-read row whose up-dot reduction runs on WMMA.
#
# A kernel plugin and nothing else: abi/ is the whole contract, so this builds in radiance's tree or
# standalone against an installed prefix
# (`cmake -S plugin/hc_up_wmma -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`; see
# cmake/RadAddPlugin.cmake, which is also where the HIP compiler has to be clang++ and not hipcc).
#
# It declares NO op of its own: it answers `hc_read`, which libr4d also answers, and it is inert
# until RADIANCE_HC_UP_WMMA=1. With the switch set it wins the op by the default load order -- its
# file sorts before libr4d.so -- so the switch is the whole opt-in; naming it first on
# RADIANCE_KERNELS only pins that order. Everything else about the op -- the operand order, the
# mixing-weight rows, the layouts, the down+inject routine -- is libr4d's, included rather than
# restated (hc_up_wmma.hip).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(hc_up_wmma LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

if(NOT RAD_HIP_FOUND)
  # The schema only. `hc_read`'s device rows are libr4d's, so a card-less tree has no M=3/4 row to
  # override and this library is not useful there; it still builds, so an install script can carry
  # one plugin list for every machine.
  message(STATUS "hc_up_wmma: this radiance has no device build -- the row is not built")
  rad_add_plugin(hc_up_wmma kernels SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/exports.cpp)
else()
  # RDNA4 only, and not as a tuning choice: the down+inject routine this row reuses is libr4d's,
  # and libr4d is gfx12xx. A build for another card has no base row to override.
  rad_plugin_gpu_targets(HCW_GPU_TARGETS "^gfx12")
  if(NOT HCW_GPU_TARGETS)
    message(STATUS "hc_up_wmma: RDNA4 (gfx12xx) only, and this build targets "
                   "${RAD_GPU_TARGETS} -- not built")
    return()
  endif()
  # The owning project must enable HIP itself; enable_language() is project scope.
  get_property(hc_up_wmma_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT "HIP" IN_LIST hc_up_wmma_langs)
    message(FATAL_ERROR "hc_up_wmma: HIP_SOURCES need `enable_language(HIP)` in this project (set "
                        "-DRAD_WITH_HIP=ON, or enable it after project()).")
  endif()
  rad_add_plugin(hc_up_wmma kernels
                 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/exports.cpp
                 HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hc_up_wmma.hip
                 GPU_TARGETS ${HCW_GPU_TARGETS})
  target_compile_definitions(hc_up_wmma PRIVATE RAD_HC_UP_WMMA_WITH_DEVICE=1)
endif()

# -ffp-contract=off, as libr4d builds: the row's arithmetic must round exactly as the base row it
# overrides does, or the two cannot be compared. The WMMA dot itself is unscaled BF16 with the FP32
# scale applied after it, which is what keeps it comparable at all.
target_compile_options(hc_up_wmma PRIVATE -ffp-contract=off)

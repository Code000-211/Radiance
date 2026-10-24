# qsatable -- the shared QSA block table (MAIN-54, prototype).
#
# A kernel plugin and a host oracle for one op, `qsa_table`, and for its read, `qsa_attn`. Same shape as plugin/spechub: abi/ is
# the whole contract, so it builds in radiance's tree or standalone against an installed prefix.
#
# PROTOTYPE, and gated on the architecture's side, not here: arch/common/rad_qsa.h issues the op only
# with RADIANCE_QSA_TABLE=1, and the loader reads no switch. Without a card only the reference row is
# built and a card-less machine still has the definition of the op to check against.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(qsatable LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

set(QSATABLE_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/qsatable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_qsatable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qsaattn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_qsaattn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exports.cpp)

if(NOT RAD_HIP_FOUND)
  message(STATUS "qsatable: this radiance has no device build -- only the host reference is built")
  rad_add_plugin(qsatable kernels SOURCES ${QSATABLE_COMMON})
else()
  get_property(qsatable_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT "HIP" IN_LIST qsatable_langs)
    message(FATAL_ERROR "qsatable: HIP_SOURCES need `enable_language(HIP)` in this project (set "
                        "-DRAD_WITH_HIP=ON, or enable it after project()).")
  endif()
  rad_add_plugin(qsatable kernels SOURCES ${QSATABLE_COMMON}
                  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/qsatable.hip
                              ${CMAKE_CURRENT_SOURCE_DIR}/qsaattn.hip)
  target_compile_definitions(qsatable PRIVATE RAD_QSATABLE_WITH_DEVICE=1)
endif()

# Arithmetic the reference states: the union and the mask are integer bookkeeping, and the masked
# read's sums must contract no differently from libref's. -ffp-contract=off over both rows.
target_compile_options(qsatable PRIVATE -ffp-contract=off)

# exl3import -- an exllamav3 (EXL3) checkpoint into a .rad container before a plugin that serves the model
# exists: the `exl3` quantiser (quant/, imports the EXL3 linears) and the convert-only architecture
# `exl3_import` (arch/, declares every tensor under its checkpoint name). Host code only. README.md says how
# to run a convert.
#
# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S plugin/exl3import -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(exl3import LANGUAGES C CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  set(EXL3_STANDALONE ON)
endif()
enable_language(C)

set(EXL3_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(EXL3_COMMON_C ${EXL3_ROOT}/common/exl3_decode.c ${EXL3_ROOT}/common/exl3_layout.c)
# The decode's roundings are its own: no FMA contraction in the C (as upstream).
set_source_files_properties(${EXL3_COMMON_C} PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
find_library(EXL3_LIBM m)

rad_add_plugin(exl3_quant quantizers
  SOURCES ${EXL3_ROOT}/quant/exl3_quant.cpp ${EXL3_ROOT}/quant/exl3_src.cpp ${EXL3_COMMON_C}
  INCLUDES ${EXL3_ROOT}/include ${EXL3_ROOT}/common ${EXL3_ROOT}/quant)
rad_add_plugin(exl3_import architectures
  SOURCES ${EXL3_ROOT}/arch/exl3_import.cpp ${EXL3_ROOT}/quant/exl3_src.cpp ${EXL3_COMMON_C}
  INCLUDES ${EXL3_ROOT}/include ${EXL3_ROOT}/common ${EXL3_ROOT}/quant)
foreach(t exl3_quant exl3_import)
  set_target_properties(${t} PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
  if(EXL3_LIBM)
    target_link_libraries(${t} PRIVATE ${EXL3_LIBM})
  endif()
endforeach()

# The end-to-end convert (label `host`): a synthetic DeepSeek-shaped EXL3 checkpoint through
# tools/exl3_convert.sh and rad-convert, every imported plane checked against the source. In radiance's tree
# only, where rad-convert is a target (defined after this directory, so named by generator expression); needs
# python3 with numpy, and is SKIPPED without it.
if(NOT EXL3_STANDALONE AND RAD_BUILD_TESTS)
  enable_testing()
  find_program(EXL3_PYTHON NAMES python3 python)
  if(NOT EXL3_PYTHON)
    set(EXL3_PYTHON python3)
  endif()
  add_test(NAME exl3_import_convert
           COMMAND ${EXL3_ROOT}/tests/convert_test.sh ${EXL3_PYTHON} $<TARGET_FILE:rad-convert>
                   ${RAD_PLUGIN_OUT} ${CMAKE_CURRENT_BINARY_DIR}/convert-test)
  set_tests_properties(exl3_import_convert PROPERTIES LABELS host SKIP_RETURN_CODE 77
                       ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
endif()

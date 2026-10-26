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
  # A plugin home of this build's own, put ahead of the engine's on $RADIANCE_HOME; set BEFORE find_package,
  # whose config would otherwise default it to the installed radiance's home.
  set(RAD_PLUGIN_INSTALL_HOME "${CMAKE_BINARY_DIR}/home" CACHE PATH "the \$RADIANCE_HOME this build installs under")
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  set(EXL3_STANDALONE ON)
  set(EXL3_ABI 26)
  if(NOT RADIANCE_ABI_VERSION EQUAL EXL3_ABI)
    message(FATAL_ERROR "exl3import is written against radiance plugin ABI ${EXL3_ABI} (radiance 1.10.0); the "
                        "radiance found has ABI ${RADIANCE_ABI_VERSION}. Point CMAKE_PREFIX_PATH at a matching "
                        "install (RADIANCE_VERSION).")
  endif()
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
# tools/exl3_convert.sh and rad-convert, every imported plane checked against the source. Needs python3 with
# numpy, and is SKIPPED without it. In radiance's tree rad-convert is a target (defined after this directory,
# so named by generator expression) and the tree's home holds every plugin; standalone it is the installed
# radiance's, found beside its home, and the home is this build's ahead of the engine's.
if(EXL3_STANDALONE)
  find_program(EXL3_RAD_CONVERT rad-convert HINTS "${RADIANCE_HOME}/../../bin" NO_CACHE)
  set(EXL3_TEST_CONVERT "${EXL3_RAD_CONVERT}")
  set(EXL3_TEST_HOME "${RAD_PLUGIN_OUT}:${RADIANCE_HOME}")
  set(EXL3_TESTS ${EXL3_RAD_CONVERT})
  if(NOT EXL3_RAD_CONVERT)
    message(STATUS "exl3import: no rad-convert beside ${RADIANCE_HOME}; the convert test is not added")
  endif()
else()
  set(EXL3_TEST_CONVERT "$<TARGET_FILE:rad-convert>")
  set(EXL3_TEST_HOME "${RAD_PLUGIN_OUT}")
  set(EXL3_TESTS ${RAD_BUILD_TESTS})
endif()
if(EXL3_TESTS)
  enable_testing()
  find_program(EXL3_PYTHON NAMES python3 python)
  if(NOT EXL3_PYTHON)
    set(EXL3_PYTHON python3)
  endif()
  add_test(NAME exl3_import_convert
           COMMAND ${EXL3_ROOT}/tests/convert_test.sh ${EXL3_PYTHON} ${EXL3_TEST_CONVERT}
                   ${EXL3_TEST_HOME} ${CMAKE_CURRENT_BINARY_DIR}/convert-test)
  set_tests_properties(exl3_import_convert PROPERTIES LABELS host SKIP_RETURN_CODE 77
                       ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
endif()

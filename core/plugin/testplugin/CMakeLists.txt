# Test plugins built from one source: a hierarchy for plugin_test to load, a schema conflict to
# refuse and a stale ABI to reject. They go under $RADIANCE_HOME/testkernels, not kernels/, because
# the engine scans kernels/ at startup and must never select a row that implements nothing
# (spec.md §17). The two that must be refused live in testkernels/bad, so loading testkernels/ is
# clean and the refusals are requested by path.
rad_add_plugin(radtest_base testkernels SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_base PRIVATE RADTEST_BASE=1)

rad_add_plugin(radtest_over testkernels SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_over PRIVATE RADTEST_OVER=1)

rad_add_plugin(radtest_conflict testkernels/bad SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_conflict PRIVATE RADTEST_CONFLICT=1)

rad_add_plugin(radtest_badabi testkernels/bad SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_badabi PRIVATE RADTEST_BADABI=1)

# Two architecture plugins claiming one architecture id: the first is identified, the second
# refused.
rad_add_plugin(radtest_arch testarch SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_arch PRIVATE RADTEST_ARCH=1)

rad_add_plugin(radtest_arch2 testarch SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_arch2 PRIVATE RADTEST_ARCH2=1)

# Two plugins for a load with no hierarchy; one declares itself a reference implementation.
# Do not rename: load_dir reads in sorted order, so the reference (`..._a`) is staged first, and
# the test checks that the loader still ranks it below every other kernel plugin. The reverse
# naming would pass on alphabetical order alone.
rad_add_plugin(radtest_ord_a testkernels/order SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_ord_a PRIVATE RADTEST_BASE=1 RADTEST_REFERENCE=1
                                                 RADTEST_BASE_NAME=\"radtest_reference\")

rad_add_plugin(radtest_ord_b testkernels/order SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_ord_b PRIVATE RADTEST_BASE=1)

# Rows for rules the base plugin does not reach (a non-interval ranged constraint, a cross-axis
# tune test, a layout hook, a capacity parameter), in their own directory so loading testkernels/
# is unchanged.
rad_add_plugin(radtest_extra testkernels/extra SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_extra PRIVATE RADTEST_EXTRA=1)

# Two tunable axes the loader must refuse, loaded by path, and a library that declines the machine.
rad_add_plugin(radtest_badtune_default testkernels/bad
               SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_badtune_default PRIVATE RADTEST_BADTUNE=1)
rad_add_plugin(radtest_badtune_twice testkernels/bad SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_badtune_twice PRIVATE RADTEST_BADTUNE=2)
rad_add_plugin(radtest_decline testkernels/decline SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_decline PRIVATE RADTEST_DECLINE=1)

# One architecture family at three quantisations, for the (id, quantisation) selection key.
foreach(q plain fp8 w4)
  set(_rad_q "${q}")
  if(q STREQUAL "plain")
    set(_rad_q "")
  endif()
  rad_add_plugin(radtest_archq_${q} testarchq SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
  target_compile_definitions(radtest_archq_${q} PRIVATE RADTEST_ARCHQ=1
                             RADTEST_ARCHQ_NAME="radtest_archq_${q}"
                             RADTEST_ARCHQ_QUANT="${_rad_q}")
endforeach()
# A fourth member, quantised but declining imported checkpoints (rad_arch_serves_imports).
rad_add_plugin(radtest_archq_w8 testarchq SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_archq_w8 PRIVATE RADTEST_ARCHQ=1
                           RADTEST_ARCHQ_NAME="radtest_archq_w8" RADTEST_ARCHQ_QUANT="w8"
                           RADTEST_ARCHQ_NO_IMPORTS=1)

# KV tier transfer test plugin. Unlike the others it computes (byte-row gather and scatter in the
# device domain, on host backend memory), so only tests/mem_test.cpp loads it, by path. It calls
# the device API for stream order, which the test executable exports.
rad_add_plugin(radtest_rows testrows SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tierrows.cc)

# A quantiser with its own scheme, read only by its decode hook. tests/quant_test.cpp loads it by
# path to check that its weights decode through the core's decoder.
rad_add_plugin(radtest_quant testquant SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testquant.cc)

# Deliberately defective kernel libraries, one per defect, for tests/kbench_mutant_test.cpp: a
# replay must fail each of them (or, for kbm_ok, pass it). One source, as above; KBM_MUTANT picks
# the defect. Under testkernels/kbmut, loaded only by name.
set(RAD_KBM_MUTANTS ok scale nan half overrun refuse shape argmax inwrite enc pair
    CACHE INTERNAL "the kbmutant.cc builds, read by tests/CMakeLists.txt")
set(i 0)
foreach(m ${RAD_KBM_MUTANTS})
  rad_add_plugin(kbm_${m} testkernels/kbmut SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kbmutant.cc)
  target_compile_definitions(kbm_${m} PRIVATE KBM_MUTANT=${i} KBM_NAME=\"kbm_${m}\")
  math(EXPR i "${i} + 1")
endforeach()

# An architecture that declares a norm, a projection and an add from a synthetic container, so
# tests/kbench_mutant_test.cpp can record a fixture on the host with libref alone.
rad_add_plugin(radtest_kbarch testarch/kbench SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kbarch.cc)

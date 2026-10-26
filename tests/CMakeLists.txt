# One executable per *_test.cpp, globbed, so adding a test is adding a file.
#
# The tiers are labels (docs/GUIDE.md §1.3 lists them with their times):
#   unit     a harness binary (rad_test.h) on the host backend
#   perf     a binary's perf_ cases, which assert wall-clock costs; RUN_SERIAL
#   slow     minutes: the kernel replays, the device selftest, the long sweeps
#   gpu      needs a card; RESOURCE_LOCK card
#   oot      rebuilds every bundled plugin out of tree against an install
#   r4d_emu  libr4d's kernel sources run on the host by the SIMT emulator
#   fast     every test with none of slow, gpu, oot and perf, set at the end of this file: the
#            inner loop (`ctest --preset fast`, or `ctest -L fast`)
#
# arch_test and glm5next_test do not link rad_core: they supply their own recording fake of the
# declare-phase C ABI (arch_fakes.h), so plugins are tested without depending on core/build/'s
# builder. With the archive linked, a new ABI call missing from the fake would pull in
# rad_builder.cpp.o and collide on every rad_decl_* symbol; without it the error is a single
# undefined reference.
set(RAD_TESTS_NO_CORE arch_test glm5next_test)

# fixture_test tests tools/kfixture.cpp and tools/opshapes.cpp (the seeded draw, the reduction and
# the on-disk recording) directly, kbench_test tools/kbench_core.cpp (rad-kbench's rules), and
# kbench_mutant_test runs rad-kbench itself against known-bad kernel libraries. The tool sources are
# not a library (tools/CMakeLists.txt), so the tests compile them in.
set(RAD_TESTS_WITH_TOOLS fixture_test kbench_test kbench_mutant_test)

# tools_test tests tools/iface.cpp, the command-line plumbing the tools share, which is compiled
# into each tool rather than built as a library.
set(RAD_TESTS_WITH_IFACE tools_test)

# The tokenizer tests compare the encoder against the reference encoder, tools/tokref.
set(RAD_TESTS_WITH_TOKREF tokenizer_test tokenizer_perf_test)

# The GBNF oracle (upstream's candidate walk), which the mask engine must match bit for bit. Test
# code, compiled only into these tests and rad-gbnf-check, never into a serving binary.
set(RAD_TESTS_WITH_GBNF_ORACLE sample_test gbnf_test)

# The legacy reply parser as an oracle for core/text/chatparse.h: a whole-reply path built on
# llama.cpp's PEG parser and a hand-written call lifter. Its own archive, linked only into the
# comparing tests and rad-chatparse; core/ cannot reach it.
add_library(rad_legacy_toolparse STATIC ${CMAKE_CURRENT_SOURCE_DIR}/legacy/toolparse_legacy.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/legacy/chatdiff.cpp)
target_link_libraries(rad_legacy_toolparse PUBLIC rad_core)
target_include_directories(rad_legacy_toolparse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/legacy)
set(RAD_TESTS_WITH_LEGACY chatparse_test chatparse_gemma4_test)

# relayout_parity_test checks each kernel library's relayout of libquant's planes against a
# recorded hash of the stored bytes. libr4d's layout sources are host-only, so they are compiled in,
# with -ffp-contract=off as libr4d uses, so scale conversions round identically.
# r4d_ar_links_test checks libr4d's header-only link reader, so it needs only the include path.
set(RAD_TESTS_WITH_R4D_HOST relayout_parity_test r4d_ar_links_test)

# Chat templates (copied from the model repositories) and media samples, passed by absolute path so
# test binaries can run from anywhere.
set(RAD_CHAT_FIXTURES ${CMAKE_CURRENT_SOURCE_DIR}/data/chat)
set(RAD_MEDIA_FIXTURES ${CMAKE_CURRENT_SOURCE_DIR}/data/media)

# libstdc++ before rad_core on every test's link line. lld extracts archive members to resolve a
# shared library's imports too: ROCm 10.1's libamdhip64 imports std::string::reserve, which g++-14
# emits weak in rad_graph.cpp.o, and that object pulls in dispatch.cpp.o, whose strong device calls
# silently replace the weak fakes sched_test, ref_test and runtime_test rely on.
set(RAD_TEST_CORE stdc++ rad_core)

# harness_test runs harness_fixture's cases through the harness's command line (rad_test.h).
set(RAD_TESTS_HARNESS harness_test)
add_executable(harness_fixture ${CMAKE_CURRENT_SOURCE_DIR}/harness_fixture.cpp)
set_target_properties(harness_fixture PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# Binaries registered as several entries, <bin>_shard<k> running `<bin> --shard k/n`, so their
# cases run in parallel and no entry is the fast tier's long pole.
set(RAD_TEST_SHARDS_gbnf_test 4)

# ------------------------------------------------------------------ the environment of every test
#
# OMP_WAIT_POLICY=passive and a team of 4: libref, libquant and libavx open OpenMP regions over
# small loops, and with libgomp's default wait every thread of a 32-thread team spins at each
# barrier. Beside other processes that costs 10-100x (a 0.2 s binary takes a minute). Four threads
# keep the multi-threaded paths, rows split across a team, under test.
#
# ROCR_VISIBLE_DEVICES=99 hides every card, so a unit test runs on the host backend on every
# machine: the same code everywhere, and nothing touches a card a server is using. The _card
# entries below run the device tests on a card.
#
# RAD_TEST_STRICT=1: a plugin or fixture this build makes is never legitimately absent, so a test
# that cannot find one fails rather than skips. RAD_TEST_TMPDIR puts radtest::tmpdir() in the build
# tree, on disk, where O_DIRECT works.
#
# RAD_TEST_CPU_BUDGET: a unit binary that spends more CPU seconds than this fails (rad_test.h), so
# a slow case or a spinning thread pool is caught when it lands. Sanitised and Debug builds run
# several times slower and have no budget.
set(RAD_TEST_OMP_THREADS 4)
set(RAD_TEST_TIMEOUT 120)
set(RAD_TEST_CPU_BUDGET 10)
set(RAD_TEST_SAN_ENV "")
if(RAD_ASAN)
  list(APPEND RAD_TEST_SAN_ENV "ASAN_OPTIONS=detect_leaks=1" "UBSAN_OPTIONS=print_stacktrace=1")
elseif(RAD_TSAN)
  # libgomp is not instrumented, so its synchronisation would read as races: one thread.
  set(RAD_TEST_OMP_THREADS 1)
  list(APPEND RAD_TEST_SAN_ENV "TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1")
endif()
if(RAD_ASAN OR RAD_TSAN OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(RAD_TEST_TIMEOUT 600)
  set(RAD_TEST_CPU_BUDGET 0)
endif()
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tmp)
set(RAD_TEST_OMP_ENV "OMP_WAIT_POLICY=passive" "OMP_NUM_THREADS=${RAD_TEST_OMP_THREADS}")
set(RAD_TEST_CARD_ENV "RADIANCE_HOME=${CMAKE_BINARY_DIR}/radiance_home" ${RAD_TEST_OMP_ENV}
                      "RAD_TEST_STRICT=1" "RAD_TEST_TMPDIR=${CMAKE_BINARY_DIR}/tmp"
                      ${RAD_TEST_SAN_ENV})
set(RAD_TEST_ENV ${RAD_TEST_CARD_ENV} "ROCR_VISIBLE_DEVICES=99")

# Binaries that need more CPU than the default budget, and what costs it. Lower an entry when its
# cost comes down, so the tripwire stays tight.
set(RAD_TEST_CPU_BUDGET_quant_ggml_test 25)  # llama.cpp's iq1_s and iq2_* grid searches, ~13 s
set(RAD_TEST_CPU_BUDGET_tokenizer_test 20)  # the reference encoder over every corpus: ~3 s built by GCC 16, ~10.5 s by GCC 14 (the image's Ubuntu)

# Binaries whose OpenMP team is busy for most of their run, so a parallel ctest counts them as a
# team's worth of slots rather than one.
set(RAD_TESTS_THREADED quant_ggml_test relayout_parity_test)

# A harness binary as a ctest entry, `<bin> -perf_ ARGS...`, labelled unit. Its perf_ cases run in
# <bin>_perf instead (below).
function(rad_unit_test name bin)
  set(budget ${RAD_TEST_CPU_BUDGET})
  if(budget AND DEFINED RAD_TEST_CPU_BUDGET_${bin})
    set(budget ${RAD_TEST_CPU_BUDGET_${bin}})
  endif()
  add_test(NAME ${name} COMMAND ${bin} -perf_ ${ARGN})
  # 77 is the harness's "nothing was checked" (rad_test.h).
  set_tests_properties(${name} PROPERTIES
    LABELS unit
    SKIP_RETURN_CODE 77
    TIMEOUT ${RAD_TEST_TIMEOUT}
    ENVIRONMENT "${RAD_TEST_ENV};RAD_TEST_CPU_BUDGET=${budget}")
  if(bin IN_LIST RAD_TESTS_THREADED)
    set_tests_properties(${name} PROPERTIES PROCESSORS ${RAD_TEST_OMP_THREADS})
  endif()
endfunction()

# The script CTestTestfile.cmake includes to register the perf_ entries (below).
set(RAD_PERF_SCRIPT "set(RAD_TEST_ENV [==[${RAD_TEST_ENV}]==])\nset(RAD_TEST_TIMEOUT ${RAD_TEST_TIMEOUT})\n")
string(APPEND RAD_PERF_SCRIPT "include([==[${CMAKE_CURRENT_SOURCE_DIR}/perf_entries.cmake]==])\n")

# shipped_plugins_test loads the shipped architecture plugins, which bind the plugin-facing C ABI
# back to the host at dlopen, so like the engine it carries all of rad_core and exports it.
set(RAD_TESTS_WHOLE_CORE shipped_plugins_test)

file(GLOB RAD_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
foreach(t ${RAD_TESTS})
  get_filename_component(n ${t} NAME_WE)
  if(n IN_LIST RAD_TESTS_NO_CORE)
    add_executable(${n} ${t} ${CMAKE_SOURCE_DIR}/core/util/rad_util.cpp)
  elseif(n IN_LIST RAD_TESTS_HARNESS)
    add_executable(${n} ${t})
    add_dependencies(${n} harness_fixture)
    target_compile_definitions(${n} PRIVATE RAD_HARNESS_FIXTURE="$<TARGET_FILE:harness_fixture>")
  elseif(n IN_LIST RAD_TESTS_WITH_TOKREF)
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE rad_tokref ${RAD_TEST_CORE})
  elseif(n IN_LIST RAD_TESTS_WITH_TOOLS)
    add_executable(${n} ${t} ${CMAKE_SOURCE_DIR}/tools/kfixture.cpp
                            ${CMAKE_SOURCE_DIR}/tools/opshapes.cpp
                            ${CMAKE_SOURCE_DIR}/tools/kbench_core.cpp)
    target_link_libraries(${n} PRIVATE ${RAD_TEST_CORE} ${CMAKE_DL_LIBS})
    target_include_directories(${n} PRIVATE ${CMAKE_SOURCE_DIR}/tools)
  elseif(n IN_LIST RAD_TESTS_WITH_IFACE)
    add_executable(${n} ${t} ${CMAKE_SOURCE_DIR}/tools/iface.cpp)
    target_link_libraries(${n} PRIVATE ${RAD_TEST_CORE})
    target_include_directories(${n} PRIVATE ${CMAKE_SOURCE_DIR}/tools)
  elseif(n IN_LIST RAD_TESTS_WITH_R4D_HOST)
    add_executable(${n} ${t} ${CMAKE_SOURCE_DIR}/libr4d/r4d_layout.cpp
                            ${CMAKE_SOURCE_DIR}/libr4d/r4d_fp8_layout.cpp)
    target_link_libraries(${n} PRIVATE ${RAD_TEST_CORE} ${CMAKE_DL_LIBS})
    target_include_directories(${n} PRIVATE ${CMAKE_SOURCE_DIR}/libr4d)
    target_compile_options(${n} PRIVATE -ffp-contract=off)
  elseif(n IN_LIST RAD_TESTS_WHOLE_CORE)
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE stdc++ "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
    set_target_properties(${n} PROPERTIES ENABLE_EXPORTS ON)
  elseif(n IN_LIST RAD_TESTS_WITH_GBNF_ORACLE)
    add_executable(${n} ${t} ${CMAKE_CURRENT_SOURCE_DIR}/gbnf_oracle.cpp)
    target_link_libraries(${n} PRIVATE ${RAD_TEST_CORE})
    # Chat templates whose tool-call grammars are checked against the oracle (checked in; the
    # tokenizers are too large).
    target_compile_definitions(${n} PRIVATE
      RAD_GBNF_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data/gbnf")
  else()
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE ${RAD_TEST_CORE})
  endif()
  if(n IN_LIST RAD_TESTS_WITH_LEGACY)
    target_link_libraries(${n} PRIVATE rad_legacy_toolparse)
  endif()
  target_compile_definitions(${n} PRIVATE RAD_CHAT_FIXTURES="${RAD_CHAT_FIXTURES}"
                                          RAD_MEDIA_FIXTURES="${RAD_MEDIA_FIXTURES}")
  # Normally inherited from rad_core; tests that do not link it still need the ABI headers.
  target_include_directories(${n} PRIVATE ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
                                          ${CMAKE_BINARY_DIR}/rad_include)
  set_target_properties(${n} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # mem_test loads a kernel plugin that calls back into the device API, so it exports it like the
  # engine binary.
  if(n STREQUAL "mem_test")
    set_target_properties(${n} PROPERTIES ENABLE_EXPORTS ON)
  endif()
  if(DEFINED RAD_TEST_SHARDS_${n})
    math(EXPR last "${RAD_TEST_SHARDS_${n}} - 1")
    foreach(k RANGE ${last})
      rad_unit_test(${n}_shard${k} ${n} --shard ${k}/${RAD_TEST_SHARDS_${n}})
    endforeach()
  else()
    rad_unit_test(${n} ${n})
  endif()
  string(APPEND RAD_PERF_SCRIPT
         "rad_perf_entry(${n} [==[${t}]==] [==[${CMAKE_BINARY_DIR}/tests/${n}]==])\n")
endforeach()

# ================================================================== perf_ cases
#
# A case named perf_... asserts a wall-clock cost, which tests running beside it would distort. The
# unit entries leave those cases out, and <bin>_perf runs them alone (RUN_SERIAL, label perf).
# Which binaries have one is read from their sources whenever ctest loads the tests
# (perf_entries.cmake), so a new perf_ case is registered without reconfiguring.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rad_perf_tests.cmake "${RAD_PERF_SCRIPT}")
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
             ${CMAKE_CURRENT_BINARY_DIR}/rad_perf_tests.cmake)

# ================================================================== the device tests on a card
#
# The unit entries run on the host backend. These run the tests whose code differs on a card (the
# AQL backend, the driver's elastic commit, the mover's copies) on the card, one at a time: a test
# that commits device memory fails beside another holding it, or beside a server.
if(RAD_HIP_FOUND)
  foreach(n device_test mem_test place_test runtime_test)
    add_test(NAME ${n}_card COMMAND ${n} -perf_)
    set_tests_properties(${n}_card PROPERTIES
      LABELS gpu
      SKIP_RETURN_CODE 77
      TIMEOUT ${RAD_TEST_TIMEOUT}
      ENVIRONMENT "${RAD_TEST_CARD_ENV}")
  endforeach()
endif()

# spechub_wiring_test dlopens the built spechub plugin and checks how its rows are classified: the
# device rows must be production kernels and the host oracle rows must stay reference. It needs the
# library path and the same RAD_SPECHUB_WITH_DEVICE answer the plugin was built with.
if(TARGET spechub)
  target_link_libraries(spechub_wiring_test PRIVATE ${CMAKE_DL_LIBS})
  target_compile_definitions(spechub_wiring_test PRIVATE
    RAD_SPECHUB_LIB="${RAD_PLUGIN_OUT}/kernels/spechub.so")
  if(RAD_HIP_FOUND)
    target_compile_definitions(spechub_wiring_test PRIVATE RAD_SPECHUB_WITH_DEVICE=1)
  endif()
  add_dependencies(spechub_wiring_test spechub)
  # sample_test's coupling case declares a Sampler with RADIANCE_MTP_COUPLED and needs the verify
  # op the spechub plugin carries (its host row); it skips when the library is absent.
  add_dependencies(sample_test spechub)
endif()

# ================================================================== whole models on the CPU
#
# engine_e2e_test runs the radiance binary on tiny synthetic checkpoints (tests/tiny_model.h) on
# the host backend; convert_test runs rad-convert on the same checkpoints. Both start the built
# binaries as children, so they depend on them and on every plugin the checkpoints select
# (llama_fp8 for convert_test's --set of an fp8 quantization_config).
set(RAD_E2E_DEPS radiance rad-convert libref libquant qwen35_bf16 lfm2_bf16 gemma4 llama_dense_fp8
                 llama_fp8 qwenimage21)
if(TARGET libavx)
  list(APPEND RAD_E2E_DEPS libavx)
endif()
foreach(t engine_e2e_test convert_test diagnose_e2e_test)
  if(TARGET ${t})
    target_compile_definitions(${t} PRIVATE
      RAD_E2E_SERVER="$<TARGET_FILE:radiance>" RAD_E2E_CONVERT="$<TARGET_FILE:rad-convert>"
      RAD_E2E_HOME="${RAD_PLUGIN_OUT}")
    add_dependencies(${t} ${RAD_E2E_DEPS})
  endif()
endforeach()
# 6 to 17 s by build type and load, mostly libref's GEMMs and decode steps and the engine's own
# startups, so labelled `e2e` and kept out of the fast tier; the host and full tiers run it.
if(TARGET engine_e2e_test)
  set_tests_properties(engine_e2e_test PROPERTIES LABELS "e2e" TIMEOUT 300)
endif()
# diagnose_e2e_test runs the engine's self-measurement over its whole fixed workload, once per case:
# ten runs of the workload, most of them seconds apart in wall time because the workload is the
# cost. Labelled `e2e` with engine_e2e_test and kept out of the fast tier.
if(TARGET diagnose_e2e_test)
  set_tests_properties(diagnose_e2e_test PROPERTIES LABELS "e2e" TIMEOUT 300)
endif()

# abi_layout_test compares the ABI's struct layouts with a C translation unit's: abi_c_test.c
# includes every abi/ header as strict C11, so a header only C++ accepts fails the build.
if(TARGET abi_layout_test)
  target_sources(abi_layout_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/abi_c_test.c)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/abi_c_test.c PROPERTIES
    COMPILE_OPTIONS "-std=c11;-pedantic-errors;-Wall;-Wextra")
endif()

# arch_test reads libref's vocabulary from $RADIANCE_HOME/kernels to check that each architecture
# declares on it, and glm5next_test the loaded libraries' op schemas, so libref is built before
# them.
foreach(t arch_test glm5next_test)
  if(TARGET libref AND TARGET ${t})
    add_dependencies(${t} libref)
  endif()
endforeach()

# kbench_mutant_test runs rad-kbench against the deliberately defective kernel libraries
# (core/plugin/testplugin/kbmutant.cc) on the host backend, so it needs the tool, libref and the
# mutants built.
if(TARGET kbench_mutant_test AND TARGET rad-kbench AND TARGET libref)
  target_compile_definitions(kbench_mutant_test PRIVATE
    RAD_KBENCH_BIN="$<TARGET_FILE:rad-kbench>" RAD_PLUGIN_DIR="${RAD_PLUGIN_OUT}")
  # It also loads a mutant in process, whose device rows call back into the device API.
  set_target_properties(kbench_mutant_test PROPERTIES ENABLE_EXPORTS ON)
  add_dependencies(kbench_mutant_test rad-kbench libref radtest_kbarch)
  foreach(m ${RAD_KBM_MUTANTS})
    add_dependencies(kbench_mutant_test kbm_${m})
  endforeach()
  # One case replays libavx's sampler stages against libref; it is reported missing without it.
  if(TARGET libavx)
    add_dependencies(kbench_mutant_test libavx)
  endif()
endif()

# ================================================================== the device selftest
#
# r4d_selftest is the only check of libr4d's fusion claims: rows declaring a fused kernel
# byte-identical to the ops it replaces, verified by memcmp against the unfused kernels on the same
# inputs. It needs a card and forks a child per row (about six minutes), so it is labelled `slow`;
# the default `ctest` still runs it.
if(RAD_HIP_FOUND AND TARGET r4d_selftest)
  add_test(NAME r4d_selftest
           COMMAND r4d_selftest ${RAD_PLUGIN_OUT}/kernels/libr4d.so)
  set_tests_properties(r4d_selftest PROPERTIES
    LABELS "slow;gpu"
    TIMEOUT 900
    ENVIRONMENT "RADIANCE_HOME=${CMAKE_BINARY_DIR}/radiance_home")
endif()

# ================================================================== the device layer without SVM
#
# Without the KFD SVM API (e.g. a KVM guest whose amdgpu registered no device memory with HMM), ROCr
# cannot lock host memory in place, and the backend allocates pinned ranges from the host pool
# instead (core/device/aql.cpp, "pinned host memory, in pieces"). HSA_USE_SVM=0 forces that path, so
# device_test runs a second time on the card under it.
if(RAD_HIP_FOUND AND TARGET device_test)
  add_test(NAME device_test_nosvm COMMAND device_test -perf_)
  set_tests_properties(device_test_nosvm PROPERTIES
    LABELS gpu
    SKIP_RETURN_CODE 77
    TIMEOUT ${RAD_TEST_TIMEOUT}
    ENVIRONMENT "${RAD_TEST_CARD_ENV};HSA_USE_SVM=0")
endif()

# ================================================================== the host kernels
#
# rad-avx-check compares libavx with libref op by op at every ISA level the machine supports, and
# runs embed_lookup_q over a file-backed table with two ranks racing the read-ahead. The table is
# written beside the binary (filesystem reports a direct-I/O alignment) and in TMPDIR (a tmpfs
# reports none, so reads align by block size, as on btrfs).
if(TARGET libavx AND TARGET rad-avx-check)
  add_test(NAME avx_check
           COMMAND rad-avx-check --ref ${RAD_PLUGIN_OUT}/kernels/libref.so
                                 --avx ${RAD_PLUGIN_OUT}/kernels/libavx.so)
  set(env ${RAD_TEST_OMP_ENV} "ROCR_VISIBLE_DEVICES=99" ${RAD_TEST_SAN_ENV})
  set_tests_properties(avx_check PROPERTIES TIMEOUT ${RAD_TEST_TIMEOUT} ENVIRONMENT "${env}")
endif()

# The same checker on another host library (avx_subset_plugin.cpp: part of libref under another
# name): the ops it lacks are listed rather than failed, cases go to the row whose constraints admit
# them, and a declining row still fails.
if(TARGET rad-avx-check AND TARGET libref)
  add_library(avx_subset_plugin MODULE avx_subset_plugin.cpp)
  target_link_libraries(avx_subset_plugin PRIVATE radiance::abi ${CMAKE_DL_LIBS})
  add_test(NAME avx_check_general
           COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/avx_check_general.sh
                   $<TARGET_FILE:rad-avx-check> ${RAD_PLUGIN_OUT}/kernels/libref.so
                   $<TARGET_FILE:avx_subset_plugin>)
  set(env ${RAD_TEST_OMP_ENV} "ROCR_VISIBLE_DEVICES=99" ${RAD_TEST_SAN_ENV})
  set_tests_properties(avx_check_general PROPERTIES TIMEOUT ${RAD_TEST_TIMEOUT} ENVIRONMENT "${env}")
endif()

# ================================================================== libr4d's kernels, on the host
#
# Some libr4d ops (the sandwich-norm joins, expert_scale) have no libref oracle by design
# (docs/OPS.md), so rad-kbench never checks them. r4d_emu runs the source of thirteen units
# written against r4d_launch.h on the host: the windowed paged attention, pair_argmax, the joins,
# expert_scale, the QSA block selection, the runtime-LoRA pair, the F32 routing projection's tile
# forms, GLM's mHC read with its chunk split, the F32-scale E4M3 GEMM's forms, GLM's KDA
# recurrence, the MXFP4 expert GEMM's forms, the MLA matrix row's pairings and GLM's MoE decode
# fusions, whose kernels a card-less build otherwise never runs. emu/hip/hip_runtime.h supplies the
# HIP names, each GPU thread is a fiber, and __syncthreads and cross-lane ops are barriers (one not
# reached uniformly is reported as a deadlock). Checked against ports of libref's ref_attn_paged and
# ref_qsa_select, the host arithmetic in ref_host.cpp, for the joins the unfused chain bit for bit,
# for the LoRA pair libref itself when the test can load it, and for KDA libref itself.
#
# Requires clang (ext_vector_type) and x86-64 (SysV fiber switch in inline assembly). With a clang
# C++ compiler the .hip files build here as C++ (`-x c++`; the property is per directory, so
# libr4d's HIP build is unaffected). Otherwise a clang++ builds them through custom commands: the
# HIP compiler (ROCm's clang++) when there is one, else RAD_EMU_CLANG, the clang++ on PATH. A
# sanitised build also takes that route, since the emulator's fiber switches would confuse the
# sanitiser. With no clang at all the tests are not registered.
set(R4D_DIR ${CMAKE_SOURCE_DIR}/libr4d)
set(R4D_EMU ${CMAKE_CURRENT_SOURCE_DIR}/r4d_emu)
set(R4D_EMU_HIP ${R4D_DIR}/r4d_attn_paged_window.hip ${R4D_DIR}/r4d_pair_argmax_f32.hip
                ${R4D_DIR}/r4d_norm_join_bf16.hip ${R4D_DIR}/r4d_expert_scale_bf16.hip
                ${R4D_DIR}/r4d_qsa_select_bf16.hip ${R4D_DIR}/r4d_lora_bf16.hip ${R4D_DIR}/r4d_glm.hip
                ${R4D_DIR}/r4d_mhc.hip ${R4D_DIR}/r4d_fp8_f32scale.hip ${R4D_DIR}/r4d_kda.hip
                ${R4D_DIR}/r4d_moe_mxfp4.hip ${R4D_DIR}/r4d_mla.hip ${R4D_DIR}/r4d_quant_act_fp8.hip)
set(R4D_EMU_TESTS attn pick norm escale qsa lora f32 mhc fp8f32 kda moemx mla moefuse)
set(R4D_EMU_BUILT OFF)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  # No fiber switch for this ISA.
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT RAD_ASAN AND NOT RAD_TSAN)
  set_source_files_properties(${R4D_EMU_HIP} PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;c++")
  add_library(r4d_emu_kernels OBJECT ${R4D_EMU_HIP} ${R4D_EMU}/ref_host.cpp ${R4D_EMU}/emu/emu.cpp)
  target_include_directories(r4d_emu_kernels PUBLIC ${R4D_EMU}/emu ${R4D_EMU} ${R4D_DIR})
  target_link_libraries(r4d_emu_kernels PUBLIC rad_abi)
  # As in libr4d: the joins must match the unfused chain bit for bit, so no FMA contraction.
  target_compile_options(r4d_emu_kernels PRIVATE -ffp-contract=off -Wno-psabi)
  foreach(t ${R4D_EMU_TESTS})
    add_executable(r4d_emu_${t} ${R4D_EMU}/test_${t}.cpp)
    target_link_libraries(r4d_emu_${t} PRIVATE r4d_emu_kernels ${CMAKE_DL_LIBS})
    target_compile_options(r4d_emu_${t} PRIVATE -ffp-contract=off -Wno-psabi)
    set_target_properties(r4d_emu_${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  endforeach()
  set(R4D_EMU_BUILT ON)
else()
  if(RAD_HIP_FOUND AND CMAKE_HIP_COMPILER_ID MATCHES "Clang")
    set(R4D_EMU_CXX ${CMAKE_HIP_COMPILER})
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(R4D_EMU_CXX ${CMAKE_CXX_COMPILER})
  else()
    find_program(RAD_EMU_CLANG NAMES clang++
                 DOC "clang++ that builds tests/r4d_emu when the C++ compiler is not clang")
    set(R4D_EMU_CXX ${RAD_EMU_CLANG})
  endif()
  if(R4D_EMU_CXX)
    set(dir ${CMAKE_CURRENT_BINARY_DIR}/r4d_emu)
    file(MAKE_DIRECTORY ${dir})
    set(flags -std=c++20 -O2 -ffp-contract=off -Wno-psabi -I${R4D_EMU}/emu -I${R4D_EMU}
              -I${CMAKE_SOURCE_DIR}/abi -I${R4D_DIR})
    set(objs "")
    foreach(src ${R4D_EMU_HIP} ${R4D_EMU}/ref_host.cpp ${R4D_EMU}/emu/emu.cpp
                ${R4D_EMU}/test_attn.cpp ${R4D_EMU}/test_pick.cpp ${R4D_EMU}/test_norm.cpp
                ${R4D_EMU}/test_escale.cpp ${R4D_EMU}/test_qsa.cpp ${R4D_EMU}/test_lora.cpp
                ${R4D_EMU}/test_f32.cpp ${R4D_EMU}/test_mhc.cpp ${R4D_EMU}/test_fp8f32.cpp
                ${R4D_EMU}/test_kda.cpp ${R4D_EMU}/test_moemx.cpp ${R4D_EMU}/test_mla.cpp
                ${R4D_EMU}/test_moefuse.cpp)
      get_filename_component(stem ${src} NAME_WE)
      set(lang "")
      if(src MATCHES "\\.hip$")
        set(lang -x c++)
      endif()
      add_custom_command(OUTPUT ${dir}/${stem}.o
        COMMAND ${R4D_EMU_CXX} ${flags} -MD -MF ${dir}/${stem}.o.d ${lang} -c ${src}
                -o ${dir}/${stem}.o
        DEPENDS ${src}
        DEPFILE ${dir}/${stem}.o.d
        COMMENT "Building r4d_emu ${stem}.o with ${R4D_EMU_CXX}"
        VERBATIM)
      if(NOT stem MATCHES "^test_")
        list(APPEND objs ${dir}/${stem}.o)
      endif()
    endforeach()
    # test_lora and test_kda open libref with dlopen to hold the emulated kernels to it.
    set(dl "")
    foreach(l ${CMAKE_DL_LIBS})
      list(APPEND dl -l${l})
    endforeach()
    set(exes "")
    foreach(t ${R4D_EMU_TESTS})
      set(exe ${CMAKE_BINARY_DIR}/tests/r4d_emu_${t})
      add_custom_command(OUTPUT ${exe}
        COMMAND ${R4D_EMU_CXX} ${dir}/test_${t}.o ${objs} ${dl} -o ${exe}
        DEPENDS ${dir}/test_${t}.o ${objs}
        COMMENT "Linking r4d_emu_${t}"
        VERBATIM)
      list(APPEND exes ${exe})
    endforeach()
    add_custom_target(r4d_emu ALL DEPENDS ${exes})
    set(R4D_EMU_BUILT ON)
  endif()
endif()
if(R4D_EMU_BUILT)
  foreach(t pick norm escale qsa lora f32 mhc fp8f32 kda moemx mla moefuse)
    add_test(NAME r4d_emu_${t} COMMAND ${CMAKE_BINARY_DIR}/tests/r4d_emu_${t})
  endforeach()
  # The attention cases take about 11 s, 7 of them in the two long prefills on the WMMA kernel,
  # which three shards put in different entries.
  foreach(k 0 1 2)
    add_test(NAME r4d_emu_attn_shard${k}
             COMMAND ${CMAKE_BINARY_DIR}/tests/r4d_emu_attn --shard ${k}/3)
  endforeach()
  set_tests_properties(r4d_emu_attn_shard0 r4d_emu_attn_shard1 r4d_emu_attn_shard2 r4d_emu_pick
                       r4d_emu_norm r4d_emu_escale r4d_emu_qsa r4d_emu_lora r4d_emu_mla r4d_emu_moefuse
                       PROPERTIES LABELS "r4d_emu" TIMEOUT ${RAD_TEST_TIMEOUT})
  # The LoRA pair has a libref oracle, and the test holds the emulated kernels to it when it can
  # load one -- expand bit for bit, shrink to its bound. kda_update is held to libref's within
  # rounding, and that test needs it.
  set_tests_properties(r4d_emu_lora r4d_emu_kda PROPERTIES
                       ENVIRONMENT "RADIANCE_HOME=${CMAKE_BINARY_DIR}/radiance_home")
  # A seeded random sweep, mostly prefill shapes on the WMMA kernel, which the emulator runs at
  # about 2 s a case: `slow`. `r4d_emu_attn sweep 200` is a longer run by hand, a superset of it.
  add_test(NAME r4d_emu_attn_sweep COMMAND ${CMAKE_BINARY_DIR}/tests/r4d_emu_attn sweep 40)
  set_tests_properties(r4d_emu_attn_sweep PROPERTIES LABELS "r4d_emu;slow" TIMEOUT 900)
  # QSA selection at serving sizes, seeded: about 15 s.
  add_test(NAME r4d_emu_qsa_sweep COMMAND ${CMAKE_BINARY_DIR}/tests/r4d_emu_qsa sweep 100)
  set_tests_properties(r4d_emu_qsa_sweep PROPERTIES LABELS "r4d_emu;slow" TIMEOUT 900)
endif()

# ================================================================== the document as a test
#
# docs/OPS.md is the authoritative op table, so this test checks it against what the plugins dump
# (dropped optional operands, a missing `:inout`, parameters moved into `params`, undeclared
# parameters). Cheap, so in the fast tier. libr4d.so is added when built; the script reports
# "  SKIP " rather than passing when a plugin will not dump.
set(RAD_OPSDOC_PLUGINS ${RAD_PLUGIN_OUT}/kernels/libref.so)
if(TARGET spechub)
  list(APPEND RAD_OPSDOC_PLUGINS ${RAD_PLUGIN_OUT}/kernels/spechub.so)
endif()
if(RAD_HIP_FOUND AND TARGET libr4d)
  list(APPEND RAD_OPSDOC_PLUGINS ${RAD_PLUGIN_OUT}/kernels/libr4d.so)
endif()
add_test(NAME ops_doc
         COMMAND ${CMAKE_SOURCE_DIR}/tools/ops_doc_check.sh
                 ${CMAKE_SOURCE_DIR}/docs/OPS.md $<TARGET_FILE:rad-schemas>
                 ${RAD_OPSDOC_PLUGINS})
set_tests_properties(ops_doc PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT ${RAD_TEST_TIMEOUT}
  ENVIRONMENT "ROCR_VISIBLE_DEVICES=99")

# tools/rad-export. Its standard-library checks always run; the merge arithmetic runs where torch
# and safetensors import, and the PEFT comparison where peft and transformers do too.
find_program(RAD_PYTHON3 NAMES python3)
if(RAD_PYTHON3)
  add_test(NAME rad_export_test COMMAND ${RAD_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/rad_export_test.py)
  set_tests_properties(rad_export_test PROPERTIES LABELS "unit;fast" TIMEOUT ${RAD_TEST_TIMEOUT})
endif()

# ================================================================== the shipped plugins, out of tree
#
# Installs the build into a prefix, then configures, builds and installs every plugin directory
# standalone against it with find_package(radiance). The result must match the tree's plugins,
# vocabulary and registration, and serve rad-kbench's fixture through the installed tools
# (tests/oot_plugins.sh). libr4d is included when the tree built it, with the same HIP compiler,
# targets and RAD_HIP_VOPD_REPEATABLE. Labelled `oot`: it rebuilds every plugin and takes minutes.
set(RAD_OOT_HIP "")
if(RAD_HIP_FOUND AND TARGET libr4d)
  get_target_property(RAD_OOT_R4D_ARCH libr4d HIP_ARCHITECTURES)
  string(REPLACE ";" "\\;" RAD_OOT_R4D_ARCH "${RAD_OOT_R4D_ARCH}")
  set(RAD_OOT_HIP ${CMAKE_HIP_COMPILER} "${RAD_OOT_R4D_ARCH}" ${RAD_HIP_VOPD_REPEATABLE})
endif()
add_test(NAME oot_plugins
         COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/oot_plugins.sh
                 ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/oot ${RAD_OOT_HIP})
# RESOURCE_LOCK kbench_memory: its replay holds operands in host memory as the kbench entries do.
set_tests_properties(oot_plugins PROPERTIES
  LABELS "oot"
  TIMEOUT 3600
  RESOURCE_LOCK kbench_memory
  ENVIRONMENT "CMAKE_GENERATOR=${CMAKE_GENERATOR};${RAD_TEST_OMP_ENV}")

# ================================================================== the kernel library, as a test
#
# rad-kbench replays data/kernels.rkb: every kernel of every loaded library against libref's
# recorded answers on seeded inputs, in guarded allocations. No model, oracle or network needed.
#
# kbench and kbench_moe check every library except the reference (--no-ref): its answers are what
# the fixture holds. --strict fails a stale fixture, and --coverage fails when a row a fixture
# reached (data/kbench_rows/) is no longer reached. Labelled `kbench`, which keeps them out of the
# fast tier (a replay takes seconds to minutes), and `gpu` only when libr4d is built, so
# `ctest -LE gpu` still runs them. `--max-cases 1` collapses a second seed of the same case; the
# depth sweep's cases differ in their operand shapes and all run. OpenMP waits passively: the host
# libraries' teams would otherwise spin between launches and starve what ctest -j runs beside them.
set(RAD_KBENCH_LABELS "kbench")
if(RAD_HIP_FOUND AND TARGET libr4d)
  set(RAD_KBENCH_LABELS "kbench;gpu")
endif()
set(RAD_KBENCH_ROWS ${CMAKE_SOURCE_DIR}/data/kbench_rows)
if(TARGET rad-kbench AND EXISTS ${CMAKE_SOURCE_DIR}/data/kernels.rkb)
  add_test(NAME kbench
           COMMAND rad-kbench --home ${RAD_PLUGIN_OUT} --max-cases 1 --no-ref --strict
                   --coverage ${RAD_KBENCH_ROWS})
  # RESOURCE_LOCK: each replay budgets a fraction of the memory available when it starts, so two
  # at once would together exceed it. They would also share the last-level cache and skew --bench.
  set_tests_properties(kbench PROPERTIES
    LABELS "${RAD_KBENCH_LABELS}"
    TIMEOUT 1800
    RESOURCE_LOCK kbench_memory
    PROCESSORS 4
    ENVIRONMENT "RADIANCE_HOME=${RAD_PLUGIN_OUT};OMP_WAIT_POLICY=passive")
endif()

# The same replay over the routed fixture, which covers the MoE ops, the gqa 8 attention dispatch
# and the MTP head (see the root CMakeLists). Separate tests, because a fixture records the one
# container it came from.
if(TARGET rad-kbench AND EXISTS ${CMAKE_SOURCE_DIR}/data/kernels_moe.rkb)
  add_test(NAME kbench_moe
           COMMAND rad-kbench --home ${RAD_PLUGIN_OUT} --max-cases 1 --no-ref --strict
                   --coverage ${RAD_KBENCH_ROWS}
                   --fixture ${RAD_PLUGIN_OUT}/kernels_moe.rkb)
  set_tests_properties(kbench_moe PROPERTIES
    LABELS "${RAD_KBENCH_LABELS}"
    TIMEOUT 1800
    RESOURCE_LOCK kbench_memory
    PROCESSORS 4
    ENVIRONMENT "RADIANCE_HOME=${RAD_PLUGIN_OUT};OMP_WAIT_POLICY=passive")
endif()

# libref against its own recordings, both fixtures, at tolerance zero: it is deterministic and
# machine-independent by construction, so any difference is a change in libref (or in the draw)
# that would otherwise become the next recording's target. This is the oracle's regression test.
# Its answers are cached under $RADIANCE_HOME/cache/kbench, keyed by libref's build-id and
# everything it links, the input bytes and the case, so an unchanged libref replays in seconds and
# a rebuilt one in full; the comparison runs either way. Labelled `slow` for the full run.
if(TARGET rad-kbench AND TARGET libref AND EXISTS ${CMAKE_SOURCE_DIR}/data/kernels.rkb AND
   EXISTS ${CMAKE_SOURCE_DIR}/data/kernels_moe.rkb)
  add_test(NAME kbench_ref
           COMMAND rad-kbench --home ${RAD_PLUGIN_OUT} --kernels libref --max-cases 1 --strict
                   --coverage ${RAD_KBENCH_ROWS}
                   --fixture ${RAD_PLUGIN_OUT}/kernels.rkb
                   --fixture ${RAD_PLUGIN_OUT}/kernels_moe.rkb)
  set_tests_properties(kbench_ref PROPERTIES
    LABELS "kbench;slow"
    TIMEOUT 3600
    RESOURCE_LOCK kbench_memory
    PROCESSORS 4
    ENVIRONMENT "RADIANCE_HOME=${RAD_PLUGIN_OUT};OMP_WAIT_POLICY=passive")
endif()

# ================================================================== tiers
#
# `fast` goes on every test registered above that carries none of slow, gpu, oot, perf, e2e and
# kbench, so the inner loop's membership is decided here alone. Every `gpu` test takes the card lock: two tests
# that each commit device memory, or each time kernels, must not share a card. COST starts the
# longest tests first in a parallel run, so they are not left on the critical path.
get_property(RAD_ALL_TESTS DIRECTORY PROPERTY TESTS)
foreach(t ${RAD_ALL_TESTS})
  get_test_property(${t} LABELS labels)
  if(NOT labels)
    set(labels "")
  endif()
  if(NOT labels MATCHES "(^|;)(slow|gpu|oot|perf|e2e|kbench)(;|$)")
    set_property(TEST ${t} APPEND PROPERTY LABELS fast)
  endif()
  if(labels MATCHES "(^|;)gpu(;|$)")
    set_property(TEST ${t} APPEND PROPERTY RESOURCE_LOCK card)
  endif()
endforeach()
foreach(cost IN ITEMS kbench_ref:500 r4d_selftest:350 r4d_emu_attn_sweep:80 kbench:20
                      kbench_moe:15 engine_e2e_test:12 r4d_emu_attn_shard0:5 r4d_emu_attn_shard2:5)
  string(REPLACE ":" ";" cost ${cost})
  list(GET cost 0 t)
  list(GET cost 1 c)
  if(TEST ${t})
    set_tests_properties(${t} PROPERTIES COST ${c})
  endif()
endforeach()

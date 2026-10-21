# One executable per *_test.cpp, globbed, so adding a test is adding a file.
#
# arch_test is the exception and says why in its own header: "No GPU, no model file, no core." It
# supplies its OWN implementation of the declare-phase C ABI -- a recording fake -- so that a
# plugin cannot pass only against core/build/'s builder, which would be an undeclared dependency.
# Linking rad_core would put a second definition of every rad_decl_* symbol in the link, so it
# gets the two utility translation units it actually needs and nothing else.
#
# The failure mode this avoids is worth naming, because it is not obvious from the error: a static
# archive only yields an object file that resolves an undefined symbol, so the clash appears the
# moment the ABI grows a call the fake has not implemented yet. The linker pulls rad_builder.cpp.o
# to satisfy that one symbol and every other rad_decl_* in it collides. Adding the call to the fake
# is the real fix; not linking the archive is what stops the next one presenting as a wall of
# duplicate-definition errors instead of one honest "undefined reference".
set(RAD_TESTS_NO_CORE arch_test)

# fixture_test is about tools/kfixture.cpp and tools/opshapes.cpp -- the seeded draw, the
# reduction and the on-disk recording that every kernel correctness number in the project is
# relative to. Those two are compiled into each tool rather than made a library (tools/
# CMakeLists.txt says why), so a test of them compiles them too. The alternative is a test that
# shells out to rad-kbench, which can only observe the harness through whatever a kernel run says
# about it -- and the properties worth pinning here are ones a passing run is equally consistent
# with.
set(RAD_TESTS_WITH_TOOLS fixture_test)

# The tokenizer tests compare the encoder against tools/tokref, the reference encoder, which is a
# library of its own for exactly that (tools/CMakeLists.txt).
set(RAD_TESTS_WITH_TOKREF tokenizer_test tokenizer_perf_test)

# The GBNF oracle is upstream's candidate walk, which the mask engine is held to bit for bit. It is
# test code and is compiled into the tests that compare against it -- and into rad-gbnf-check,
# which runs the same comparison on a real vocabulary -- so that no serving binary contains it.
set(RAD_TESTS_WITH_GBNF_ORACLE sample_test gbnf_test)

# THE LEGACY REPLY PARSER, AS AN ORACLE. The engine reads replies with core/text/chatparse.h; a
# whole-reply path built on llama.cpp's PEG parser and a by-hand call lifter is kept in legacy/ so
# that the tests can compare the two on every input. Built as its own archive and linked into the
# tests that compare -- nothing in core/ can reach it. tools/ links it for the acceptance tool's
# side-by-side numbers.
add_library(rad_legacy_toolparse STATIC ${CMAKE_CURRENT_SOURCE_DIR}/legacy/toolparse_legacy.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/legacy/chatdiff.cpp)

# Shared library export for rad_drafter C-ABI execution and Python E2E parity harness
add_library(rad_drafter SHARED ${CMAKE_CURRENT_SOURCE_DIR}/drafter_parity_test.cpp)
target_link_libraries(rad_drafter PRIVATE rad_core)
target_include_directories(rad_drafter PRIVATE ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi)
target_compile_definitions(rad_drafter PRIVATE RAD_DRAFTER_SHARED_LIB=1)
set_target_properties(rad_drafter PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(rad_legacy_toolparse PUBLIC rad_core)
target_include_directories(rad_legacy_toolparse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/legacy)
set(RAD_TESTS_WITH_LEGACY chatparse_test)

# relayout_parity_test holds each kernel library's relayout of libquant's planes to the recorded
# hash of the bytes that library stores. libr4d needs a HIP toolchain to build and its layouts do
# not, so the test compiles their host sources in -- -ffp-contract=off, as libr4d compiles every
# unit, so a scale converted on the way through rounds here as it does in the plugin.
set(RAD_TESTS_WITH_R4D_HOST relayout_parity_test)

# The chat templates the reply-format tests render through, copied from the model repositories,
# and the media samples the decoder is held to. Passed by path so a test binary run from anywhere
# finds them.
set(RAD_CHAT_FIXTURES ${CMAKE_CURRENT_SOURCE_DIR}/data/chat)
set(RAD_MEDIA_FIXTURES ${CMAKE_CURRENT_SOURCE_DIR}/data/media)

# Milestone 3: Paged KV-Cache Tree Rollback & Compaction unit test (kv_rollback_test)
file(GLOB RAD_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
# This model-owned GPU harness has an explicit guarded target below.
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/dloop_gpu_parity_test.cpp)
# The DFlash2 fusion oracle also needs an explicit target: it loads two plugin .so files and pins
# rows by name, so it is not the plain rad_core test the glob would build.
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/dflash2_fusion_gpu_test.cpp)
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/gdn_tree_fusion_gpu_test.cpp)
# Same for the host-side argument test: it links the three candidate .cpp files and the ABI, with
# no rad_core and no HIP, which is not what the glob's plain branch would give it.
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/dflash2_fusion_args_test.cpp)
# The tree selector's device comparator loads two plugin .so files and pins the device row by name,
# so it needs a card and an explicit target rather than the glob's plain rad_core test.
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/dflash_tree_gpu_test.cpp)
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/mtp_tree_accum_gpu_test.cpp)
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/mtp_scored_tree_gpu_test.cpp)
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/qsa_tree_commit_gpu_test.cpp)
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/attn_paged_tree_gpu_test.cpp)
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/mtp_pruned_tree_test.cpp)
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/mtp_pruned_gpu_test.cpp)
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/spechub_test.cpp)
# The PLE lookup regression loads a plugin .so and needs a card to itself, so it runs as its own
# binary with an explicit target below, not inside a shared fixture.
list(REMOVE_ITEM RAD_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/ple_lookup_gpu_test.cpp)
foreach(t ${RAD_TESTS})
  get_filename_component(n ${t} NAME_WE)
  if(n IN_LIST RAD_TESTS_NO_CORE)
    add_executable(${n} ${t} ${CMAKE_SOURCE_DIR}/core/util/rad_util.cpp)
  elseif(n IN_LIST RAD_TESTS_WITH_TOKREF)
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE rad_tokref rad_core)
  elseif(n IN_LIST RAD_TESTS_WITH_TOOLS)
    add_executable(${n} ${t} ${CMAKE_SOURCE_DIR}/tools/kfixture.cpp
                            ${CMAKE_SOURCE_DIR}/tools/opshapes.cpp)
    target_link_libraries(${n} PRIVATE rad_core)
    target_include_directories(${n} PRIVATE ${CMAKE_SOURCE_DIR}/tools)
  elseif(n IN_LIST RAD_TESTS_WITH_R4D_HOST)
    add_executable(${n} ${t} ${CMAKE_SOURCE_DIR}/libr4d/r4d_layout.cpp
                            ${CMAKE_SOURCE_DIR}/libr4d/r4d_fp8_layout.cpp)
    target_link_libraries(${n} PRIVATE rad_core ${CMAKE_DL_LIBS})
    target_include_directories(${n} PRIVATE ${CMAKE_SOURCE_DIR}/libr4d)
    target_compile_options(${n} PRIVATE -ffp-contract=off)
  elseif(n IN_LIST RAD_TESTS_WITH_GBNF_ORACLE)
    add_executable(${n} ${t} ${CMAKE_CURRENT_SOURCE_DIR}/gbnf_oracle.cpp)
    target_link_libraries(${n} PRIVATE rad_core)
    # The chat templates whose tool-call grammars the engine is held to the oracle on. Checked
    # in, because a template is a few kilobytes and the tokenizer beside it is not.
    target_compile_definitions(${n} PRIVATE
      RAD_GBNF_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data/gbnf")
  else()
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE rad_core)
  endif()
  if(n STREQUAL "gdn_tree_ops_test" OR n STREQUAL "dloop_ref_test" OR n STREQUAL "dloop_full_ref_test" OR n STREQUAL "ple_tree_ops_test" OR n STREQUAL "dflash_tree_select_test" OR n STREQUAL "attn_paged_tree_test")
    # Exercise the same model-owned host objects that mine_ref publishes.
    target_sources(${n} PRIVATE $<TARGET_OBJECTS:mine_ref_ops>)
    add_dependencies(${n} mine_ref)
    target_compile_options(${n} PRIVATE -ffp-contract=off)
  endif()
  if(n STREQUAL "ple_tree_ops_test")
    # Unchanged conventional linear PLE operations are independent replay oracles.
    target_sources(${n} PRIVATE ${CMAKE_SOURCE_DIR}/libref/ref_ngram.cpp
                               ${CMAKE_SOURCE_DIR}/libref/ref_ple.cpp)
  endif()
  if(n STREQUAL "dloop_ref_test")
    target_compile_definitions(${n} PRIVATE
      RAD_DLOOP_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/dloop_tiny")
  endif()
  if(n STREQUAL "dloop_full_ref_test")
    target_compile_definitions(${n} PRIVATE
      RAD_DLOOP_FULL_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/dloop_full_tiny")
  endif()
  if(n IN_LIST RAD_TESTS_WITH_LEGACY)
    target_link_libraries(${n} PRIVATE rad_legacy_toolparse)
  endif()
  target_compile_definitions(${n} PRIVATE RAD_CHAT_FIXTURES="${RAD_CHAT_FIXTURES}"
                                          RAD_MEDIA_FIXTURES="${RAD_MEDIA_FIXTURES}")
  # include/ normally arrives through rad_core's PUBLIC interface; a test that does not link it
  # still needs the ABI headers, since that is the whole surface it is testing against.
  target_include_directories(${n} PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
                                          ${CMAKE_BINARY_DIR}/rad_include)
  set_target_properties(${n} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # The tier transfer test loads a kernel plugin that waits on a stream through the device API, so
  # the executable exports it the way the engine binary does for its plugins.
  if(n STREQUAL "mem_test")
    set_target_properties(${n} PROPERTIES ENABLE_EXPORTS ON)
  endif()
  add_test(NAME ${n} COMMAND ${n})
  # AND mem_test NEEDS THE CARD TO ITSELF, which nothing about its name says. The KV pool is
  # elastic on every run, so Pools::configure reserves and commits through the driver whatever
  # allocators the fixture injects -- the host-allocator seam is real for the weight pool and
  # dead for this one. Beside a server holding the card the pool cannot be backed and the whole
  # file fails with "the card would not back it", which reads as a regression and is contention.
  # The `gpu` label is how the rest of the tree already says that, and `ctest -LE gpu` still runs
  # everything else on a machine that is busy.
  if(${n} STREQUAL "mem_test")
    set_tests_properties(${n} PROPERTIES LABELS "gpu")
  endif()
  if(n STREQUAL "adversarial_kv_compact_stress_test" AND RAD_HIP_FOUND)
    # This C++ target exercises the engine's runtime dispatcher, not a HIP-header test branch.
    set_tests_properties(${n} PROPERTIES LABELS "gpu" RUN_SERIAL TRUE)
  endif()
  # A TEST THAT SKIPPED IS NOT A TEST THAT PASSED, and a test that FAILED must not read as one
  # either. Several files here guard cases on a model file, a vocabulary or a backend and print a
  # "  SKIP " line when it is absent; ctest swallows output on success, so without a skip signal
  # those cases report Passed having asserted nothing.
  #
  # SKIP_REGULAR_EXPRESSION cannot be that signal. CMake evaluates it BEFORE the return code, so a
  # binary that prints one skip line and then fails is reported Skipped and counted toward "100%
  # tests passed" -- and a partly-skipped binary is exactly the one that prints skip lines.
  #
  # So the skip signal is the RETURN CODE, and rad_test.h derives it rather than declaring it:
  # every CHECK increments a counter and main returns 77 only when nothing failed and nothing was
  # checked at all. A file where some cases ran and others skipped passes on the ones that ran,
  # which is the honest answer, and a failure anywhere is a failure.
  set_tests_properties(${n} PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties(${n} PROPERTIES ENVIRONMENT
    "RADIANCE_HOME=${CMAKE_BINARY_DIR}/radiance_home")
endforeach()

# SpecHub checks both the proposal law and the full-target verification transport.
add_executable(spechub_test spechub_test.cpp
  ${CMAKE_SOURCE_DIR}/mine/spechub.cpp ${CMAKE_SOURCE_DIR}/mine/ref_spechub.cpp)
target_link_libraries(spechub_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
target_include_directories(spechub_test PRIVATE
  ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
  ${CMAKE_BINARY_DIR}/rad_include)
target_compile_options(spechub_test PRIVATE -ffp-contract=off)
set_target_properties(spechub_test PROPERTIES ENABLE_EXPORTS ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_dependencies(sample_test mine_ref)
add_test(NAME spechub_host COMMAND spechub_test)
set_tests_properties(spechub_host PROPERTIES LABELS "host;spechub")
if(RAD_HIP_FOUND AND TARGET mine AND TARGET mine_ref AND TARGET libr4d)
  add_dependencies(spechub_test mine mine_ref libr4d)
  add_test(NAME spechub_gpu COMMAND spechub_test
    --mine $<TARGET_FILE:mine> --ref $<TARGET_FILE:mine_ref> --libr4d $<TARGET_FILE:libr4d>)
  set_tests_properties(spechub_gpu PROPERTIES LABELS "gpu;spechub"
    TIMEOUT 300 RUN_SERIAL TRUE)
endif()

# ================================================================== the DFlash2 host contract
#
# The host half of the three fusion candidates -- schema, shape and every refusal in
# `rad_dflash_*_args` -- runs with no device, no stream and no launch, so it is an ORDINARY test.
# It compiles the three candidate .cpp files directly and links the header-only ABI: no rad_core
# (nothing here resolves a plugin or opens a card) and no HIP. That is also why the files are
# removed from the glob above -- the glob's plain branch would add rad_core and turn a test that
# must stay device-free into one that needs the engine's registry.
#
# This claims nothing about numerics; tests/dflash2_fusion_gpu_test.cpp is what compares bytes.
add_executable(dflash2_fusion_args_test dflash2_fusion_args_test.cpp
  ${CMAKE_SOURCE_DIR}/mine/dflash_conv_add.cpp
  ${CMAKE_SOURCE_DIR}/mine/dflash_k_norm_rope_store.cpp
  ${CMAKE_SOURCE_DIR}/mine/dflash_gated_bf16.cpp)
target_link_libraries(dflash2_fusion_args_test PRIVATE radiance::abi)
target_include_directories(dflash2_fusion_args_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/mine)
# The same setting the plugin compiles these units with, so a rounding decision taken on the host
# path is the one the plugin makes.
target_compile_options(dflash2_fusion_args_test PRIVATE -ffp-contract=off)
set_target_properties(dflash2_fusion_args_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_test(NAME dflash2_fusion_args_test COMMAND dflash2_fusion_args_test)
set_tests_properties(dflash2_fusion_args_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

# ================================================================== the DFlash2 tree selector
#
# `dflash_tree_select` is a branching search whose output is a DISCRETE structure, so the device row
# and the host oracle disagree in whole nodes rather than in low bits: the comparison is a memcmp
# with no tolerance. Stock rad-kbench cannot check it either -- the row declares no `replaces`
# chain, deliberately, because the op is a different selector and not a fusion of `dflash_select`.
if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS AND TARGET mine AND TARGET mine_ref AND TARGET libr4d)
  add_executable(dflash_tree_gpu_test dflash_tree_gpu_test.cpp)
  target_link_libraries(dflash_tree_gpu_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(dflash_tree_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  # libr4d is NOT optional here: the single-path equivalence is checked against the shipping
  # `dflash_select` kernel, and without it the run would prove only that two of this op's own
  # implementations agree.
  add_dependencies(dflash_tree_gpu_test mine mine_ref libr4d)
  set_target_properties(dflash_tree_gpu_test PROPERTIES
    ENABLE_EXPORTS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME dflash_tree_gpu_test COMMAND dflash_tree_gpu_test
    --mine $<TARGET_FILE:mine> --ref $<TARGET_FILE:mine_ref> --libr4d $<TARGET_FILE:libr4d>)
  set_tests_properties(dflash_tree_gpu_test PROPERTIES LABELS "gpu;dflash2" TIMEOUT 300)
endif()

# The MTP tree compares the fused device path with libr4d merge and the host topology oracle.
if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS AND TARGET mine AND TARGET mine_ref AND TARGET libr4d)
  add_executable(mtp_tree_accum_gpu_test mtp_tree_accum_gpu_test.cpp)
  target_link_libraries(mtp_tree_accum_gpu_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(mtp_tree_accum_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  add_dependencies(mtp_tree_accum_gpu_test mine mine_ref libr4d)
  set_target_properties(mtp_tree_accum_gpu_test PROPERTIES
    ENABLE_EXPORTS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME mtp_tree_accum_gpu_test COMMAND mtp_tree_accum_gpu_test
    --mine $<TARGET_FILE:mine> --ref $<TARGET_FILE:mine_ref> --libr4d $<TARGET_FILE:libr4d>)
  set_tests_properties(mtp_tree_accum_gpu_test PROPERTIES
    LABELS "gpu;mtp" TIMEOUT 300 RUN_SERIAL TRUE)
endif()

if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS AND TARGET mine AND TARGET libref)
  add_executable(attn_paged_tree_gpu_test attn_paged_tree_gpu_test.cpp)
  target_link_libraries(attn_paged_tree_gpu_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(attn_paged_tree_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  add_dependencies(attn_paged_tree_gpu_test mine libref)
  set_target_properties(attn_paged_tree_gpu_test PROPERTIES
    ENABLE_EXPORTS ON RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME attn_paged_tree_gpu_test COMMAND attn_paged_tree_gpu_test
    --mine $<TARGET_FILE:mine> --libref $<TARGET_FILE:libref>)
  set_tests_properties(attn_paged_tree_gpu_test PROPERTIES
    LABELS "gpu;attention;mtp" TIMEOUT 900 RUN_SERIAL TRUE)
endif()

# Scored MTP uses full-vector normalization and an independent ancestor-closed subset oracle.
if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS AND TARGET mine AND TARGET mine_ref AND TARGET libr4d)
  add_executable(mtp_scored_tree_gpu_test mtp_scored_tree_gpu_test.cpp)
  target_link_libraries(mtp_scored_tree_gpu_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(mtp_scored_tree_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  add_dependencies(mtp_scored_tree_gpu_test mine mine_ref libr4d)
  set_target_properties(mtp_scored_tree_gpu_test PROPERTIES
    ENABLE_EXPORTS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME mtp_scored_tree_gpu_test COMMAND mtp_scored_tree_gpu_test
    --mine $<TARGET_FILE:mine> --ref $<TARGET_FILE:mine_ref> --libr4d $<TARGET_FILE:libr4d>)
  set_tests_properties(mtp_scored_tree_gpu_test PROPERTIES
    LABELS "gpu;mtp;scored" TIMEOUT 300 RUN_SERIAL TRUE)
endif()

# Pruned MTP exercises real parent gathering and branch attention, not source wiring.
if(TARGET mine_ref AND TARGET libref)
  add_executable(mtp_pruned_tree_test mtp_pruned_tree_test.cpp)
  target_link_libraries(mtp_pruned_tree_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(mtp_pruned_tree_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  target_compile_options(mtp_pruned_tree_test PRIVATE -ffp-contract=off)
  add_dependencies(mtp_pruned_tree_test mine_ref libref)
  set_target_properties(mtp_pruned_tree_test PROPERTIES
    ENABLE_EXPORTS ON RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME mtp_pruned_tree_test COMMAND mtp_pruned_tree_test
    --ref $<TARGET_FILE:mine_ref> --libref $<TARGET_FILE:libref>)
  set_tests_properties(mtp_pruned_tree_test PROPERTIES LABELS "mtp;pruned" TIMEOUT 120)
endif()
if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS AND TARGET mine AND TARGET mine_ref
   AND TARGET libr4d AND TARGET libref)
  add_executable(mtp_pruned_gpu_test mtp_pruned_gpu_test.cpp mtp_pruned_tree_test.cpp)
  target_compile_definitions(mtp_pruned_gpu_test PRIVATE RAD_MTP_PRUNED_GPU_TEST=1)
  target_link_libraries(mtp_pruned_gpu_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(mtp_pruned_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  target_compile_options(mtp_pruned_gpu_test PRIVATE -ffp-contract=off)
  add_dependencies(mtp_pruned_gpu_test mine mine_ref libr4d libref)
  set_target_properties(mtp_pruned_gpu_test PROPERTIES
    ENABLE_EXPORTS ON RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME mtp_pruned_gpu_test COMMAND mtp_pruned_gpu_test
    --mine $<TARGET_FILE:mine> --ref $<TARGET_FILE:mine_ref>
    --libr4d $<TARGET_FILE:libr4d> --libref $<TARGET_FILE:libref>)
  set_tests_properties(mtp_pruned_gpu_test PROPERTIES
    LABELS "gpu;mtp;pruned" TIMEOUT 300 RUN_SERIAL TRUE)
endif()

if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS AND TARGET mine AND TARGET mine_ref AND TARGET libr4d)
  add_executable(qsa_tree_commit_gpu_test qsa_tree_commit_gpu_test.cpp)
  target_link_libraries(qsa_tree_commit_gpu_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(qsa_tree_commit_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  add_dependencies(qsa_tree_commit_gpu_test mine mine_ref libr4d)
  set_target_properties(qsa_tree_commit_gpu_test PROPERTIES
    ENABLE_EXPORTS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME qsa_tree_commit_gpu_test COMMAND qsa_tree_commit_gpu_test
    --mine $<TARGET_FILE:mine> --ref $<TARGET_FILE:mine_ref> --libr4d $<TARGET_FILE:libr4d>)
  set_tests_properties(qsa_tree_commit_gpu_test PROPERTIES
    LABELS "gpu;qsa;mtp" TIMEOUT 900 RUN_SERIAL TRUE)
endif()

# ================================================================== the device selftest
#
# r4d_selftest IS THE ONLY THING THAT CHECKS THE FUSION CLAIMS, so it is a registered test.
#
# Sixteen rows in libr4d's registry say things like "byte-identical to the two ops it replaces",
# and that claim is what makes a fused kernel SUBSTITUTABLE rather than a second numerical path --
# it is the difference between a fusion and a fork. Six of those claims are verified, by memcmp,
# against the unfused kernels run on the same inputs. Every one of those verifications lives in
# r4d_selftest.cpp, which libr4d/CMakeLists.txt builds; a binary built and registered with nothing
# leaves `ctest` green without executing a single one of them.
#
# It needs a card, so it is guarded on the HIP backend; it forks a child per row and takes about
# two minutes, against 1.2 seconds for everything else, so it carries a `slow` label and
# `ctest -LE slow` is the fast inner loop. The default `ctest` runs it.
if(RAD_HIP_FOUND AND TARGET r4d_selftest)
  add_test(NAME r4d_selftest
           COMMAND r4d_selftest ${RAD_PLUGIN_OUT}/kernels/libr4d.so)
  set_tests_properties(r4d_selftest PROPERTIES
    LABELS "slow;gpu"
    TIMEOUT 900
    ENVIRONMENT "RADIANCE_HOME=${CMAKE_BINARY_DIR}/radiance_home")
endif()

# Compile the actual FullDrafter selector into a standalone HIP regression; no core or
# teacher model is needed. Keep host-only and non-gfx1201 builds free of this target.
if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS)
  add_executable(fulldrafter_markov_select_gpu_test fulldrafter_markov_select_gpu_test.hip)
  set_source_files_properties(fulldrafter_markov_select_gpu_test.hip PROPERTIES LANGUAGE HIP)
  target_link_libraries(fulldrafter_markov_select_gpu_test PRIVATE radiance::abi)
  rad_hipify(fulldrafter_markov_select_gpu_test gfx1201)
  set_target_properties(fulldrafter_markov_select_gpu_test PROPERTIES
    HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME fulldrafter_markov_select_gpu_test COMMAND fulldrafter_markov_select_gpu_test)
  set_tests_properties(fulldrafter_markov_select_gpu_test PROPERTIES
    LABELS "gpu" SKIP_RETURN_CODE 77 TIMEOUT 180)
endif()

# Model-owned device/reference/golden parity. This binary must never be run
# while training owns the GPUs; missing devices or skipped ops are failures.
if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS)
  add_executable(dloop_gpu_parity_test dloop_gpu_parity_test.cpp)
  target_sources(dloop_gpu_parity_test PRIVATE $<TARGET_OBJECTS:mine_ref_ops>)
  target_link_libraries(dloop_gpu_parity_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(dloop_gpu_parity_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  add_dependencies(dloop_gpu_parity_test mine mine_ref)
  set_target_properties(dloop_gpu_parity_test PROPERTIES
    ENABLE_EXPORTS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME dloop_gpu_parity_test COMMAND dloop_gpu_parity_test
    $<TARGET_FILE:mine> $<TARGET_FILE:mine_ref> ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/dloop_tiny)
  set_tests_properties(dloop_gpu_parity_test PROPERTIES LABELS "gpu;dloop" TIMEOUT 900)
endif()

# ================================================================== the DFlash2 fusion oracle
#
# Stock rad-kbench cannot check these rows: it resolves a `replaces` chain only inside the fused
# row's OWN plugin (tools/rad_kbench.cpp:2682), and `mine` deliberately publishes no unfused copy
# of libr4d's arithmetic. A rad-kbench `--fuse` run over these ops is therefore a SKIP, and a skip
# is not a pass. This comparator names both sides -- the `mine` candidate and each unfused libr4d
# row -- and compares device bytes with zero tolerance, failing closed on a missing row.
if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS AND TARGET mine AND TARGET libr4d)
  add_executable(dflash2_fusion_gpu_test dflash2_fusion_gpu_test.cpp)
  target_link_libraries(dflash2_fusion_gpu_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(dflash2_fusion_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  add_dependencies(dflash2_fusion_gpu_test mine libr4d)
  # The plugin .so files resolve the engine's device and registry symbols from this image.
  set_target_properties(dflash2_fusion_gpu_test PROPERTIES
    ENABLE_EXPORTS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  # `--cases all`, not the binary's default: `mine` now publishes all three candidate families and
  # a default that leaves one of them out would report green over an uncompared row.
  add_test(NAME dflash2_fusion_gpu_test COMMAND dflash2_fusion_gpu_test
    --mine $<TARGET_FILE:mine> --ref $<TARGET_FILE:libr4d> --cases all)
  set_tests_properties(dflash2_fusion_gpu_test PROPERTIES LABELS "gpu;dflash2" TIMEOUT 900)
endif()

# Compare the fused tree convolution/recurrence/INT8 epilog against retained unfused rows.
if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS AND TARGET mine AND TARGET libr4d)
  add_executable(gdn_tree_fusion_gpu_test gdn_tree_fusion_gpu_test.cpp)
  target_link_libraries(gdn_tree_fusion_gpu_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(gdn_tree_fusion_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  add_dependencies(gdn_tree_fusion_gpu_test mine libr4d)
  set_target_properties(gdn_tree_fusion_gpu_test PROPERTIES
    ENABLE_EXPORTS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME gdn_tree_fusion_gpu_test COMMAND gdn_tree_fusion_gpu_test
    --mine $<TARGET_FILE:mine> --libr4d $<TARGET_FILE:libr4d>)
  set_tests_properties(gdn_tree_fusion_gpu_test PROPERTIES
    LABELS "gpu;gdn;dflash2" TIMEOUT 900 RUN_SERIAL TRUE)
endif()

# PLE n-gram lookup regression: the specialised 160-wide walk must return each id's own literal
# BF16 row. One test, because there is one path -- the id-keyed VRAM cache and its
# RADIANCE_PLE_VRAM_CACHE_SETS knob are gone, so there is no second mode to run.
if(RAD_HIP_FOUND AND "gfx1201" IN_LIST RAD_GPU_TARGETS AND TARGET libr4d)
  add_executable(ple_lookup_gpu_test ple_lookup_gpu_test.cpp)
  target_link_libraries(ple_lookup_gpu_test PRIVATE rad_core radiance::abi ${CMAKE_DL_LIBS})
  target_include_directories(ple_lookup_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
    ${CMAKE_BINARY_DIR}/rad_include)
  add_dependencies(ple_lookup_gpu_test libr4d)
  # The plugin .so resolves the engine's device and registry symbols from this image.
  set_target_properties(ple_lookup_gpu_test PROPERTIES
    ENABLE_EXPORTS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_test(NAME ple_lookup_gpu_test COMMAND ple_lookup_gpu_test
    $<TARGET_FILE:libr4d> 1000)
  set_tests_properties(ple_lookup_gpu_test PROPERTIES
    LABELS "gpu;ple" TIMEOUT 900 RUN_SERIAL TRUE)
endif()

# ================================================================== the host kernels
#
# rad-avx-check holds libavx to libref, op for op and at every ISA level the machine executes, and
# runs embed_lookup_q over a file-backed table with two ranks and the read-ahead racing: the path a
# model's host gather takes and that no geometry reaches. It writes its table beside itself, where
# the filesystem reports its direct-I/O alignment, and again in TMPDIR, where a tmpfs reports none
# and the gather aligns its reads by the block size, as it does on btrfs.
if(TARGET libavx AND TARGET rad-avx-check)
  add_test(NAME avx_check
           COMMAND rad-avx-check --ref ${RAD_PLUGIN_OUT}/kernels/libref.so
                                 --avx ${RAD_PLUGIN_OUT}/kernels/libavx.so)
  set_tests_properties(avx_check PROPERTIES TIMEOUT 600)
endif()

# ================================================================== the document as a test
#
# docs/OPS.md calls its resolved table "THE AUTHORITY FOR THE OPS IN IT" and says of itself that
# ref's op list is the conventional vocabulary and this file is that list. A hand-maintained table
# drifts from the registries -- dropped optional operands, a missing `:inout`, parameters that
# have moved into the `params` operand, a parameter no plugin declares -- so this test holds the
# document to what the plugins dump.
#
# The cost is named in OPS.md itself: a header written against a GUESS at a schema the document
# omits gets the schema wrong. A document that is WRONG is worse than one that is missing, because
# it is read instead of the source.
#
# It is cheap (a dlopen and an awk pass) so it carries no label and runs in the default ctest.
# libref alone would catch most of it; libr4d.so is added when there is a card to load it, and
# the script reports "  SKIP " rather than green when a plugin will not dump.
set(RAD_OPSDOC_PLUGINS ${RAD_PLUGIN_OUT}/kernels/libref.so)
if(TARGET mine_ref)
  list(APPEND RAD_OPSDOC_PLUGINS $<TARGET_FILE:mine_ref>)
endif()
if(RAD_HIP_FOUND AND TARGET libr4d)
  list(APPEND RAD_OPSDOC_PLUGINS ${RAD_PLUGIN_OUT}/kernels/libr4d.so)
endif()
add_test(NAME ops_doc
         COMMAND ${CMAKE_SOURCE_DIR}/tools/ops_doc_check.sh
                 ${CMAKE_SOURCE_DIR}/docs/OPS.md $<TARGET_FILE:rad-schemas>
                 ${RAD_OPSDOC_PLUGINS})
set_tests_properties(ops_doc PROPERTIES SKIP_RETURN_CODE 77)

# ================================================================== RDNA4 ISA compliance check
#
# Holds libr4d.so and built HIP kernels to the RDNA4 (gfx1201) ISA engineering standards
# documented in docs/RDNA4_ISA_ENGINEERING_RULES.md: 24-VGPR allocation boundaries, zero-waterfall
# loops on hot decode/draft paths, and zero scratch spills on prefill/decode kernels.
if(RAD_HIP_FOUND AND TARGET libr4d)
  add_test(NAME isa_compliance
           COMMAND ${CMAKE_SOURCE_DIR}/scripts/check_isa_compliance.sh
                   ${RAD_PLUGIN_OUT}/kernels/libr4d.so)
  set_tests_properties(isa_compliance PROPERTIES
    TIMEOUT 120
    ENVIRONMENT "RADIANCE_HOME=${CMAKE_BINARY_DIR}/radiance_home")
endif()

# ================================================================== the shipped plugins, out of tree
#
# Every plugin radiance ships is a plugin like any other, and this is what holds that to be true:
# the build is installed into a prefix, every plugin directory is configured ON ITS OWN against it
# with find_package(radiance), built and installed into a home of its own, and the result has to
# hold the same plugins, declare the same vocabulary and register the same way as the tree's build
# -- then serve rad-kbench's fixture from that home through the installed tools
# (tests/oot_plugins.sh). A first-party plugin that quietly depended on being built in the tree
# fails here; so does an installed package a third party could not build against.
#
# libr4d is included where the tree built it: the same HIP compiler, its own targets. Labelled
# `oot` because it rebuilds every plugin, which is minutes, not seconds.
set(RAD_OOT_HIP "")
if(RAD_HIP_FOUND AND TARGET libr4d)
  get_target_property(RAD_OOT_R4D_ARCH libr4d HIP_ARCHITECTURES)
  string(REPLACE ";" "\\;" RAD_OOT_R4D_ARCH "${RAD_OOT_R4D_ARCH}")
  set(RAD_OOT_HIP ${CMAKE_HIP_COMPILER} "${RAD_OOT_R4D_ARCH}")
endif()
add_test(NAME oot_plugins
         COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/oot_plugins.sh
                 ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/oot ${RAD_OOT_HIP})
set_tests_properties(oot_plugins PROPERTIES LABELS "oot" TIMEOUT 3600
                     ENVIRONMENT "CMAKE_GENERATOR=${CMAKE_GENERATOR}")

# ================================================================== the kernel library, as a test
#
# rad-kbench replaying data/kernels.rkb: every kernel every loaded library supplies, run against
# libref's recorded answers on seeded inputs, inside guarded allocations. No model, no oracle at
# run time, no network -- the fixture is checked in.
#
# r4d_selftest checks what its author wrote down and is compiled by the plugin; rad-kbench checks
# every row in every registry against an independent implementation, and the checked-in fixture
# lets it run without a model container.
#
# NOT GATED ON A CARD, because the reference library is under test in a replay too. The answers
# in the fixture were recorded by libref and checked in, so libref against them is a real
# comparison and the only regression test the oracle has -- which matters more than it sounds,
# since every correctness number this project reports about any kernel is defined relative to
# libref. It also means a build machine with no card still checks something, and a host
# implementation is exactly what regresses there.
#
# SLOW: libref is a straight-loop host implementation, so a full replay is minutes rather than the
# seconds the device libraries take. `ctest -LE slow` is the fast inner loop, as it is for
# r4d_selftest. `gpu` is added only where there is one, so `ctest -LE gpu` on a build machine
# still runs this.
#
# `--max-cases 1` because a fixture recorded at max-cases 1 has one case per point already and
# the cap is then only a guard against a future fixture that does not.
if(TARGET rad-kbench AND EXISTS ${CMAKE_SOURCE_DIR}/data/kernels.rkb)
  add_test(NAME kbench
           COMMAND rad-kbench --home ${RAD_PLUGIN_OUT} --max-cases 1)
  set(RAD_KBENCH_LABELS "slow")
  if(RAD_HIP_FOUND AND TARGET libr4d)
    set(RAD_KBENCH_LABELS "slow;gpu")
  endif()
  # RESOURCE_LOCK, because the case budget each of these takes is a fraction of what the machine
  # has FREE at the moment it starts. Two replays launched together both read the same number and
  # both take their share of it, so the pair arrives at the total the budget exists to stay under.
  # They also thrash a shared last-level cache against each other, which makes --bench meaningless.
  set_tests_properties(kbench PROPERTIES
    LABELS "${RAD_KBENCH_LABELS}"
    TIMEOUT 1800
    RESOURCE_LOCK kbench_memory
    ENVIRONMENT "RADIANCE_HOME=${RAD_PLUGIN_OUT}")
endif()

# The same replay over the ROUTED fixture. A recording is of one container and no container in
# scope declares the whole vocabulary, so the MoE ops, the gqa 8 attention dispatch and the MTP
# head have no coverage in data/kernels.rkb at all -- see the note beside it in the root
# CMakeLists. Two tests rather than one merged fixture because a fixture names the container it
# was recorded from, and merging two would make that field a lie.
if(TARGET rad-kbench AND EXISTS ${CMAKE_SOURCE_DIR}/data/kernels_moe.rkb)
  add_test(NAME kbench_moe
           COMMAND rad-kbench --home ${RAD_PLUGIN_OUT} --max-cases 1
                   --fixture ${RAD_PLUGIN_OUT}/kernels_moe.rkb)
  set_tests_properties(kbench_moe PROPERTIES
    LABELS "${RAD_KBENCH_LABELS}"
    TIMEOUT 1800
    RESOURCE_LOCK kbench_memory
    ENVIRONMENT "RADIANCE_HOME=${RAD_PLUGIN_OUT}")
endif()

add_executable(dloop_loader_test ${CMAKE_SOURCE_DIR}/candidates/dloop_dartree/loader_test.cpp)
target_include_directories(dloop_loader_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
add_test(NAME dloop_loader COMMAND dloop_loader_test ${CMAKE_SOURCE_DIR}/tests/fixtures/dloop_full_tiny)
set_tests_properties(dloop_loader PROPERTIES LABELS "host;dloop")

# THE ARCHITECTURES libr4d SERVES, declared here and nowhere else. Its kernels are written for
# RDNA4: wave32, the gfx12 WMMA fragment layouts, and gfx12-only instructions such as
# global_load_tr_b128. Compiled for anything else they would launch and compute garbage, so the
# coverage is a property of the library and the library is what states it.
#
# A build targeting other cards skips libr4d and keeps going. What it gets is an engine with no
# device kernel library for that card, which REFUSES TO SERVE an op only a reference library
# implements unless --debug-accept-reference-kernels is passed (spec §17) -- an unserved card that
# says so, rather than a project that will not configure. Supplying a kernel library for that card
# is the supported answer, and it is the same one an out-of-tree author gives.
#
# A plugin like any other: built here inside radiance's tree, or on its own against an installed
# radiance (`cmake -S libr4d -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`, with the HIP compiler
# named as for the tree) -- the same rad_add_plugin either way (cmake/RadAddPlugin.cmake).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libr4d LANGUAGES CXX HIP)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_plugin_gpu_targets(R4D_GPU_TARGETS "^gfx12")
if(NOT R4D_GPU_TARGETS)
  message(STATUS
    "libr4d: RDNA4 (gfx12xx) only, and this build targets ${RAD_GPU_TARGETS} -- not built")
  return()
endif()

# The libr4d kernel plugin, and THE ONLY BUILD libr4d HAS: this list is the definition of its
# translation units.
#
# THE PER-UNIT FLAGS ARE NOT DECORATION: the GDN chunk scan needs -mcumode, and every quantised
# GEMM needs its group size, which must match the group the weights were packed at or the kernel
# reads the wrong scales.
#
# Every unit is compiled -ffp-contract=off. The quantised all-reduce REQUIRES it: it sums two
# dequantised products per element, the two ranks see the products in the opposite order, and
# contracting either into an FMA makes them disagree by ~1 ULP -- which breaks the replicated-state
# invariant. For the rest it is reproducibility: the costs the kernels' comments state hold at
# this setting.

# ONE DIRECTORY: the kernels and libr4d's own declaration -- the row table, the layouts -- live
# together. There is no launch adapter between the plugin ABI and the kernels: every entry point
# takes `(const RadArgs*, RadStream)` and sits beside the kernel it launches.
#
# THIS PLUGIN LINKS rad_abi AND NOTHING ELSE: it is a standalone plugin built on abi/, exactly like
# one written out of tree.

# The weight group size, as the DEVICE units see it. It is NOT one variable covering both sides:
# the packer is r4d_layout.cpp, a host source in R4D_DECLARATION below, which no -D here reaches --
# it reads R4D_W4_GROUP from r4d_args.h, a hardcoded 128. So changing this alone re-groups the
# kernel and leaves the packer at 128, which is precisely the silently-wrong-output the single
# variable was supposed to rule out.
# CHANGING THE GROUP MEANS CHANGING BOTH, and the enum is the one that decides what is on disk.
set(RAD_W4_GROUP 128 CACHE STRING "libr4d quantised-weight group size (must match R4D_W4_GROUP in r4d_args.h)")

set(R4D_UNITS
  "r4d_attn_paged_h256_gqa6|"
  "r4d_attn_paged_h128_gqa4|"
  # Eight queries per KV head, at both head widths: the routed 35B-A3B is 16 over 2 at 256
  # and MiniCPM5-2B is 16 over 2 at 128. Same device code, different instantiations.
  "r4d_attn_paged_gqa8|"
  # Twelve, head_dim 256 only: Qwen3.8-Flash-Next's QSA layers are 24 over 2. Four lines that
  # set the ratio and include the unit above, which is the parameterised body.
  "r4d_attn_paged_h256_gqa12|"
  # Six, head_dim 256 and the same 4-token page: the same QSA layers at four ranks, where each KV
  # head is replicated across a rank pair. Again four lines that set the ratio and include the body.
  "r4d_attn_paged_h256_gqa6_bs4|"
  "r4d_attn_vit_h72_bf16|"
  # The vision tower's own small ops: layer norm, the learned-grid position embedding, and the
  # biased GEMM with a linear layer's activation and residual in its store.
  "r4d_vit_bf16|"
  "r4d_gdn_chunk_scan_k128_v128_c64_bf16|-mcumode"
  "r4d_gdn_conv_w4_h128_bf16|"
  "r4d_gdn_kkt_solve_k128_c64_bf16|"
  "r4d_gdn_recurrent_update_k128_v128_bf16_fp32state|"
  "r4d_gdn_gated_rmsnorm_h128_bf16|"
  # The gated residual, which is what a hyper-connection model has instead of a pre-norm and a
  # residual add. ONE unit: hc_read is two launches behind one entry point, because the mix
  # depends on every column of the low-rank waist and a kernel boundary is the grid-wide
  # barrier that says so.
  "r4d_hc_bf16|"
  "r4d_ngram_ids|"
  "r4d_ple_gate_bf16|"
  "r4d_ple_conv_bf16|"
  # The all-reduce family is ONE unit and not one per kernel: they share a single instance --
  # peer scratch, IPC-equivalent pointers, per-block flags, device sequence counters -- and one
  # rendezvous that establishes it across every rank before the first launch. Splitting the entry
  # points across the six kernel units would put that state in a header and give each unit its own
  # copy of it, which is the one thing it must not have.
  "r4d_ar_entry|"
  "r4d_ar_gather_2rank|"
  "r4d_ar_gather_Nrank|"
  "r4d_ar_oneshot_2rank_exact|"
  "r4d_ar_oneshot_2rank_wht6|"
  "r4d_ar_oneshot_Nrank_exact|"
  "r4d_ar_twoshot_Nrank_exact|"
  "r4d_ar_twoshot_Nrank_ti8|"
  "r4d_gemm_bf16_nt_m16|"
  "r4d_gemm_bf16_nt_m64|"
  "r4d_gemm_bf16_nt_tiled|"
  "r4d_gemm_w4a16_nt_m64|-DR4D_GEMM_W4_GROUP=${RAD_W4_GROUP}"
  "r4d_gemm_w8a16_nt_m64|-DR4D_GEMM_W8_GROUP=${RAD_W4_GROUP}"
  "r4d_gemm_w2a8_nt|-DR4D_GEMM_W2A8_GROUP=${RAD_W4_GROUP}"
  "r4d_gemm_w4a8_nt_m64|-DR4D_GEMM_W4A8_GROUP=${RAD_W4_GROUP}"
  "r4d_gemm_w8a8_nt_m64|-DR4D_GEMM_W8A8_GROUP=${RAD_W4_GROUP}"
  "r4d_gemm_w8a8_tiled|-DR4D_GEMM_W8A8_GROUP=${RAD_W4_GROUP}"
  "r4d_gemm_w4a8_tiled|-DR4D_GEMM_W4A8_GROUP=${RAD_W4_GROUP}"
  "r4d_gemm_w4a8_prefill|-DR4D_GEMM_W4A8_GROUP=${RAD_W4_GROUP}"
  "r4d_gemm_mxfp4a8_nt_m64|"
  # Block-scaled FP8 -- the format Qwen3.8-27B-FP8 ships in.
  # No -D: the 128x128 block is a property of the checkpoint on disk and not a quantiser choice
  # like RAD_W4_GROUP, so it is a constant in the source and the entry points reject a mismatch.
  "r4d_gemm_fp8a16_nt_m1|"
  "r4d_gemm_fp8a8|"
  "r4d_quant_act_fp8|"
  "r4d_fused_quant_fp8|"
  "r4d_gram_accum_fp8|"
  "r4d_qsa_block_key_bf16|"
  "r4d_qsa_score_bf16|"
  "r4d_qsa_select_bf16|"
  "r4d_qsa_tail_store_bf16|"
  "r4d_qsa_work_bf16|"
  "r4d_sample_chain_f32|"
  "r4d_model_bf16|"
  "r4d_sample_stages_f32|"
  "r4d_quant_act_i8|"
  "r4d_had_quant_act_i8|"
  "r4d_rmsnorm_had_quant_i8|"
  "r4d_ar_ln_had_quant_i8|"
  "r4d_gated_had_quant_i8|"
  "r4d_qk_norm_rope_gate|"
  "r4d_gdn_gated_norm_had_quant_i8|"
  "r4d_dequant_w4_bf16|"
  "r4d_rowtopk_bf16|"
  "r4d_dflash_conv_t2_g16_bf16|"
  "r4d_dflash2_prep_h128|"
  "r4d_dflash_select_bf16|"
  "r4d_fulldrafter|"
  # The routed mixture of experts: router, scatter, the grouped block-scaled fp8 GEMM and
  # gather, in one unit because the other three exist only to feed the GEMM.
  "r4d_moe|"
)

set(R4D_SRC "")
foreach(u ${R4D_UNITS})
  string(REPLACE "|" ";" parts "${u}")
  list(GET parts 0 unit)
  list(LENGTH parts n)
  set(extra "")
  if(n GREATER 1)
    list(GET parts 1 extra)
  endif()
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${unit}.hip")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "libr4d unit ${unit}.hip is missing from ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(extra)
    set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${extra}")
  endif()
  list(APPEND R4D_SRC "${src}")
endforeach()

# THE HOST SOURCES IN libr4d/ ARE DECLARATION. There is no launch adapter: every entry point
# takes `(const RadArgs*, RadStream)` and lives beside the kernel it launches, in libr4d.
# These are the plugin's statements ABOUT those kernels -- the row table, the weight layouts and
# their relayouts, the operand descriptions, the fusion claims -- and all of them are PURE HOST
# C++ that include neither r4d.h nor hip_runtime.h.
#
# That split is what lets the tables a transcription error hides in be compiled and read on a box
# with no ROCm, and it is why an operand description (answered before anything is allocated) and a
# declared fusion (a statement about the op vocabulary, answered with no tensors in hand) belong
# here rather than in a translation unit that needs a compiler for the card.
set(R4D_DECLARATION r4d_rows.cpp r4d_layout.cpp r4d_fp8_layout.cpp r4d_shapes.cpp r4d_fuse.cpp)

rad_add_plugin(libr4d kernels
  SOURCES     ${R4D_DECLARATION}
  HIP_SOURCES ${R4D_SRC}
  GPU_TARGETS ${R4D_GPU_TARGETS}
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR})

# The catalogue dump: prints every row and, on a card, calls each launch with a minimal valid
# input. This is how a transcription error in the constraint table is found -- otherwise it is
# invisible until a model silently picks the wrong kernel. It dlopens the plugin rather than
# linking it, because an undefined symbol in the .so is itself one of the things being tested.
add_executable(r4d_selftest ${CMAKE_CURRENT_SOURCE_DIR}/r4d_selftest.cpp)
# It dlopens the plugin but allocates the device buffers it feeds the kernels itself, so it needs
# the HIP runtime even though it links no kernel.
target_link_libraries(r4d_selftest PRIVATE radiance::abi ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME})
target_include_directories(r4d_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(r4d_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# libr4d is built at -std=c++17: at C++20 clang tightens the address-space check on
# __builtin_amdgcn_global_load_tr_b128_v8i16 in r4d_common.h, and C++17 is the configuration every
# cost stated in the kernels' comments was taken in.
set_target_properties(libr4d PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)

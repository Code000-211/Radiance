# Model-owned kernels and their host-domain reference plugin.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(mine LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  if(RAD_HIP_FOUND)
    enable_language(HIP)
  endif()
endif()

add_library(mine_ref_ops OBJECT dloop_ops.cpp dloop_full.cpp tree_gdn.cpp tree_gdn_fused_ref.cpp ple_tree.cpp
  ref_dloop.cpp ref_dloop_full.cpp ref_gdn_tree.cpp ref_ple_tree.cpp
  dflash_tree_select.cpp ref_dflash_tree.cpp mtp_tree_accum.cpp ref_mtp_tree_accum.cpp
  qsa_tree_commit.cpp qsa_tree_commit_ref.cpp mtp_scored_tree.cpp ref_mtp_scored_tree.cpp
  mtp_pruned_tree.cpp ref_mtp_pruned_tree.cpp mtp_overlay.cpp ref_mtp_overlay.cpp
  spechub.cpp ref_spechub.cpp attn_paged_tree.cpp ref_attn_paged_tree.cpp)
target_link_libraries(mine_ref_ops PRIVATE radiance::abi)
set_target_properties(mine_ref_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(mine_ref_ops PRIVATE -ffp-contract=off)
rad_add_plugin(mine_ref kernels SOURCES ref_rows.cpp $<TARGET_OBJECTS:mine_ref_ops>)
target_compile_options(mine_ref PRIVATE -ffp-contract=off)

if(RAD_HIP_FOUND)
  rad_add_plugin(mine kernels
    SOURCES rows.cpp tree_gdn.cpp ple_tree.cpp dloop_ops.cpp dloop_full.cpp dloop_rows.cpp
            hc_mean.cpp
            # DFlash2 experimental fusions: host schema/args/row beside the launcher below.
            dflash_conv_add.cpp dflash_k_norm_rope_store.cpp dflash_gated_bf16.cpp
            dflash_tree_select.cpp mtp_tree_accum.cpp qsa_tree_commit.cpp mtp_scored_tree.cpp
            mtp_pruned_tree.cpp mtp_overlay.cpp spechub.cpp attn_paged_tree.cpp
    HIP_SOURCES tree_gdn.hip ple_tree.hip dloop_kernels.hip dloop_tree.hip dloop_skinny.hip
                hc_mean.hip hc_up_wmma.hip shared_i8a8_down.hip
                dflash_conv_add.hip dflash_k_norm_rope_store.hip dflash_gated_bf16.hip
                dflash_tree_select.hip mtp_tree_accum.hip qsa_tree_commit.hip mtp_scored_tree.hip
                mtp_pruned_tree.hip mtp_overlay.hip spechub.hip attn_paged_tree.hip
    GPU_TARGETS gfx1201)
  target_compile_options(mine PRIVATE -ffp-contract=off)
  # The HC override reuses upstream helpers with hidden visibility; discard unused helpers.
  set_source_files_properties(hc_up_wmma.hip PROPERTIES
    COMPILE_OPTIONS "-ffunction-sections;-fdata-sections")
  target_link_options(mine PRIVATE -Wl,--gc-sections)
  # dloop_full.cpp carries the device row table next to the schemas: this is what tells it the HIP
  # entry points exist in this build. Without it the plugin would publish the schemas and the host
  # oracle while quietly registering no device row at all, which selection reads as "no kernel".
  target_compile_definitions(mine PRIVATE RAD_DLOOP_FULL_DEVICE=1)
endif()

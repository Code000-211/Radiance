# libbolegdn -- the Bole tree-GDN state update, out of tree and opt-in.
#
# Two device rows on one new op (`gdn_tree_bole`), for gfx1201 only. Nothing in the engine selects
# this plugin's op: a caller reaches it by declaring the op and naming this library in --kernels.
#
# The device source needs libr4d's private headers (r4d_common.h, r4d_gdn_wmma.h, r4d_q8.h,
# r4d_gdn_conv_tap.h) because every rounding boundary in it is libr4d's, restated. They are private
# include directories, not symbols, so the plugin still links nothing of the engine's -- but it is
# built in tree rather than through find_package(radiance) for that reason. See README.md.
if(NOT COMMAND rad_add_plugin)
  message(FATAL_ERROR
    "libbolegdn includes libr4d's private device headers and is built in radiance's tree; configure "
    "it with the tree, not standalone against an installed prefix.")
endif()

rad_add_plugin(libbolegdn kernels
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/bole_rows.cpp
  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/bole_tree.hip
  INCLUDES ${CMAKE_SOURCE_DIR}/libr4d)

# The comparison the row's byte claim rests on is a floating-point one: `a*b + c` must round twice,
# as libr4d's own device targets are compiled (cmake/RadAddPlugin.cmake, rad_hipify).
target_compile_options(libbolegdn PRIVATE -ffp-contract=off)

# One static library. The engine is monolithic on purpose (spec §head): the plugin boundary is
# where substitution happens, and inside it there is nothing to gain from .so seams.
#
# Sources are globbed per subdirectory with CONFIGURE_DEPENDS so that adding a file to a component
# does not mean editing a file another component owns. Re-run cmake if a new file does not appear.
set(RAD_CORE_DIRS util device plugin build runtime mem sched place sample text server format mm)

set(RAD_CORE_SRC "")
foreach(d ${RAD_CORE_DIRS})
  file(GLOB_RECURSE _s CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/${d}/*.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/${d}/*.c)
  list(APPEND RAD_CORE_SRC ${_s})
endforeach()
# The top level, MINUS main.cpp. main() belongs to the `radiance` executable and not to the
# library: with the archive linked whole -- which every host of an architecture plugin must do, see
# tools/CMakeLists.txt -- a main() inside it collides with each tool's own. A plain archive link
# hides this, because the tools reference nothing in main.cpp.o and the linker skips it.
file(GLOB _top CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM _top ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
list(APPEND RAD_CORE_SRC ${_top})

add_library(rad_core STATIC ${RAD_CORE_SRC})
target_link_libraries(rad_core PUBLIC rad_abi ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_definitions(rad_core PUBLIC RAD_VERSION="${PROJECT_VERSION}")
target_include_directories(rad_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/abi
  ${CMAKE_SOURCE_DIR}/vendor)

if(TARGET rad_vendor)
  target_link_libraries(rad_core PUBLIC rad_vendor)
endif()

if(RAD_FFMPEG_FOUND)
  target_compile_definitions(rad_core PRIVATE RAD_HAVE_FFMPEG=1)
  target_link_libraries(rad_core PUBLIC PkgConfig::RAD_FFMPEG)
endif()

if(RAD_HIP_FOUND)
  target_compile_definitions(rad_core PUBLIC RAD_HAVE_HIP=1)
  target_link_libraries(rad_core PUBLIC ${RAD_HIP_RUNTIME} ${RAD_HSA_RUNTIME})
  # THE AQL BACKEND'S OWN KERNELS: copies, fills and ordered winning-branch KV compaction.
  # The host-compiled device dispatcher selects these only when the runtime backend is AQL.
  target_sources(rad_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device/aql_blit.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/device/aql_blit.hip PROPERTIES LANGUAGE HIP)
  # THE STEP'S OWN BOOKKEEPING ON THE CARD: acceptance and the next pass's positions (sched/advance.h).
  target_sources(rad_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sched/advance.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/sched/advance.hip PROPERTIES LANGUAGE HIP)
  # WEIGHT-TABLE ENTRIES WRITTEN IN STREAM ORDER, their values carried in the arguments (runtime/wtab.h).
  target_sources(rad_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/wtab.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/runtime/wtab.hip PROPERTIES LANGUAGE HIP)
  # CED LOW-RANK GEMV PROJECTION KERNEL FOR PREDICTIVE MOE PREFETCHING (place/ced_gemv.h).
  target_sources(rad_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/place/ced_gemv.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/place/ced_gemv.hip PROPERTIES LANGUAGE HIP)
  # THE CORE'S OWN KERNELS ARE BUILT FOR EVERY AMD GPU FAMILY, not only for the cards this build
  # targets. They are what every copy and fill on a card runs, so an engine without a code object
  # for a card cannot start on it -- whatever kernel plugin is installed for that card, and a plugin
  # for a card the build machine does not have is exactly what a third party ships. The generic
  # targets cost three small files a family; the loader prefers the exact chip where both exist
  # (core/device/aql_code.cpp), so a targeted card still runs code built for it.
  set(RAD_CORE_GPU_TARGETS ${RAD_GPU_TARGETS}
      gfx9-generic gfx9-4-generic gfx10-1-generic gfx10-3-generic gfx11-generic gfx12-generic)
  list(REMOVE_DUPLICATES RAD_CORE_GPU_TARGETS)
  set_target_properties(rad_core PROPERTIES HIP_ARCHITECTURES "${RAD_CORE_GPU_TARGETS}")
  # EVERY EXECUTABLE THAT LINKS THE CORE EXPORTS THE HIP ENTRY POINTS THE BACKEND DEFINES. A kernel
  # plugin is dlopen'ed and its hipLaunchKernel binds to the first definition in the global scope,
  # which is the executable's only if the executable exports it. Without this a plugin would launch
  # through HIP onto a queue HIP does not own; aql.cpp checks at startup and refuses to run.
  foreach(sym hipLaunchKernel hipMemcpy2DAsync hipMemcpyAsync hipMemsetAsync hipMemset hipStreamSynchronize
              hipEventRecord hipStreamWaitEvent hipGetLastError hipPeekAtLastError
              __hipRegisterFatBinary __hipRegisterFunction __hipUnregisterFatBinary)
    target_link_options(rad_core INTERFACE "LINKER:--export-dynamic-symbol=${sym}")
  endforeach()
endif()

# The engine binary. core/main.cpp is owned by the engine component.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
  add_executable(radiance main.cpp)
  target_link_libraries(radiance PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
  # ENABLE_EXPORTS: the architecture plugin binds the declare-phase C ABI back to this binary at
  # dlopen. See the same note in tools/CMakeLists.txt.
  set_target_properties(radiance PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ENABLE_EXPORTS ON)
  # Installed beside the tools, so a prefix that holds rad-convert and rad-kbench also holds the
  # binary they exist to feed. RUNTIME and not a bare DESTINATION so that ENABLE_EXPORTS -- which
  # is what lets an architecture plugin bind the declare-phase C ABI back to this binary at
  # dlopen, see the note above -- is carried by the same rule that carries the executable.
  install(TARGETS radiance RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# The plugin layer's test plugins. They are .so files rather than part of rad_core, so they need a
# CMakeLists of their own -- and it needs to be reached. (Owned by core/plugin/.)
#
# RAD_PLUGIN_INSTALL=OFF for that whole subtree, and this is the one place in the build that turns
# it off. rad_add_plugin installs what it builds, because a plugin that is built and not installed
# leaves $RADIANCE_HOME working only when it points at a build directory. But the .so files under
# core/plugin/testplugin exist only for tests -- to be refused, to conflict with each other, to
# report a stale ABI, or to stand in for a card in one test -- and none is a kernel a real run may
# select. Shipping those into a real prefix is the same mistake testplugin/CMakeLists.txt already
# refuses to make in the build tree -- "a test plugin that a real run could select is a fallback
# nobody knows about" -- so the opt-out is set here, where the subtree is reached, rather than in a
# file that component owns.
set(RAD_PLUGIN_INSTALL OFF)
add_subdirectory(plugin)

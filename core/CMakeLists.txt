# One static library. The engine is monolithic on purpose (spec §head): substitution happens at the
# plugin boundary, so .so seams inside it buy nothing.
#
# Sources are globbed per subdirectory with CONFIGURE_DEPENDS. Re-run cmake if a new file does not
# appear.
set(RAD_CORE_DIRS util device plugin build runtime mem sched place sample text server format mm diag gen
                  lora)

set(RAD_CORE_SRC "")
foreach(d ${RAD_CORE_DIRS})
  file(GLOB_RECURSE _s CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/${d}/*.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/${d}/*.c)
  list(APPEND RAD_CORE_SRC ${_s})
endforeach()
# The top level minus main.cpp, which belongs to the `radiance` executable. Every architecture plugin
# host links the archive whole (tools/CMakeLists.txt), so a main() inside it would collide with each
# tool's own.
file(GLOB _top CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM _top ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
list(APPEND RAD_CORE_SRC ${_top})

add_library(rad_core STATIC ${RAD_CORE_SRC})
target_link_libraries(rad_core PUBLIC rad_abi ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_definitions(rad_core PUBLIC RAD_VERSION="${PROJECT_VERSION}")
if(RAD_BUILD_TESTS)
  # The test suite's only seam inside core: a weak "fail the next build" hook in batch.cpp, so the
  # scheduler's "a pass was owed and could not be built" path is reachable without a runtime fault
  # switch. A build configured without tests compiles neither the call nor the definition.
  target_compile_definitions(rad_core PRIVATE RAD_TEST_HOOKS=1)
endif()
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
  # Run the ROCm the tree was built against. The HIP driver reports the runtime's directory as an
  # implicit link directory, so CMake writes no RPATH for it and the loader cache would pick the
  # system's libamdhip64.so.7. A linker flag instead of CMake's RPATH, because install rewrites the
  # latter and, next to ROCm 10 clang's entry, leaves an empty element (the current directory).
  get_filename_component(RAD_HIP_LIBDIR "${RAD_HIP_RUNTIME}" DIRECTORY)
  target_link_options(rad_core INTERFACE "LINKER:-rpath,${RAD_HIP_LIBDIR}")
  # AQL backend kernels: the copies and fills behind rad_memcpy_async and rad_memset_async.
  target_sources(rad_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device/aql_blit.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/device/aql_blit.hip PROPERTIES LANGUAGE HIP)
  # Step bookkeeping on the card: acceptance and the next pass's positions (sched/advance.h).
  target_sources(rad_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sched/advance.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/sched/advance.hip PROPERTIES LANGUAGE HIP)
  # Weight-table entries written in stream order, values passed as arguments (runtime/wtab.h).
  target_sources(rad_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/wtab.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/runtime/wtab.hip PROPERTIES LANGUAGE HIP)
  # Next-layer expert prefetch: prediction, plan and table patches (runtime/xprefetch.h).
  target_sources(rad_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/xprefetch.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/runtime/xprefetch.hip PROPERTIES LANGUAGE HIP)
  # Core kernels are built for every AMD GPU family, not only RAD_GPU_TARGETS: every copy and fill
  # runs them, so without a code object the engine cannot start on a card even when a third-party
  # kernel plugin supports it. The loader prefers the exact chip when both exist
  # (core/device/aql_code.cpp).
  set(RAD_CORE_GPU_TARGETS ${RAD_GPU_TARGETS}
      gfx9-generic gfx9-4-generic gfx10-1-generic gfx10-3-generic gfx11-generic gfx12-generic)
  list(REMOVE_DUPLICATES RAD_CORE_GPU_TARGETS)
  set_target_properties(rad_core PROPERTIES HIP_ARCHITECTURES "${RAD_CORE_GPU_TARGETS}")
  rad_hip_repeatable(rad_core)
  # Executables linking the core export the HIP entry points the backend defines, so a dlopen'ed
  # kernel plugin binds to them instead of real HIP (which does not own the queues). aql.cpp checks
  # this at startup and refuses to run without it.
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
  # dlopen (tools/CMakeLists.txt).
  set_target_properties(radiance PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ENABLE_EXPORTS ON)
  # Installed beside the tools. RUNTIME rather than a bare DESTINATION, so the rule that carries
  # the executable also carries ENABLE_EXPORTS.
  install(TARGETS radiance RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# The plugin layer's test plugins (.so files, so they need their own CMakeLists).
#
# RAD_PLUGIN_INSTALL=OFF for this subtree only. rad_add_plugin normally installs what it builds,
# but these plugins exist only for tests and must never be selectable from a real prefix.
set(RAD_PLUGIN_INSTALL OFF)
add_subdirectory(plugin)

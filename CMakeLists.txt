cmake_minimum_required(VERSION 3.16)
project(ab3d1 C)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Emscripten: default single-threaded wasm (plain `emcmake cmake ..` + `cmake --build .`). Native: default threaded renderer.
# Use CMAKE_SYSTEM_NAME (not EMSCRIPTEN): reliable as soon as the Emscripten toolchain has run.
if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set(_AB3D_NO_THREADS_DEFAULT ON)
else()
  set(_AB3D_NO_THREADS_DEFAULT OFF)
endif()
option(AB3D_NO_THREADS "Build without renderer threading support (render_threads is ignored at runtime)" ${_AB3D_NO_THREADS_DEFAULT})
unset(_AB3D_NO_THREADS_DEFAULT)

# -----------------------------------------------------------------------
# Data staging (build/data)
#
# All intermediate sync/conversion is done in the build tree under data/.
# Runtime copies then come from build/data to the executable-local data/.
# -----------------------------------------------------------------------
set(AB3D_STAGE_DATA_DIR "${CMAKE_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${AB3D_STAGE_DATA_DIR}")
file(MAKE_DIRECTORY "${AB3D_STAGE_DATA_DIR}/vectorobjects")
file(MAKE_DIRECTORY "${AB3D_STAGE_DATA_DIR}/sounds")
file(MAKE_DIRECTORY "${AB3D_STAGE_DATA_DIR}/includes")
file(MAKE_DIRECTORY "${AB3D_STAGE_DATA_DIR}/levels")
file(MAKE_DIRECTORY "${AB3D_STAGE_DATA_DIR}/gfx")
file(MAKE_DIRECTORY "${AB3D_STAGE_DATA_DIR}/pal")
file(MAKE_DIRECTORY "${AB3D_STAGE_DATA_DIR}/helper")
file(MAKE_DIRECTORY "${AB3D_STAGE_DATA_DIR}/texturemaps")

find_package(Python3 QUIET)

# Stage palettes at configure time so sprite_palettes_data.h can be generated.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/pal")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/amiga/pal" DESTINATION "${AB3D_STAGE_DATA_DIR}")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/pal")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/pal" DESTINATION "${AB3D_STAGE_DATA_DIR}")
endif()

# -----------------------------------------------------------------------
# SDL2 (FetchContent on native; Emscripten uses -sUSE_SDL=2)
# -----------------------------------------------------------------------
if(NOT EMSCRIPTEN)
  include(FetchContent)
  set(SDL_SHARED OFF CACHE BOOL "" FORCE)
  set(SDL_STATIC ON CACHE BOOL "" FORCE)
  set(SDL_TEST OFF CACHE BOOL "" FORCE)
  set(SDL2_DISABLE_INSTALL ON CACHE BOOL "" FORCE)

  FetchContent_Declare(
      SDL2
      GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
      GIT_TAG        release-2.30.11
      GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(SDL2)
endif()

# All source files
set(SOURCES
    src/main.c
    src/game_state.c
    src/game_data.c
    src/math_tables.c
    src/ai.c
    src/movement.c
    src/level.c
    src/objects.c
    src/visibility.c
    src/control_loop.c
    src/game_loop.c
    src/player.c
    src/renderer.c
    src/renderer_3dobj.c
    src/display.c
    src/input.c
    src/audio.c
    src/io.c
    src/settings.c
    src/logging.c
    src/sb_decompress.c
    src/sprite_palettes.c
)

file(GLOB_RECURSE PROJECT_HEADERS
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
)

add_executable(ab3d1 ${SOURCES})
if(EMSCRIPTEN)
  target_sources(ab3d1 PRIVATE src/emscripten_loop.c)
  if(AB3D_NO_THREADS)
    target_compile_definitions(ab3d1 PRIVATE AB3D_NO_THREADS)
  endif()
elseif(AB3D_NO_THREADS)
  target_compile_definitions(ab3d1 PRIVATE AB3D_NO_THREADS)
endif()

target_sources(ab3d1 PRIVATE ${PROJECT_HEADERS})
set_source_files_properties(${PROJECT_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)

target_include_directories(ab3d1 PRIVATE src)

# Generate sprite_palettes_data.h from staged build/data/pal/*.pal (no runtime .pal load)
set(PAL_DIR "${AB3D_STAGE_DATA_DIR}/pal")
set(PAL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/sprite_palettes_data.h")
if(EXISTS "${PAL_DIR}")
  if(Python3_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/pal_to_header.py"
              --pal-dir "${PAL_DIR}"
              --output "${PAL_HEADER}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE PAL_RESULT
      OUTPUT_VARIABLE PAL_OUT
      ERROR_VARIABLE PAL_ERR
    )
    if(PAL_RESULT EQUAL 0)
      message(STATUS "Sprite palettes: ${PAL_OUT}")
    else()
      message(WARNING "Sprite palette generation failed: ${PAL_ERR}")
    endif()
  endif()
endif()
if(NOT EXISTS "${PAL_HEADER}")
  message(FATAL_ERROR "sprite_palettes_data.h not found. Put .pal files in amiga/pal or amiga/data/pal and run: python tools/pal_to_header.py --pal-dir \"${PAL_DIR}\" --output \"${PAL_HEADER}\"")
endif()

# Link SDL2 (no SDL2main); Emscripten links SDL2 via -sUSE_SDL=2
if(NOT EMSCRIPTEN)
  target_link_libraries(ab3d1 PRIVATE SDL2-static)
endif()
target_include_directories(ab3d1 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# Incremental build: copy ab3d.ini.template (repo root) -> ab3d.ini beside the exe only if ab3d.ini is missing
# (so local edits next to the exe are kept). Full Clean/Rebuild: ab3d.ini is listed in ADDITIONAL_CLEAN_FILES,
# so Clean deletes it and the next Build recreates it from the template.
add_custom_command(TARGET ab3d1 POST_BUILD
  COMMAND ${CMAKE_COMMAND}
    -DDEST_DIR=$<TARGET_FILE_DIR:ab3d1>
    -DSRC_FILE=${CMAKE_CURRENT_SOURCE_DIR}/ab3d.ini.template
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_ab3d_ini_if_missing.cmake
  COMMENT "Ensure ab3d.ini beside ab3d1 (from template if missing)"
)
add_custom_command(TARGET ab3d1 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/README.txt"
          "$<TARGET_FILE_DIR:ab3d1>/README.txt"
  COMMENT "Copy README.txt beside ab3d1"
)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fonts")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/fonts"
            "$<TARGET_FILE_DIR:ab3d1>/fonts"
    COMMENT "Copy fonts -> output fonts/"
  )
endif()
function(ab3d_stage_dir source_dir subdir)
  if(EXISTS "${source_dir}")
    add_custom_command(TARGET ab3d1 POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E remove_directory "${AB3D_STAGE_DATA_DIR}/${subdir}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${AB3D_STAGE_DATA_DIR}/${subdir}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              "${source_dir}"
              "${AB3D_STAGE_DATA_DIR}/${subdir}"
      COMMENT "Stage ${subdir}: ${source_dir} -> ${AB3D_STAGE_DATA_DIR}/${subdir}"
    )
  endif()
endfunction()

function(ab3d_copy_stage_dir subdir)
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ab3d1>/data/${subdir}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${AB3D_STAGE_DATA_DIR}/${subdir}"
            "$<TARGET_FILE_DIR:ab3d1>/data/${subdir}"
    COMMENT "Copy staged ${subdir} -> output data/${subdir}"
  )
endfunction()

ab3d_stage_dir("${CMAKE_CURRENT_SOURCE_DIR}/amiga/vectorobjects" "vectorobjects")
ab3d_copy_stage_dir("vectorobjects")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/sounds")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${AB3D_STAGE_DATA_DIR}/sounds"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${AB3D_STAGE_DATA_DIR}/sounds"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/amiga/sounds"
            "${AB3D_STAGE_DATA_DIR}/sounds"
    COMMENT "Stage sounds: amiga/sounds -> build/data/sounds"
  )
endif()
ab3d_copy_stage_dir("sounds")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/sounds" AND Python3_FOUND)
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/raw_to_wav.py"
            --sounds-dir "$<TARGET_FILE_DIR:ab3d1>/data/sounds"
            --wav-only
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generate/update .wav files in output data/sounds and keep wav-only assets"
  )
endif()

ab3d_stage_dir("${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/includes" "includes")
ab3d_copy_stage_dir("includes")

ab3d_stage_dir("${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/levels" "levels")
ab3d_copy_stage_dir("levels")

ab3d_stage_dir("${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/gfx" "gfx")
ab3d_copy_stage_dir("gfx")

ab3d_stage_dir("${CMAKE_CURRENT_SOURCE_DIR}/amiga/pal" "pal")
ab3d_stage_dir("${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/pal" "pal")
ab3d_copy_stage_dir("pal")

ab3d_stage_dir("${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/helper" "helper")
ab3d_copy_stage_dir("helper")

ab3d_stage_dir("${CMAKE_CURRENT_SOURCE_DIR}/amiga/texturemaps" "texturemaps")
ab3d_copy_stage_dir("texturemaps")
set_target_properties(ab3d1 PROPERTIES
  ADDITIONAL_CLEAN_FILES "$<TARGET_FILE_DIR:ab3d1>/ab3d.ini"
)

# Release build: start fullscreen desktop with fixed renderer (see display.c)
target_compile_definitions(ab3d1 PRIVATE $<$<CONFIG:Release>:AB3D_RELEASE>)

# Platform-specific settings
if(EMSCRIPTEN)
  # Required at compile time too; otherwise emcc uses fakesdl/SDL.h stubs.
  target_compile_options(ab3d1 PRIVATE "SHELL:-sUSE_SDL=2")
  if(NOT AB3D_NO_THREADS)
    target_compile_options(ab3d1 PRIVATE "SHELL:-pthread")
  endif()
  set(AB3D_EMPRELOAD "${CMAKE_BINARY_DIR}/em_preload")
  file(MAKE_DIRECTORY "${AB3D_EMPRELOAD}")
  # Re-run preload when any staged asset changes (recursive so new .wad/.ptr files are not missed).
  file(GLOB_RECURSE _AB3D_AMIGA_PRELOAD LIST_DIRECTORIES false CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/amiga/vectorobjects/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/amiga/texturemaps/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/amiga/pal/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/amiga/sounds/*")
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/em_preload/.stamp"
    COMMAND ${CMAKE_COMMAND}
      "-DAB3D_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DAB3D_EMPRELOAD=${AB3D_EMPRELOAD}"
      "-DAB3D_BUILD_DIR=${CMAKE_BINARY_DIR}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/emscripten_sync_data.cmake
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/emscripten_sync_data.cmake"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/raw_to_wav.py"
      ${_AB3D_AMIGA_PRELOAD}
    COMMENT "Stage em_preload/ for WebAssembly (MEMFS)"
    VERBATIM
  )
  add_custom_target(ab3d_em_preload_data DEPENDS "${CMAKE_BINARY_DIR}/em_preload/.stamp")
  add_dependencies(ab3d1 ab3d_em_preload_data)
  if(NOT AB3D_NO_THREADS)
    set(AB3D_EXPECTS_CROSS_ORIGIN_ISOLATION "true")
  else()
    set(AB3D_EXPECTS_CROSS_ORIGIN_ISOLATION "false")
  endif()
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/shell.html.in"
    "${CMAKE_BINARY_DIR}/emscripten_shell.html"
    @ONLY
  )
  file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/emscripten_shell.html" AB3D_SHELL_HTML_POSIX)
  string(REPLACE "\\" "/" AB3D_SHELL_HTML_POSIX "${AB3D_SHELL_HTML_POSIX}")
  file(TO_CMAKE_PATH "${AB3D_EMPRELOAD}" AB3D_EMPRELOAD_POSIX)
  string(REPLACE "\\" "/" AB3D_EMPRELOAD_POSIX "${AB3D_EMPRELOAD_POSIX}")
  target_link_options(ab3d1 PRIVATE
    "SHELL:--shell-file=${AB3D_SHELL_HTML_POSIX}"
    "SHELL:-sUSE_SDL=2"
    "SHELL:-sFULL_ES3=1"
    "SHELL:-sMIN_WEBGL_VERSION=2"
    "SHELL:-sMAX_WEBGL_VERSION=2"
    "SHELL:-sALLOW_MEMORY_GROWTH=1"
    "SHELL:-sINITIAL_MEMORY=134217728"
    "SHELL:--preload-file=${AB3D_EMPRELOAD_POSIX}@/"
  )
  if(NOT AB3D_NO_THREADS)
    target_link_options(ab3d1 PRIVATE
      "SHELL:-pthread"
      "SHELL:-sUSE_PTHREADS=1"
      # Pre-spawn workers so SDL_CreateThread does not wait on first frame.
      "SHELL:-sPTHREAD_POOL_SIZE=16"
      # Gate before wasm/workers: Module.preRun is too late (pthread postMessage needs crossOriginIsolated).
      "SHELL:--pre-js=${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten_pthread_gate.js"
    )
    message(STATUS "Emscripten: pthread renderer workers enabled; host with COOP/COEP (SharedArrayBuffer)")
  else()
    # Single-threaded: force non-shared Wasm memory (no SharedArrayBuffer for the module).
    target_compile_options(ab3d1 PRIVATE "SHELL:-sSHARED_MEMORY=0")
    target_link_options(ab3d1 PRIVATE "SHELL:-sSHARED_MEMORY=0")
    message(STATUS "Emscripten: single-threaded build; -sSHARED_MEMORY=0 (no SharedArrayBuffer for wasm)")
  endif()
  set_target_properties(ab3d1 PROPERTIES
    OUTPUT_NAME "index"
    SUFFIX ".html"
  )
  if(NOT AB3D_NO_THREADS)
    add_custom_command(TARGET ab3d1 POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CMAKE_CURRENT_SOURCE_DIR}/deployment/cloudflare/_headers"
              "$<TARGET_FILE_DIR:ab3d1>/_headers"
      COMMENT "Copy Cloudflare Pages _headers (COOP/COEP) beside index.html"
    )
  endif()
elseif(WIN32)
    # Windows: link winmm for SDL2
    target_link_libraries(ab3d1 PRIVATE winmm imm32 version setupapi)
elseif(UNIX)
    target_link_libraries(ab3d1 PRIVATE m)
endif()

# Debug build by default for development
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Force WINDOWS subsystem on Windows (no DOS console; logs go to ab3d.log)
if(MSVC AND NOT EMSCRIPTEN)
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT ab3d1)
    set_property(TARGET ab3d1 PROPERTY VS_DPI_AWARE OFF)
    target_link_options(ab3d1 PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
endif()

# Warnings
if(EMSCRIPTEN)
    target_compile_options(ab3d1 PRIVATE -Wall -Wextra -Wno-unused-parameter)
elseif(MSVC)
    target_compile_options(ab3d1 PRIVATE /W4)
    target_compile_definitions(ab3d1 PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(ab3d1 PRIVATE -Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------
# Standalone level parser (Amiga format only; outputs door/switch/lift logs)
# -----------------------------------------------------------------------
if(NOT EMSCRIPTEN)
add_executable(parse_level tools/parse_level.c src/sb_decompress.c)
target_include_directories(parse_level PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(parse_level PROPERTIES C_STANDARD 11)
target_compile_definitions(parse_level PRIVATE AB3D_STANDALONE_TOOL)
if(MSVC)
    target_compile_definitions(parse_level PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
endif()

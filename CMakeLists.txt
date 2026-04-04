cmake_minimum_required(VERSION 3.16)
project(ab3d1 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
option(AB3D_NO_THREADS "Build without renderer threading support (render_threads is ignored at runtime)" OFF)

# -----------------------------------------------------------------------
# Data: ensure data/pal exists and copy from amiga/pal and amiga/data/pal
# -----------------------------------------------------------------------
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/pal")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/pal")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/amiga/pal" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/pal")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/pal" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()

# Data: use amiga/data/gfx as source and sync into data/gfx.
# Keep backfile/backpal mirrored next to repo-root includes/ for io_load_sky search path.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/gfx")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/includes")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/gfx")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/gfx" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/gfx/backfile")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/gfx/backfile" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/includes")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/gfx/backpal")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/gfx/backpal" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/includes")
endif()

# -----------------------------------------------------------------------
# Data: use amiga/sounds as source; generate .wav into data/sounds
# -----------------------------------------------------------------------
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/sounds")
find_package(Python3 QUIET)

# -----------------------------------------------------------------------
# Data: use amiga/data/includes as source and sync into data/includes
# -----------------------------------------------------------------------
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/includes")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/includes")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/includes" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()

# -----------------------------------------------------------------------
# Data: use amiga/data/levels as source and sync into data/levels
# -----------------------------------------------------------------------
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/levels")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/levels")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/levels" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()

# -----------------------------------------------------------------------
# Data: vector objects are copied to the build output (beside the exe)
# from amiga/vectorobjects.
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# SDL2 (auto-download via FetchContent)
# -----------------------------------------------------------------------
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
    src/sb_decompress.c
    src/sprite_palettes.c
)

file(GLOB_RECURSE PROJECT_HEADERS
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
)

add_executable(ab3d1 ${SOURCES})
if(AB3D_NO_THREADS)
  target_compile_definitions(ab3d1 PRIVATE AB3D_NO_THREADS)
endif()

target_sources(ab3d1 PRIVATE ${PROJECT_HEADERS})
set_source_files_properties(${PROJECT_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)

target_include_directories(ab3d1 PRIVATE src)

# Generate sprite_palettes_data.h from data/pal/*.pal (no runtime .pal load)
set(PAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/pal")
set(PAL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/sprite_palettes_data.h")
if(EXISTS "${PAL_DIR}")
  if(Python3_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/pal_to_header.py"
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
  message(FATAL_ERROR "sprite_palettes_data.h not found. Put .pal files in data/pal and run: python tools/pal_to_header.py")
endif()

# Link SDL2 (no SDL2main - we use CONSOLE subsystem for printf output)
target_link_libraries(ab3d1 PRIVATE SDL2-static)

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
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/vectorobjects")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ab3d1>/data/vectorobjects"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/amiga/vectorobjects"
            "$<TARGET_FILE_DIR:ab3d1>/data/vectorobjects"
    COMMENT "Copy amiga/vectorobjects -> build data/vectorobjects"
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/sounds")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/data/sounds"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/amiga/sounds"
            "${CMAKE_CURRENT_SOURCE_DIR}/data/sounds"
    COMMENT "Sync amiga/sounds -> data/sounds"
  )
  if(Python3_FOUND)
    add_custom_command(TARGET ab3d1 POST_BUILD
      COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/raw_to_wav.py"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMENT "Generate .wav files in data/sounds from raw Amiga sounds"
    )
  endif()
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ab3d1>/data/sounds"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/data/sounds"
            "$<TARGET_FILE_DIR:ab3d1>/data/sounds"
    COMMENT "Copy data/sounds -> build data/sounds"
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/includes")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ab3d1>/data/includes"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/includes"
            "$<TARGET_FILE_DIR:ab3d1>/data/includes"
    COMMENT "Copy amiga/data/includes -> build data/includes"
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/levels")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ab3d1>/data/levels"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/levels"
            "$<TARGET_FILE_DIR:ab3d1>/data/levels"
    COMMENT "Copy amiga/data/levels -> build data/levels"
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/gfx")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ab3d1>/data/gfx"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/gfx"
            "$<TARGET_FILE_DIR:ab3d1>/data/gfx"
    COMMENT "Copy amiga/data/gfx -> build data/gfx"
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/pal")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ab3d1>/data/pal"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/pal"
            "$<TARGET_FILE_DIR:ab3d1>/data/pal"
    COMMENT "Copy amiga/data/pal -> build data/pal"
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/helper")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ab3d1>/data/helper"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/amiga/data/helper"
            "$<TARGET_FILE_DIR:ab3d1>/data/helper"
    COMMENT "Copy amiga/data/helper -> build data/helper"
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/amiga/texturemaps")
  add_custom_command(TARGET ab3d1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ab3d1>/data/texturemaps"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/amiga/texturemaps"
            "$<TARGET_FILE_DIR:ab3d1>/data/texturemaps"
    COMMENT "Copy amiga/texturemaps -> build data/texturemaps"
  )
endif()
set_target_properties(ab3d1 PROPERTIES
  ADDITIONAL_CLEAN_FILES "$<TARGET_FILE_DIR:ab3d1>/ab3d.ini"
)

# Release build: start fullscreen desktop with fixed renderer (see display.c)
target_compile_definitions(ab3d1 PRIVATE $<$<CONFIG:Release>:AB3D_RELEASE>)

# Platform-specific settings
if(WIN32)
    # Windows: link winmm for SDL2
    target_link_libraries(ab3d1 PRIVATE winmm imm32 version setupapi)
elseif(UNIX)
    target_link_libraries(ab3d1 PRIVATE m)
endif()

# Debug build by default for development
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Force CONSOLE subsystem on Windows (override SDL2's WINDOWS subsystem)
if(MSVC)
    target_link_options(ab3d1 PRIVATE /SUBSYSTEM:CONSOLE)
endif()

# Warnings
if(MSVC)
    target_compile_options(ab3d1 PRIVATE /W4)
    target_compile_definitions(ab3d1 PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(ab3d1 PRIVATE -Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------
# Standalone level parser (Amiga format only; outputs door/switch/lift logs)
# -----------------------------------------------------------------------
add_executable(parse_level tools/parse_level.c src/sb_decompress.c)
target_include_directories(parse_level PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(parse_level PROPERTIES C_STANDARD 11)
if(MSVC)
    target_compile_definitions(parse_level PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

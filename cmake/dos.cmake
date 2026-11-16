file(TO_CMAKE_PATH "${DOS_SDL_ROOT}" DOS_SDL_ROOT)
if(NOT EXISTS "${DOS_SDL_ROOT}/SDL-build/libSDL3.a")
  message(FATAL_ERROR "Run tools/setup_dos.ps1 to build the pinned DOS SDL dependency")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/game_sources.cmake")
list(REMOVE_ITEM SOURCES src/audio.c)
add_executable(ab3d1 ${SOURCES} src/platform/dos/audio_dos.c src/platform/dos/platform_dos.c
  src/platform/dos/ui_dos.c src/platform/dos/ui_surface_dos.c src/platform/dos/raster_dos.c
  src/platform/dos/text_dos.c)
target_include_directories(ab3d1 BEFORE PRIVATE src/platform/dos)
target_include_directories(ab3d1 PRIVATE src third_party "${DOS_SDL_ROOT}/SDL/include" "${CMAKE_BINARY_DIR}/generated")
option(AB3D_DOS_INDEXED "Direct 8-bit DOS rendering with source-derived per-level palettes" OFF)
target_compile_definitions(ab3d1 PRIVATE AB3D_DOS AB3D_NO_THREADS AB3D_RELEASE
  AB3D_ENABLE_FLOOR_COL_FAST=1 SKY_BILINEAR_FILTER=0 AB3D_CW_COL_MAJOR=0 SDL_MAIN_HANDLED)
if(AB3D_DOS_INDEXED)
  target_compile_definitions(ab3d1 PRIVATE AB3D_DOS_INDEXED)
  target_sources(ab3d1 PRIVATE src/platform/dos/palette_dos.c)
else()
  target_compile_definitions(ab3d1 PRIVATE AB3D_NATIVE_RGB565)
  target_sources(ab3d1 PRIVATE src/platform/dos/sprite_alpha_dos.c)
endif()
target_compile_options(ab3d1 PRIVATE -march=i486 -mtune=i486 -O3 -flto -ffunction-sections -fdata-sections -fomit-frame-pointer -fno-strict-aliasing)
# Keep the orchestration/clipping unit compact enough for 4 MB. Pixel loops
# live in raster_dos.c and retain -O3; -O2 avoids duplicating large setup paths.
set_source_files_properties(src/renderer.c PROPERTIES COMPILE_OPTIONS "-O2")
# Make room for the coprocessor emulator within the existing 4 MB budget.
# Keep startup, loading, control/UI and gameplay dispatch compact. World
# drawing, movement and mixing retain their normal optimization level.
set_source_files_properties(src/main.c src/display.c src/input.c src/settings.c
  src/renderer_3dobj.c src/ai.c src/objects.c src/player.c
  src/level_setup.c src/game_data.c src/io.c
  src/control_loop.c src/platform/dos/platform_dos.c src/platform/dos/text_dos.c
  PROPERTIES COMPILE_OPTIONS "-Os")
target_link_options(ab3d1 PRIVATE -O3 -flto -fwhole-program -Wl,--gc-sections)

set(AB3D_DOS_PROFILE_DIR "" CACHE PATH "Compiler feedback prepared by tools/profile_dos.py")
if(AB3D_DOS_PROFILE_DIR)
  if(AB3D_DOS_INDEXED)
    message(FATAL_ERROR "DOS compiler feedback currently targets native RGB565")
  endif()
  include("${AB3D_DOS_PROFILE_DIR}/sources.cmake")
  # Keep the large billboard helpers separate and avoid register-heavy loop
  # expansion on a 486 after dense scene feedback makes those paths hot.
  set_property(SOURCE src/platform/dos/raster_dos.c APPEND_STRING PROPERTY COMPILE_FLAGS
    " -fno-inline-functions-called-once -fno-unroll-loops -fno-peel-loops")
  file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}" dos_profile_prefix)
  set(dos_profile_options "-fprofile-use=${AB3D_DOS_PROFILE_DIR}"
    "-fprofile-prefix-path=${dos_profile_prefix}" -fprofile-correction -fno-tracer
    -Werror=coverage-mismatch -Werror=missing-profile)
  target_compile_options(ab3d1 PRIVATE ${dos_profile_options})
  target_link_options(ab3d1 PRIVATE ${dos_profile_options})
  file(GLOB dos_profile_data CONFIGURE_DEPENDS "${AB3D_DOS_PROFILE_DIR}/*.gcda")
  if(dos_profile_data)
    get_target_property(dos_profile_sources ab3d1 SOURCES)
    set_property(SOURCE ${dos_profile_sources} APPEND PROPERTY OBJECT_DEPENDS ${dos_profile_data})
  endif()
endif()
target_compile_options(ab3d1 PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/dos/math_compat.h")
# SDL and DJGPP's C runtime retain floating-point entry points. Link DJGPP's
# emulator so those calls also work on a 486SX. Native raster and audio mixing
# use integer arithmetic; libemu installs only when startup detects no x87.
target_link_libraries(ab3d1 PRIVATE "${DOS_SDL_ROOT}/SDL-build/libSDL3.a" m emu)
set_target_properties(ab3d1 PROPERTIES OUTPUT_NAME ABHD)

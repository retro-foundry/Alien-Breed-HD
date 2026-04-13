# Sync game assets into em_preload/ for Emscripten --preload-file (MEMFS root layout).
# Invoked with: cmake -DAB3D_SRC_DIR=... -DAB3D_EMPRELOAD=... [-DAB3D_BUILD_DIR=...] -P emscripten_sync_data.cmake
#
# Native copies ${CMAKE_BINARY_DIR}/data/ next to the executable (POST_BUILD). We mirror that by staging
# the same Amiga sources into build/data, then copying the entire tree to em_preload/data in one step
# so nothing is missed (includes/*.wad, levels, gfx, etc.).
if(NOT AB3D_SRC_DIR OR NOT AB3D_EMPRELOAD)
  message(FATAL_ERROR "emscripten_sync_data: AB3D_SRC_DIR and AB3D_EMPRELOAD required")
endif()

set(OUT "${AB3D_EMPRELOAD}")
file(MAKE_DIRECTORY "${OUT}/data")

function(ab3d_copy_tree src dest_sub)
  if(EXISTS "${src}")
    file(MAKE_DIRECTORY "${OUT}/${dest_sub}")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E copy_directory "${src}" "${OUT}/${dest_sub}"
      RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
      message(WARNING "emscripten_sync_data: copy failed ${src} -> ${OUT}/${dest_sub}")
    endif()
  endif()
endfunction()

# Copy Amiga game data into ${CMAKE_BINARY_DIR}/data (same sources as native POST_BUILD staging).
function(ab3d_stage_to_build src_rel dest_sub)
  if(NOT DEFINED AB3D_BUILD_DIR)
    return()
  endif()
  set(_src "${AB3D_SRC_DIR}/${src_rel}")
  if(NOT EXISTS "${_src}")
    return()
  endif()
  file(MAKE_DIRECTORY "${AB3D_BUILD_DIR}/data/${dest_sub}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${_src}" "${AB3D_BUILD_DIR}/data/${dest_sub}"
    RESULT_VARIABLE _rc
  )
  if(NOT _rc EQUAL 0)
    message(WARNING "emscripten_sync_data: stage ${_src} -> ${AB3D_BUILD_DIR}/data/${dest_sub} failed")
  endif()
endfunction()

# ab3d.ini at MEMFS root (settings.c looks in SDL_GetBasePath(), "/" on web)
if(EXISTS "${AB3D_SRC_DIR}/ab3d.ini.template")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${AB3D_SRC_DIR}/ab3d.ini.template"
            "${OUT}/ab3d.ini"
  )
endif()
if(EXISTS "${AB3D_SRC_DIR}/README.txt")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${AB3D_SRC_DIR}/README.txt"
            "${OUT}/README.txt"
  )
endif()
if(EXISTS "${AB3D_SRC_DIR}/fonts")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${AB3D_SRC_DIR}/fonts"
            "${OUT}/fonts"
  )
endif()

if(DEFINED AB3D_BUILD_DIR)
  file(MAKE_DIRECTORY "${AB3D_BUILD_DIR}/data")
  # Same order / sources as CMakeLists.txt ab3d_stage_dir + pal configure-time copy (build/data/pal).
  ab3d_stage_to_build("amiga/vectorobjects" "vectorobjects")
  ab3d_stage_to_build("amiga/sounds" "sounds")
  ab3d_stage_to_build("amiga/data/includes" "includes")
  ab3d_stage_to_build("amiga/data/levels" "levels")
  ab3d_stage_to_build("amiga/data/gfx" "gfx")
  ab3d_stage_to_build("amiga/pal" "pal")
  ab3d_stage_to_build("amiga/data/pal" "pal")
  ab3d_stage_to_build("amiga/data/helper" "helper")
  ab3d_stage_to_build("amiga/texturemaps" "texturemaps")

  # Single copy: native exe data/ is identical to this tree (after POST_BUILD staging).
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${AB3D_BUILD_DIR}/data"
            "${OUT}/data"
    RESULT_VARIABLE _full_rc
  )
  if(NOT _full_rc EQUAL 0)
    message(FATAL_ERROR "emscripten_sync_data: copy ${AB3D_BUILD_DIR}/data -> ${OUT}/data failed")
  endif()
else()
  # Fallback if AB3D_BUILD_DIR not passed (should not happen for Emscripten targets).
  ab3d_copy_tree("${AB3D_SRC_DIR}/amiga/vectorobjects" "data/vectorobjects")
  ab3d_copy_tree("${AB3D_SRC_DIR}/amiga/sounds" "data/sounds")
  ab3d_copy_tree("${AB3D_SRC_DIR}/amiga/data/includes" "data/includes")
  ab3d_copy_tree("${AB3D_SRC_DIR}/amiga/data/levels" "data/levels")
  ab3d_copy_tree("${AB3D_SRC_DIR}/amiga/data/gfx" "data/gfx")
  ab3d_copy_tree("${AB3D_SRC_DIR}/amiga/pal" "data/pal")
  ab3d_copy_tree("${AB3D_SRC_DIR}/amiga/data/pal" "data/pal")
  ab3d_copy_tree("${AB3D_SRC_DIR}/amiga/data/helper" "data/helper")
  ab3d_copy_tree("${AB3D_SRC_DIR}/amiga/texturemaps" "data/texturemaps")
endif()

# Optional runtime overrides (e.g. replacements/weapons from gun_frames_png.py import)
ab3d_copy_tree("${AB3D_SRC_DIR}/data/replacements" "data/replacements")

# Match desktop: convert extensionless/raw SFX to .wav; --wav-only drops non-wav sources to save MEMFS.
if(EXISTS "${OUT}/data/sounds")
  find_program(PYTHON3_EXE NAMES python3 python py)
  if(PYTHON3_EXE)
    execute_process(
      COMMAND "${PYTHON3_EXE}" "${AB3D_SRC_DIR}/tools/raw_to_wav.py"
              --sounds-dir "${OUT}/data/sounds"
              --wav-only
      WORKING_DIRECTORY "${AB3D_SRC_DIR}"
      RESULT_VARIABLE _wav_rc
    )
    if(NOT _wav_rc EQUAL 0)
      message(WARNING "emscripten_sync_data: raw_to_wav.py failed (exit ${_wav_rc})")
    endif()
  else()
    message(WARNING "emscripten_sync_data: Python not found; preload sounds may not load (run raw_to_wav manually on em_preload/data/sounds)")
  endif()
endif()

if(NOT EXISTS "${OUT}/data/includes/rockets.wad")
  message(WARNING "emscripten_sync_data: missing ${OUT}/data/includes/rockets.wad — ensure amiga/data/includes contains the full game data set")
endif()

file(WRITE "${OUT}/.stamp" "ok\n")

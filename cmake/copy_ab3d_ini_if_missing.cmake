# POST_BUILD helper: create ab3d.ini from template only when the user has no local file.
# Invoked with: cmake -DDEST_DIR=... -DSRC_FILE=... -P copy_ab3d_ini_if_missing.cmake
if(NOT DEST_DIR OR NOT SRC_FILE)
  message(FATAL_ERROR "copy_ab3d_ini_if_missing: DEST_DIR and SRC_FILE required")
endif()
set(OUT_INI "${DEST_DIR}/ab3d.ini")
if(EXISTS "${OUT_INI}")
  return()
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy "${SRC_FILE}" "${OUT_INI}"
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  message(WARNING "copy_ab3d_ini_if_missing: failed to create ${OUT_INI}")
endif()

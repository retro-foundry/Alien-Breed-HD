cmake_minimum_required(VERSION 3.16)
project(ab3d_dos_assets LANGUAGES C)
add_executable(sb_unpack ../unpack_sb.c ../../src/sb_decompress.c)
target_include_directories(sb_unpack PRIVATE ../../src)
target_compile_definitions(sb_unpack PRIVATE AB3D_STANDALONE_TOOL)
set_target_properties(sb_unpack PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")

cmake_minimum_required(VERSION 3.24)
project(gen2_image_backend LANGUAGES C CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(FATAL_ERROR "This image backend requires Apple silicon macOS")
endif()
if(NOT DEFINED SD_SOURCE_DIR)
    message(FATAL_ERROR "Set SD_SOURCE_DIR to the pinned stable-diffusion.cpp checkout")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
set(SD_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
add_subdirectory("${SD_SOURCE_DIR}" sd EXCLUDE_FROM_ALL)

add_library(gen2_image SHARED bridge.cpp)
target_link_libraries(gen2_image PRIVATE stable-diffusion)
# Both runtimes carry different GGML revisions. Keep SD and GGML private to
# this dylib; Rust resolves only the small, versioned bridge below.
target_link_options(gen2_image PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.txt")
set_target_properties(gen2_image PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

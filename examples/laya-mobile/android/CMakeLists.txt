cmake_minimum_required(VERSION 3.22)
project(gen2_laya_jni LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
# Set GEN2_LAYA_LIBRARY to the matching ABI's Rust shared library.
add_library(gen2_laya_mobile SHARED IMPORTED)
set_target_properties(gen2_laya_mobile PROPERTIES IMPORTED_LOCATION "${GEN2_LAYA_LIBRARY}")
add_library(gen2_laya_jni SHARED laya_jni.cpp)
target_link_libraries(gen2_laya_jni PRIVATE gen2_laya_mobile)

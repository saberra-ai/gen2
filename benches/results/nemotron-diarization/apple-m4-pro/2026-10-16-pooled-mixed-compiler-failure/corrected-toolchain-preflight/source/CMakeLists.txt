cmake_minimum_required(VERSION 3.20)
project(FrozenCompilerProbe LANGUAGES C CXX)
add_library(probe STATIC probe.c probe.cpp)

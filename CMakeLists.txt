cmake_minimum_required(VERSION 3.21)

project(adaptive_predicates)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(CompilerFlags)

add_library(adaptive_predicates INTERFACE)
target_include_directories(adaptive_predicates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

option(ADP_ENABLE_TESTS "Test suite" ON)
if(ADP_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

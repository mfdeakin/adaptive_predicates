cmake_minimum_required(VERSION 3.21)

project(adaptive_predicates)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(CompilerFlags)

find_package(fmt REQUIRED)

add_library(adaptive_predicates INTERFACE 
            src/ae_expr.hpp src/ae_expr_utils.hpp src/ae_fp_eval.hpp src/ae_expr_utils.hpp 
            src/ae_fp_eval_impl.hpp src/ae_adaptive_predicate_eval.hpp)
target_include_directories(adaptive_predicates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(adaptive_predicates INTERFACE fmt::fmt)

add_subdirectory(extern)

option(ADP_ENABLE_TESTS "Test suite" ON)
if(ADP_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

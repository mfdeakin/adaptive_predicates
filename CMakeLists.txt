cmake_minimum_required(VERSION 3.21)

project(adaptive_predicates)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(CompilerFlags)

add_library(adaptive_predicates INTERFACE 
  src/ae_expr.hpp src/ae_expr_utils.hpp src/ae_fp_eval.hpp src/ae_fp_eval_impl.hpp
  src/ae_expr_utils.hpp src/ae_expr_iostream.hpp src/ae_expr_print.hpp src/ae_expr_fmt.hpp
  src/ae_adaptive_predicate_eval.hpp)
target_compile_features(adaptive_predicates INTERFACE cxx_std_20)
target_include_directories(adaptive_predicates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_sources(adaptive_predicates PUBLIC FILE_SET HEADERS
  BASE_DIRS src
  FILES src/ae_expr.hpp src/ae_expr_utils.hpp src/ae_fp_eval.hpp src/ae_fp_eval_impl.hpp
  src/ae_expr_utils.hpp src/ae_expr_iostream.hpp src/ae_expr_print.hpp src/ae_expr_fmt.hpp
  src/ae_adaptive_predicate_eval.hpp)

install(TARGETS adaptive_predicates FILE_SET HEADERS)

option(ADP_ENABLE_EXAMPLES "Examples" ON)
if(ADP_ENABLE_EXAMPLES)
  find_package(fmt REQUIRED)
  add_subdirectory(examples)
endif()

option(ADP_ENABLE_TESTS "Test suite" ON)
if(ADP_ENABLE_TESTS)
  add_subdirectory(extern)
  find_package(fmt REQUIRED)

  enable_testing()
  set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
  add_subdirectory(tests)
endif()


find_package(Catch2 3 REQUIRED)
include(Catch)

option(BENCHMARK_CGAL "Include benchmark comparison to CGAL's Exact_predicates_exact_constructions_kernel" OFF)
if(BENCHMARK_CGAL)
  find_package(CGAL)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  option(CLANG_TIDY "Enable building with clang tidy's static analyzer" OFF)
  if(CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY
      clang-tidy;
      -format-style='file';
      -header-filter=${CMAKE_SOURCE_DIR};
    )
  endif()
endif()

add_executable(tests test_adaptive_expr.cpp test_exact_eval.cpp test_geom_exprs.cpp test_determinant_4x4.cpp test_determinant_5x5.cpp)
target_compile_features(tests PRIVATE cxx_std_23)
target_compile_options(tests PRIVATE -fprofile-arcs -ftest-coverage)
target_link_libraries(tests PRIVATE adaptive_predicates shewchuk Catch2::Catch2WithMain fmt)
target_link_options(tests PRIVATE -fprofile-arcs -ftest-coverage)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/extern ${CMAKE_SOURCE_DIR}/extern/simd_vec)

# CGAL adds -frounding-math for GCC, but that breaks our tests,
# wheras CGAL is only required for performance comparison
# https://github.com/CGAL/cgal/issues/3180#issuecomment-406697718
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set_source_files_properties(test_adaptive_expr.cpp PROPERTIES COMPILE_OPTIONS -fno-rounding-math)
endif()
catch_discover_tests(tests)

add_executable(test_matrix test_matrix.cpp)
target_compile_features(test_matrix PRIVATE cxx_std_23)
target_compile_options(test_matrix PRIVATE -fprofile-arcs -ftest-coverage)
target_link_libraries(test_matrix PRIVATE adaptive_predicates Catch2::Catch2WithMain fmt::fmt)
target_link_options(test_matrix PRIVATE -fprofile-arcs -ftest-coverage)
target_include_directories(test_matrix PRIVATE ${CMAKE_SOURCE_DIR}/extern)
catch_discover_tests(test_matrix)

add_executable(test_benchmarks test_benchmarks.cpp)
target_compile_features(test_benchmarks PRIVATE cxx_std_23)
target_link_libraries(test_benchmarks PRIVATE adaptive_predicates shewchuk Catch2::Catch2WithMain fmt::fmt)
if (CGAL_FOUND)
  target_compile_definitions(test_benchmarks PRIVATE HAS_CGAL)
  target_link_libraries(test_benchmarks PRIVATE CGAL::CGAL)
endif ()
target_include_directories(test_benchmarks PRIVATE ${CMAKE_SOURCE_DIR}/extern ${CMAKE_SOURCE_DIR}/extern/simd_vec)
catch_discover_tests(test_benchmarks)

add_executable(perf_benchmark perf_benchmark.cpp)
target_compile_features(perf_benchmark PRIVATE cxx_std_23)
target_compile_options(perf_benchmark PUBLIC -fno-omit-frame-pointer)
target_link_libraries(perf_benchmark PRIVATE adaptive_predicates fmt::fmt)
target_include_directories(perf_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/extern)

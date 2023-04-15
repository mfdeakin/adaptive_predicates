
find_package(Catch2 3 REQUIRED)
include(Catch)

find_package(CGAL)

add_executable(tests test_adaptive_expr.cpp)
target_link_libraries(tests PRIVATE adaptive_predicates shewchuk Catch2::Catch2WithMain CGAL::CGAL)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/extern)

# CGAL adds -frounding-math for GCC, but that breaks our tests,
# wheras CGAL is only required for performance comparison
# https://github.com/CGAL/cgal/issues/3180#issuecomment-406697718
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set_source_files_properties(test_adaptive_expr.cpp PROPERTIES COMPILE_OPTIONS -fno-rounding-math)
endif()

add_executable(perf_benchmark perf_benchmark.cpp)
target_compile_options(perf_benchmark PUBLIC -fno-omit-frame-pointer)
target_link_libraries(perf_benchmark PRIVATE adaptive_predicates)

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE adaptive_predicates Catch2::Catch2WithMain)
target_include_directories(test_matrix PRIVATE ${CMAKE_SOURCE_DIR}/extern)

add_executable(generate_cases generate_cases.cpp)
target_link_libraries(generate_cases PRIVATE adaptive_predicates)

catch_discover_tests(tests)

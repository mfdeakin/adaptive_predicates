
find_package(Catch2 3 REQUIRED)
include(Catch)

find_package(CGAL)

add_executable(tests test_adaptive_expr.cpp)
target_link_libraries(tests PRIVATE adaptive_predicates shewchuk Catch2::Catch2WithMain CGAL::CGAL)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/extern)

add_executable(perf_benchmark perf_benchmark.cpp)
target_compile_options(perf_benchmark PUBLIC -fno-omit-frame-pointer)
target_link_libraries(perf_benchmark PRIVATE adaptive_predicates)

catch_discover_tests(tests)


find_package(Catch2 3 REQUIRED)
include(Catch)

add_executable(tests test_adaptive_expr.cpp)

target_link_libraries(tests PRIVATE adaptive_predicates Catch2::Catch2WithMain)

catch_discover_tests(tests)

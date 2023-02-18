
find_package(Catch2 3 REQUIRED)
include(Catch)

add_executable(tests test_adaptive_expr.cpp)
target_link_libraries(tests PRIVATE adaptive_predicates shewchuk Catch2::Catch2WithMain)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/extern)

catch_discover_tests(tests)

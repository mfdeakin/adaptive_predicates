
add_executable(point_inclusion point_inclusion.cpp)
target_link_libraries(point_inclusion adaptive_predicates Kokkos::kokkos fmt::fmt)

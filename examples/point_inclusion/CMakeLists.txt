
add_executable(point_inclusion point_inclusion.cpp)
target_link_libraries(point_inclusion adaptive_predicates Kokkos::kokkos fmt::fmt)
target_include_directories(point_inclusion PRIVATE ${CMAKE_SOURCE_DIR}/extern)
target_compile_features(point_inclusion PRIVATE cxx_std_23)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(point_inclusion PRIVATE -Wno-unknown-cuda-version)
endif()


find_package(Kokkos QUIET)

if(Kokkos_FOUND)
  option(EX_POINT_INCLUSION "Point inclusion example" ON)
  if(EX_POINT_INCLUSION)
    add_subdirectory(point_inclusion)
  endif()
endif()

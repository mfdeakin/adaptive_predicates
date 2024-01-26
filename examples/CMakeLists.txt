
find_package(Kokkos QUIET)
find_package(argparse QUIET)

if(Kokkos_FOUND)
  option(EX_POINT_INCLUSION "Point inclusion example" ON)
  if(EX_POINT_INCLUSION)
    add_subdirectory(point_inclusion)
  endif()
endif()

if(argparse_FOUND)
  option(EX_GENERATE_CASES "Generate pathological points example" ON)
  if(EX_GENERATE_CASES)
    add_subdirectory(generate_cases)
  endif()
endif()

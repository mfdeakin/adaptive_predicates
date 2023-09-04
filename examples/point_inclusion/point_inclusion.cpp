
#include <fmt/format.h>

#include <Kokkos_Core.hpp>

class KokkosScope {
public:
  KokkosScope(int &argc, char *argv[]) { Kokkos::initialize(argc, argv); }
  ~KokkosScope() { Kokkos::finalize(); }
};

int main(int argc, char *argv[]) {
  KokkosScope(argc, argv);
  fmt::println("Kokkos yay!\n");
  return 0;
}

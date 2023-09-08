
#include <cmath>
#include <random>
#include <type_traits>

#include <fmt/format.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "ae_expr.hpp"

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_fp_eval.hpp"

using real = double;

using ExecSpace = Kokkos::DefaultExecutionSpace;

enum class PointLocation { Inside, OnSurface, Outside, Indeterminant };

#ifdef KOKKOS_ENABLE_CUDA
constexpr bool is_cpu() { return !std::is_same_v<Kokkos::Cuda, ExecSpace>; }
#else
constexpr bool is_cpu() { return true; }
#endif // KOKKOS_ENABLE_CUDA

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  fmt::println("On CPU? {}\n", is_cpu());

  const auto seed = std::random_device()();
  fmt::print("Random seed: {}\n", seed);

  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_gen(seed);

  const int num_test_points = 1 << 18;
  Kokkos::View<real *> x_pos("Test Point X", num_test_points);
  Kokkos::View<real *> y_pos("Test Point Y", num_test_points);
  Kokkos::View<real *> z_pos("Test Point Z", num_test_points);

  const int num_ellipsoids = 1 << 12;
  Kokkos::View<real *> x_center("Ellipsoid X Center", num_ellipsoids);
  Kokkos::View<real *> y_center("Ellipsoid Y Center", num_ellipsoids);
  Kokkos::View<real *> z_center("Ellipsoid Z Center", num_ellipsoids);
  Kokkos::View<real *> x_scale("Ellipsoid X Scale", num_ellipsoids);
  Kokkos::View<real *> y_scale("Ellipsoid Y Scale", num_ellipsoids);
  Kokkos::View<real *> z_scale("Ellipsoid Z Scale", num_ellipsoids);

  const real coord_max = 1024.0;
  const real scale_min = 1.0 / 32.0;
  const real scale_max = 32.0;
  Kokkos::parallel_for(
      "generate test points", num_test_points, KOKKOS_LAMBDA(int i) {
        auto generator = rand_gen.get_state();
        x_pos(i) = generator.drand(-coord_max, coord_max);
        y_pos(i) = generator.drand(-coord_max, coord_max);
        z_pos(i) = generator.drand(-coord_max, coord_max);
        rand_gen.free_state(generator);
      });
  Kokkos::parallel_for(
      "generate ellipsoids", num_ellipsoids, KOKKOS_LAMBDA(int i) {
        auto generator = rand_gen.get_state();
        x_center(i) = generator.drand(-coord_max, coord_max);
        y_center(i) = generator.drand(-coord_max, coord_max);
        z_center(i) = generator.drand(-coord_max, coord_max);
        x_scale(i) = generator.drand(scale_min, scale_max);
        y_scale(i) = generator.drand(scale_min, scale_max);
        z_scale(i) = generator.drand(scale_min, scale_max);
        rand_gen.free_state(generator);
      });

  Kokkos::View<PointLocation **> ellipsoid_locs(
      "Ellipsoid Locations", num_ellipsoids, num_test_points);

  const int vec_size = 8;

  Kokkos::parallel_for(
      "FP point locations ellipsoid",
      Kokkos::TeamPolicy<>(num_ellipsoids, Kokkos::AUTO_t{}, vec_size),
      KOKKOS_LAMBDA(const auto &team) {
        const int i = team.league_rank();
        const real &x_c = x_center(i);
        const real &y_c = y_center(i);
        const real &z_c = z_center(i);
        const real &x_s = x_scale(i);
        const real &y_s = y_scale(i);
        const real &z_s = z_scale(i);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, num_test_points / vec_size),
            [&](const int j) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team, vec_size), [&](const int k) {
                    const real &x_p = x_pos(j * vec_size + k);
                    const real &y_p = y_pos(j * vec_size + k);
                    const real &z_p = z_pos(j * vec_size + k);
                    using adaptive_expr::minus_expr;
                    const auto x_diff = x_s * minus_expr(x_p, x_c);
                    const auto y_diff = y_s * minus_expr(y_p, y_c);
                    const auto z_diff = z_s * minus_expr(z_p, z_c);
                    // This form of the expression has at most one subtraction,
                    // as x_diff^2, y_diff^2, and z_diff^2 are all positive
                    const auto ex =
                        (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff) -
                        real{1};
                    const auto [result, _] =
                        adaptive_expr::eval_checked_fast<real>(ex);

                    PointLocation &loc = ellipsoid_locs(i, j * vec_size + k);
                    using std::isnan;
                    if (isnan(result)) {
                      loc = PointLocation::Indeterminant;
                    } else if (result < 0.0) {
                      loc = PointLocation::Inside;
                    } else if (result == 0.0) {
                      loc = PointLocation::OnSurface;
                    } else {
                      loc = PointLocation::Outside;
                    }
                  });
            });
      });
  int num_indeterminant = 0;
  Kokkos::parallel_reduce(
      num_ellipsoids * num_test_points,
      KOKKOS_LAMBDA(const int i, int &count) {
        if (ellipsoid_locs(i / num_test_points, i % num_test_points) ==
            PointLocation::Indeterminant) {
          ++count;
        }
      },
      num_indeterminant);
  fmt::println("{} / {} points locations evaluated incorrectly",
               num_indeterminant, num_ellipsoids * num_test_points);
  return 0;
}

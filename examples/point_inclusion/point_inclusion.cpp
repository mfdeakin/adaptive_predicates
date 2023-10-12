
#include <cmath>
#include <random>
#include <type_traits>

#include <fmt/format.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "ae_expr.hpp"

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_fp_eval.hpp"
#include "ae_gpu_vector.hpp"

#include <simd_vec/vectorclass.h>

using std::signbit;

using real = double;

using ExecSpace = Kokkos::DefaultExecutionSpace;

enum class PointLocation { Inside, OnSurface, Outside, Indeterminant, Wrong };

#ifdef KOKKOS_ENABLE_CUDA
constexpr bool is_cpu() { return !std::is_same_v<Kokkos::Cuda, ExecSpace>; }
#else
constexpr bool is_cpu() { return true; }
#endif // KOKKOS_ENABLE_CUDA

constexpr int threads = is_cpu() ? 1 : 32;

using eval_type =
    std::conditional_t<is_cpu(), Vec4d, adaptive_expr::GPUVec<real, threads>>;

constexpr int vec_size = eval_type::size() / threads;

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  fmt::println("On CPU? {}\n", is_cpu());

  const auto seed = std::random_device()();
  fmt::print("Random seed: {}\n", seed);

  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_gen(seed);

  const int num_test_points = 1 << 12;
  Kokkos::View<real *> x_pos("Test Point X", num_test_points);
  Kokkos::View<real *> y_pos("Test Point Y", num_test_points);
  Kokkos::View<real *> z_pos("Test Point Z", num_test_points);

  const int num_ellipsoids = 1 << 10;
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

  using MinusExprType = adaptive_expr::arith_expr<std::minus<>, real, real>;
  using EllipseScaleExprType =
      adaptive_expr::arith_expr<std::multiplies<>, real, MinusExprType>;
  using EllipseTermsExprType =
      adaptive_expr::arith_expr<std::multiplies<>, EllipseScaleExprType,
                                EllipseScaleExprType>;
  using ExprType = adaptive_expr::arith_expr<
      std::minus<>,
      adaptive_expr::arith_expr<
          std::plus<>,
          adaptive_expr::arith_expr<std::plus<>, EllipseTermsExprType,
                                    EllipseTermsExprType>,
          EllipseTermsExprType>,
      real>;

  using ScratchView = Kokkos::View<
      eval_type * [adaptive_expr::num_partials_for_exact<ExprType>() + 8],
      ExecSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  fmt::println("Shared memory needed: {}", ScratchView::shmem_size(1));

  constexpr int thread_range = threads * vec_size;
  const auto policy =
      Kokkos::TeamPolicy<>(num_ellipsoids * num_test_points / thread_range,
                           threads)
          .set_scratch_size(0, Kokkos::PerTeam(ScratchView::shmem_size(1)));
  Kokkos::parallel_for(
      "FP point locations ellipsoid", policy, KOKKOS_LAMBDA(const auto &team) {
        const int i = team.league_rank();
        const int i_ell = i / (num_test_points / thread_range);
        const int i_pt_offset =
            (i % (num_test_points / thread_range)) * thread_range;
        const real x_c = x_center(i_ell);
        const real y_c = y_center(i_ell);
        const real z_c = z_center(i_ell);
        const real x_s = x_scale(i_ell);
        const real y_s = y_scale(i_ell);
        const real z_s = z_scale(i_ell);
        eval_type x_p;
        eval_type y_p;
        eval_type z_p;
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, thread_range),
                             [&](const int j) {
                               const int i_pt = i_pt_offset + j;
                               x_p.insert(j, x_pos(i_pt));
                               y_p.insert(j, y_pos(i_pt));
                               z_p.insert(j, z_pos(i_pt));
                             });
        using adaptive_expr::minus_expr;
        const auto x_diff = x_s * minus_expr(x_p, x_c);
        const auto y_diff = y_s * minus_expr(y_p, y_c);
        const auto z_diff = z_s * minus_expr(z_p, z_c);
        // This form of the expression has at most one subtraction,
        // as x_diff^2, y_diff^2, and z_diff^2 are all positive
        const auto ex =
            (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff) - real{1};
        const auto eval_results =
            adaptive_expr::eval_checked_fast<eval_type>(ex);
        const auto result = eval_results.first;
        const auto exact = adaptive_expr::exactfp_eval<eval_type>(ex);

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, thread_range), [&](const int j) {
              const int i_pt = i_pt_offset + j;
              PointLocation &loc = ellipsoid_locs(i_ell, i_pt);
              using std::isnan;
              if (isnan(static_cast<real>(result[j]))) {
                loc = PointLocation::Indeterminant;
              } else {
                if (exact[j] != 0 && static_cast<real>(result[j]) != 0 &&
                    signbit(exact[j]) !=
                        signbit(static_cast<real>(result[j]))) {
                  loc = PointLocation::Wrong;
                } else if (static_cast<real>(result[j]) < 0.0) {
                  loc = PointLocation::Inside;
                } else if (static_cast<real>(result[j]) == 0.0) {
                  loc = PointLocation::OnSurface;
                } else {
                  loc = PointLocation::Outside;
                }
              }
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
  int num_wrong = 0;
  Kokkos::parallel_reduce(
      num_ellipsoids * num_test_points,
      KOKKOS_LAMBDA(const int i, int &count) {
        if (ellipsoid_locs(i / num_test_points, i % num_test_points) ==
            PointLocation::Wrong) {
          ++count;
        }
      },
      num_wrong);
  fmt::println("({}, {}) / {} points locations evaluated incorrectly",
               num_indeterminant, num_wrong, num_ellipsoids * num_test_points);
  return 0;
}

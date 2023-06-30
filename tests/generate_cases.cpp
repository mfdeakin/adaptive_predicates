
#include <array>
#include <numbers>
#include <random>
#include <span>

#include <fmt/format.h>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr_io.hpp"
#include "ae_fp_eval.hpp"
#include "testing_utils.hpp"

using namespace adaptive_expr;
using std::numbers::pi;

using real = double;

void generate_incircle_cases(std::mt19937_64 rng);
void generate_orient_cases(std::mt19937_64 rng);

int main() {
  const auto seed = std::random_device()();
  std::mt19937_64 rng(seed);
  fmt::print("Random seed: {}\n", seed);
  generate_incircle_cases(rng);
  return 0;
}

void generate_incircle_cases(std::mt19937_64 rng) {
  const auto eps = std::numeric_limits<real>::epsilon();
  std::uniform_real_distribution<real> dist(-1.0, 1.0);
  std::uniform_real_distribution<real> rad_dist(1.0 - 16.0 * eps,
                                                1.0 + 16.0 * eps);
  std::uniform_real_distribution<real> theta_dist(-pi - 16.0 * eps,
                                                  pi + 16.0 * eps);
  const auto gen_points = [&rng, &dist, &rad_dist, &theta_dist]() {
    std::array<std::array<real, 2>, 4> points;
    for (auto &pt : points | std::ranges::views::take(3)) {
      for (real &coord : pt) {
        coord = dist(rng);
      }
    }
    const std::array perp{-(points[1][1] - points[0][1]),
                          points[1][0] - points[0][0]};
    const std::array edge1_vec{points[2][0] - points[0][0],
                               points[2][1] - points[0][1]};
    const std::array bisector1_vec{(points[1][0] - points[2][0]) / 2,
                                   (points[1][1] - points[2][1]) / 2};
    const real perp_vec_dist =
        (bisector1_vec[0] * edge1_vec[0] + bisector1_vec[1] * edge1_vec[1]) /
        (perp[0] * edge1_vec[0] + perp[1] * edge1_vec[1]);
    const std::array bisector0{(points[1][0] + points[0][0]) / 2,
                               (points[1][1] + points[0][1]) / 2};
    const std::array center{bisector0[0] - perp[0] * perp_vec_dist,
                            bisector0[1] - perp[1] * perp_vec_dist};
    const std::array p0_vec{points[0][0] - center[0], points[0][1] - center[1]};
    const real radius = rad_dist(rng) * std::sqrt(std::pow(p0_vec[0], 2) +
                                                  std::pow(p0_vec[1], 2));
    const real theta = theta_dist(rng);
    points[3][0] = center[0] + radius * std::cos(theta);
    points[3][1] = center[1] + radius * std::sin(theta);
    return points;
  };
  for (;;) {
    const auto points = gen_points();
    const auto expr = build_incircle2d_case(points);
    if (!correct_eval<real>(expr)) {
      if (adaptive_eval<real>(expr) == 0.0 &&
          !check_sign(adaptive_eval<real>(expr), fp_eval<real>(expr))) {
        fmt::print("std::pair{{\n"
                   "  std::array{{\n"
                   "    std::array<real, 2>{{{: .55e}, {: .55e}}},\n"
                   "    std::array<real, 2>{{{: .55e}, {: .55e}}},\n"
                   "    std::array<real, 2>{{{: .55e}, {: .55e}}},\n"
                   "    std::array<real, 2>{{{: .55e}, {: .55e}}}}},\n"
                   "  real{{{: .55e}}}}},\n",
                   points[0][0], points[0][1], points[1][0], points[1][1],
                   points[2][0], points[2][1], points[3][0], points[3][1],
                   adaptive_eval<real>(expr));
        fmt::print("Expression:\n{}\n", expr);
        fmt::print("exact sign: {: .55e}, fp sign: {: .55e}\n",
                   adaptive_eval<real>(expr), fp_eval<real>(expr));
      }
    }
  }
}
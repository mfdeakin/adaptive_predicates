
#include <array>
#include <random>
#include <span>

#include <fmt/format.h>

#include "testing_utils.hpp"
#include "ae_expr_io.hpp"
#include "ae_fp_eval.hpp"
#include "ae_adaptive_predicate_eval.hpp"

using namespace adaptive_expr;

int main() {
  const auto seed = std::random_device()();
  fmt::print("Random seed: {}\n", seed);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto gen_points = [&dist, &rng]() {
    std::array<std::array<double, 2>, 3> points;
    for (auto &pt : std::span{points}.first(2)) {
      for (double &coord : pt) {
        coord = dist(rng);
      }
    }
    points[2][1] =
        (points[1][0] * points[0][1] - points[0][0] * points[1][1]) /
        ((points[1][0] - points[0][0]) +
         (points[0][1] - points[1][1]) * points[1][0] / points[1][1]);
    points[2][0] = points[1][0] / points[1][1] * points[2][1];
    return points;
  };
  for (;;) {
    const auto points = gen_points();
    const auto expr = build_orient2d_case(points);
    const auto print_dbl = [](const double v) {
      const int exp = std::ilogb(v);
      const double normalized =
          std::pow(2, std::numeric_limits<double>::digits - exp - 1) * v;
      const long int mantissa = std::lrint(normalized);
      fmt::print("{: 10},   {: 64b} ({: .17f}, {: 17.1f})\n", exp, mantissa,
                 std::pow(2, -exp - 1) * v, normalized);
    };
    if (!correct_eval<double>(expr)) {
      if (!check_sign(adaptive_eval<double>(expr), fp_eval<double>(expr))) {
        fmt::print("std::pair{{\n"
                   "  std::array{{\n"
                   "    std::array<real, 2>{{{: .55e}, {: .55e}}},\n"
                   "    std::array<real, 2>{{{: .55e}, {: .55e}}},\n"
                   "    std::array<real, 2>{{{: .55e}, {: .55e}}}}},\n"
                   "  real{{0.0}}}},\n",
                   points[0][0], points[0][1], points[1][0], points[1][1],
                   points[2][0], points[2][1]);
        fmt::print("Expression:\n{}\n", expr);
        fmt::print("exact sign: {: .55e}, fp sign: {: .55e}\n",
                   adaptive_eval<double>(expr), fp_eval<double>(expr));
      }
    }
  }
  return 0;
}
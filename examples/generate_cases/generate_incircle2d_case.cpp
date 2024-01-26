
#include <array>
#include <numbers>
#include <random>
#include <span>

#include <fmt/format.h>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr_fmt.hpp"
#include "ae_fp_eval.hpp"
#include "ae_geom_exprs.hpp"

#include "shewchuk.h"

#include "generate_cases.hpp"

using namespace adaptive_expr;

std::array<std::array<double, 2>, 4>
generate_incircle2d_case(std::mt19937_64 &rng) {
  const auto oriented_pts = generate_orient2d_case(rng);
  std::array<std::array<double, 2>, 4> incircle_pts;
  for (std::size_t i = 0; i < oriented_pts.size(); ++i) {
    for (std::size_t j = 0; j < oriented_pts[i].size(); ++j) {
      incircle_pts[i][j] = oriented_pts[i][j];
    }
  }
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  auto &test_pt = incircle_pts.back();
  for (double &coord : test_pt) {
    coord = dist(rng);
  }
  constexpr std::size_t num_steps = 10;
  for (std::size_t i = 0; i < num_steps; ++i) {
    for (double &coord : test_pt) {
      const double x0 = coord;
      const double y0 = exactfp_eval<double>(pt_incircle_expr(incircle_pts));
      coord = std::nextafter(x0, std::numeric_limits<double>::infinity());
      const double dx = coord - x0;
      const double dy =
          exactfp_eval<double>(pt_incircle_expr(incircle_pts) - y0);
      coord = exactfp_eval<double>(minus_expr(x0 / dx, y0 / dy) * dx);
    }
  }
  Shewchuk::incircle(incircle_pts[0].data(), incircle_pts[1].data(),
                     incircle_pts[2].data(), incircle_pts[3].data());
  return incircle_pts;
}

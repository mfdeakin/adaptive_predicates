
#include <array>
#include <random>
#include <span>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_fp_eval.hpp"
#include "ae_geom_exprs.hpp"

#include "generate_cases.hpp"

using namespace adaptive_expr;

std::array<std::array<double, 3>, 4>
generate_orient3d_case(std::mt19937_64 &rng) {
  const std::array<std::array<double, 2>, 3> pts_2d =
      generate_orient2d_case(rng);
  std::array<std::array<double, 3>, 4> orient3d_pts;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < pts_2d.size(); ++i) {
    orient3d_pts[i][0] = dist(rng);
    for (std::size_t j = 0; j < pts_2d[i].size(); ++j) {
      orient3d_pts[i][j + 1] = pts_2d[i][j];
    }
  }
  auto &test_pt = orient3d_pts.back();
  for (double &coord : test_pt) {
    coord = 0.0;
  }
  do {
    test_pt[0] = dist(rng);
    for (std::size_t i = 1; i < test_pt.size(); ++i) {
      const double start = exactfp_eval<double>(pt_orient_expr(orient3d_pts));
      test_pt[i] = 1.0;
      const double delta =
          exactfp_eval<double>(pt_orient_expr(orient3d_pts) - start);
      if (std::abs(delta) > 1e-6) {
        test_pt[i] = -start / delta;
      }
    }
  } while (correct_eval<double>(pt_orient_expr(orient3d_pts)).has_value() ||
           check_sign(adaptive_eval<double>(pt_orient_expr(orient3d_pts)),
                      fp_eval<double>(pt_orient_expr(orient3d_pts))));
  return orient3d_pts;
}

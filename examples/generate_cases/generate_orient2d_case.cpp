
#include <array>
#include <random>
#include <span>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_fp_eval.hpp"
#include "ae_geom_exprs.hpp"

#include "shewchuk.h"

#include "generate_cases.hpp"

using namespace adaptive_expr;

std::array<std::array<double, 2>, 3>
generate_orient2d_case(std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::array<std::array<double, 2>, 3> points;
  do {
    for (auto &pt : std::span{points}.first(2)) {
      for (double &coord : pt) {
        coord = dist(rng);
      }
    }
    auto &test_pt = points.back();
    for (double &coord : test_pt) {
      coord = 0.0;
    }
    for (double &coord : test_pt) {
      const double initial = exactfp_eval<double>(pt_orient_expr(points));
      coord = 1.0;
      const double delta =
          exactfp_eval<double>(pt_orient_expr(points) - initial);
      coord = -initial / delta;
    }
  } while (correct_eval<double>(pt_orient_expr(points)).has_value() ||
           check_sign(adaptive_eval<double>(pt_orient_expr(points)),
                      fp_eval<double>(pt_orient_expr(points))));
  return points;
}


#include <array>
#include <numbers>
#include <random>
#include <span>

#include <fmt/format.h>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr_fmt.hpp"
#include "ae_fp_eval.hpp"
#include "testing_utils.hpp"

#include "shewchuk.h"

using namespace adaptive_expr;

int main() {
  Shewchuk::exactinit();
  // Known functioning seeds: 1229189403, 1989977017, 1990231328, 4039919784,
  // 1466186244, 197479037
  const auto seed = std::random_device()();
  fmt::print("Random seed: {}\n", seed);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> center_dist(-1.0, 1.0);
  std::uniform_real_distribution<double> rad_dist(
      std::numeric_limits<double>::epsilon(), 1.0);
  std::uniform_real_distribution<double> theta_dist(
      0.0, 2.0 * std::numbers::pi_v<double>);
  std::uniform_real_distribution<double> phi_dist(0.0,
                                                  std::numbers::pi_v<double>);

  for (;;) {
    std::array<double, 3> center;
    for (double &coord : center) {
      coord = center_dist(rng);
    }
    const double rad = rad_dist(rng);
    std::array<std::array<double, 3>, 5> points;
    for (std::array<double, 3> &pt : points) {
      const double theta = theta_dist(rng);
      const double phi = theta_dist(rng);
      pt[0] = rad * std::cos(theta) * std::sin(phi) + center[0];
      pt[1] = rad * std::sin(theta) * std::sin(phi) + center[1];
      pt[2] = rad * std::cos(phi) + center[2];
    }
    const double orientation = Shewchuk::orient3d(
        points[0].data(), points[1].data(), points[2].data(), points[3].data());
    if (orientation < 0.0) {
      std::swap(points[0], points[1]);
    } else if (orientation == 0.0) {
      points[0][0] =
          std::nextafter(points[0][0], std::numeric_limits<double>::infinity());
      if (Shewchuk::orient3d(points[0].data(), points[1].data(),
                             points[2].data(), points[3].data()) == 0.0) {
        points[0][1] = std::nextafter(points[0][1],
                                      std::numeric_limits<double>::infinity());
        if (Shewchuk::orient3d(points[0].data(), points[1].data(),
                               points[2].data(), points[3].data()) == 0.0) {
          points[0][2] = std::nextafter(
              points[0][2], std::numeric_limits<double>::infinity());
        }
      }
    }
    const std::array<double, 3> radial_vec{points[4][0] - center[0],
                                           points[4][1] - center[1],
                                           points[4][2] - center[2]};

    const double starting_sign = Shewchuk::insphereexact(
        points[0].data(), points[1].data(), points[2].data(), points[3].data(),
        points[4].data());
    const auto update_test_pt = [radial_vec](std::array<double, 3> &pt,
                                             const double sign) {
      if (sign >= 0) {
        if (radial_vec[0] >= 0) {
          pt[0] =
              std::nextafter(pt[0], std::numeric_limits<double>::infinity());
        } else {
          pt[0] =
              std::nextafter(pt[0], -std::numeric_limits<double>::infinity());
        }
      } else if (radial_vec[0] < 0) {
        // the test point is outside of the sphere, we need to move it inside of
        // it
        pt[0] = std::nextafter(pt[0], std::numeric_limits<double>::infinity());
      } else if (radial_vec[0] > 0) {
        pt[0] = std::nextafter(pt[0], -std::numeric_limits<double>::infinity());
      } else if (radial_vec[1] < 0) {
        pt[1] = std::nextafter(pt[1], std::numeric_limits<double>::infinity());
      } else if (radial_vec[1] > 0) {
        pt[1] = std::nextafter(pt[1], -std::numeric_limits<double>::infinity());
      } else if (radial_vec[2] < 0) {
        pt[2] = std::nextafter(pt[2], std::numeric_limits<double>::infinity());
      } else {
        pt[2] = std::nextafter(pt[2], -std::numeric_limits<double>::infinity());
      }
    };
    update_test_pt(points[4], starting_sign);
    auto radius_expr_gen = [center, rad](const std::array<double, 3> &pt) {
      const auto [dx, dy, dz] = pt_diff_expr(pt, center);
      return mult_expr(rad, rad) - (dx * dx + dy * dy + dz * dz);
    };
    double cur_sign = Shewchuk::insphereexact(
        points[0].data(), points[1].data(), points[2].data(), points[3].data(),
        points[4].data());
    while (check_sign<double>(starting_sign, cur_sign)) {
      update_test_pt(points[4], starting_sign);
      cur_sign = Shewchuk::insphereexact(points[0].data(), points[1].data(),
                                         points[2].data(), points[3].data(),
                                         points[4].data());
    }
    const auto expr = pt_incircle_expr(points);

    fmt::print("std::pair{{\n"
               "  std::array{{\n"
               "    std::array<real, 3>{{{: .55e}, {: .55e}, {: .55e}}},\n"
               "    std::array<real, 3>{{{: .55e}, {: .55e}, {: .55e}}},\n"
               "    std::array<real, 3>{{{: .55e}, {: .55e}, {: .55e}}},\n"
               "    std::array<real, 3>{{{: .55e}, {: .55e}, {: .55e}}},\n"
               "    std::array<real, 3>{{{: .55e}, {: .55e}, {: .55e}}}}},\n"
               "  real{{0.0}}}},\n\n",
               points[0][0], points[0][1], points[0][2], points[1][0],
               points[1][1], points[1][2], points[2][0], points[2][1],
               points[2][2], points[3][0], points[3][1], points[3][2],
               points[4][0], points[4][1], points[4][2]);
    fmt::print("Distance Expression:\n{}\n\n", radius_expr_gen(points[4]));
    fmt::print("Expression:\n{}\n\n", expr);
    Shewchuk::insphere(points[0].data(), points[1].data(), points[2].data(),
                       points[3].data(), points[4].data());
    const double adaptive = adaptive_eval<double>(expr);
    fmt::print("exact sign: {: .55e}, shewchuk exact sign: {: .55e}, "
               "fp sign: {: .55e}\n",
               adaptive, cur_sign, fp_eval<double>(expr));
  }
  return 0;
}
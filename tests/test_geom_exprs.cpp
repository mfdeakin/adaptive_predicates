
#include <catch2/catch_test_macros.hpp>

#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"
#include "ae_geom_exprs.hpp"

#include <random>
#include <tuple>
#include <type_traits>

using namespace adaptive_expr;

static_assert(std::is_same_v<_impl::matrix_index<0, 0>,
                             decltype(determinant(std::make_tuple(std::tuple{
                                 _impl::matrix_index<0, 0>{}})))>);

TEST_CASE("2x2 determinant expression", "[geom_exprs][determinant][2x2]") {
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 0.0}, std::array{0.0, 0.0}})) == 0.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0}, std::array{0.0, 1.0}})) == 1.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 1.0}, std::array{1.0, 0.0}})) == -1.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 1.0}, std::array{1.0, 1.0}})) == 0.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 0.0}, std::array{0.0, 1.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0}, std::array{0.0, 2.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 0.0}, std::array{0.0, 2.0}})) == 4.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 2.0}, std::array{1.0, 0.0}})) == -2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 1.0}, std::array{2.0, 0.0}})) == -2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 2.0}, std::array{2.0, 0.0}})) == -4.0);

  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 1.0}, std::array{0.0, 1.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0}, std::array{1.0, 2.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 1.0}, std::array{1.0, 2.0}})) == 3.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 2.0}, std::array{1.0, 0.0}})) == -2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 1.0}, std::array{2.0, 1.0}})) == -2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 2.0}, std::array{2.0, 1.0}})) == -3.0);
}

TEST_CASE("3x3 determinant expression", "[geom_exprs][determinant][3x3]") {
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 0.0, 0.0}, std::array{0.0, 0.0, 0.0},
                         std::array{0.0, 0.0, 0.0}})) == 0.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0, 0.0}, std::array{0.0, 1.0, 0.0},
                         std::array{0.0, 0.0, 0.0}})) == 0.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0, 0.0}, std::array{0.0, 1.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == 1.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 1.0, 0.0}, std::array{1.0, 0.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == -1.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 1.0, 0.0}, std::array{1.0, 1.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == 0.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 1.0, 1.0}, std::array{1.0, 1.0, 1.0},
                         std::array{1.0, 1.0, 1.0}})) == 0.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 0.0, 0.0}, std::array{0.0, 1.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0, 0.0}, std::array{0.0, 2.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 0.0, 0.0}, std::array{0.0, 2.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == 4.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 2.0, 0.0}, std::array{1.0, 0.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == -2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 1.0, 0.0}, std::array{2.0, 0.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == -2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 2.0, 0.0}, std::array{2.0, 0.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == -4.0);

  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 1.0, 0.0}, std::array{0.0, 1.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 0.0, 1.0}, std::array{0.0, 1.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0, 0.0}, std::array{1.0, 2.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0, 0.0}, std::array{0.0, 2.0, 1.0},
                         std::array{0.0, 0.0, 1.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0, 0.0}, std::array{0.0, 1.0, 0.0},
                         std::array{0.0, 1.0, 2.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 0.0, 0.0}, std::array{0.0, 1.0, 0.0},
                         std::array{1.0, 0.0, 2.0}})) == 2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 1.0, 0.0}, std::array{1.0, 2.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == 3.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 1.0, 0.0}, std::array{1.0, 2.0, 0.0},
                         std::array{0.0, 0.0, 2.0}})) == 6.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 0.0, 1.0}, std::array{0.0, 2.0, 0.0},
                         std::array{1.0, 0.0, 2.0}})) == 6.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{2.0, 0.0, 0.0}, std::array{0.0, 2.0, 1.0},
                         std::array{0.0, 1.0, 2.0}})) == 6.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 2.0, 0.0}, std::array{1.0, 0.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == -2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{0.0, 1.0, 0.0}, std::array{2.0, 1.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == -2.0);
  REQUIRE(fp_eval<double>(determinant(
              std::array{std::array{1.0, 2.0, 0.0}, std::array{2.0, 1.0, 0.0},
                         std::array{0.0, 0.0, 1.0}})) == -3.0);
}

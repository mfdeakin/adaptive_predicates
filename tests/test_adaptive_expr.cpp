
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"

#include "shewchuk.h"

using namespace adaptive_expr;

static_assert(is_expr<arith_expr<std::plus<>, float, float>>::value);
static_assert(is_expr<const arith_expr<std::plus<>, float, float>>::value);
static_assert(is_expr<arith_expr<std::plus<>, float, float> &>::value);
static_assert(is_expr<const arith_expr<std::plus<>, float, float> &>::value);
static_assert(is_expr<arith_expr<std::plus<>, float, float> &&>::value);
static_assert(is_expr<const arith_expr<std::plus<>, float, float> &&>::value);

TEST_CASE("expr_template_structure", "[expr_template]") {
  auto e = ((arith_expr{} + 4 - 7) * 5 + 3.0 / 6.0) / 2;
  using E = typeof(e);
  static_assert(std::is_same_v<additive_id, E::LHS::LHS::LHS::LHS::LHS::LHS>);
  static_assert(std::is_same_v<additive_id, E::LHS::LHS::LHS::LHS::LHS::RHS>);
  static_assert(std::is_same_v<int, E::LHS::LHS::LHS::LHS::RHS>);
  static_assert(std::is_same_v<std::plus<>, E::LHS::LHS::LHS::LHS::Op>);
  static_assert(std::is_same_v<std::minus<>, E::LHS::LHS::LHS::Op>);
  static_assert(std::is_same_v<std::multiplies<>, E::LHS::LHS::Op>);
  static_assert(std::is_same_v<std::plus<>, E::LHS::Op>);
  static_assert(std::is_same_v<std::divides<>, E::Op>);
  REQUIRE(e.lhs().lhs().lhs().lhs().rhs() == 4);
  REQUIRE(e.lhs().lhs().lhs().rhs() == 7);
  REQUIRE(e.lhs().lhs().rhs() == 5);
  REQUIRE(e.lhs().rhs() == 0.5);
  REQUIRE(e.rhs() == 2);
  REQUIRE(fp_eval<float>(e) == -7.25);
}

TEST_CASE("expr_template_eval_simple", "[expr_template_eval]") {
  static_assert(num_partials_for_exact(arith_expr{}) == 0);
  static_assert(num_partials_for_exact(arith_expr{} + 4) == 1);
  static_assert(num_partials_for_exact(arith_expr{} + 4 - 7) == 2);
  static_assert(num_partials_for_exact((arith_expr{} + 4 - 7) * 5) == 4);
  auto e = ((arith_expr{} + 4 - 7) * 5 + 3.0 / 6.0);
  REQUIRE(exactfp_eval<float>(e.lhs().lhs().lhs().lhs()) == 0.0);
  REQUIRE(exactfp_eval<float>(e.lhs().lhs().lhs()) == 4.0);
  REQUIRE(exactfp_eval<float>(e.lhs().lhs()) == -3.0);
  REQUIRE(exactfp_eval<float>(e.lhs()) == -15.0);
  REQUIRE(exactfp_eval<float>(e) == -14.5);
  std::vector<float> fp_vals{5.0, 10.0, 11.0, 11.0, 44.0};
  auto s = std::span{fp_vals};
  merge_sum(s, s.first(2), s.subspan(2, 3));
  for (auto [expected, val] :
       std::ranges::views::zip(std::array{0.0, 0.0, 0.0, 0.0, 81.0}, s)) {
    REQUIRE(val == expected);
  }
}

TEST_CASE("BenchmarkDeterminant", "[benchmark]") {
  // Points used in the orientation expression
  std::array<double, 6> points{1.0, 5.1, 323.04, -33.5, 234.1, 8.6};
  using mult_expr = arith_expr<std::multiplies<>, double, double>;
  BENCHMARK("build expr") {
    return (mult_expr{points[1], points[5]} - mult_expr{points[2], points[4]}) -
           (mult_expr{points[0], points[5]} - mult_expr{points[2], points[3]}) +
           (mult_expr{points[0], points[4]} - mult_expr{points[1], points[3]});
  };
  BENCHMARK("no expr floating point") {
    return points[1] * points[5] - points[2] * points[4] -
           points[0] * points[5] + points[2] * points[3] +
           points[0] * points[4] - points[1] * points[3];
  };
  BENCHMARK("floating point") {
    auto e =
        (mult_expr{points[1], points[5]} - mult_expr{points[2], points[4]}) -
        (mult_expr{points[0], points[5]} - mult_expr{points[2], points[3]}) +
        (mult_expr{points[0], points[4]} - mult_expr{points[1], points[3]});
    return fp_eval<double>(e);
  };
  BENCHMARK("exact rounded") {
    auto e =
        (mult_expr{points[1], points[5]} - mult_expr{points[2], points[4]}) -
        (mult_expr{points[0], points[5]} - mult_expr{points[2], points[3]}) +
        (mult_expr{points[0], points[4]} - mult_expr{points[1], points[3]});
    return exactfp_eval<double>(e);
  };
  BENCHMARK("shewchuk floating point") {
    return orient2dfast(&points[0], &points[2], &points[4]);
  };
  BENCHMARK("shewchuk exact rounded") {
    return orient2d(&points[0], &points[2], &points[4]);
  };
}

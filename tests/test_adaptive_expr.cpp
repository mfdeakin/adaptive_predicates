
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"

using namespace adaptive_expr;

static_assert(is_expr<arith_expr<std::plus<>, float, float>>::value);
static_assert(is_expr<const arith_expr<std::plus<>, float, float>>::value);
static_assert(is_expr<arith_expr<std::plus<>, float, float> &>::value);
static_assert(is_expr<const arith_expr<std::plus<>, float, float> &>::value);
static_assert(is_expr<arith_expr<std::plus<>, float, float> &&>::value);
static_assert(is_expr<const arith_expr<std::plus<>, float, float> &&>::value);

TEST_CASE("simple", "[expr_template]") {
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

TEST_CASE("simple", "[expr_template_exact_eval]") {
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
}

TEST_CASE("Benchmark Approx Determinant", "[!benchmark]") {}


#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"

#include <iostream>

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
  std::cout << exactfp_eval<float>(e) << "\n";
}

TEST_CASE("simple", "[expr_template_eval]") {
  auto e = ((arith_expr{} + 4 - 7) * 5 + 3.0 / 6.0);
  std::cout << exactfp_eval<float>(e) << "\n";
}

TEST_CASE("Benchmark Approx Determinant", "[!benchmark]") {}

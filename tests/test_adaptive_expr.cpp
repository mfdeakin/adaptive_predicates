
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <fmt/format.h>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"

#include "shewchuk.h"

using namespace adaptive_expr;
using namespace _impl;

using real = float;

static_assert(is_expr_v<arith_expr<std::plus<>, real, real>>);
static_assert(is_expr_v<const arith_expr<std::plus<>, real, real>>);
static_assert(is_expr_v<arith_expr<std::plus<>, real, real> &>);
static_assert(is_expr_v<const arith_expr<std::plus<>, real, real> &>);
static_assert(is_expr_v<arith_expr<std::plus<>, real, real> &&>);
static_assert(is_expr_v<const arith_expr<std::plus<>, real, real> &&>);

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
  REQUIRE(fp_eval<real>(e) == -7.25);
}

class constructor_test {
public:
  class copy_ref_ex {};
  class move_ex {};
  constructor_test() = default;
  constructor_test(const constructor_test &) { throw copy_ref_ex{}; }
  constructor_test(constructor_test &&) { throw move_ex{}; }
  constructor_test operator=(const constructor_test &) { return *this; }
};

TEST_CASE("expr_template_construction", "[expr_template]") {
  using E = arith_expr<std::plus<>, constructor_test, constructor_test>;
  CHECK_THROWS_AS((E{constructor_test{}, constructor_test{}}),
                  constructor_test::move_ex);
  constructor_test c1;
  constructor_test c2;
  CHECK_THROWS_AS((E{c1, c2}), constructor_test::copy_ref_ex);

  CHECK_THROWS_AS(E{} + E{}, constructor_test::move_ex);
  CHECK_THROWS_AS(E{} - E{}, constructor_test::move_ex);
  CHECK_THROWS_AS(E{} * E{}, constructor_test::move_ex);
  CHECK_THROWS_AS(E{} / E{}, constructor_test::move_ex);
  CHECK_THROWS_AS(E{} < E{}, constructor_test::move_ex);
  CHECK_THROWS_AS(E{} <= E{}, constructor_test::move_ex);
  CHECK_THROWS_AS(E{} == E{}, constructor_test::move_ex);
  CHECK_THROWS_AS(E{} > E{}, constructor_test::move_ex);
  CHECK_THROWS_AS(E{} >= E{}, constructor_test::move_ex);

  REQUIRE_NOTHROW(E{});
  E e1;
  E e2;
  CHECK_THROWS_AS(e1 + e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 - e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 * e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 / e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 < e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 <= e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 == e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 > e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 >= e2, constructor_test::copy_ref_ex);
}

// expression rewriting
static_assert(
    std::is_same_v<
        std::invoke_result_t<decltype(trim_expr<additive_id>), additive_id>,
        additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} + additive_id{})>),
                  decltype(additive_id{} + additive_id{})>,
              additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} - additive_id{})>),
                  decltype(additive_id{} - additive_id{})>,
              additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} * additive_id{})>),
                  decltype(additive_id{} * additive_id{})>,
              additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} + additive_id{} +
                                              additive_id{})>),
                  decltype(additive_id{} + additive_id{} + additive_id{})>,
              additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} + additive_id{} -
                                              additive_id{})>),
                  decltype(additive_id{} + additive_id{} - additive_id{})>,
              additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} - additive_id{} +
                                              additive_id{})>),
                  decltype(additive_id{} - additive_id{} + additive_id{})>,
              additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} - additive_id{} -
                                              additive_id{})>),
                  decltype(additive_id{} - additive_id{} - additive_id{})>,
              additive_id>);

static_assert(std::is_same_v<
              std::invoke_result_t<decltype(trim_expr<real>), real>, real>);
static_assert(
    std::is_same_v<std::invoke_result_t<
                       decltype(trim_expr<decltype(additive_id{} + real{})>),
                       decltype(additive_id{} + real{})>,
                   real>);
static_assert(
    std::is_same_v<std::invoke_result_t<
                       decltype(trim_expr<decltype(additive_id{} - real{})>),
                       decltype(additive_id{} - real{})>,
                   decltype(additive_id{} - real{})>);
static_assert(
    std::is_same_v<std::invoke_result_t<
                       decltype(trim_expr<decltype(additive_id{} * real{})>),
                       decltype(additive_id{} * real{})>,
                   additive_id>);

static_assert(
    std::is_same_v<std::invoke_result_t<
                       decltype(trim_expr<decltype(real{} + additive_id{})>),
                       decltype(real{} + additive_id{})>,
                   real>);
static_assert(
    std::is_same_v<std::invoke_result_t<
                       decltype(trim_expr<decltype(real{} - additive_id{})>),
                       decltype(real{} - additive_id{})>,
                   real>);
static_assert(
    std::is_same_v<std::invoke_result_t<
                       decltype(trim_expr<decltype(real{} * additive_id{})>),
                       decltype(real{} * additive_id{})>,
                   additive_id>);

static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} + additive_id{} +
                                              additive_id{})>),
                  decltype(additive_id{} + additive_id{} + additive_id{})>,
              additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} + additive_id{} -
                                              additive_id{})>),
                  decltype(additive_id{} + additive_id{} - additive_id{})>,
              additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} - additive_id{} +
                                              additive_id{})>),
                  decltype(additive_id{} - additive_id{} + additive_id{})>,
              additive_id>);
static_assert(std::is_same_v<
              std::invoke_result_t<
                  decltype(trim_expr<decltype(additive_id{} - additive_id{} -
                                              additive_id{})>),
                  decltype(additive_id{} - additive_id{} - additive_id{})>,
              additive_id>);

TEST_CASE("expr_template_trim", "[expr_template_rewrite]") {
  REQUIRE(trim_expr(additive_id{} + 5) == 5);
  REQUIRE(fp_eval<real>(trim_expr(additive_id{} - real(5))) == real(-5));
  REQUIRE(fp_eval<real>(balance_expr(additive_id{} - real(5))) == real(-5));
  REQUIRE(fp_eval<real>(balance_expr(additive_id{} - real(5))) == real(-5));
}

// num_partials
static_assert(num_partials_for_exact<decltype(arith_expr{})>() == 0);
static_assert(num_partials_for_exact<decltype(arith_expr{} + 4)>() == 1);
static_assert(num_partials_for_exact<decltype(arith_expr{} + 4 - 7)>() == 2);
static_assert(num_partials_for_exact<decltype((arith_expr{} + 4 - 7) * 5)>() ==
              4);

// enumerate_branches_functor
struct branch_token_tag : public branch_token_s {
  template <template <class> class branch_dir>
  using append_branch = branch_dir<branch_token_tag>;
};

static_assert(
    std::is_same_v<std::invoke_result_t<
                       enumerate_branches_functor<branch_token_tag>, float>,
                   std::tuple<>>);
static_assert(std::is_same_v<
              std::invoke_result_t<enumerate_branches_functor<branch_token_tag>,
                                   arith_expr<std::plus<>, real, real>>,
              std::tuple<branch_token_tag>>);
static_assert(std::is_same_v<
              std::invoke_result_t<enumerate_branches_functor<branch_token_tag>,
                                   decltype(arith_expr{})>,
              std::tuple<branch_token_tag>>);
static_assert(
    std::is_same_v<
        std::invoke_result_t<enumerate_branches_functor<branch_token_tag>,
                             decltype(arith_expr{} + 1)>,
        std::tuple<branch_token_tag, branch_token_left<branch_token_tag>>>);
static_assert(
    std::is_same_v<
        std::invoke_result_t<enumerate_branches_functor<branch_token_tag>,
                             decltype(1 + arith_expr{})>,
        std::tuple<branch_token_tag, branch_token_right<branch_token_tag>>>);
static_assert(std::is_same_v<
              std::invoke_result_t<enumerate_branches_functor<branch_token_tag>,
                                   decltype(arith_expr{} + arith_expr{})>,
              std::tuple<branch_token_tag, branch_token_left<branch_token_tag>,
                         branch_token_right<branch_token_tag>>>);
static_assert(
    std::is_same_v<
        std::invoke_result_t<enumerate_branches_functor<branch_token_tag>,
                             decltype((arith_expr{} + arith_expr{}) +
                                      arith_expr{})>,
        std::tuple<branch_token_tag, branch_token_left<branch_token_tag>,
                   branch_token_left<branch_token_left<branch_token_tag>>,
                   branch_token_left<branch_token_right<branch_token_tag>>,
                   branch_token_right<branch_token_tag>>>);

TEST_CASE("adaptive_construction", "[adaptive_eval_functor]") {
  REQUIRE((adaptive_eval<decltype(arith_expr{}), real>{}).eval(arith_expr{}) ==
          0.0);
  REQUIRE((adaptive_eval<decltype(arith_expr{} + 7.0), real>{})
              .eval(arith_expr{} + 7.0) == 7.0);
  REQUIRE((adaptive_eval<decltype((arith_expr{} + 7.0) * 2.0), real>{})
              .eval((arith_expr{} + 7.0) * 2.0) == 14.0);
  REQUIRE((adaptive_eval<decltype((arith_expr{} + 7.0) * 2.0 - 14.0), real>{})
              .eval((arith_expr{} + 7.0) * 2.0 - 14.0) == 0.0);

  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  std::array<std::array<real, 2>, 3> points{
      std::array<real, 2>{-0.257641255855560303, 0.282396793365478516},
      std::array<real, 2>{-0.734969973564147949, 0.716774165630340576},
      std::array<real, 2>{0.48675835132598877, -0.395019501447677612}};

  using mult_expr = arith_expr<std::multiplies<>, real, real>;
  const auto build_e = [points]() constexpr {
    const auto cross_expr = [](const std::array<real, 2> &lhs,
                               const std::array<real, 2> &rhs) constexpr {
      return mult_expr{lhs[x], rhs[y]} - mult_expr{lhs[y], rhs[x]};
    };
    return cross_expr(points[1], points[2]) - cross_expr(points[0], points[2]) +
           cross_expr(points[0], points[1]);
  };
  // std::tuple<> t = enumerate_branches_functor<branch_token_tag>()(build_e());
  REQUIRE(adaptive_eval<std::invoke_result_t<decltype(build_e)>, real>{}.eval(build_e()) < 0.0);
}

// adaptive relative error bounds
static_assert(max_rel_error<real, decltype(additive_id{})>() == 0);

static_assert(max_rel_error<real, decltype(arith_expr{})>() ==
              std::numeric_limits<real>::epsilon() / 2);

static_assert(max_rel_error<real, decltype(arith_expr{} + 4)>() ==
              std::numeric_limits<real>::epsilon());

static_assert(max_rel_error<real, decltype(arith_expr{} + 4 - 7)>() ==
              std::numeric_limits<real>::epsilon() * 2);

static_assert(max_rel_error<real, decltype((arith_expr{} + 4 - 7) * 5)>() ==
              std::numeric_limits<real>::epsilon() * 4);

static_assert(max_rel_error<real, decltype((arith_expr{} + 4 - 7) *
                                           (arith_expr{} + 5))>() >
              std::numeric_limits<real>::epsilon() * 4);
static_assert(max_rel_error<real, decltype((arith_expr{} + 4 - 7) *
                                           (arith_expr{} + 5))>() <
              std::numeric_limits<real>::epsilon() * 16);

TEST_CASE("expr_template_eval_simple", "[expr_template_eval]") {
  auto e = ((arith_expr{} + 4 - 7) * 5 + 3.0 / 6.0);
  REQUIRE(exactfp_eval<real>(e.lhs().lhs().lhs().lhs()) == 0.0);
  REQUIRE(exactfp_eval<real>(e.lhs().lhs().lhs()) == 4.0);
  REQUIRE(exactfp_eval<real>(e.lhs().lhs()) == -3.0);
  REQUIRE(exactfp_eval<real>(e.lhs()) == -15.0);
  REQUIRE(exactfp_eval<real>(e) == -14.5);
  std::vector<real> fp_vals{5.0, 10.0, 11.0, 11.0, 44.0};
  REQUIRE(merge_sum(std::span{fp_vals}) == 81.0);
  REQUIRE(correct_eval<real>(e));
  REQUIRE(!correct_eval<real>(e + 14.5));
  REQUIRE(*correct_eval<real>(e.lhs().lhs().lhs().lhs()) == 0.0);
  REQUIRE(*correct_eval<real>(e.lhs().lhs().lhs()) == 4.0);
  REQUIRE(*correct_eval<real>(e.lhs().lhs()) == -3.0);
  REQUIRE(*correct_eval<real>(e.lhs()) == -15.0);
  REQUIRE(*correct_eval<real>(e) == -14.5);

  using mult_expr = arith_expr<std::multiplies<>, real, real>;
  const auto build_e = []() constexpr {
    constexpr std::size_t x = 0;
    constexpr std::size_t y = 1;
    const auto cross_expr = [](const std::array<real, 2> &lhs,
                               const std::array<real, 2> &rhs) constexpr {
      return mult_expr{lhs[x], rhs[y]} - mult_expr{lhs[y], rhs[x]};
    };

    // Points used in the orientation expression
    // These points require adaptive evaluation of the determinant for the
    // orientation result to be correct when real=float
    std::array<std::array<real, 2>, 3> points{
        std::array<real, 2>{-0.257641255855560303, 0.282396793365478516},
        std::array<real, 2>{-0.734969973564147949, 0.716774165630340576},
        std::array<real, 2>{0.48675835132598877, -0.395019501447677612}};
    return cross_expr(points[1], points[2]) - cross_expr(points[0], points[2]) +
           cross_expr(points[0], points[1]);
  };

  const real result =
      exactfp_eval<real>(build_e()); // The exact answer is -9.392445044e-8
  REQUIRE(result < 0.0);

  REQUIRE(!correct_eval<real>(build_e()));
}

TEST_CASE("BenchmarkDeterminant", "[benchmark]") {
  exactinit();
  // Points used in the orientation expression
  // These points require adaptive evaluation of the determinant for the
  // orientation result to be correct when real=float
  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  std::array<std::array<real, 2>, 3> points{
      std::array<real, 2>{-0.257641255855560303, 0.282396793365478516},
      std::array<real, 2>{-0.734969973564147949, 0.716774165630340576},
      std::array<real, 2>{0.48675835132598877, -0.395019501447677612}};

  using mult_expr = arith_expr<std::multiplies<>, real, real>;
  const auto build_e = [points]() constexpr {
    const auto cross_expr = [](const std::array<real, 2> &lhs,
                               const std::array<real, 2> &rhs) constexpr {
      return mult_expr{lhs[x], rhs[y]} - mult_expr{lhs[y], rhs[x]};
    };
    return cross_expr(points[1], points[2]) - cross_expr(points[0], points[2]) +
           cross_expr(points[0], points[1]);
  };
  BENCHMARK("build expr") { return build_e(); };
  BENCHMARK("no expr floating point") {
    return points[1][x] * points[2][y] - points[1][y] * points[2][x] -
           points[0][x] * points[2][y] + points[0][y] * points[2][x] +
           points[0][x] * points[1][y] - points[0][y] * points[1][x];
  };
  BENCHMARK("floating point") {
    const auto e = build_e();
    return fp_eval<real>(e);
  };
  BENCHMARK("exact rounded") {
    const auto e = build_e();
    return exactfp_eval<real>(e);
  };
  BENCHMARK("shewchuk floating point") {
    return orient2dfast(points[0].data(), points[1].data(), points[2].data());
  };
  BENCHMARK("shewchuk exact rounded") {
    return orient2d(points[0].data(), points[1].data(), points[2].data());
  };
}

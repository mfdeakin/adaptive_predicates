
#include <random>

#include <catch2/catch_test_macros.hpp>

#include <fmt/format.h>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"

#include "double_testing_data.hpp"
#include "testing_utils.hpp"

using namespace adaptive_expr;
using namespace _impl;

static_assert(arith_number<int>);
static_assert(arith_number<long long>);
static_assert(arith_number<unsigned int>);
static_assert(arith_number<unsigned long long>);
static_assert(arith_number<real>);
static_assert(arith_number<Vec4f>);

static_assert(scalar_type<int>);
static_assert(scalar_type<long long>);
static_assert(scalar_type<unsigned int>);
static_assert(scalar_type<unsigned long long>);
static_assert(scalar_type<real>);
static_assert(!scalar_type<Vec4f>);

static_assert(!vector_type<int>);
static_assert(!vector_type<long long>);
static_assert(!vector_type<unsigned int>);
static_assert(!vector_type<unsigned long long>);
static_assert(!vector_type<real>);
static_assert(vector_type<Vec4f>);

static_assert(is_expr_v<arith_expr<std::plus<>, real, real>>);
static_assert(is_expr_v<const arith_expr<std::plus<>, real, real>>);
static_assert(is_expr_v<arith_expr<std::plus<>, real, real> &>);
static_assert(is_expr_v<const arith_expr<std::plus<>, real, real> &>);
static_assert(is_expr_v<arith_expr<std::plus<>, real, real> &&>);
static_assert(is_expr_v<const arith_expr<std::plus<>, real, real> &&>);

TEST_CASE("expr_template_structure", "[expr_template]") {
  auto e = ((arith_expr{} + 4 - 7) * 5 + 3.0 / 6.0) / 2;
  using E = decltype(e);
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

  REQUIRE_NOTHROW(E{});
  E e1;
  E e2;
  CHECK_THROWS_AS(e1 + e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 - e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 * e2, constructor_test::copy_ref_ex);
  CHECK_THROWS_AS(e1 / e2, constructor_test::copy_ref_ex);
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

static_assert(
    std::is_same_v<
        decltype(rewrite_minus(mult_expr(3.0, 5.1) - mult_expr(0.0, 1.0))),
        decltype(mult_expr(3.0, 5.1) + (additive_id{} - mult_expr(0.0, 1.0)))>);
static_assert(
    std::is_same_v<decltype(_balance_expr_impl(mult_expr(3.0, 5.1) -
                                               mult_expr(0.0, 1.0))),
                   decltype(mult_expr(3.0, 5.1) - mult_expr(0.0, 1.0))>);
static_assert(std::is_same_v<
              decltype(trim_expr(mult_expr(3.0, 5.1) - mult_expr(0.0, 1.0))),
              decltype(mult_expr(3.0, 5.1) - mult_expr(0.0, 1.0))>);
static_assert(std::is_same_v<
              decltype(balance_expr(mult_expr(3.0, 5.1) - mult_expr(0.0, 1.0))),
              decltype(mult_expr(3.0, 5.1) - mult_expr(0.0, 1.0))>);

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

TEST_CASE("adaptive_construction", "[eval_checked_fast_functor]") {
  REQUIRE(eval_checked_fast<real>(arith_expr{}).first == 0.0);
  REQUIRE(eval_checked_fast<real>(arith_expr{} + 7.0).first == 7.0);
  REQUIRE(eval_checked_fast<real>((arith_expr{} + 7.0) * 2.0).first == 14.0);
  REQUIRE(std::isnan(
      eval_checked_fast<real>((arith_expr{} + 7.0) * 2.0 - 14.0).first));

  for (auto [points, expected] : orient2d_cases) {
    const auto e = build_orient2d_case(points);
    const auto result = eval_checked_fast<real>(e).first;
    CHECK((std::isnan(result) || check_sign(expected, result)));
  }
}

TEST_CASE("adaptive_construction", "[eval_with_err_functor]") {
  REQUIRE(eval_with_err<real>(arith_expr{}).first == 0.0);
  REQUIRE(eval_with_err<real>(arith_expr{} + 7.0).first == 7.0);
  REQUIRE(eval_with_err<real>((arith_expr{} + 7.0) * 2.0).first == 14.0);
  REQUIRE(
      std::isnan(eval_with_err<real>((arith_expr{} + 7.0) * 2.0 - 14.0).first));

  for (auto [points, expected] : orient2d_cases) {
    const auto e = build_orient2d_case(points);
    const auto result = eval_with_err<real>(e).first;
    CHECK((std::isnan(result) || check_sign(expected, result)));
  }
}

TEST_CASE("adaptive_construction", "[exact_eval_functor]") {
  REQUIRE(exactfp_eval<real>(arith_expr{}) == 0.0);
  REQUIRE(exactfp_eval<real>(arith_expr{} + 7.0) == 7.0);
  REQUIRE(exactfp_eval<real>((arith_expr{} + 7.0) * 2.0) == 14.0);
  REQUIRE(exactfp_eval<real>((arith_expr{} + 7.0) * 2.0 - 14.0) == 0.0);

  for (auto [points, expected] : orient2d_cases) {
    const auto e = build_orient2d_case(points);
    CHECK(check_sign(expected, exactfp_eval<real>(e)));
  }
}

TEST_CASE("adaptive_construction", "[adaptive_eval_functor]") {
  REQUIRE(adaptive_eval<real>(arith_expr{}) == 0.0);
  REQUIRE(adaptive_eval<real>(arith_expr{} + 7.0) == 7.0);
  REQUIRE(adaptive_eval<real>((arith_expr{} + 7.0) * 2.0) == 14.0);
  REQUIRE(adaptive_eval<real>((arith_expr{} + 7.0) * 2.0 - 14.0) == 0.0);

  for (auto [points, expected] : orient2d_cases) {
    const auto e = build_orient2d_case(points);
    CHECK(check_sign(expected, adaptive_eval<real>(e)));
  }
}

#if defined(__cpp_lib_ranges_slide) && defined(__FMA__)
TEST_CASE("eval_checked_fast_simd", "[simd_eval_checked_fast_functor]") {
  for (auto window : orient2d_cases | std::views::slide(vec_size)) {
    const auto [e, expected] = build_orient2d_vec_case(window);
    const auto result = eval_checked_fast<Vec4d>(e).first;
    for (size_t i = 0; i < vec_size; ++i) {
      CHECK((std::isnan(result[i]) || check_sign(expected[i], result[i])));
    }
  }
}

TEST_CASE("eval_with_err_simd", "[simd_eval_with_err_functor]") {
  for (auto window : orient2d_cases | std::views::slide(vec_size)) {
    const auto [e, expected] = build_orient2d_vec_case(window);
    const auto [result, _] = eval_with_err<Vec4d>(e);
    for (size_t i = 0; i < vec_size; ++i) {
      CHECK((std::isnan(result[i]) || check_sign(expected[i], result[i])));
    }
  }
}

TEST_CASE("exact_eval_simd", "[simd_exact_eval_functor]") {
  for (auto window : orient2d_cases | std::views::slide(vec_size)) {
    const auto [e, expected] = build_orient2d_vec_case(window);
    const Vec4d result = exactfp_eval<Vec4d>(e);
    for (size_t i = 0; i < vec_size; ++i) {
      CHECK(check_sign(expected[i], result[i]));
    }
  }
}
#endif // defined(__cpp_lib_ranges_slide) && defined(__FMA__)

// adaptive relative error bounds
static_assert(max_rel_error<real, decltype(additive_id{})>() == 0);

static_assert(max_rel_error<real, decltype(arith_expr{})>() ==
              std::numeric_limits<real>::epsilon() / 2);

static_assert(max_rel_error<real, decltype(arith_expr{} + 4)>() ==
              std::numeric_limits<real>::epsilon());

static_assert(max_rel_error<real, decltype(arith_expr{} + 4 - 7)>() <=
              std::numeric_limits<real>::epsilon() * 2);

static_assert(max_rel_error<real, decltype((arith_expr{} + 4 - 7) * 5)>() <=
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

  REQUIRE(!std::isnan(eval_with_err<real>(e).first));
  REQUIRE(std::isnan(eval_with_err<real>(e + 14.5).first));
  REQUIRE(eval_with_err<real>(e.lhs().lhs().lhs().lhs()).first == 0.0);
  REQUIRE(eval_with_err<real>(e.lhs().lhs().lhs()).first == 4.0);
  REQUIRE(eval_with_err<real>(e.lhs().lhs()).first == -3.0);
  REQUIRE(eval_with_err<real>(e.lhs()).first == -15.0);
  REQUIRE(eval_with_err<real>(e).first == -14.5);

  const auto points = orient2d_cases[0].first;
  const auto expected = orient2d_cases[0].second;
  const auto expr = build_orient2d_case(points);
  REQUIRE(correct_eval<real>(expr));
  REQUIRE(check_sign(*correct_eval<real>(expr), expected));
  REQUIRE(!std::isnan(eval_with_err<real>(expr).first));
  REQUIRE(check_sign(eval_with_err<real>(expr).first, expected));
}

TEST_CASE("nonoverlapping", "[eval_utils]") {
  CHECK(is_nonoverlapping(std::vector<real>{}));
  CHECK(is_nonoverlapping(std::vector<real>{0}));
  CHECK(is_nonoverlapping(std::vector<real>{0, 0, 0}));
  CHECK(is_nonoverlapping(std::vector<real>{0.125}));
  CHECK(is_nonoverlapping(std::vector<real>{0.125, 0.25}));
  CHECK(is_nonoverlapping(std::vector<real>{0.125, -0.25, 1.5}));
  CHECK(is_nonoverlapping(
      std::vector<real>{0.0, 0.0, 0.0, 0.0, -0.375, 0.5, -1.0, 14.0}));
  CHECK(is_nonoverlapping(std::vector<real>{-0.375, 0.5, -1.0, 14.0}));
  CHECK(is_nonoverlapping(std::vector<real>{-0.375, 0.5, -1.0, 14.0}));
  CHECK(is_nonoverlapping(std::vector<real>{-0.375, 0.5, -1.0, 14.0}));
  CHECK(is_nonoverlapping(std::vector<real>{-0.375, 0.5, -1.0, 14.0}));
  CHECK(is_nonoverlapping(std::vector<real>{-0.375, 0.5, -1.0, 14.0}));
  CHECK(!is_nonoverlapping(std::vector<real>{0.125, 0.375}));
  CHECK(!is_nonoverlapping(std::vector<real>{0.125, -0.25, 1.75}));
  CHECK(!is_nonoverlapping(std::vector<real>{-0.375, 0.5, 0, -1.0, 15.0}));
  CHECK(!is_nonoverlapping(std::vector<real>{-0.375, 0, -0.75, 0, 1.0, 14.0}));
  CHECK(
      !is_nonoverlapping(std::vector<real>{-0.375, 0.5, 1.5, 0, 0, 0, -14.0}));

  std::vector<real> merge_test1{0,
                                -1.5436178396078065e-49,
                                -2.184158631330676e-33,
                                -1.1470824290427116e-16,
                                0,
                                1.0353799381025734e-34,
                                -1.7308376953906192e-17,
                                -1.2053999999999998};
  const auto midpoint1 = merge_test1.begin() + 4;
  CHECK(*midpoint1 == real{0});
  CHECK(is_nonoverlapping(std::span{merge_test1.begin(), midpoint1}));
  CHECK(is_nonoverlapping(std::span{midpoint1, merge_test1.end()}));
  REQUIRE(midpoint1 > merge_test1.begin());
  REQUIRE(midpoint1 < merge_test1.end());

  merge_sum_linear(merge_test1, midpoint1);
  CHECK(is_nonoverlapping(merge_test1));

  // Same strongly non-overlapping sequence but for merge_sum_linear_fast
  std::vector<real> merge_test2{0,
                                -1.5436178396078065e-49,
                                -2.184158631330676e-33,
                                -1.1470824290427116e-16,
                                0,
                                1.0353799381025734e-34,
                                -1.7308376953906192e-17,
                                -1.2053999999999998};
  const auto midpoint2 = merge_test2.begin() + 4;
  merge_sum_linear_fast(merge_test2, midpoint2);
  CHECK(is_nonoverlapping(merge_test2));
}

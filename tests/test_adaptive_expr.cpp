
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

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Point_2.h>

#include "testing_data.hpp"

using namespace adaptive_expr;
using namespace _impl;

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
  REQUIRE(adaptive_eval<real>(arith_expr{}) == 0.0);
  REQUIRE(adaptive_eval<real>(arith_expr{} + 7.0) == 7.0);
  REQUIRE(adaptive_eval<real>((arith_expr{} + 7.0) * 2.0) == 14.0);
  REQUIRE(adaptive_eval<real>((arith_expr{} + 7.0) * 2.0 - 14.0) == 0.0);

  std::mt19937_64 gen(std::random_device{}());
  std::uniform_real_distribution<real> dist(-1.0, 1.0);
  for (auto [points, expected] : orient2d_cases) {
    const auto e = build_orient2d_case(points);
    CHECK(check_sign(expected, adaptive_eval<real>(e)));
    CHECK(check_sign(expected, exactfp_eval<real>(e)));
  }
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

  const auto points = orient2d_cases[0].first;
  const real result = exactfp_eval<real>(
      build_orient2d_case(points)); // The exact answer is -9.392445044e-8
  REQUIRE(result < 0.0);

  REQUIRE(!correct_eval<real>(build_orient2d_case(points)));
}

TEST_CASE("BenchmarkDeterminant", "[benchmark]") {
  exactinit();
  const auto points1 = orient2d_cases[0].first;
  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel> pt0(
      points1[0][x], points1[0][y]);
  CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel> pt1(
      points1[1][x], points1[1][y]);
  CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel> pt2(
      points1[2][x], points1[2][y]);
  BENCHMARK("build expr 1") { return build_orient2d_case(points1); };
  BENCHMARK("no expr floating point 1") {
    return points1[1][x] * points1[2][y] - points1[1][y] * points1[2][x] -
           points1[0][x] * points1[2][y] + points1[0][y] * points1[2][x] +
           points1[0][x] * points1[1][y] - points1[0][y] * points1[1][x];
  };
  BENCHMARK("floating point 1") {
    const auto e = build_orient2d_case(points1);
    return fp_eval<real>(e);
  };
  BENCHMARK("correct or nothing 1") {
    return correct_eval<real>(build_orient2d_case(points1));
  };
  BENCHMARK("exact rounded 1") {
    const auto e = build_orient2d_case(points1);
    return exactfp_eval<real>(e);
  };
  BENCHMARK("adaptive 1") {
    const auto e = build_orient2d_case(points1);
    return adaptive_eval<real>(e);
  };
  BENCHMARK("shewchuk floating point 1") {
    return orient2dfast(points1[0].data(), points1[1].data(),
                        points1[2].data());
  };
  BENCHMARK("shewchuk exact rounded 1") {
    return orient2d(points1[0].data(), points1[1].data(), points1[2].data());
  };
  BENCHMARK("cgal exact rounded 1") {
    return CGAL::orientation(pt0, pt1, pt2);
  };

  const auto points2 = orient2d_cases[1].first;
  pt0 = CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel>(
      points2[0][x], points2[0][y]);
  pt1 = CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel>(
      points2[1][x], points2[1][y]);
  pt2 = CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel>(
      points2[2][x], points2[2][y]);
  BENCHMARK("build expr 2") { return build_orient2d_case(points2); };
  BENCHMARK("no expr floating point 2") {
    return points2[1][x] * points2[2][y] - points2[1][y] * points2[2][x] -
           points2[0][x] * points2[2][y] + points2[0][y] * points2[2][x] +
           points2[0][x] * points2[1][y] - points2[0][y] * points2[1][x];
  };
  BENCHMARK("floating point 2") {
    const auto e = build_orient2d_case(points2);
    return fp_eval<real>(e);
  };
  BENCHMARK("correct or nothing 2") {
    return correct_eval<real>(build_orient2d_case(points2));
  };
  BENCHMARK("exact rounded 2") {
    const auto e = build_orient2d_case(points2);
    return exactfp_eval<real>(e);
  };
  BENCHMARK("adaptive 2") {
    const auto e = build_orient2d_case(points2);
    return adaptive_eval<real>(e);
  };
  BENCHMARK("shewchuk floating point 2") {
    return orient2dfast(points2[0].data(), points2[1].data(),
                        points2[2].data());
  };
  BENCHMARK("shewchuk exact rounded 2") {
    return orient2d(points2[0].data(), points2[1].data(), points2[2].data());
  };
  BENCHMARK("cgal exact rounded 2") {
    return CGAL::orientation(pt0, pt1, pt2);
  };

  const auto points3 = orient2d_cases[15].first;
  pt0 = CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel>(
      points3[0][x], points3[0][y]);
  pt1 = CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel>(
      points3[1][x], points3[1][y]);
  pt2 = CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel>(
      points3[2][x], points3[2][y]);
  BENCHMARK("build expr 3") { return build_orient2d_case(points3); };
  BENCHMARK("no expr floating point 3") {
    return points3[1][x] * points3[2][y] - points3[1][y] * points3[2][x] -
           points3[0][x] * points3[2][y] + points3[0][y] * points3[2][x] +
           points3[0][x] * points3[1][y] - points3[0][y] * points3[1][x];
  };
  BENCHMARK("floating point 3") {
    const auto e = build_orient2d_case(points3);
    return fp_eval<real>(e);
  };
  BENCHMARK("correct or nothing 3") {
    return correct_eval<real>(build_orient2d_case(points3));
  };
  BENCHMARK("exact rounded 3") {
    const auto e = build_orient2d_case(points3);
    return exactfp_eval<real>(e);
  };
  BENCHMARK("adaptive 3") {
    const auto e = build_orient2d_case(points3);
    return adaptive_eval<real>(e);
  };
  BENCHMARK("shewchuk floating point 3") {
    return orient2dfast(points3[0].data(), points3[1].data(),
                        points3[2].data());
  };
  BENCHMARK("shewchuk exact rounded 3") {
    return orient2d(points3[0].data(), points3[1].data(), points3[2].data());
  };
  BENCHMARK("cgal exact rounded 3") {
    return CGAL::orientation(pt0, pt1, pt2);
  };
}

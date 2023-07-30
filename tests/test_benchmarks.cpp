
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"

#include "shewchuk.h"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Point_2.h>

#include "double_testing_data.hpp"
#include "testing_utils.hpp"

using namespace adaptive_expr;

TEST_CASE("BenchmarkDeterminant", "[benchmark]") {
  Shewchuk::exactinit();
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
  BENCHMARK("matrix adaptive 1") {
    auto mtx = build_orient2d_matrix(points1);
    return determinant<real>(
        mdspan<real, extents<std::size_t, 3, 3>>(mtx.data()));
  };
  BENCHMARK("shewchuk floating point 1") {
    return Shewchuk::orient2dfast(points1[0].data(), points1[1].data(),
                                  points1[2].data());
  };
  BENCHMARK("shewchuk exact rounded 1") {
    return Shewchuk::orient2d(points1[0].data(), points1[1].data(),
                              points1[2].data());
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
  BENCHMARK("matrix adaptive 2") {
    auto mtx = build_orient2d_matrix(points2);
    return determinant<real>(
        mdspan<real, extents<std::size_t, 3, 3>>(mtx.data()));
  };
  BENCHMARK("shewchuk floating point 2") {
    return Shewchuk::orient2dfast(points2[0].data(), points2[1].data(),
                                  points2[2].data());
  };
  BENCHMARK("shewchuk exact rounded 2") {
    return Shewchuk::orient2d(points2[0].data(), points2[1].data(),
                              points2[2].data());
  };
  BENCHMARK("cgal exact rounded 2") {
    return CGAL::orientation(pt0, pt1, pt2);
  };

  const auto points3 = orient2d_cases[2].first;
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
  BENCHMARK("matrix adaptive 3") {
    auto mtx = build_orient2d_matrix(points3);
    return determinant<real>(
        mdspan<real, extents<std::size_t, 3, 3>>(mtx.data()));
  };
  BENCHMARK("shewchuk floating point 3") {
    return Shewchuk::orient2dfast(points3[0].data(), points3[1].data(),
                                  points3[2].data());
  };
  BENCHMARK("shewchuk exact rounded 3") {
    return Shewchuk::orient2d(points3[0].data(), points3[1].data(),
                              points3[2].data());
  };
  BENCHMARK("cgal exact rounded 3") {
    return CGAL::orientation(pt0, pt1, pt2);
  };

  const auto points4 = orient2d_cases[17].first;
  pt0 = CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel>(
      points4[0][x], points4[0][y]);
  pt1 = CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel>(
      points4[1][x], points4[1][y]);
  pt2 = CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel>(
      points4[2][x], points4[2][y]);
  BENCHMARK("build expr 4") { return build_orient2d_case(points4); };
  BENCHMARK("no expr floating point 4") {
    return points4[1][x] * points4[2][y] - points4[1][y] * points4[2][x] -
           points4[0][x] * points4[2][y] + points4[0][y] * points4[2][x] +
           points4[0][x] * points4[1][y] - points4[0][y] * points4[1][x];
  };
  BENCHMARK("floating point 4") {
    const auto e = build_orient2d_case(points4);
    return fp_eval<real>(e);
  };
  BENCHMARK("correct or nothing 4") {
    return correct_eval<real>(build_orient2d_case(points4));
  };
  BENCHMARK("exact rounded 4") {
    const auto e = build_orient2d_case(points4);
    return exactfp_eval<real>(e);
  };
  BENCHMARK("adaptive 4") {
    const auto e = build_orient2d_case(points4);
    return adaptive_eval<real>(e);
  };
  BENCHMARK("matrix adaptive 4") {
    auto mtx = build_orient2d_matrix(points4);
    return determinant<real>(
        mdspan<real, extents<std::size_t, 3, 3>>(mtx.data()));
  };
  BENCHMARK("shewchuk floating point 4") {
    return Shewchuk::orient2dfast(points4[0].data(), points4[1].data(),
                                  points4[2].data());
  };
  BENCHMARK("shewchuk exact rounded 4") {
    return Shewchuk::orient2d(points4[0].data(), points4[1].data(),
                              points4[2].data());
  };
  BENCHMARK("cgal exact rounded 4") {
    return CGAL::orientation(pt0, pt1, pt2);
  };
}

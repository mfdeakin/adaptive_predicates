
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"

#include "shewchuk.h"

#ifdef HAS_CGAL
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Point_2.h>
#endif // HAS_CGAL

#include "double_testing_data.hpp"
#include "testing_utils.hpp"

using namespace adaptive_expr;

#define BENCHMARK_CASE(TEST_PREAMBLE, BENCHMARK_CALL_METHOD, name, tags,       \
                       method, points)                                         \
  TEST_CASE(name " - [benchmark]" tags, "[benchmark]" tags) {                  \
    TEST_PREAMBLE(points)                                                      \
    BENCHMARK(name) { return BENCHMARK_CALL_METHOD(method); };                 \
  }

#ifdef HAS_CGAL
static constexpr std::size_t x = 0;
static constexpr std::size_t y = 1;
static constexpr std::size_t z = 2;

#define CGAL_ORIENT2D_PREAMBLE(points)                                         \
  CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel> cgal_pt0(   \
      points[0][x], points[0][y]);                                             \
  CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel> cgal_pt1(   \
      points[1][x], points[1][y]);                                             \
  CGAL::Point_2<CGAL::Exact_predicates_exact_constructions_kernel> cgal_pt2(   \
      points[2][x], points[2][y]);

#define CGAL_ORIENT2D_CALL_METHOD(method) method(cgal_pt0, cgal_pt1, cgal_pt2)

#define CGAL_ORIENT2D_CASE(name, tags, points)                                 \
  BENCHMARK_CASE(CGAL_ORIENT2D_PREAMBLE, CGAL_ORIENT2D_CALL_METHOD, name,      \
                 "[cgal][orient2d]" tags, CGAL::orientation, points)
#else // HAS_CGAL
#define CGAL_ORIENT2D_CASE(name, tags, points)
#endif // HAS_CGAL

#define SHEWCHUK_ORIENT2D_PREAMBLE(points)                                     \
  Shewchuk::exactinit();                                                       \
  const real *sh_pt0 = points[0].data();                                       \
  const real *sh_pt1 = points[1].data();                                       \
  const real *sh_pt2 = points[2].data();

#define SHEWCHUK_ORIENT2D_CALL_METHOD(method) method(sh_pt0, sh_pt1, sh_pt2)

#define SHEWCHUK_ORIENT2D_CASE(name, tags, points)                             \
  BENCHMARK_CASE(SHEWCHUK_ORIENT2D_PREAMBLE, SHEWCHUK_ORIENT2D_CALL_METHOD,    \
                 name, "[shewchuk][orient2d]" tags, Shewchuk::orient2d,        \
                 points)

#define AE_ORIENT2D_PREAMBLE(points)                                           \
  const auto expression = pt_orient_expr(points);

#define AE_ORIENT2D_CALL_METHOD(method) method(expression)

#define AE_ORIENT2D_CASE(name, tags, method, points)                           \
  BENCHMARK_CASE(AE_ORIENT2D_PREAMBLE, AE_ORIENT2D_CALL_METHOD, name,          \
                 "[ae_expression][orient2d]" tags, method, points)

#define AE_MTX_ORIENT2D_PREAMBLE(points)                                       \
  auto mtx_array = build_orient2d_matrix(points);                              \
  const mdspan<real, extents<std::size_t, 3, 3>> expression(mtx_array.data());

#define AE_MTX_ORIENT2D_CALL_METHOD(method) method(expression)

#define AE_MTX_ORIENT2D_CASE(name, tags, method, points)                       \
  BENCHMARK_CASE(AE_MTX_ORIENT2D_PREAMBLE, AE_MTX_ORIENT2D_CALL_METHOD, name,  \
                 "[ae_expression][matrix][orient2d]" tags, method, points)

#define ORIENT2D_POINTS_CASE(points, tag)                                      \
  AE_ORIENT2D_CASE("Floating point evaluation", "[fp_eval]" tag,               \
                   fp_eval<real>, points)                                      \
  AE_ORIENT2D_CASE("Fast checked evaluation", "[eval_checked_fast]" tag,       \
                   eval_checked_fast<real>, points)                            \
  AE_ORIENT2D_CASE("Correct or nothing", "[correct_eval]" tag,                 \
                   correct_eval<real>, points)                                 \
  AE_ORIENT2D_CASE("Exact evaluation", "[exactfp_eval]" tag,                   \
                   exactfp_eval<real>, points)                                 \
  AE_ORIENT2D_CASE("Adaptive evaluation", "[adaptive_eval]" tag,               \
                   adaptive_eval<real>, points)                                \
  AE_MTX_ORIENT2D_CASE("Matrix Adaptive evaluation", "[adaptive_eval]" tag,    \
                       determinant<real>, points)                              \
  SHEWCHUK_ORIENT2D_CASE("Shewchuk Orient2D", tag, points)                     \
  CGAL_ORIENT2D_CASE("CGAL Orient2D", tag, points)

ORIENT2D_POINTS_CASE(orient2d_cases[0].first, "[points0]")
ORIENT2D_POINTS_CASE(orient2d_cases[1].first, "[points1]")
ORIENT2D_POINTS_CASE(orient2d_cases[2].first, "[points2]")
ORIENT2D_POINTS_CASE(orient2d_cases[17].first, "[points3]")

#ifdef HAS_CGAL
#define CGAL_INCIRCLE3D_PREAMBLE(points)                                       \
  CGAL::Point_3<CGAL::Exact_predicates_exact_constructions_kernel> cgal_pt0(   \
      points[0][x], points[0][y], points[0][z]);                               \
  CGAL::Point_3<CGAL::Exact_predicates_exact_constructions_kernel> cgal_pt1(   \
      points[1][x], points[1][y], points[1][z]);                               \
  CGAL::Point_3<CGAL::Exact_predicates_exact_constructions_kernel> cgal_pt2(   \
      points[2][x], points[2][y], points[2][z]);                               \
  CGAL::Point_3<CGAL::Exact_predicates_exact_constructions_kernel> cgal_pt3(   \
      points[3][x], points[3][y], points[3][z]);                               \
  CGAL::Sphere_3<CGAL::Exact_predicates_exact_constructions_kernel>            \
      cgal_sphere(cgal_pt0, cgal_pt1, cgal_pt2, cgal_pt3);                     \
  CGAL::Point_3<CGAL::Exact_predicates_exact_constructions_kernel> cgal_pt4(   \
      points[4][x], points[4][y], points[4][z]);

#define CGAL_INCIRCLE3D_CALL_METHOD(method) method(cgal_pt4)

#define CGAL_INCIRCLE3D_CASE(name, tags, points)                               \
  BENCHMARK_CASE(CGAL_INCIRCLE3D_PREAMBLE, CGAL_INCIRCLE3D_CALL_METHOD, name,  \
                 "[cgal][incircle3d]" tags, cgal_sphere.bounded_side, points)
#else // HAS_CGAL
#define CGAL_INCIRCLE3D_CASE(name, tags, points)
#endif // HAS_CGAL

#define SHEWCHUK_INCIRCLE3D_PREAMBLE(points)                                   \
  Shewchuk::exactinit();                                                       \
  const real *sh_pt0 = points[0].data();                                       \
  const real *sh_pt1 = points[1].data();                                       \
  const real *sh_pt2 = points[2].data();                                       \
  const real *sh_pt3 = points[3].data();                                       \
  const real *sh_pt4 = points[4].data();

#define SHEWCHUK_INCIRCLE3D_CALL_METHOD(method)                                \
  method(sh_pt0, sh_pt1, sh_pt2, sh_pt3, sh_pt4)

#define SHEWCHUK_INCIRCLE3D_CASE(name, tags, points)                           \
  BENCHMARK_CASE(SHEWCHUK_INCIRCLE3D_PREAMBLE,                                 \
                 SHEWCHUK_INCIRCLE3D_CALL_METHOD, name,                        \
                 "[shewchuk][incircle3d]" tags, Shewchuk::insphere, points)

#define AE_INCIRCLE3D_PREAMBLE(points)                                         \
  const auto expression = pt_incircle_expr(points);

#define AE_INCIRCLE3D_CALL_METHOD(method) method(expression)

#define AE_INCIRCLE3D_CASE(name, tags, method, points)                         \
  BENCHMARK_CASE(AE_INCIRCLE3D_PREAMBLE, AE_INCIRCLE3D_CALL_METHOD, name,      \
                 "[ae_expression][incircle3d]" tags, method, points)

#define INCIRCLE3D_POINTS_CASE(points, tag)                                    \
  AE_INCIRCLE3D_CASE("Floating point evaluation", "[fp_eval]" tag,             \
                     fp_eval<real>, points)                                    \
  AE_INCIRCLE3D_CASE("Fast checked evaluation", "[eval_checked_fast]" tag,     \
                     eval_checked_fast<real>, points)                          \
  AE_INCIRCLE3D_CASE("Correct or nothing", "[correct_eval]" tag,               \
                     correct_eval<real>, points)                               \
  AE_INCIRCLE3D_CASE("Exact evaluation", "[exactfp_eval]" tag,                 \
                     exactfp_eval<real>, points)                               \
  AE_INCIRCLE3D_CASE("Adaptive evaluation", "[adaptive_eval]" tag,             \
                     adaptive_eval<real>, points)                              \
  SHEWCHUK_INCIRCLE3D_CASE("Shewchuk InCircle3D", tag, points)                 \
  CGAL_INCIRCLE3D_CASE("CGAL InCircle3D", tag, points)

INCIRCLE3D_POINTS_CASE(in_circle3d_cases[0].first, "[points0]")
INCIRCLE3D_POINTS_CASE(in_circle3d_cases[1].first, "[points1]")
INCIRCLE3D_POINTS_CASE(in_circle3d_cases[2].first, "[points2]")
INCIRCLE3D_POINTS_CASE(in_circle3d_cases[3].first, "[points3]")
INCIRCLE3D_POINTS_CASE(in_circle3d_cases[4].first, "[points4]")
INCIRCLE3D_POINTS_CASE(in_circle3d_cases[5].first, "[points5]")

#define AE_VEC_ORIENT2D_PREAMBLE(points)                                       \
  const auto [expression, _] = build_orient2d_vec_case(points);

#define AE_VEC_ORIENT2D_CALL_METHOD(method) method(expression)

#define AE_VEC_ORIENT2D_CASE(name, tags, method, points)                       \
  BENCHMARK_CASE(AE_VEC_ORIENT2D_PREAMBLE, AE_VEC_ORIENT2D_CALL_METHOD, name,  \
                 "[ae_expression][orient2d][vectorized]" tags, method, points)

#define ORIENT2D_VEC_POINTS_CASE(points, tag)                                  \
  AE_VEC_ORIENT2D_CASE("Floating point vector evaluation", "[fp_eval]" tag,    \
                       fp_eval<Vec4d>, points)                                 \
  AE_VEC_ORIENT2D_CASE("Fast checked vector evaluation",                       \
                       "[eval_checked_fast]" tag, eval_checked_fast<Vec4d>,    \
                       points)                                                 \
  AE_VEC_ORIENT2D_CASE("Interval checked vector evaluation",                   \
                       "[eval_with_err]" tag, eval_with_err<Vec4d>, points)    \
  AE_VEC_ORIENT2D_CASE("Exact vector evaluation", "[exactfp_eval]" tag,        \
                       exactfp_eval<Vec4d>, points)

ORIENT2D_VEC_POINTS_CASE((std::array{
                             orient2d_cases[0],
                             orient2d_cases[1],
                             orient2d_cases[2],
                             orient2d_cases[17],
                         }),
                         "")


#ifndef TESTING_UTILS_HPP
#define TESTING_UTILS_HPP

#include <cmath>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"

#if __cpp_lib_mdspan >= 202207L
#include <mdspan>
using std::extents;
using std::mdarray;
using std::mdspan;
#else
#include <mdspan.hpp>
using std::experimental::extents;
using std::experimental::mdarray;
using std::experimental::mdspan;
#endif

template <adaptive_expr::arith_number eval_type>
constexpr bool check_sign(eval_type correct, eval_type check) {
  if (correct == eval_type{0.0}) {
    return check == eval_type{0.0};
  } else if (check == eval_type{0.0}) {
    return false;
  } else {
    return std::signbit(correct) == std::signbit(check);
  }
}

template <adaptive_expr::arith_number eval_type>
constexpr auto
build_orient2d_case(const std::array<std::array<eval_type, 2>, 3> &points) {
  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  const auto cross_expr = [](const std::array<eval_type, 2> &lhs,
                             const std::array<eval_type, 2> &rhs) constexpr {
    return adaptive_expr::mult_expr(lhs[x], rhs[y]) -
           adaptive_expr::mult_expr(lhs[y], rhs[x]);
  };
  return cross_expr(points[1], points[2]) - cross_expr(points[0], points[2]) +
         cross_expr(points[0], points[1]);
}

template <adaptive_expr::arith_number eval_type>
constexpr auto
build_orient2d_matrix(const std::array<std::array<eval_type, 2>, 3> &points) {
  mdarray<eval_type, extents<std::size_t, 3, 3>> array;
  array[0, 0] = points[0][0];
  array[0, 1] = points[0][1];
  array[0, 2] = eval_type{1};
  array[1, 0] = points[1][0];
  array[1, 1] = points[1][1];
  array[1, 2] = eval_type{1};
  array[2, 0] = points[2][0];
  array[2, 1] = points[2][1];
  array[2, 2] = eval_type{1};
  return array;
}

template <adaptive_expr::arith_number eval_type>
constexpr auto
build_incircle2d_case(const std::array<std::array<eval_type, 2>, 4> &points) {
  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  const auto dist_expr = [](const std::array<eval_type, 2> pt) constexpr {
    return adaptive_expr::mult_expr(pt[x], pt[x]) +
           adaptive_expr::mult_expr(pt[y], pt[y]);
  };
  const auto cross_expr = [](const std::array<eval_type, 2> &lhs,
                             const std::array<eval_type, 2> &rhs) constexpr {
    return adaptive_expr::mult_expr(lhs[x], rhs[y]) -
           adaptive_expr::mult_expr(lhs[y], rhs[x]);
  };
  const auto incircle_subexpr =
      [dist_expr, cross_expr](const std::array<eval_type, 2> &p0,
                              const std::array<eval_type, 2> &p1,
                              const std::array<eval_type, 2> &p2) constexpr {
        return dist_expr(p0) * cross_expr(p1, p2) -
               dist_expr(p1) * cross_expr(p0, p2) +
               dist_expr(p2) * cross_expr(p0, p1);
      };
  return balance_expr(-incircle_subexpr(points[1], points[2], points[3]) +
                      incircle_subexpr(points[0], points[2], points[3]) -
                      incircle_subexpr(points[0], points[1], points[3]) +
                      incircle_subexpr(points[0], points[1], points[2]));
}

template <typename eval_type, typename row_exprs, std::size_t N>
auto exchange_pivot(const mdspan<row_exprs, extents<std::size_t, N, N>> mtx) {
  eval_type approx_val{0};
  decltype(adaptive_expr::balance_expr(mtx[0, 0])) v;
  std::size_t swap_row;
  for (swap_row = 0; swap_row < N && approx_val == eval_type{0}; ++swap_row) {
    v = adaptive_expr::balance_expr(mtx[swap_row + 1, 0]);
    approx_val = adaptive_expr::adaptive_eval<eval_type>(v);
  }
  if (approx_val != eval_type{0}) {
    for (std::size_t col = 0; col < N; ++col) {
      std::swap(mtx[0, col], mtx[swap_row, col]);
    }
  }
  return std::pair{v, approx_val};
}

// Based on LU Factorization, except written to avoid divisions in the exact
// evaluation.
// Returns an approximate value with sign matching the sign of the determinant.
// Terminates as early as possible for a non-full rank matrix returning 0
//
// LU factorization decomposes a matrix M into a lower triangular matrix L and
// an upper triangular matrix U. L has 1's on its diagonal, so the determinant
// of M is the determinant of U, which is the product of its diagonal.
// This computes U multiplied by some scaling matrices to avoid divisions;
// the determinant is then the determinant of the resulting matrix divided by
// the determinants of the scaling matrices
template <typename eval_type, typename row_exprs, std::size_t N>
constexpr eval_type
determinant(const mdspan<row_exprs, extents<std::size_t, N, N>> mtx) {
  auto v = adaptive_expr::balance_expr(mtx[0, 0]);
  eval_type approx_val = adaptive_expr::adaptive_eval<eval_type>(v);
  if constexpr (N == 1) {
    return approx_val;
  } else {
    bool swap = approx_val == eval_type{0};
    if (swap) {
      std::tie(v, approx_val) = exchange_pivot<eval_type>(mtx);
    }
    using submtx_expr = decltype(adaptive_expr::mult_expr(v, mtx[0, 0]) -
                                 adaptive_expr::mult_expr(v, mtx[0, 0]));
    mdarray<submtx_expr, extents<std::size_t, N - 1, N - 1>> submtx;
    for (std::size_t i = 1; i < N; ++i) {
      const auto s = adaptive_expr::balance_expr(mtx[i, 0]);
      for (std::size_t j = 1; j < N; ++j) {
        submtx[i - 1, j - 1] = adaptive_expr::mult_expr(v, mtx[i, j]) -
                               adaptive_expr::mult_expr(s, mtx[0, j]);
      }
    }
    const mdspan<submtx_expr, extents<std::size_t, N - 1, N - 1>> submtx_span{
        submtx.data()};
    const eval_type subdet = determinant<eval_type>(submtx_span);
    eval_type det = subdet / std::pow(approx_val, N - 2);
    if (swap) {
      det *= eval_type{-1};
    }
    return det;
  }
}

#endif // TESTING_UTILS_HPP

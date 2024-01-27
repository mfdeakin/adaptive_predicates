
#ifndef TESTING_UTILS_HPP
#define TESTING_UTILS_HPP

#include <cmath>

#include <simd_vec/vectorclass.h>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"
#include "ae_geom_exprs.hpp"

constexpr std::size_t vec_size = 4;

// Returns true if the signs match or both are zero
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

constexpr auto build_orient2d_vec_case(std::ranges::range auto window) {
  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  std::array<std::array<Vec4d, 2>, 3> points;
  Vec4d expected;
  for (size_t i = 0; i < vec_size; ++i) {
    for (size_t j = 0; j < points.size(); ++j) {
      points[j][x].insert(i, window[i].first[j][x]);
      points[j][y].insert(i, window[i].first[j][y]);
    }
    expected.insert(i, window[i].second);
  }
  return std::pair{adaptive_expr::pt_orient_expr(points), expected};
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
// Returns an approximate value with sign matching the sign of the
// determinant. Terminates as early as possible for a non-full rank matrix
// returning 0
//
// LU factorization decomposes a matrix M into a lower triangular matrix L
// and an upper triangular matrix U. L has 1's on its diagonal, so the
// determinant of M is the determinant of U, which is the product of its
// diagonal. This computes U multiplied by some scaling matrices to avoid
// divisions; the determinant is then the determinant of the resulting
// matrix divided by the determinants of the scaling matrices
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

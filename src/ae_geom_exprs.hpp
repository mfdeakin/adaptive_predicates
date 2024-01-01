
#ifndef ADAPTIVE_PREDICATES_AE_GEOM_EXPRS_HPP
#define ADAPTIVE_PREDICATES_AE_GEOM_EXPRS_HPP

#include <tuple>
#include <utility>

#include "ae_expr.hpp"

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

#include "ae_expr.hpp"

namespace adaptive_expr {

template <typename left_type, typename right_type, std::size_t N>
constexpr auto pt_diff_expr(const std::array<left_type, N> &left,
                            const std::array<right_type, N> &right);

template <typename left_type, typename right_type>
constexpr auto vec_cross_expr(const std::array<left_type, 2> &left,
                              const std::array<right_type, 2> &right);

template <typename left_type, typename right_type>
constexpr auto vec_cross_expr(const std::array<left_type, 3> &left,
                              const std::array<right_type, 3> &right);

template <adaptive_expr::arith_number coord_type, std::size_t N>
constexpr auto
pt_orient_expr(const std::array<std::array<coord_type, N>, N + 1> &&points);

template <adaptive_expr::arith_number coord_type>
constexpr auto
pt_incircle_expr(const std::array<std::array<coord_type, 2>, 4> &&points);

template <adaptive_expr::arith_number coord_type>
constexpr auto
pt_incircle_expr(const std::array<std::array<coord_type, 3>, 5> &&points);

template <std::size_t row = 0, typename entry_type, std::size_t N>
constexpr auto
determinant(const mdspan<entry_type, extents<std::size_t, N, N>> matrix);

namespace _impl {
template <typename tuple_type>
concept tuple_like = requires(tuple_type t) {
  std::get<0>(t);
  static_cast<
      typename std::tuple_element<0, std::remove_cvref_t<tuple_type>>::type>(
      std::get<0>(t));
  std::tuple_size<std::remove_cvref_t<tuple_type>>::value;
};

template <typename matrix_type, std::size_t row = 0>
concept square_matrix_tuple = requires {
  requires tuple_like<matrix_type>;
  requires tuple_like<typename std::tuple_element<row, matrix_type>::type>;
  std::tuple_size<matrix_type>::value ==
      std::tuple_size<
          typename std::tuple_element<row, matrix_type>::type>::value;
};
} // namespace _impl

// Takes a tuple-like object of tuple-like objects
// Elements of the outer tuple-like object are rows of the matrix,
template <std::size_t row = 0, typename tuple_type>
  requires _impl::square_matrix_tuple<tuple_type, row>
constexpr auto determinant(tuple_type &&matrix);

template <typename left_type, typename right_type, std::size_t N>
constexpr auto pt_diff_expr(const std::array<left_type, N> &left,
                            const std::array<right_type, N> &right) {
  std::array<arith_expr<std::minus<>, left_type, right_type>, N> diff;
  for (std::size_t i = 0; i < N; ++i) {
    diff[i] = minus_expr(left[i], right[i]);
  }
  return diff;
}

template <typename left_type, typename right_type>
constexpr auto vec_cross_expr(const std::array<left_type, 2> &left,
                              const std::array<right_type, 2> &right) {
  static constexpr std::size_t x = 0;
  static constexpr std::size_t y = 1;
  return adaptive_expr::mult_expr(left[x], right[y]) -
         adaptive_expr::mult_expr(left[y], right[x]);
}

template <adaptive_expr::arith_number coord_type, std::size_t N>
constexpr auto
pt_orient_expr(const std::array<std::array<coord_type, N>, N + 1> &points) {
  using diff_type =
      std::remove_cvref_t<decltype(pt_diff_expr(points[0], points[0])[0])>;
  std::array<diff_type, N * N> matrix_mem;
  mdspan<diff_type, extents<std::size_t, N, N>> matrix{matrix_mem.data()};
  for (std::size_t i = 1; i < N + 1; ++i) {
    const auto vec_diff = pt_diff_expr(points[i], points[0]);
    for (std::size_t j = 0; j < N; ++j) {
      matrix[i - 1, j] = vec_diff[j];
    }
  }
  return determinant(matrix);
} // namespace adaptive_expr

template <adaptive_expr::arith_number coord_type>
constexpr auto
build_incircle2d_case(const std::array<std::array<coord_type, 2>, 4> &points) {
  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  const auto dist_expr = [](const std::array<coord_type, 2> pt) constexpr {
    return adaptive_expr::mult_expr(pt[x], pt[x]) +
           adaptive_expr::mult_expr(pt[y], pt[y]);
  };
  const auto cross_expr = [](const std::array<coord_type, 2> &lhs,
                             const std::array<coord_type, 2> &rhs) constexpr {
    return adaptive_expr::mult_expr(lhs[x], rhs[y]) -
           adaptive_expr::mult_expr(lhs[y], rhs[x]);
  };
  const auto incircle_subexpr =
      [dist_expr, cross_expr](const std::array<coord_type, 2> &p0,
                              const std::array<coord_type, 2> &p1,
                              const std::array<coord_type, 2> &p2) constexpr {
        return dist_expr(p0) * cross_expr(p1, p2) -
               dist_expr(p1) * cross_expr(p0, p2) +
               dist_expr(p2) * cross_expr(p0, p1);
      };
  return balance_expr(-incircle_subexpr(points[1], points[2], points[3]) +
                      incircle_subexpr(points[0], points[2], points[3]) -
                      incircle_subexpr(points[0], points[1], points[3]) +
                      incircle_subexpr(points[0], points[1], points[2]));
}

template <adaptive_expr::arith_number coord_type>
constexpr auto
build_incircle3d_case(const std::array<std::array<coord_type, 3>, 5> &points) {
  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  constexpr std::size_t z = 2;
  const auto ptdiff_expr = [](const auto &lhs, const auto &rhs) constexpr {
    return std::array{adaptive_expr::minus_expr(lhs[x], rhs[x]),
                      adaptive_expr::minus_expr(lhs[y], rhs[y]),
                      adaptive_expr::minus_expr(lhs[z], rhs[z])};
  };
  const auto delta_0 = ptdiff_expr(points[0], points[4]);
  const auto delta_1 = ptdiff_expr(points[1], points[4]);
  const auto delta_2 = ptdiff_expr(points[2], points[4]);
  const auto delta_3 = ptdiff_expr(points[3], points[4]);

  const auto cross_2d_expr = [](const auto &lhs, const auto &rhs) constexpr {
    return adaptive_expr::mult_expr(lhs[x], rhs[y]) -
           adaptive_expr::mult_expr(lhs[y], rhs[x]);
  };
  const auto cross_0_1 = cross_2d_expr(delta_0, delta_1);
  const auto cross_1_2 = cross_2d_expr(delta_1, delta_2);
  const auto cross_2_3 = cross_2d_expr(delta_2, delta_3);
  const auto cross_3_0 = cross_2d_expr(delta_3, delta_0);
  const auto cross_0_2 = cross_2d_expr(delta_0, delta_2);
  const auto cross_1_3 = cross_2d_expr(delta_1, delta_3);

  // Computes the terms for the determinant of the following matrix:
  //
  // [ d0x, d0y, d0z, 1 ]
  // [ d1x, d1y, d1z, 1 ]
  // [ d2x, d2y, d2z, 1 ]
  // [ d3x, d3y, d3z, 1 ]
  //
  const auto det_3 = (delta_0[z] * cross_1_2 - delta_1[z] * cross_0_2 +
                      delta_2[z] * cross_0_1);
  const auto det_0 = -(delta_1[z] * cross_2_3 - delta_2[z] * cross_1_3 +
                       delta_3[z] * cross_1_2);
  const auto det_1 = (delta_2[z] * cross_3_0 + delta_3[z] * cross_0_2 +
                      delta_0[z] * cross_2_3);
  const auto det_2 = -(delta_3[z] * cross_0_1 + delta_0[z] * cross_1_3 +
                       delta_1[z] * cross_3_0);

  const auto magsq_expr = [](const auto &vec) constexpr {
    return vec[x] * vec[x] + vec[y] * vec[y] + vec[z] * vec[z];
  };
  const auto magsq_0 = magsq_expr(delta_0);
  const auto magsq_1 = magsq_expr(delta_1);
  const auto magsq_2 = magsq_expr(delta_2);
  const auto magsq_3 = magsq_expr(delta_3);

  return magsq_3 * det_3 + magsq_2 * det_2 + magsq_1 * det_1 + magsq_0 * det_0;
}

namespace _impl {

template <tuple_like tuple_type, std::size_t... col_pre,
          std::size_t... col_post>
constexpr auto submatrix_row_impl(tuple_type &&row,
                                  std::index_sequence<col_pre...>,
                                  std::index_sequence<col_post...>) {
  static constexpr std::size_t post_start = sizeof...(col_pre) + 1;
  return std::tuple{std::get<col_pre>(row)...,
                    std::get<post_start + col_post>(row)...};
}

template <std::size_t remove_col, square_matrix_tuple matrix_type,
          std::size_t... row_pre, std::size_t... row_post>
constexpr auto submatrix_impl(matrix_type &&matrix,
                              std::index_sequence<row_pre...>,
                              std::index_sequence<row_post...>) {
  constexpr std::size_t N = std::tuple_size<matrix_type>::value;
  static constexpr std::size_t post_start = sizeof...(row_pre) + 1;
  return std::tuple{
      submatrix_row_impl(std::get<row_pre>(matrix),
                         std::make_index_sequence<remove_col>(),
                         std::make_index_sequence<N - 1 - remove_col>())...,
      submatrix_row_impl(std::get<post_start + row_post>(matrix),
                         std::make_index_sequence<remove_col>(),
                         std::make_index_sequence<N - 1 - remove_col>())...};
}

// Constructs an (N-1)x(N-1) from the passed matrix, excluding the specified row
// and the specified column
template <std::size_t remove_row = 0, std::size_t remove_col = 0,
          square_matrix_tuple matrix_type>
constexpr auto submatrix(matrix_type &&matrix) {
  constexpr std::size_t N = std::tuple_size<matrix_type>::value;
  return submatrix_impl<remove_col>(
      matrix, std::make_index_sequence<remove_row>(),
      std::make_index_sequence<N - 1 - remove_row>());
}

} // namespace _impl

// Takes a tuple-like object of tuple-like objects
// Elements of the outer tuple-like object are rows of the matrix,
template <std::size_t row, typename tuple_type>
  requires _impl::square_matrix_tuple<tuple_type, row>
constexpr auto determinant(tuple_type &&matrix) {
  static constexpr std::size_t N = std::tuple_size<tuple_type>::value;
  if constexpr (N == 1) {
    return std::get<0>(std::get<0>(matrix));
  } else {
    constexpr auto minor = _impl::submatrix(std::forward<tuple_type>(matrix));
    if constexpr (row < N - 1) {
      return balance_expr(
          mult_expr(std::get<0>(std::get<row>(matrix)), determinant(minor)) -
          determinant<row + 1>(matrix));
    } else {
      return balance_expr(
          mult_expr(std::get<0>(std::get<row>(matrix)), determinant(minor)));
    }
  }
}

template <std::size_t row, typename entry_type, std::size_t N>
constexpr auto
determinant(const mdspan<entry_type, extents<std::size_t, N, N>> matrix) {
  if constexpr (N == 1) {
    return matrix[0, 0];
  } else {
    std::array<entry_type, (N - 1) * (N - 1)> submatrix_mem;
    mdspan<entry_type, extents<std::size_t, N - 1, N - 1>> submatrix{
        submatrix_mem.data()};
    for (std::size_t j = 1; j < N; ++j) {
      for (std::size_t i = 0; i < row; ++i) {
        submatrix[i, j - 1] = matrix[i, j];
      }
      for (std::size_t i = row + 1; i < N; ++i) {
        submatrix[i - 1, j - 1] = matrix[i, j];
      }
    }
    if constexpr (row < N - 1) {
      return balance_expr(mult_expr(matrix[row, 0], determinant(submatrix)) -
                          determinant<row + 1>(matrix));
    } else {
      return balance_expr(mult_expr(matrix[row, 0], determinant(submatrix)));
    }
  }
}

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_GEOM_EXPRS_HPP

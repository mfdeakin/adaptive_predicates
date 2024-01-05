
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

namespace _impl {
template <typename tuple_type>
concept tuple_like = requires(tuple_type t) {
  std::get<0>(t);
  static_cast<std::tuple_element<
      0, typename std::remove_cvref<tuple_type>::type>::type>(std::get<0>(t));
  std::tuple_size<typename std::remove_cvref<tuple_type>::type>::value;
};

template <typename matrix_type, std::size_t row = 0>
concept square_matrix_tuple = requires {
  requires tuple_like<matrix_type>;
  requires tuple_like<typename std::remove_cvref<typename std::tuple_element<
      row, typename std::remove_cvref<matrix_type>::type>::type>::type>;
  std::tuple_size<typename std::remove_cvref<matrix_type>::type>::value ==
      std::tuple_size<typename std::tuple_element<
          row, typename std::remove_cvref<matrix_type>::type>::type>::value;
};

template <typename points_type>
concept orient_tuple = requires {
  requires tuple_like<points_type>;
  requires tuple_like<typename std::tuple_element<
      0, typename std::remove_cvref<points_type>::type>::type>;
  std::tuple_size<typename std::remove_cvref<points_type>::type>::value ==
      1 + std::tuple_size<typename std::tuple_element<
              0, typename std::remove_cvref<points_type>::type>::type>::value;
};

template <typename points_type>
concept incircle_tuple = requires {
  requires tuple_like<points_type>;
  requires tuple_like<typename std::tuple_element<
      0, typename std::remove_cvref<points_type>::type>::type>;
  std::tuple_size<typename std::remove_cvref<points_type>::type>::value ==
      2 + std::tuple_size<typename std::tuple_element<
              0, typename std::remove_cvref<points_type>::type>::type>::value;
};
} // namespace _impl

template <_impl::tuple_like vec_type>
constexpr auto vec_mag_expr(vec_type &&vec);

template <_impl::tuple_like left_type, _impl::tuple_like right_type>
  requires(
      std::tuple_size<typename std::remove_cvref<left_type>::type>::value ==
      std::tuple_size<typename std::remove_cvref<right_type>::type>::value)
constexpr auto pt_diff_expr(left_type &&left, right_type &&right);

template <_impl::tuple_like left_type, _impl::tuple_like right_type>
  requires(
      std::tuple_size<typename std::remove_cvref<left_type>::type>::value ==
          2 &&
      std::tuple_size<typename std::remove_cvref<right_type>::type>::value == 2)
constexpr auto vec_cross_expr(left_type &&left, right_type &&right);

template <_impl::tuple_like left_type, _impl::tuple_like right_type>
  requires(
      std::tuple_size<typename std::remove_cvref<left_type>::type>::value ==
          3 &&
      std::tuple_size<typename std::remove_cvref<right_type>::type>::value == 3)
constexpr auto vec_cross_expr(left_type &&left, right_type &&right);

template <_impl::orient_tuple points_type>
constexpr auto pt_orient_expr(points_type &&points);

template <_impl::incircle_tuple points_type>
constexpr auto pt_incircle_expr(points_type &&points);

template <std::size_t row = 0, typename entry_type, std::size_t N>
constexpr auto
determinant(const mdspan<entry_type, extents<std::size_t, N, N>> matrix);

template <typename tuple_type>
  requires _impl::square_matrix_tuple<tuple_type>
constexpr auto determinant(tuple_type &&matrix);

#pragma region implementations

namespace _impl {
template <tuple_like vec_type, std::size_t... indices>
constexpr auto vec_mag_expr(vec_type &&vec, std::index_sequence<indices...>) {
  return (mult_expr(std::get<indices>(vec), std::get<indices>(vec)) + ...);
}

template <tuple_like left_type, tuple_like right_type, std::size_t... indices>
constexpr auto pt_diff_expr(left_type &&left, right_type &&right,
                            std::index_sequence<indices...>) {
  return std::tuple{
      minus_expr(std::get<indices>(left), std::get<indices>(right))...};
}
} // namespace _impl

template <_impl::tuple_like vec_type>
constexpr auto vec_mag_expr(vec_type &&vec) {
  return _impl::vec_mag_expr(
      std::forward<vec_type>(vec),
      std::make_index_sequence<std::tuple_size<
          typename std::remove_cvref<vec_type>::type>::value>());
}

template <_impl::tuple_like left_type, _impl::tuple_like right_type>
  requires(
      std::tuple_size<typename std::remove_cvref<left_type>::type>::value ==
      std::tuple_size<typename std::remove_cvref<right_type>::type>::value)
constexpr auto pt_diff_expr(left_type &&left, right_type &&right) {
  return _impl::pt_diff_expr(
      std::forward<left_type>(left), std::forward<right_type>(right),
      std::make_index_sequence<std::tuple_size<
          typename std::remove_cvref<left_type>::type>::value>());
}

template <_impl::tuple_like left_type, _impl::tuple_like right_type>
  requires(
      std::tuple_size<typename std::remove_cvref<left_type>::type>::value ==
          2 &&
      std::tuple_size<typename std::remove_cvref<right_type>::type>::value == 2)
constexpr auto vec_cross_expr(left_type &&left, right_type &&right) {
  static constexpr std::size_t x = 0;
  static constexpr std::size_t y = 1;
  return mult_expr(std::get<x>(left), std::get<y>(right)) -
         mult_expr(std::get<y>(left), std::get<x>(right));
}

template <_impl::tuple_like left_type, _impl::tuple_like right_type>
  requires(
      std::tuple_size<typename std::remove_cvref<left_type>::type>::value ==
          3 &&
      std::tuple_size<typename std::remove_cvref<right_type>::type>::value == 3)
constexpr auto vec_cross_expr(left_type &&left, right_type &&right) {
  static constexpr std::size_t x = 0;
  static constexpr std::size_t y = 1;
  static constexpr std::size_t z = 2;
  return std::array{
      vec_cross_expr(std::tuple{std::get<y>(std::forward<left_type>(left)),
                                std::get<z>(std::forward<left_type>(left))},
                     std::tuple{std::get<y>(std::forward<right_type>(right)),
                                std::get<z>(std::forward<right_type>(right))}),
      vec_cross_expr(std::tuple{std::get<z>(std::forward<left_type>(left)),
                                std::get<x>(std::forward<left_type>(left))},
                     std::tuple{std::get<z>(std::forward<right_type>(right)),
                                std::get<x>(std::forward<right_type>(right))}),
      vec_cross_expr(std::tuple{std::get<x>(std::forward<left_type>(left)),
                                std::get<y>(std::forward<left_type>(left))},
                     std::tuple{std::get<x>(std::forward<right_type>(right)),
                                std::get<y>(std::forward<right_type>(right))})};
}

namespace _impl {

template <orient_tuple points_type, std::size_t... indices>
constexpr auto pt_orient_expr(points_type &&points,
                              std::index_sequence<indices...>) {
  static constexpr std::size_t last_pt =
      std::tuple_size<typename std::remove_cvref<points_type>::type>::value - 1;
  static constexpr std::size_t dim = last_pt;
  const auto ref_pt = std::get<last_pt>(points);
  return determinant(std::make_tuple(adaptive_expr::pt_diff_expr(
      std::get<indices>(std::forward<points_type>(points)), ref_pt)...));
}

} // namespace _impl

template <_impl::orient_tuple points_type>
constexpr auto pt_orient_expr(points_type &&points) {
  static constexpr std::size_t num_facet_pts =
      std::tuple_size<typename std::remove_cvref<points_type>::type>::value - 1;
  return _impl::pt_orient_expr(points,
                               std::make_index_sequence<num_facet_pts>());
}

namespace _impl {

template <_impl::incircle_tuple points_type, std::size_t... indices>
constexpr auto pt_incircle_expr(points_type &&points,
                                std::index_sequence<indices...>) {
  static constexpr std::size_t last_idx =
      std::tuple_size<typename std::remove_cvref<points_type>::type>::value - 1;
  static constexpr std::size_t dim = last_idx - 1;
  const auto ref_pt = std::get<last_idx>(points);
  const auto offset_pts = std::make_tuple(adaptive_expr::pt_diff_expr(
      std::get<indices>(std::forward<points_type>(points)), ref_pt)...);
  const auto insphere_matrix = std::make_tuple(
      std::tuple_cat(std::get<indices>(offset_pts),
                     std::make_tuple(adaptive_expr::vec_mag_expr(
                         std::get<indices>(offset_pts))))...);
  return determinant(insphere_matrix);
}

} // namespace _impl

template <_impl::incircle_tuple points_type>
constexpr auto pt_incircle_expr(points_type &&points) {
  static constexpr std::size_t num_circle_pts =
      std::tuple_size<typename std::remove_cvref<points_type>::type>::value - 1;
  return _impl::pt_incircle_expr(points,
                                 std::make_index_sequence<num_circle_pts>());
}

namespace _impl {

template <tuple_like tuple_type, std::size_t... col_pre,
          std::size_t... col_post>
constexpr auto submatrix_rm_row(tuple_type &&row,
                                std::index_sequence<col_pre...>,
                                std::index_sequence<col_post...>) {
  static constexpr std::size_t post_start = sizeof...(col_pre) + 1;
  const auto left =
      std::make_tuple(std::get<col_pre>(std::forward<tuple_type>(row))...);
  const auto right = std::make_tuple(
      std::get<post_start + col_post>(std::forward<tuple_type>(row))...);
  return std::tuple_cat(left, right);
}

template <std::size_t remove_col, square_matrix_tuple matrix_type,
          std::size_t... row_pre, std::size_t... row_post>
constexpr auto submatrix_rm_col(matrix_type &&matrix,
                                std::index_sequence<row_pre...>,
                                std::index_sequence<row_post...>) {
  constexpr std::size_t N =
      std::tuple_size<typename std::remove_cvref<matrix_type>::type>::value;
  static constexpr std::size_t post_start = sizeof...(row_pre) + 1;

  const auto top = std::make_tuple(submatrix_rm_row(
      std::get<row_pre>(matrix), std::make_index_sequence<remove_col>(),
      std::make_index_sequence<N - 1 - remove_col>())...);

  const auto bottom = std::make_tuple(
      submatrix_rm_row(std::get<post_start + row_post>(matrix),
                       std::make_index_sequence<remove_col>(),
                       std::make_index_sequence<N - 1 - remove_col>())...);

  return std::tuple_cat(top, bottom);
}

// Constructs an (N-1)x(N-1) from the passed matrix, excluding the specified row
// and the specified column
template <std::size_t remove_row = 0, std::size_t remove_col = 0,
          square_matrix_tuple matrix_type>
constexpr auto submatrix(matrix_type &&matrix) {
  constexpr std::size_t N =
      std::tuple_size<typename std::remove_cvref<matrix_type>::type>::value;
  return submatrix_rm_col<remove_col>(
      matrix, std::make_index_sequence<remove_row>(),
      std::make_index_sequence<N - 1 - remove_row>());
}

template <typename tuple_type, std::size_t... indices>
  requires _impl::square_matrix_tuple<tuple_type>
constexpr auto determinant(tuple_type &&matrix,
                           std::index_sequence<indices...>) {
  static constexpr std::size_t N =
      std::tuple_size<typename std::remove_cvref<tuple_type>::type>::value;
  if constexpr (N == 1) {
    return std::get<0>(std::get<0>(std::forward<tuple_type>(matrix)));
  } else {
    return balance_expr((
        (std::get<0>(std::get<indices>(std::forward<tuple_type>(matrix))) *
         determinant(submatrix<indices, 0>(std::forward<tuple_type>(matrix)))) -
        ... - additive_id{}));
  }
}

} // namespace _impl

// Takes a tuple-like object of tuple-like objects
// Elements of the outer tuple-like object are rows of the matrix,
template <typename tuple_type>
  requires _impl::square_matrix_tuple<tuple_type>
constexpr auto determinant(tuple_type &&matrix) {
  static constexpr std::size_t N =
      std::tuple_size<typename std::remove_cvref<tuple_type>::type>::value;
  return _impl::determinant(std::forward<tuple_type>(matrix),
                            std::make_index_sequence<N>());
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
#pragma endregion implementations

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_GEOM_EXPRS_HPP

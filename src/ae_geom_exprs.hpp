
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
  return std::tuple{
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
constexpr auto pt_incircle_expr_impl(points_type &&points,
                                     std::index_sequence<indices...>) {
  static constexpr std::size_t last_idx =
      std::tuple_size<typename std::remove_cvref<points_type>::type>::value - 1;
  const auto ref_pt = std::get<last_idx>(points);
  const auto offset_pts = std::make_tuple(adaptive_expr::pt_diff_expr(
      std::get<indices>(std::forward<points_type>(points)), ref_pt)...);
  const auto insphere_matrix = std::make_tuple(
      std::tuple_cat(std::make_tuple(adaptive_expr::vec_mag_expr(
                         std::get<indices>(offset_pts))),
                     std::get<indices>(offset_pts))...);
  const auto det = determinant(insphere_matrix);
  // We rotated the columns of the following matrix:
  // x0 y0 x0^2+y0^2 1
  // x1 y1 x1^2+y1^2 1
  // x2 y2 x2^2+y2^2 1
  // x3 y3 x3^2+y3^2 1
  //
  // to become
  //
  // x0^2+y0^2 x0 y0 1
  // x1^2+y1^2 x1 y1 1
  // x2^2+y2^2 x2 y2 1
  // x3^2+y3^2 x3 y3 1
  //
  // This requires N-1 column swaps, multiplying our determinant by (-1)^(N-1),
  // so we need to account for the possible negation
  // We perform this rotation so that the resulting expression is significantly
  // simpler than if the paraboloid lifting term was in the sub-determinants
  if constexpr (last_idx % 2 == 1) {
    return det;
  } else {
    return -det;
  }
}

} // namespace _impl

template <_impl::incircle_tuple points_type>
constexpr auto pt_incircle_expr(points_type &&points) {
  static constexpr std::size_t num_circle_pts =
      std::tuple_size<typename std::remove_cvref<points_type>::type>::value - 1;
  return _impl::pt_incircle_expr_impl(
      points, std::make_index_sequence<num_circle_pts>());
}

namespace _impl {

template <std::size_t _row, std::size_t _col> struct matrix_index {
  static constexpr std::size_t row = _row;
  static constexpr std::size_t col = _col;

  template <tuple_like lookup> static constexpr auto operator()(lookup &&l) {
    return std::get<col>(std::get<row>(l));
  }
  template <std::size_t other_row, std::size_t other_col>
  constexpr auto operator+(matrix_index<other_row, other_col>) {
    return *this;
  }
  template <std::size_t other_row, std::size_t other_col>
  constexpr auto operator-(matrix_index<other_row, other_col>) {
    return *this;
  }
  template <std::size_t other_row, std::size_t other_col>
  constexpr auto operator*(matrix_index<other_row, other_col>) {
    return *this;
  }
  template <std::size_t other_row, std::size_t other_col>
  constexpr auto operator/(matrix_index<other_row, other_col>) {
    return *this;
  }
  template <std::size_t other_row, std::size_t other_col>
  constexpr auto operator>=(matrix_index<other_row, other_col>) {
    return false;
  }
};
template <std::size_t row, std::size_t col>
constexpr auto abs(matrix_index<row, col>) {
  return matrix_index<row, col>{};
}
template <std::size_t row, std::size_t col>
constexpr auto mul_sub(matrix_index<row, col>, matrix_index<row, col>,
                       matrix_index<row, col>) {
  return matrix_index<row, col>{};
}

template <std::size_t row, std::size_t... col_inds>
constexpr auto index_row(std::index_sequence<col_inds...> is) {
  return std::make_tuple(matrix_index<row, col_inds>{}...);
}

template <std::size_t... row_inds>
constexpr auto index_square_matrix(std::index_sequence<row_inds...> is) {
  return std::make_tuple(index_row<row_inds>(is)...);
}

template <std::size_t step_row, std::size_t step_col, typename _expr_t>
constexpr auto adjust_mtx_indices(_expr_t &&e) {
  using expr_t = std::remove_cvref_t<_expr_t>;
  if constexpr (is_expr_v<expr_t>) {
    return make_expr<typename expr_t::Op>(
        adjust_mtx_indices<step_row, step_col>(e.lhs()),
        adjust_mtx_indices<step_row, step_col>(e.rhs()));
  } else {
    constexpr std::size_t row = expr_t::row;
    constexpr std::size_t col = expr_t::col;
    return matrix_index<(row < step_row ? row : row + 1),
                        (col < step_col ? col : col + 1)>{};
  }
}

template <tuple_like tuple_type, std::size_t... indices>
  requires _impl::square_matrix_tuple<tuple_type>
consteval auto determinant_impl(tuple_type &&matrix,
                                std::index_sequence<indices...>) {
  static constexpr std::size_t N =
      std::tuple_size<typename std::remove_cvref<tuple_type>::type>::value;
  static_assert(N == sizeof...(indices));
  if constexpr (N == 1) {
    return std::get<0>(std::get<0>(std::forward<tuple_type>(matrix)));
  } else if constexpr (N == 2) {
    return vec_cross_expr(
        std::tuple{std::get<0>(std::get<0>(std::forward<tuple_type>(matrix))),
                   std::get<1>(std::get<0>(std::forward<tuple_type>(matrix)))},
        std::tuple{std::get<0>(std::get<1>(std::forward<tuple_type>(matrix))),
                   std::get<1>(std::get<1>(std::forward<tuple_type>(matrix)))});
  } else {
    constexpr auto subdet =
        determinant_impl(index_square_matrix(std::make_index_sequence<N - 1>()),
                         std::make_index_sequence<N - 1>());
    return balance_expr((mult_expr(std::get<0>(std::get<indices>(
                                       std::forward<tuple_type>(matrix))),
                                   adjust_mtx_indices<indices, 0>(subdet)) -
                         ... - additive_id{}));
  }
}

template <typename indexed_expr, typename tuple_type>
constexpr auto apply_indices(tuple_type &&matrix) {
  if constexpr (is_expr_v<indexed_expr>) {
    return make_expr<typename indexed_expr::Op>(
        apply_indices<typename indexed_expr::LHS>(
            std::forward<tuple_type>(matrix)),
        apply_indices<typename indexed_expr::RHS>(
            std::forward<tuple_type>(matrix)));
  } else {
    static_assert(requires {
      indexed_expr::row;
      indexed_expr::col;
    });
    return std::get<indexed_expr::col>(
        std::get<indexed_expr::row>(std::forward<tuple_type>(matrix)));
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

  if constexpr (N == 1) {
    return std::get<0>(std::get<0>(std::forward<tuple_type>(matrix)));
  } else if constexpr (N == 2) {
    return vec_cross_expr(
        std::tuple{std::get<0>(std::get<0>(std::forward<tuple_type>(matrix))),
                   std::get<1>(std::get<0>(std::forward<tuple_type>(matrix)))},
        std::tuple{std::get<0>(std::get<1>(std::forward<tuple_type>(matrix))),
                   std::get<1>(std::get<1>(std::forward<tuple_type>(matrix)))});
  } else {
    using index_expr = decltype(_impl::determinant_impl(
        _impl::index_square_matrix(std::make_index_sequence<N>()),
        std::make_index_sequence<N>()));
    return _impl::apply_indices<index_expr>(std::forward<tuple_type>(matrix));
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
#pragma endregion implementations

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_GEOM_EXPRS_HPP

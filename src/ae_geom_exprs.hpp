
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
constexpr auto pt_incircle_expr(points_type &&points,
                                std::index_sequence<indices...>) {
  static constexpr std::size_t last_idx =
      std::tuple_size<typename std::remove_cvref<points_type>::type>::value - 1;
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

template <adaptive_expr::arith_number coord_type>
constexpr auto
pt_incircle_explicit_expr(const std::array<std::array<coord_type, 3>, 5> &points) {
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

template <adaptive_expr::arith_number coord_type>
constexpr auto
pt_incircle_explicit_expr(std::array<std::array<coord_type, 3>, 5> &&points) {
  return pt_incircle_explicit_expr(points);
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

template <tuple_like tuple_type, std::size_t... indices>
  requires _impl::square_matrix_tuple<tuple_type>
consteval auto determinant(tuple_type &&matrix,
                           std::index_sequence<indices...> is) {
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
  } else if constexpr (N == 3) {
    const auto cross_prod = vec_cross_expr(
        std::tuple{std::get<0>(std::get<0>(std::forward<tuple_type>(matrix))),
                   std::get<1>(std::get<0>(std::forward<tuple_type>(matrix))),
                   std::get<2>(std::get<0>(std::forward<tuple_type>(matrix)))},
        std::tuple{std::get<0>(std::get<1>(std::forward<tuple_type>(matrix))),
                   std::get<1>(std::get<1>(std::forward<tuple_type>(matrix))),
                   std::get<2>(std::get<1>(std::forward<tuple_type>(matrix)))});
    return std::get<0>(std::get<2>(std::forward<tuple_type>(matrix))) *
               std::get<0>(cross_prod) -
           std::get<1>(std::get<2>(std::forward<tuple_type>(matrix))) *
               std::get<1>(cross_prod) +
           std::get<2>(std::get<2>(std::forward<tuple_type>(matrix))) *
               std::get<2>(cross_prod);
  } else {
    return balance_expr(
        ((std::get<0>(std::get<indices>(std::forward<tuple_type>(matrix))) *
          determinant(submatrix<indices, 0>(std::forward<tuple_type>(matrix)),
                      is)) -
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
  using index_expr = decltype(_impl::determinant(
      _impl::index_square_matrix(std::make_index_sequence<N>()),
      std::make_index_sequence<N>()));
  return _impl::apply_indices<index_expr>(std::forward<tuple_type>(matrix));
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

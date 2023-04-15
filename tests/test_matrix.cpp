
#include <array>
#include <span>
#include <tuple>
#include <type_traits>

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr.hpp"

using namespace adaptive_expr;

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
constexpr auto determinant(std::array<std::array<row_exprs, N>, N> mtx,
                           const int swaps = 0) {
  const auto v = balance_expr(mtx[0][0]);
  const auto approx_val = adaptive_eval<eval_type>(v);
  if constexpr (N == 1) {
    return approx_val;
  } else {
    if (approx_val == eval_type{0}) {
      if (swaps == N - 1) {
        return eval_type{0};
      } else {
        std::swap(mtx[0], mtx[swaps + 1]);
        return determinant<eval_type>(mtx, swaps + 1);
      }
    } else {
      using submtx_expr = decltype(mult_expr(v, std::declval<row_exprs>()) -
                                   mult_expr(std::declval<row_exprs>(),
                                             std::declval<row_exprs>()));
      std::array<std::array<submtx_expr, N - 1>, N - 1> submtx;
      for (std::size_t i = 1; i < N; ++i) {
        const auto s = mtx[i][0];
        for (std::size_t j = 1; j < N; ++j) {
          submtx[i - 1][j - 1] =
              mult_expr(v, mtx[i][j]) - mult_expr(s, mtx[0][j]);
        }
      }
      const auto subdet = determinant<eval_type>(submtx);
      return subdet * std::pow(approx_val, -N + 2) * std::pow(-1.0, swaps);
    }
  }
}

TEST_CASE("determinant_sign", "[matrix]") {
  REQUIRE(determinant<double>(
              std::array{std::array{5.0, 2.0}, std::array{-1.0, 0.0}}) == 2.0);
  REQUIRE(determinant<double>(std::array{std::array{0.0, 5.0, 2.0},
                                         std::array{0.0, 1.0, 0.0},
                                         std::array{1.0, 86.0, 35.0}}) == -2.0);
}

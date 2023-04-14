
#include <array>
#include <span>
#include <tuple>
#include <type_traits>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr.hpp"

using namespace adaptive_expr;

// Based on LU Factorization, except written to avoid divisions in the exact
// evaluation.
// Returns an approximate value with sign matching the sign of the determinant.
// Terminates as early as possible for a non-full rank matrix returning 0
template <typename eval_type, typename row_exprs, std::size_t N>
constexpr auto determinant(std::array<std::array<row_exprs, N>, N> mtx,
                           const int swaps = 0) {
  const auto v = balance_expr(mtx[0, 0]);
  const auto approx_val = adaptive_eval<eval_type>(v);
  if constexpr (N == 1) {
    return approx_val;
  } else {
    if (approx_val == eval_type{0}) {
      if (swaps == N - 1) {
        return eval_type{0};
      } else {
        std::swap(mtx[0], mtx[swaps + 1]);
        return determinant_impl<eval_type>(mtx, swaps + 1);
      }
    } else {
      using submtx_expr =
          decltype(mult_expr(declval(row_exprs), declval(row_exprs)) -
                   mult_expr(declval(row_exprs), declval(row_exprs)));
      std::array<std::array<submtx_expr, N - 1>, N - 1> submtx;
      for (std::size_t i = 1; i < N; ++i) {
        const auto s = mtx[i][0];
        for (std::size_t j = 1; j < N; ++j) {
          submtx[i - 1][j - 1] =
              mult_expr(v, mtx[i][j]) - mult_expr(s, mtx[0][j]);
        }
      }
      const auto subdet = determinant_impl<eval_type>(mtx);
      return subdet * std::pow(approx_val, -N + 2) * std::pow(-1.0, swaps);
    }
  }
}

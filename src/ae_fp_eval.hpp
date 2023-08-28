
#ifndef ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

#include <algorithm>
#include <optional>
#include <ranges>
#include <span>

#include <cassert>
#include <cmath>

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"

#include "ae_fp_eval_impl.hpp"

namespace adaptive_expr {

template <arith_number eval_type, typename E_>
  requires expr_type<E_> || arith_number<E_>
constexpr eval_type fp_eval(E_ &&e) noexcept {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr_v<E>) {
    using Op = typename E::Op;
    return Op()(fp_eval<eval_type>(e.lhs()), fp_eval<eval_type>(e.rhs()));
  } else {
    return static_cast<eval_type>(e);
  }
}

template <arith_number eval_type, typename E>
  requires expr_type<E> || arith_number<E>
constexpr eval_type exactfp_eval(E &&e) noexcept {
  if constexpr (is_expr_v<std::remove_reference_t<E>>) {
    auto partial_results = []() {
      constexpr std::size_t max_stack_storage = 1024 / sizeof(eval_type);
      constexpr std::size_t storage_needed = num_partials_for_exact<E>();
      if constexpr (storage_needed > max_stack_storage) {
        return std::vector<eval_type>(storage_needed);
      } else {
        return std::array<eval_type, storage_needed>{};
      }
    }();
    std::span<eval_type, num_partials_for_exact<E>()> partial_span{
        partial_results};
    _impl::exactfp_eval_impl<eval_type>(std::forward<E>(e), partial_span);
    return _impl::merge_sum(partial_span);
    // return std::accumulate(partial_results.begin(), partial_results.end(),
    // eval_type(0));
  } else {
    return static_cast<eval_type>(e);
  }
}

// Returns the approximate value and an upper bound on the absolute error
// If the sign may be wrong, return a pair of NaNs
template <arith_number eval_type, typename E_>
  requires(expr_type<E_> || arith_number<E_>) && (!vector_type<E_>)
constexpr std::pair<eval_type, eval_type> eval_with_err(E_ &&e) noexcept {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr_v<E>) {
    const auto [left_result, left_abs_err] = eval_with_err<eval_type>(e.lhs());
    auto [right_result, right_abs_err] = eval_with_err<eval_type>(e.rhs());
    using Op = typename E::Op;
    const auto [result, max_abs_err] = _impl::eval_with_max_abs_err<Op>(
        left_result, left_abs_err, right_result, right_abs_err);

    if constexpr (!std::is_same_v<std::multiplies<>, Op>) {
      constexpr eval_type nan{std::numeric_limits<double>::signaling_NaN()};
      if constexpr (std::is_same_v<std::minus<>, Op>) {
        right_result = -right_result;
      }
      const auto selector = (!same_sign_or_zero(left_result, right_result)) &&
                            _impl::error_overlaps(left_result, left_abs_err,
                                                  -right_result, right_abs_err);
      if constexpr (vector_type<eval_type>) {
        return {select(selector, result, nan),
                select(selector, max_abs_err, nan)};
      } else {
        if (selector) {
          return {nan, nan};
        } else {
          return {result, max_abs_err};
        }
      }
    }
    return {result, max_abs_err};
  } else {
    return {e, eval_type{0}};
  }
}

// correct_eval either returns the result with an accuracy bounded by
// max_rel_error<E>, or it returns std::nullopt
// This is intended to make it easier to effectively run on GPUs - results that
// need to be evaluated more accurately can be aggregated into a separate
// collection; small expressions will likely use the same code-paths
template <arith_number eval_type, typename E>
  requires(expr_type<E> || arith_number<E>) && (!vector_type<E>)
constexpr std::optional<eval_type> correct_eval(E &&e) noexcept {
  const auto [result, _] = eval_with_err<eval_type>(std::forward<E>(e)); 
  if(std::isnan(result)) {
    return std::nullopt;
  } else {
    return result;
  }
}

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

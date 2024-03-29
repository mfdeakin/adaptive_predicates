
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

template <arith_number eval_type,
          typename allocator_type_ = std::pmr::polymorphic_allocator<eval_type>,
          typename E>
  requires expr_type<E> || arith_number<E>
constexpr eval_type
exactfp_eval(E &&e, allocator_type_ &&mem_pool =
                        std::remove_cvref_t<allocator_type_>()) noexcept {
  if constexpr (is_expr_v<std::remove_reference_t<E>>) {
    using allocator_type = std::remove_cvref_t<allocator_type_>;
    constexpr std::size_t storage_needed = num_partials_for_exact<E>();

    _impl::constexpr_unique<eval_type, allocator_type> partial_results_ptr{
        mem_pool, storage_needed};
    std::span<eval_type, storage_needed> partial_span{partial_results_ptr.get(),
                                                      storage_needed};
    _impl::exactfp_eval_impl<eval_type>(std::forward<E>(e), partial_span);
    const eval_type result = _impl::merge_sum(partial_span);
    return result;
  } else {
    return static_cast<eval_type>(e);
  }
}

// Returns the approximate value and an upper bound on the absolute error
// If the sign may be wrong, return a pair of NaNs
template <arith_number eval_type, typename E_>
  requires(expr_type<E_> || arith_number<E_>)
constexpr std::pair<eval_type, eval_type> eval_with_err(E_ &&e) noexcept {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr_v<E>) {
    const auto [left_result, left_abs_err] = eval_with_err<eval_type>(e.lhs());
    auto [right_result, right_abs_err] = eval_with_err<eval_type>(e.rhs());
    using Op = typename E::Op;
    const auto [result, max_abs_err] = _impl::eval_with_max_abs_err<Op>(
        left_result, left_abs_err, right_result, right_abs_err);

    if constexpr (!std::is_same_v<std::multiplies<>, Op> && depth(E{}) > 2) {
      const eval_type nan{std::numeric_limits<double>::signaling_NaN()};
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
        } else [[likely]] {
          return {result, max_abs_err};
        }
      }
    }
    return {result, max_abs_err};
  } else {
    return {e, eval_type{0}};
  }
}

// Returns the approximate value or a NaN if its sign can't be guaranteed
// This only performs an extra addition and multiplication by a compile time
// value, followed by a comparison to filter out obviously correct results.
// The additional bool return indicates whether future relative error analysis
// with max_rel_error is useful
template <arith_number eval_type, typename E_>
  requires(expr_type<E_> || arith_number<E_>)
constexpr auto eval_checked_fast(E_ &&e) noexcept {
  using E = std::remove_cvref_t<E_>;
  using bool_type = decltype(eval_type{} < eval_type{});
  if constexpr (is_expr_v<E>) {
    const auto [left_result, left_no_subtract] =
        eval_checked_fast<eval_type>(e.lhs());
    auto [right_result, right_no_subtract] =
        eval_checked_fast<eval_type>(e.rhs());
    const auto no_lower_subtract = left_no_subtract && right_no_subtract;
    using Op = typename E::Op;
    const eval_type result = Op{}(left_result, right_result);

    if constexpr (depth(E{}) > 2 && (std::is_same_v<std::plus<>, Op> ||
                                     std::is_same_v<std::minus<>, Op>)) {
      const eval_type nan{std::numeric_limits<double>::signaling_NaN()};
      if constexpr (std::is_same_v<std::minus<>, Op>) {
        right_result = -right_result;
      }
      const auto signs_match = same_sign_or_zero(left_result, right_result);
      if constexpr (vector_type<eval_type>) {
        using fp_type = decltype(left_result[0]);
        const auto selector =
            signs_match ||
            (no_lower_subtract &&
             (abs(result) > abs(left_result - right_result) *
                                _impl::max_rel_error<fp_type, E>()));
        return std::pair<eval_type, bool_type>{
            select(selector, result, nan), no_lower_subtract && signs_match};
      } else {
        if (signs_match ||
            (no_lower_subtract &&
             (abs(result) > abs(left_result - right_result) *
                                _impl::max_rel_error<eval_type, E>())))
            [[likely]] {
          return std::pair<eval_type, bool_type>{result, no_lower_subtract &&
                                                             signs_match};
        } else {
          return std::pair<eval_type, bool_type>{nan, bool_type{false}};
        }
      }
    }
    return std::pair<eval_type, bool_type>{result, no_lower_subtract};
  } else {
    return std::pair<eval_type, bool_type>{static_cast<eval_type>(e),
                                           bool_type{true}};
  }
}

// correct_eval either returns the result with an accuracy bounded by
// max_rel_error<E>, or it returns std::nullopt
// This is intended to make it easier to effectively run on GPUs - results that
// need to be evaluated more accurately can be aggregated into a separate
// collection; small expressions will likely use the same code-paths
template <arith_number eval_type, typename E>
  requires(expr_type<E> || arith_number<E>) && scalar_type<eval_type>
constexpr std::optional<eval_type> correct_eval(E &&e) noexcept {
  const auto [result, _] = eval_checked_fast<eval_type>(std::forward<E>(e));
  if (std::isnan(result)) {
    const auto [new_result, _] = eval_with_err<eval_type>(std::forward<E>(e));
    if (std::isnan(new_result)) {
      return std::nullopt;
    } else [[likely]] {
      return new_result;
    }
  } else [[likely]] {
    return result;
  }
}

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

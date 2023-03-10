
#ifndef ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

#include <optional>
#include <ranges>
#include <span>

#define FP_FAST_FMA
#define FP_FAST_FMAF
#define FP_FAST_FMAL

#include <cmath>

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"

#include "ae_fp_eval_impl.hpp"

namespace adaptive_expr {

template <std::floating_point eval_type, typename E_>
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

template <std::floating_point eval_type, typename E>
  requires expr_type<E> || arith_number<E>
constexpr eval_type exactfp_eval(E &&e) noexcept {
  if constexpr (is_expr_v<std::remove_reference_t<E>>) {
    auto partial_results = []() {
      constexpr std::size_t max_stack_storage = 1024 / sizeof(eval_type);
      constexpr std::size_t storage_needed = num_partials_for_exact<E>();
      if constexpr (storage_needed > max_stack_storage) {
        return std::vector<eval_type>(storage_needed);
      } else {
        auto arr = std::array<eval_type, storage_needed>{};
        return arr;
      }
    }();
    _impl::exactfp_eval_impl<eval_type>(
        std::forward<E>(e),
        std::span{partial_results.begin(), partial_results.end()});
    return _impl::merge_sum(std::span<eval_type>{partial_results});
    // return std::accumulate(partial_results.begin(), partial_results.end(),
    // eval_type(0));
  } else {
    return static_cast<eval_type>(e);
  }
}

// correct_eval either returns the result with an accuracy bounded by
// max_rel_error<E>, or it returns std::nullopt
// This is intended to make it easier to effectively run on GPUs - results that
// need to be evaluated more accurately can be aggregated into a separate
// collection; small expressions will likely use the same code-paths
template <std::floating_point eval_type, typename E_>
  requires expr_type<E_> || arith_number<E_>
constexpr std::optional<eval_type> correct_eval(E_ &&e) noexcept {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr_v<E>) {
    const eval_type max_result_rel_err = _impl::max_rel_error<eval_type, E>();
    const auto left_opt = correct_eval<eval_type>(e.lhs());
    const auto right_opt = correct_eval<eval_type>(e.rhs());
    // Make sure that the result can be evaulated correctly
    // Ensuring that at least the sign of the final answer can be determined,
    // with some wiggle room for floating point error, and that left and right
    // have been evaluated correctly
    if (max_result_rel_err >= 1.0 - 0.001 || !left_opt || !right_opt) {
      return std::nullopt;
    }
    const auto left = *left_opt;
    const auto right = *right_opt;
    using Op = typename E::Op;
    const eval_type max_left_rel_err =
        _impl::max_rel_error<eval_type, typename E::LHS>();
    const eval_type max_right_rel_err =
        _impl::max_rel_error<eval_type, typename E::RHS>();

    const eval_type result = Op()(left, right);
    if constexpr (std::is_same_v<Op, std::plus<>> ||
                  std::is_same_v<Op, std::minus<>>) {
      const eval_type max_err = (max_left_rel_err * std::abs(left) +
                                 max_right_rel_err * std::abs(right)) +
                                std::abs(result) *
                                    std::numeric_limits<eval_type>::epsilon() /
                                    eval_type(2);
      if (max_err > max_result_rel_err * std::abs(result)) {
        return std::nullopt;
      } else {
        return result;
      }
    } else if constexpr (std::is_same_v<Op, std::multiplies<>>) {
      const eval_type max_err = std::abs(left) * std::abs(right) *
                                (max_left_rel_err + max_right_rel_err +
                                 max_left_rel_err * max_right_rel_err);
      // Shouldn't happen unless my multiplication invariant on max_rel_error is
      // wrong...
      assert(max_err <= max_result_rel_err * std::abs(result));
      return result;
    } else {
      static_assert(std::is_same_v<std::plus<>, Op> ||
                    std::is_same_v<std::minus<>, Op> ||
                    std::is_same_v<std::multiplies<>, Op>);
      return std::nullopt;
    }
  } else {
    return static_cast<eval_type>(e);
  }
}

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

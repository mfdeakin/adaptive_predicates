
#ifndef ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

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
} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP


#ifndef ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

#include "ae_expr.hpp"

namespace adaptive_expr {

template <std::floating_point eval_type, typename E>
  requires expr_type<E> || arith_number<E>
constexpr eval_type fp_eval(E &&e) noexcept {
  if constexpr (is_expr<std::remove_reference_t<E>>::value) {
    using Op = typename std::remove_cvref_t<E>::Op;
    auto l = fp_eval<eval_type>(e.lhs());
    auto r = fp_eval<eval_type>(e.rhs());
    return Op()(fp_eval<eval_type>(l), fp_eval<eval_type>(r));
    return 0.0;
  } else {
    return static_cast<eval_type>(e);
  }
}

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

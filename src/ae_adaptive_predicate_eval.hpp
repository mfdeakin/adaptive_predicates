
#ifndef ADAPTIVE_PREDICATES_AE_ADAPTIVE_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_ADAPTIVE_EVAL_HPP

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"
#include "ae_fp_eval.hpp"

#include <limits>

template <std::floating_point eval_type, typename E>
requires expr_type<E> || arith_number<E>
constexpr eval_type exactfp_eval(E &&e) noexcept {
  throw std::runtime_error("Not currently implemented!");
  return std::numeric_limits<eval_type>::signaling_NaN();
}

#endif //ADAPTIVE_PREDICATES_AE_ADAPTIVE_EVAL_HPP

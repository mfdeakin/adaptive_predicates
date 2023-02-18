
#ifndef ADAPTIVE_PREDICATES_AE_ADAPTIVE_PREDICATE_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_ADAPTIVE_PREDICATE_EVAL_HPP

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"
#include "ae_fp_eval.hpp"

#include <compare>
#include <limits>

namespace adaptive_expr {

template <typename E>
  requires predicate<E>
constexpr bool adaptive_compare(E &&e) noexcept {

  return E::Op()(fp_eval<float>(e.lhs()), fp_eval<float>(e.lhs()));
}

template <typename E1, typename E2>
  requires expr_type<E1> && expr_type<E2>
constexpr std::strong_ordering adaptive_compare(E1 &&e1, E2 &&e2) noexcept {
  return;
}

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_ADAPTIVE_EVAL_HPP

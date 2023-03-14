
#ifndef ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP
#define ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP

#include "ae_expr.hpp"

namespace adaptive_expr {

template <typename Op>
concept comparison_op = std::is_same_v<std::less<>, Op> ||
                        std::is_same_v<std::greater_equal<>, Op> ||
                        std::is_same_v<std::greater<>, Op> ||
                        std::is_same_v<std::less_equal<>, Op> ||
                        std::is_same_v<std::equal_to<>, Op>;

template <typename E>
concept predicate = expr_type<E> && comparison_op<typename E::Op>;

template <typename E> constexpr std::size_t num_ops(const E &&e) {
  if constexpr (is_expr_v<E>) {
    return num_ops(e.lhs()) + num_ops(e.rhs()) + 1;
  } else {
    return 0;
  }
}

template <typename E> constexpr std::size_t num_values(const E &&e) {
  if constexpr (is_expr_v<E>) {
    return num_ops(e.lhs()) + num_ops(e.rhs());
  } else {
    return 1;
  }
}

template <typename E> constexpr auto trim_expr(E &&e) {
  // Performs the following conversions (pattern matching would be nice here):
  //
  // ** additive_id + additive_id => additive_id
  // ** additive_id + right => right
  // ** left + additive_id => left
  // ** left + (additive_id - right) => left - right
  // ** (additive_id - left) + right => right - left
  // ** additive_id * additive_id => additive_id
  // ** additive_id * right => additive_id
  // ** left * additive_id => additive_id
  //
  // In the default case, e => e
  //
  if constexpr (is_expr_v<E>) {
    using Op = typename std::remove_reference_t<E>::Op;
    const auto new_left = trim_expr(e.lhs());
    const auto new_right = trim_expr(e.rhs());
    using left_t = std::remove_cvref_t<decltype(new_left)>;
    using right_t = std::remove_cvref_t<decltype(new_right)>;
    if constexpr (std::is_same_v<std::plus<>, Op>) {
      if constexpr (std::is_same_v<additive_id, left_t> &&
                    std::is_same_v<additive_id, right_t>) {
        return additive_id{};
      } else if constexpr (std::is_same_v<additive_id, left_t>) {
        return new_right;
      } else if constexpr (std::is_same_v<additive_id, right_t>) {
        return new_left;
      } else {
        if constexpr (negate_expr_type<right_t>) {
          return new_left - new_right;
        } else if constexpr (negate_expr_type<left_t>) {
          return new_right - new_left;
        } else {
          return new_left + new_right;
        }
      }
    } else if constexpr (std::is_same_v<std::minus<>, Op>) {
      if constexpr (std::is_same_v<additive_id, left_t> &&
                    std::is_same_v<additive_id, right_t>) {
        return additive_id{};
      } else if constexpr (std::is_same_v<additive_id, right_t>) {
        return new_left;
      } else {
        if constexpr (negate_expr_type<left_t> && negate_expr_type<right_t>) {
          return new_right.rhs() - new_left.rhs();
        } else if constexpr (negate_expr_type<right_t>) {
          if constexpr (std::is_same_v<additive_id, left_t>) {
            return new_right.rhs();
          } else {
            return new_left + new_right.rhs();
          }
        } else {
          return new_left - new_right;
        }
      }
    } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
      if constexpr (std::is_same_v<additive_id, left_t> ||
                    std::is_same_v<additive_id, right_t>) {
        return additive_id{};
      } else {
        return new_left * new_right;
      }
    } else {
      return Op()(new_left, new_right);
    }
  } else {
    return e;
  }
}

template <typename E> constexpr auto rewrite_minus(E &&e) {
  if constexpr (is_expr_v<E>) {
    if constexpr (std::is_same_v<std::minus<>,
                                 typename std::remove_cvref_t<E>::Op>) {
      auto new_left = rewrite_minus(e.lhs());
      auto new_right = rewrite_minus(e.rhs());
      return new_left + (additive_id{} - new_right);
    } else {
      return e;
    }
  } else {
    return e;
  }
}

template <typename E> constexpr std::size_t depth(E &&e) {
  if constexpr (is_expr_v<E>) {
    return std::max(depth(e.lhs()), depth(e.rhs())) + 1;
  } else {
    return 1;
  }
}

template <typename param, typename... tail> struct contains {
  static constexpr bool value = (std::is_same_v<param, tail> || ...);
};

template <typename E> constexpr auto _balance_expr_impl(E &&e);

template <typename E> constexpr auto balance_expr(E &&e) {
  return trim_expr(_balance_expr_impl(rewrite_minus(trim_expr(e))));
}

template <typename E1, typename E2>
concept associative_commutative =
    is_expr_v<E1> && is_expr_v<E2> &&
    std::is_same_v<typename std::remove_cvref_t<E1>::Op,
                   typename std::remove_cvref_t<E2>::Op> &&
    (std::is_same_v<std::plus<>, typename std::remove_cvref_t<E1>::Op> ||
     std::is_same_v<std::multiplies<>, typename std::remove_cvref_t<E1>::Op>);

template <typename E_> constexpr auto _balance_expr_impl(E_ &&e) {
  if constexpr (is_expr_v<E_>) {
    using E = std::remove_cvref_t<E_>;
    using Op = typename E::Op;
    const auto balanced_left = _balance_expr_impl(e.lhs());
    const auto balanced_right = _balance_expr_impl(e.rhs());
    using LHS = decltype(balanced_left);
    using RHS = decltype(balanced_right);
    if constexpr (depth(balanced_left) > depth(balanced_right) + 1 &&
                  associative_commutative<E, LHS>) {
      if constexpr (depth(balanced_left.lhs()) > depth(balanced_left.rhs())) {
        return Op{}(Op{}(balanced_right, balanced_left.rhs()),
                    balanced_left.lhs());
      } else {
        return Op{}(Op{}(balanced_left.lhs(), balanced_right),
                    balanced_left.rhs());
      }
    } else if constexpr (depth(balanced_right) > depth(balanced_left) + 1 &&
                         associative_commutative<E, RHS>) {
      if constexpr (depth(balanced_right.lhs()) > depth(balanced_right.rhs())) {
        return Op{}(balanced_right.lhs(),
                    Op{}(balanced_left, balanced_right.rhs()));
      } else {
        return Op{}(balanced_right.rhs(),
                    Op{}(balanced_right.lhs(), balanced_left));
      }
    } else {
      return e;
    }
  } else {
    return e;
  }
}

template <typename E_> consteval std::size_t num_partials_for_exact() {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr_v<E>) {
    using Op = typename E::Op;
    if constexpr (std::is_same_v<std::plus<>, Op> ||
                  std::is_same_v<std::minus<>, Op>) {
      return num_partials_for_exact<typename E::LHS>() +
             num_partials_for_exact<typename E::RHS>();
    } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
      auto num_left = num_partials_for_exact<typename E::LHS>();
      auto num_right = num_partials_for_exact<typename E::RHS>();
      return 2 * num_left * num_right;
    } else {
      // always triggers a static assert in a way that doesn't trip up the
      // compiler
      static_assert(!std::is_same_v<std::divides<>, Op>,
                    "No limit on number of partials for division!");
      static_assert(std::is_same_v<std::plus<>, Op> ||
                        std::is_same_v<std::multiplies<>, Op>,
                    "Unhandled operation!");
      return 0;
    }
  } else if constexpr (std::is_same_v<additive_id, E>) {
    return 0;
  } else {
    return 1;
  }
}

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP

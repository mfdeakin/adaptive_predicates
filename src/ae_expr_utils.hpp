
#ifndef ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP
#define ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP

#include "ae_expr.hpp"

#include <ranges>

namespace adaptive_expr {

template <typename Op>
concept comparison_op = std::is_same_v<std::less<>, Op> ||
                        std::is_same_v<std::greater_equal<>, Op> ||
                        std::is_same_v<std::greater<>, Op> ||
                        std::is_same_v<std::less_equal<>, Op> ||
                        std::is_same_v<std::equal_to<>, Op>;

template <typename E>
concept predicate = expr_type<E> && comparison_op<typename E::Op>;

template <typename eval_type> eval_type lowest_order_exp(const eval_type v) {
  if (v == eval_type{0}) {
    return 0;
  } else {
    const auto next = lowest_order_exp(
        v - std::copysign(std::ldexp(eval_type{1}, std::ilogb(v)), v));
    if (next == eval_type{0}) {
      return v;
    } else {
      return next;
    }
  }
}

bool is_nonoverlapping(std::ranges::range auto vals) {
  using eval_type = typename decltype(vals)::value_type;
#if __cpp_lib_ranges_zip > 202110L // std::ranges::views::adjacent implemented
  const auto compared_pairs =
      vals | std::ranges::views::transform([](const eval_type &v) {
        return std::pair{lowest_order_exp(std::abs(v)), std::abs(v)};
      }) |
      std::ranges::views::adjacent<2> |
      std::ranges::views::transform(
          [](const std::pair<std::pair<eval_type, eval_type>,
                             std::pair<eval_type, eval_type>> &cmp) {
            return cmp.first.second < cmp.second.first;
          });
  return std::reduce(compared_pairs.begin(), compared_pairs.end(), true,
                     std::logical_and<>());
#else
  if (vals.size() == 0) {
    return true;
  }
  eval_type last = std::abs(vals[0]);
  for (auto cur : vals | std::ranges::views::drop(1) |
                      std::ranges::views::filter([](const eval_type v) {
                        return v != eval_type{0};
                      })) {
    const auto v = lowest_order_exp(std::abs(cur));
    if (last >= v) {
      return false;
    }
    last = v;
  }
  return true;
#endif
}

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
          return minus_expr(new_left, new_right.rhs());
        } else if constexpr (negate_expr_type<left_t>) {
          return minus_expr(new_right, new_left.rhs());
        } else {
          return plus_expr(new_left, new_right);
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
          return minus_expr(new_right.rhs(), new_left.rhs());
        } else if constexpr (negate_expr_type<right_t>) {
          if constexpr (std::is_same_v<additive_id, left_t>) {
            return new_right.rhs();
          } else {
            return plus_expr(new_left, new_right.rhs());
          }
        } else {
          return minus_expr(new_left, new_right);
        }
      }
    } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
      if constexpr (std::is_same_v<additive_id, left_t> ||
                    std::is_same_v<additive_id, right_t>) {
        return additive_id{};
      } else {
        return mult_expr(new_left, new_right);
      }
    } else {
      return make_expr<Op>(new_left, new_right);
    }
  } else {
    return e;
  }
}

template <typename E_> constexpr auto rewrite_minus(E_ &&e) {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr_v<E>) {
    if constexpr (std::is_same_v<std::minus<>,
                                 typename std::remove_cvref_t<E>::Op> &&
                  !std::is_same_v<additive_id, typename E::LHS>) {
      auto new_left = rewrite_minus(e.lhs());
      auto new_right = rewrite_minus(e.rhs());
      return new_left + (additive_id{} - new_right);
    } else {
      return make_expr<typename E::Op>(rewrite_minus(e.lhs()),
                                       rewrite_minus(e.rhs()));
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

// associative_commutative is false when the expression operators fail to
// satisfy the following properties:
//
// a (E1::Op) b = b (E1::Op) a
// a (E2::Op) b = b (E2::Op) a
// (a (E1::Op) b) (E2::Op) c = b E1::Op (a (E2::Op) c)
// (a (E2::Op) b) (E1::Op) c = b E2::Op (a (E1::Op) c)
//
// This is a difficult set of properties to satisfy, so currently this only
// returns true when both operators are '+' or when both operators are '-'
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
        /*       +''          +''
         *      / \         /   \
         *     +'  d  =>   +'    +
         *    / \         / \   / \
         *   +   c       d   c a   b
         *  / \
         * a   b
         *
         * (((a + b) +' d) +'' c) => ((d +' c) +'' (a + b))
         */
        return make_expr<Op>(
            make_expr<typename LHS::Op>(balanced_right, balanced_left.rhs()),
            balanced_left.lhs());
      } else {
        /*       +''          +'
         *      / \         /   \
         *     +'  d  =>   +''   +
         *    / \         / \   / \
         *   a   +       a   d b   c
         *      / \
         *     b   c
         *
         * ((a + (b +' d)) +'' c) => ((a +'' d) +' (b + c))
         */
        return make_expr<typename LHS::Op>(
            make_expr<Op>(balanced_left.lhs(), balanced_right),
            balanced_left.rhs());
      }
    } else if constexpr (depth(balanced_right) > depth(balanced_left) + 1 &&
                         associative_commutative<E, RHS>) {
      if constexpr (depth(balanced_right.lhs()) > depth(balanced_right.rhs())) {
        /*       +             +
         *      / \          /   \
         *     a   +'  =>   +''   +'
         *        / \      / \   / \
         *       +'' d    b   c a   d
         *      / \
         *     b   c
         *
         * (a + ((b +'' c) +' d)) => ((b +'' c) + (a +' d))
         */
        return make_expr<Op>(
            balanced_right.lhs(),
            make_expr<typename RHS::Op>(balanced_left, balanced_right.rhs()));
      } else {
        /*       +             +
         *      / \          /   \
         *     a   +'  =>   +''   +'
         *        / \      / \   / \
         *       b   +''  c   d b   a
         *          / \
         *         c   d
         *
         * (a + (b +' (c +'' d)) => ((c +'' d) + (b +' a))
         */
        return make_expr<Op>(
            balanced_right.rhs(),
            make_expr<typename RHS::Op>(balanced_right.lhs(), balanced_left));
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

template <typename Op> consteval std::size_t op_latency() {
  if constexpr (std::is_same_v<std::plus<>, Op> ||
                std::is_same_v<std::minus<>, Op>) {
    return 1;
  } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
    return 4;
  }
}

template <typename Op> consteval std::size_t error_contrib_latency() {
  if constexpr (std::is_same_v<std::plus<>, Op> ||
                std::is_same_v<std::minus<>, Op>) {
    return 0;
  } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
    return 2 * op_latency<Op>();
  }
}

consteval std::size_t cmp_latency() { return 1; }
consteval std::size_t abs_latency() { return 1; }
consteval std::size_t fma_latency() { return 4; }
consteval std::size_t overshoot_latency() {
  return abs_latency() + op_latency<std::plus<>>();
}

template <typename E_> consteval std::size_t exact_fp_latency() {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr_v<E>) {
    using Op = typename E::Op;
    using LHS = typename E::LHS;
    using RHS = typename E::RHS;
    const std::size_t left_ops = exact_fp_latency<LHS>();
    const std::size_t right_ops = exact_fp_latency<RHS>();

    if constexpr (std::is_same_v<std::plus<>, Op>) {
      return left_ops + right_ops;
    } else {
      const std::size_t left_mem = num_partials_for_exact<LHS>();
      const std::size_t right_mem = num_partials_for_exact<RHS>();

      if (std::is_same_v<std::minus<>, Op>) {
        return left_ops + right_ops + op_latency<std::minus<>>() * right_mem;
      } else if (std::is_same_v<std::multiplies<>, Op>) {
        // exact_mult has 1 mult, 1 fma, and 1 negation; this is done with
        // every pair of elements of left and right
        return (fma_latency() + op_latency<std::multiplies<>>() +
                op_latency<std::minus<>>()) *
                   left_mem * right_mem +
               left_ops + right_ops;
      }
    }
  } else {
    return 0;
  }
}

template <typename E_> consteval std::size_t exact_fp_rounding_latency() {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr_v<E>) {
    const std::size_t fp_vals = num_partials_for_exact<E>();
    // dekker_sum has 2 additions, 2 abs(), 1 comparison.
    // Assume branch prediction is optimized away
    const std::size_t two_sum_cost =
        2 * op_latency<std::plus<>>() + 2 * abs_latency() + cmp_latency();

    const std::size_t exact_latency = exact_fp_latency<E_>();
    // Note that this ignores the zero elimination and is overly pessimistic
    const std::size_t merge_latency =
        two_sum_cost * fp_vals * (fp_vals - 1) / 2;
    const std::size_t accumulate_latency = fp_vals - 1;
    return exact_latency + merge_latency + accumulate_latency;
  } else {
    return 0;
  }
}

class branch_token_s {};

template <typename S>
concept branch_token = std::is_base_of_v<branch_token_s, S>;

template <typename S>
concept branch_token_inner_node = branch_token<S> && requires(S s) { s.get(); };

template <branch_token S_>
class branch_token_inner_node_s : public branch_token_s {
public:
  using S = S_;

  branch_token_inner_node_s() : s{} {}
  branch_token_inner_node_s(S _s) : s{_s} {}
  constexpr auto &get() { return s.get(); }

private:
  S s;
};

template <branch_token S>
class branch_token_left : public branch_token_inner_node_s<S> {
public:
  branch_token_left() : branch_token_inner_node_s<S>{} {}
  branch_token_left(S _s) : branch_token_inner_node_s<S>{_s} {}

  static constexpr bool is_left() { return true; }
  static constexpr bool is_right() { return false; }

  template <template <class> class branch_dir>
  using append_branch =
      branch_token_left<typename S::template append_branch<branch_dir>>;
};

template <branch_token S>
class branch_token_right : public branch_token_inner_node_s<S> {
public:
  branch_token_right() : branch_token_inner_node_s<S>{} {}
  branch_token_right(S _s) : branch_token_inner_node_s<S>{_s} {}

  static constexpr bool is_left() { return false; }
  static constexpr bool is_right() { return true; }

  template <template <class> class branch_dir>
  using append_branch =
      branch_token_right<typename S::template append_branch<branch_dir>>;
};

template <typename storage_t> struct apply_strings {
  template <typename... L, typename... R>
  auto operator()(std::tuple<L...>, std::tuple<R...>) {
    if constexpr (sizeof...(L) > 0 && sizeof...(R) > 0) {
      return std::tuple<storage_t, branch_token_left<L>...,
                        branch_token_right<R>...>{};
    } else if constexpr (sizeof...(L) > 0) {
      return std::tuple<storage_t, branch_token_left<L>...>{};
    } else if constexpr (sizeof...(R) > 0) {
      return std::tuple<storage_t, branch_token_right<R>...>{};
    } else {
      return std::tuple<storage_t>{};
    }
  }
};

template <typename storage_t> struct enumerate_branches_functor {
  template <typename E_> constexpr auto operator()(E_ &&e) {
    using E = std::remove_cvref_t<E_>;
    if constexpr (is_expr_v<E>) {
      if constexpr (is_expr_v<typename E::LHS> || is_expr_v<typename E::RHS>) {
        return apply_strings<storage_t>{}((*this)(e.lhs()), (*this)(e.rhs()));
      } else {
        return std::tuple<storage_t>{};
      }
    } else {
      return std::tuple<>{};
    }
  }
};

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP

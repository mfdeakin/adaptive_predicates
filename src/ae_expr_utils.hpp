
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

template <arith_number eval_type>
  requires scalar_type<eval_type>
constexpr bool same_sign_or_zero(eval_type a, eval_type b) {
  if (a > 0) {
    return b >= 0;
  } else if (a < 0) {
    return b <= 0;
  } else {
    return true;
  }
}

template <vector_type eval_type>
constexpr auto same_sign_or_zero(eval_type a, eval_type b) {
  return ((a > 0) && (b >= 0)) || ((a < 0) && (b <= 0)) || (a == 0);
}

template <typename eval_type>
  requires scalar_type<eval_type>
eval_type lowest_order_exp(const eval_type v) {
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

template <std::ranges::range range_t>
  requires(arith_number<typename range_t::value_type> &&
           scalar_type<typename range_t::value_type>)
bool is_nonoverlapping(range_t vals) {
  using eval_type = typename range_t::value_type;
  if (vals.size() == 0) {
    return true;
  }
  eval_type last = abs(vals[0]);
  for (auto cur : std::span{vals.begin() + 1, vals.end()} |
                      std::ranges::views::filter([](const eval_type v) {
                        return v != eval_type{0};
                      })) {
    const auto v = lowest_order_exp(abs(cur));
    if (last >= v) {
      return false;
    }
    last = v;
  }
  return true;
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

template <typename E> constexpr std::size_t num_internal_nodes() {
  if constexpr (is_expr_v<E>) {
    return num_internal_nodes<typename E::LHS>() +
           num_internal_nodes<typename E::RHS>() + 1;
  } else {
    return 0;
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

template <typename E>
  requires arith_number<E> || expr_type<E>
consteval bool sign_guaranteed(E expr) {
  if constexpr (depth(expr) <= 2) {
    return true;
  } else {
    using Op = typename E::Op;
    if (std::is_same_v<std::multiplies<>, Op> && sign_guaranteed(expr.lhs()) &&
        sign_guaranteed(expr.rhs())) {
      return true;
    }
  }
  return false;
}

namespace _impl {

constexpr std::size_t left_branch_id(const std::size_t branch_id) {
  return branch_id + 1;
}

template <evaluatable expr_t>
constexpr std::size_t right_branch_id(const std::size_t branch_id) {
  // Note that if asked, this will tell you that a left leaf node has the same
  // id as its right sibiling.
  // We don't use this id for leaf nodes, so it's not an issue here
  return branch_id + num_internal_nodes<typename expr_t::LHS>() + 1;
}

template <std::size_t branch_id, evaluatable root_t>
constexpr std::size_t get_memory_begin_idx() {
  static_assert(branch_id < num_internal_nodes<root_t>(),
                "The branch_id passed is too large for this expression tree");
  constexpr std::size_t root_branch_id = 0;
  if constexpr (branch_id == root_branch_id) {
    return 0;
  } else {
    if constexpr (branch_id < _impl::right_branch_id<root_t>(root_branch_id)) {
      return get_memory_begin_idx<branch_id -
                                      _impl::left_branch_id(root_branch_id),
                                  typename root_t::LHS>();
    } else {
      return num_partials_for_exact<typename root_t::LHS>() +
             get_memory_begin_idx<branch_id - _impl::right_branch_id<root_t>(
                                                  root_branch_id),
                                  typename root_t::RHS>();
    }
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

// Until unique_ptr with a custom deleter or vector with a custom allocator
// is cuda compatible, we unfortunately have to use a custom equivalent
template <typename eval_type, typename allocator_type> class constexpr_unique {
public:
  explicit constexpr constexpr_unique(allocator_type &mem_pool,
                                      std::size_t storage_needed)
      : mem_pool{&mem_pool}, storage_needed{storage_needed},
        ptr{mem_pool.allocate(storage_needed)} {}
  constexpr_unique(constexpr_unique &) = delete;
  constexpr ~constexpr_unique() { mem_pool->deallocate(ptr, storage_needed); }
  constexpr eval_type *get() const { return ptr; }

private:
  allocator_type *mem_pool;
  std::size_t storage_needed;
  eval_type *ptr;
};

} // namespace _impl

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP

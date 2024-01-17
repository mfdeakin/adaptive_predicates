
#ifndef ADAPTIVE_PREDICATES_AE_ADAPTIVE_PREDICATE_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_ADAPTIVE_PREDICATE_EVAL_HPP

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"
#include "ae_fp_eval.hpp"

#include <compare>
#include <limits>
#include <numeric>

namespace adaptive_expr {

namespace _impl {
template <typename E_, typename eval_type, typename allocator_type_>
  requires expr_type<E_> || arith_number<E_>
class adaptive_eval_impl;
} // namespace _impl

template <arith_number eval_type,
          typename allocator_type_ = std::pmr::polymorphic_allocator<eval_type>,
          typename E>
  requires(expr_type<E> || arith_number<E>) && scalar_type<eval_type>
eval_type adaptive_eval(E &&expr, allocator_type_ &&mem_pool =
                                      std::remove_cvref_t<allocator_type_>()) {
  using E_nr = std::remove_cvref_t<E>;
  if constexpr (sign_guaranteed(E_nr{})) {
    return fp_eval<eval_type>(expr);
  } else if constexpr (std::is_same_v<std::multiplies<>, typename E_nr::Op> ||
                       std::is_same_v<std::divides<>, typename E_nr::Op>) {
    return (typename E_nr::Op){}(
        adaptive_eval<eval_type>(expr.lhs(),
                                 std::forward<allocator_type_>(mem_pool)),
        adaptive_eval<eval_type>(expr.rhs(),
                                 std::forward<allocator_type_>(mem_pool)));
  } else {
    auto [checked_result, _] = eval_checked_fast<eval_type>(expr);
    if (std::isnan(checked_result)) {
      return _impl::adaptive_eval_impl<E, eval_type, allocator_type_>(
                 std::forward<allocator_type_>(mem_pool))
          .eval(expr);
    } else [[likely]] {
      return checked_result;
    }
  }
}

namespace _impl {

template <typename E_, typename eval_type, typename allocator_type_>
  requires expr_type<E_> || arith_number<E_>
class adaptive_eval_impl {
public:
  using E = std::remove_cvref_t<E_>;

  explicit adaptive_eval_impl(allocator_type_ mem_pool)
      : exact_storage{num_partials_for_exact<E>(),
                      std::forward<allocator_type_>(mem_pool)} {}

  explicit adaptive_eval_impl(const E_ &) : adaptive_eval_impl() {}

  template <typename EE>
    requires std::is_same_v<std::remove_cvref_t<EE>, E>
  eval_type eval(EE &&e) {
    const auto v = eval_impl(std::forward<EE>(e), 0).first;
    return v;
  }

private:
  template <evaluatable subexpr_t, evaluatable root_t = E>
  std::span<eval_type, num_partials_for_exact<subexpr_t>()>
  get_memory(const std::size_t branch_id,
             std::span<eval_type, num_partials_for_exact<root_t>()> superspan) {
    static_assert(num_partials_for_exact<root_t>() >=
                  num_partials_for_exact<subexpr_t>());
    if constexpr (num_partials_for_exact<root_t>() ==
                  num_partials_for_exact<subexpr_t>()) {
      // This means branch_id == 0
      return superspan;
    } else {
      if constexpr (expr_type<typename root_t::LHS>) {
        if (branch_id < right_branch_id<root_t>(branch_id)) {
          // branch_id > left_branch_id(0)
          return get_memory<subexpr_t, typename root_t::LHS>(
              branch_id - left_branch_id(0),
              superspan.template subspan<
                  0, num_partials_for_exact<typename root_t::LHS>()>());
        }
      }
      // branch_id > right_branch_id(0)
      return get_memory<subexpr_t, typename root_t::RHS>(
          branch_id - right_branch_id<root_t>(0),
          superspan.template subspan<
              num_partials_for_exact<typename root_t::LHS>(),
              num_partials_for_exact<typename root_t::RHS>()>());
    }
  }

  // exact_eval_root computes the requested result of the (sub)expression to 1/2
  // epsilon precision
  std::pair<eval_type, eval_type> exact_eval_root(evaluatable auto &&expr,
                                                  const std::size_t branch_id) {
    using sub_expr = decltype(expr);
    auto memory = get_memory<sub_expr>(
        branch_id, std::span<eval_type, num_partials_for_exact<E>()>{
                       exact_storage.data(), num_partials_for_exact<E>()});
    exact_eval(std::forward<sub_expr>(expr), branch_id, memory);
    _impl::merge_sum(memory);

    const eval_type exact_result = std::reduce(memory.begin(), memory.end());
    cache[branch_id] = exact_result;
    int exponent;
    std::frexp(exact_result, &exponent);
    const eval_type rel_err = std::ldexp(
        eval_type{1.0}, exponent - std::numeric_limits<eval_type>::digits - 1);
    return {exact_result, rel_err};
  }

  // handle_overshoot attempts to determine the most efficient method of
  // reducing error to levels that guarantee the correct sign
  std::pair<eval_type, eval_type>
  handle_overshoot(expr_type auto &&expr, const std::size_t branch_id,
                   const eval_type result, const eval_type left_result,
                   const eval_type left_abs_err, const eval_type right_result,
                   const eval_type right_abs_err, const eval_type max_abs_err) {
    using sub_expr_ = decltype(expr);
    using sub_expr = std::remove_cvref_t<sub_expr_>;
    using LHS = typename sub_expr::LHS;
    using RHS = typename sub_expr::RHS;
    using Op = typename sub_expr::Op;

    constexpr std::size_t subexpr_choice_latency =
        std::max(exact_fp_rounding_latency<LHS>(),
                 exact_fp_rounding_latency<RHS>()) +
        2 * overshoot_latency() + 2 * cmp_latency() +
        error_contrib_latency<Op>();

    if constexpr (exact_fp_rounding_latency<sub_expr>() >
                      subexpr_choice_latency &&
                  is_expr_v<LHS> && is_expr_v<RHS>) {
      // We need to reduce error efficiently, so don't just exactly evaluate
      // the whole expression, try to only evaluate the largest contributor
      // to the error
      const auto [left_contrib, right_contrib] = _impl::error_contributions<Op>(
          left_result, left_abs_err, right_result, right_abs_err);
      const eval_type left_overshoot =
          _impl::error_overshoot(result, left_contrib);
      const eval_type right_overshoot =
          _impl::error_overshoot(result, right_contrib);
      if (left_overshoot < 0.0 || right_overshoot < 0.0) {
        if (left_overshoot > right_overshoot) {
          // This comparison gaurantees that we exactly evaluate left or
          // right if they cause an overshoot greater than 0 It also
          // guarantees we deal with the largest part of the error, making
          // the fall-through case very unlikely
          const auto [new_left, new_left_err] =
              exact_eval_root(expr.lhs(), left_branch_id(branch_id));

          const auto [new_result, new_abs_err] =
              _impl::eval_with_max_abs_err<Op>(new_left, new_left_err,
                                               right_result, right_abs_err);
          const eval_type new_overshoot =
              _impl::error_overshoot(new_result, new_abs_err);
          if (new_overshoot < 0.0) {
            return {new_result, new_abs_err};
          }
        } else {
          const auto [new_right, new_right_err] =
              exact_eval_root(expr.rhs(), right_branch_id<sub_expr>(branch_id));
          const auto [new_result, new_abs_err] =
              _impl::eval_with_max_abs_err<Op>(left_result, left_abs_err,
                                               new_right, new_right_err);
          const eval_type new_overshoot =
              _impl::error_overshoot(new_result, new_abs_err);
          if (new_overshoot < 0.0) {
            return {new_result, new_abs_err};
          }
        }
      }
    }
    return exact_eval_root(std::forward<sub_expr_>(expr), branch_id);
  }

  std::size_t left_branch_id(const std::size_t branch_id) {
    return branch_id + 1;
  }

  template <evaluatable expr_t>
  std::size_t right_branch_id(const std::size_t branch_id) {
    // Note that if asked, this will tell you that a left leaf node has the same
    // id as its right sibiling.
    // We don't use this id for leaf nodes, so it's not an issue here
    return branch_id + num_internal_nodes<typename expr_t::LHS>() + 1;
  }

  // Returns the result and maximum absolute error from computing the expression
  std::pair<eval_type, eval_type> eval_impl(evaluatable auto &&expr,
                                            std::size_t branch_id) {
    using sub_expr = std::remove_cvref_t<decltype(expr)>;
    if constexpr (is_expr_v<sub_expr>) {
      const auto exact_eval_info = cache[branch_id];
      if (exact_eval_info) {
        return {*exact_eval_info,
                abs(*exact_eval_info) *
                    std::numeric_limits<eval_type>::epsilon() / 2.0};
      }
      using Op = typename sub_expr::Op;
      auto [left_result, left_abs_err] =
          eval_impl(expr.lhs(), left_branch_id(branch_id));
      auto [right_result, right_abs_err] =
          eval_impl(expr.rhs(), right_branch_id<sub_expr>(branch_id));
      const auto [result, max_abs_err] = _impl::eval_with_max_abs_err<Op>(
          left_result, left_abs_err, right_result, right_abs_err);

      // Multiplication doesn't affect the final sign, so don't exactly evaluate
      // it unless the upper part of the expression requires it
      if constexpr (!std::is_same_v<std::multiplies<>, Op> &&
                    depth(sub_expr{}) > 2) {
        // If we're adding values of the same sign, or subtracting opposing
        // signs, the final sign is unaffected unless the relative error is too
        // large. Do a cheap check first, then a slightly more expensive check
        constexpr bool is_plus = std::is_same_v<std::plus<>, Op>;
        constexpr bool is_minus = std::is_same_v<std::minus<>, Op>;
        if (!((is_plus && same_sign_or_zero(left_result, right_result)) ||
              (is_minus && same_sign_or_zero(left_result, -right_result))))
            [[unlikely]] {
          // We need to mirror over zero one of the values being added and
          // ensure that when mirrored, its possible range doesn't overlap the
          // other values possible range
          // This ensures that we aren't subtracting values which might have too
          // similar magnitude.
          if ((is_plus &&
               _impl::error_overlaps(left_result, left_abs_err, -right_result,
                                     right_abs_err)) ||
              (is_minus && _impl::error_overlaps(left_result, left_abs_err,
                                                 right_result, right_abs_err)))
              [[unlikely]] {
            const eval_type overshoot =
                _impl::error_overshoot(result, max_abs_err);
            if (overshoot > 0.0) {
              return handle_overshoot(expr, branch_id, result, left_result,
                                      left_abs_err, right_result, right_abs_err,
                                      max_abs_err);
            }
          }
        }
      }
      return {result, max_abs_err};
    } else {
      return {expr, 0.0};
    }
  }

  template <typename sub_expr_>
    requires expr_type<sub_expr_> || arith_number<sub_expr_>
  constexpr void
  exact_eval(sub_expr_ &&e, const std::size_t branch_id,
             std::span<eval_type, num_partials_for_exact<sub_expr_>()>
                 partial_results) noexcept {
    using sub_expr = std::remove_cvref_t<sub_expr_>;
    if constexpr (is_expr_v<sub_expr>) {
      if (cache[branch_id]) {
        return;
      }
      constexpr std::size_t reserve_left =
          num_partials_for_exact<typename sub_expr::LHS>();
      const auto storage_left = partial_results.template first<reserve_left>();
      exact_eval(e.lhs(), left_branch_id(branch_id), storage_left);

      constexpr std::size_t reserve_right =
          num_partials_for_exact<typename sub_expr::RHS>();
      const auto storage_right =
          partial_results.template subspan<reserve_left, reserve_right>();
      exact_eval(e.rhs(), right_branch_id<sub_expr>(branch_id), storage_right);

      using Op = typename sub_expr::Op;
      if constexpr (std::is_same_v<std::plus<>, Op> ||
                    std::is_same_v<std::minus<>, Op>) {
        if constexpr (std::is_same_v<std::minus<>, Op>) {
          for (eval_type &v : storage_right) {
            v = -v;
          }
        }
      } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
        const auto storage_mult =
            partial_results.template last<partial_results.size() -
                                          reserve_left - reserve_right>();
        _impl::sparse_mult(storage_left, storage_right, storage_mult);
      }
    } else if constexpr (!std::is_same_v<additive_id, sub_expr>) {
      // additive_id is zero, so we don't actually have memory allocated for it
      *partial_results.begin() = eval_type(e);
    }
  }

  std::array<std::optional<eval_type>, num_internal_nodes<E>()> cache;

  std::vector<eval_type, std::remove_cvref_t<allocator_type_>> exact_storage;
};

} // namespace _impl

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_ADAPTIVE_EVAL_HPP

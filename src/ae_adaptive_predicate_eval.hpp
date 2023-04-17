
#ifndef ADAPTIVE_PREDICATES_AE_ADAPTIVE_PREDICATE_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_ADAPTIVE_PREDICATE_EVAL_HPP

#include "ae_expr.hpp"
#include "ae_expr_io.hpp"
#include "ae_expr_utils.hpp"
#include "ae_fp_eval.hpp"

#include <compare>
#include <limits>
#include <numeric>

namespace adaptive_expr {

namespace _impl {
template <typename E_, typename eval_type>
  requires expr_type<E_> || arith_number<E_>
class adaptive_eval_impl;
} // namespace _impl

template <arith_number eval_type, typename E>
  requires expr_type<E> || arith_number<E>
eval_type adaptive_eval(E &&expr) {
  if constexpr (arith_number<E>) {
    return expr;
  } else {
    return _impl::adaptive_eval_impl<E, eval_type>().eval(expr);
  }
}

namespace _impl {

template <typename E_, typename eval_type>
  requires expr_type<E_> || arith_number<E_>
class adaptive_eval_impl {
public:
  using E = std::remove_cvref_t<E_>;

  adaptive_eval_impl() : exact_storage{} {
    if constexpr (!use_array) {
      exact_storage.reserve(num_partials_for_exact<E>());
    }
  }

  explicit adaptive_eval_impl(const E_ &) : adaptive_eval_impl() {}

  template <typename EE>
    requires std::is_same_v<std::remove_cvref_t<EE>, E>
  eval_type eval(EE &&e) {
    const auto v = eval_impl<EE, branch_token_leaf>(std::forward<EE>(e)).first;
    return v;
  }

private:
  class branch_token_leaf : public branch_token_s {
  public:
    template <template <class> class branch_dir>
    using append_branch = branch_dir<branch_token_leaf>;

    branch_token_leaf &get() { return *this; }

    std::optional<eval_type> result;
  };

  template <branch_token branch> constexpr auto get_memory() {
    // since exact_storage might be a vector, ensure the compiler knows how
    // large the vector is
    return get_memory_impl<branch, E>(
        std::span<eval_type, num_partials_for_exact<E>()>{exact_storage});
  }

  template <branch_token branch, typename sub_expr, std::ranges::range span_t>
    requires expr_type<sub_expr> || arith_number<sub_expr>
  constexpr auto get_memory_impl(span_t span) {
    if constexpr (std::is_same_v<branch_token_leaf, branch>) {
      return span;
    } else if constexpr (branch::is_left()) {
      static_assert(!branch::is_right());
      return get_memory_impl<typename branch::S, typename sub_expr::LHS>(
          span.template first<
              num_partials_for_exact<typename sub_expr::LHS>()>());
    } else {
      static_assert(branch::is_right());
      return get_memory_impl<typename branch::S, typename sub_expr::RHS>(
          span.template subspan<
              num_partials_for_exact<typename sub_expr::LHS>(),
              num_partials_for_exact<typename sub_expr::RHS>()>());
    }
  }

  template <typename sub_expr_, branch_token branch = branch_token_leaf>
    requires expr_type<sub_expr_> || arith_number<sub_expr_>
  std::pair<eval_type, eval_type> eval_impl(sub_expr_ &&expr) {
    using sub_expr = std::remove_cvref_t<sub_expr_>;

    if constexpr (is_expr_v<sub_expr>) {
      branch_token_leaf &exact_eval_info = std::get<branch>(cache).get();
      if (exact_eval_info.result) {
        return {*exact_eval_info.result,
                std::abs(*exact_eval_info.result) *
                    std::numeric_limits<eval_type>::epsilon() / 2.0};
      }
      using Op = typename sub_expr::Op;
      auto [left_result, left_abs_err] =
          eval_impl<decltype(expr.lhs()),
                    typename branch::template append_branch<branch_token_left>>(
              expr.lhs());
      auto [right_result, right_abs_err] = eval_impl<
          decltype(expr.rhs()),
          typename branch::template append_branch<branch_token_right>>(
          expr.rhs());
      const auto [result, max_abs_err] = _impl::eval_with_max_abs_err<Op>(
          left_result, left_abs_err, right_result, right_abs_err);
      if (max_abs_err >
          std::abs(result) *
              eval_type(1.0 -
                        std::numeric_limits<eval_type>::epsilon() * 8.0)) {
        auto memory = get_memory<branch>();
        exact_eval<branch>(std::forward<sub_expr_>(expr), memory);
        const eval_type exact_result =
            std::reduce(memory.begin(), memory.end());
        exact_eval_info.result = exact_result;
        return {exact_result, std::abs(exact_result) *
                                  std::numeric_limits<eval_type>::epsilon() /
                                  2.0};
      } else {
        return {result, max_abs_err};
      }
    } else {
      return {expr, 0.0};
    }
  }

  template <branch_token branch, typename sub_expr_>
    requires expr_type<sub_expr_> || arith_number<sub_expr_>
  constexpr void
  exact_eval(sub_expr_ &&e,
             std::span<eval_type, num_partials_for_exact<sub_expr_>()>
                 partial_results) noexcept {
    using sub_expr = std::remove_cvref_t<sub_expr_>;
    if constexpr (is_expr_v<sub_expr>) {
      branch_token_leaf &exact_eval_info = std::get<branch>(cache).get();
      if (exact_eval_info.result) {
        return;
      }
      constexpr std::size_t reserve_left =
          num_partials_for_exact<typename sub_expr::LHS>();
      const auto storage_left = partial_results.template first<reserve_left>();
      exact_eval<typename branch::template append_branch<branch_token_left>>(
          e.lhs(), storage_left);
      constexpr std::size_t reserve_right =
          num_partials_for_exact<typename sub_expr::RHS>();
      const auto storage_right =
          partial_results.template subspan<reserve_left, reserve_right>();
      exact_eval<typename branch::template append_branch<branch_token_right>>(
          e.rhs(), storage_right);
      using Op = typename sub_expr::Op;
      if constexpr (std::is_same_v<std::plus<>, Op> ||
                    std::is_same_v<std::minus<>, Op>) {
        if constexpr (std::is_same_v<std::minus<>, Op>) {
          for (eval_type &v : storage_right) {
            v = -v;
          }
        }
        _impl::merge_sum(partial_results);
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

  using cache_tuple =
      std::invoke_result_t<enumerate_branches_functor<branch_token_leaf>, E>;

  cache_tuple cache;

  static constexpr bool use_array = num_partials_for_exact<E>() < 512;
  std::conditional_t<use_array,
                     std::array<eval_type, num_partials_for_exact<E>()>,
                     std::vector<eval_type>>
      exact_storage;
};

} // namespace _impl

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_ADAPTIVE_EVAL_HPP

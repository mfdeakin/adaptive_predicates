
#ifndef ADAPTIVE_PREDICATES_AE_ADAPTIVE_PREDICATE_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_ADAPTIVE_PREDICATE_EVAL_HPP

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"
#include "ae_fp_eval.hpp"

#include <compare>
#include <limits>

namespace adaptive_expr {

template <typename E_, typename eval_type>
  requires expr_type<E_> || arith_number<E_>
class adaptive_eval {
public:
  using E = std::remove_cvref_t<E_>;

  adaptive_eval() : exact_storage{} {
    if constexpr (!use_array) {
      exact_storage.reserve(num_partials_for_exact<E>());
    }
    init_cache<branch_token_leaf, E>(exact_storage);
  }

  explicit adaptive_eval(const E_ &) : adaptive_eval() {}

  template <typename EE>
    requires std::is_same_v<std::remove_cvref_t<EE>, E>
  eval_type eval(EE &&e) {
    return eval_impl<EE, branch_token_leaf>(std::forward<EE>(e)).first;
  }

private:
  template <branch_token branch, typename sub_expr>
    requires expr_type<sub_expr> || arith_number<sub_expr>
  void init_cache(std::span<eval_type> memory) {
    if constexpr (is_expr_v<sub_expr>) {
      branch_token_leaf &block = std::get<branch>(cache).get();
      block.memory = memory;
      using LHS = typename sub_expr::LHS;
      using RHS = typename sub_expr::RHS;
      init_cache<typename branch::append_branch<branch_token_left>, LHS>(
          memory.first(num_partials_for_exact<LHS>()));
      init_cache<typename branch::append_branch<branch_token_right>, RHS>(
          memory.last(num_partials_for_exact<RHS>()));
    }
  }

  class branch_token_leaf : public branch_token_s {
  public:
    template <template <class> class branch_dir>
    using append_branch = branch_dir<branch_token_leaf>;

    branch_token_leaf &get() { return *this; }
    std::span<eval_type> memory;
    eval_type result = std::numeric_limits<eval_type>::signaling_NaN();
    bool computed = false;
  };

  template <typename sub_expr_, branch_token cache_type = branch_token_leaf>
    requires expr_type<sub_expr_> || arith_number<sub_expr_>
  std::pair<eval_type, eval_type> eval_impl(sub_expr_ &&expr) {
    using sub_expr = std::remove_cvref_t<sub_expr_>;

    if constexpr (is_expr_v<sub_expr>) {
      branch_token_leaf &exact_eval_info = std::get<cache_type>(cache).get();
      if (exact_eval_info.computed) {
        return {exact_eval_info.result,
                std::abs(exact_eval_info.result) *
                    std::numeric_limits<eval_type>::epsilon() / 2.0};
      }
      using Op = typename sub_expr::Op;
      auto [left_result, left_abs_err] =
          eval_impl<decltype(expr.lhs()), typename cache_type::append_branch<branch_token_left>>(
              expr.lhs());
      auto [right_result, right_abs_err] =
          eval_impl<decltype(expr.rhs()), typename cache_type::append_branch<branch_token_right>>(
              expr.rhs());
      const auto [result, max_abs_err] = _impl::eval_with_max_abs_err<Op>(
          left_result, left_abs_err, right_result, right_abs_err);
      if (max_abs_err >
          std::abs(result) *
              eval_type(1.0 -
                        std::numeric_limits<eval_type>::epsilon() * 8.0)) {
        _impl::exactfp_eval_impl<eval_type>(std::forward<sub_expr_>(expr),
                                            exact_eval_info.memory);
        const eval_type exact_result = std::reduce(
            exact_eval_info.memory.begin(), exact_eval_info.memory.end());
        exact_eval_info.computed = true;
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

  using cache_tuple =
      std::invoke_result_t<enumerate_branches_functor<branch_token_leaf>, E>;

  cache_tuple cache;

  static constexpr bool use_array = num_partials_for_exact<E>() < 512;
  std::conditional_t<use_array,
                     std::array<eval_type, num_partials_for_exact<E>()>,
                     std::vector<eval_type>>
      exact_storage;
};

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_ADAPTIVE_EVAL_HPP

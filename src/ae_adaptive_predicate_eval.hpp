
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

private:
  template <branch_token branch, typename sub_expr>
    requires expr_type<sub_expr> || arith_number<sub_expr>
  void init_cache(std::span<eval_type> memory) {
    if constexpr (is_expr_v<sub_expr>) {
      branch_token_leaf &block = std::get<branch>(cache).get();
      block.memory = memory;
      using LHS = typename sub_expr::LHS;
      using RHS = typename sub_expr::RHS;
      init_cache<branch_token_left<branch>, LHS>(
          memory.first(num_partials_for_exact<LHS>()));
      init_cache<branch_token_right<branch>, RHS>(
          memory.last(num_partials_for_exact<RHS>()));
    }
  }

  template <typename Op>
  static constexpr std::pair<eval_type, eval_type>
  compute_max_err(const eval_type left, const eval_type left_abs_err,
                  const eval_type right, const eval_type right_abs_err) {
    const eval_type result = Op()(left, right);
    if constexpr (std::is_same_v<Op, std::plus<>> ||
                  std::is_same_v<Op, std::minus<>>) {
      return {result, left_abs_err + right_abs_err +
                          std::abs(result) *
                              std::numeric_limits<eval_type>::epsilon() / 2};
    } else if constexpr (std::is_same_v<Op, std::multiplies<>>) {
      return {result, right * left_abs_err + left * right_abs_err +
                          left_abs_err * right_abs_err +
                          std::abs(result) *
                              std::numeric_limits<eval_type>::epsilon() / 2};
    } else {
      return {result, std::numeric_limits<eval_type>::signaling_NaN()};
    }
  }

  class branch_token_leaf : public branch_token_s {
  public:
    branch_token_leaf &get() { return *this; }
    std::span<eval_type> memory;
    eval_type result = std::numeric_limits<eval_type>::signaling_NaN();
    bool computed = false;
  };

  template <branch_token cache_type, typename sub_expr_>
    requires expr_type<sub_expr_> || arith_number<sub_expr_>
  std::pair<eval_type, eval_type> eval(sub_expr_ &&expr) {
    using sub_expr = std::remove_reference_t<sub_expr_>;
    branch_token_leaf &exact_eval_info = std::get<cache_type>(cache).get();

    if constexpr (expr_type<sub_expr>) {
      using Op = typename sub_expr::Op;
      auto [left_result, left_abs_err] =
          eval<branch_token_left<cache_type>>(expr.lhs());
      auto [right_result, right_abs_err] =
          eval<branch_token_right<cache_type>>(expr.rhs());
      const auto [result, max_abs_err] = compute_max_err<Op>(
          left_result, left_abs_err, right_result, right_abs_err);
      if (max_abs_err > std::abs(result) * eval_type(1.0 - 0.00001)) {
        const eval_type exact_result = exact_eval<cache_type>(expr);
        return {exact_result, std::numeric_limits<eval_type>::epsilon() / 2.0};
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

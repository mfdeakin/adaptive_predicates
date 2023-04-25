
#ifndef ADAPTIVE_PREDICATES_AE_FP_EVAL_IMPL_HPP
#define ADAPTIVE_PREDICATES_AE_FP_EVAL_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <ranges>
#include <span>

#include <fmt/ranges.h>

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"

namespace adaptive_expr {

namespace _impl {

auto merge_sum_linear(
    std::ranges::range auto &&storage,
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type;
auto merge_sum_linear_fast(std::ranges::range auto &&storage,
                           const typename decltype(storage)::iterator midpoint)
    -> typename decltype(storage)::value_type;
auto merge_sum_quadratic(std::ranges::range auto &&storage) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type;

auto merge_sum(std::ranges::range auto storage) ->
    typename decltype(storage)::value_type {
  return merge_sum_quadratic(storage);
}

template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_m>
void sparse_mult(span_l storage_left, span_r storage_right,
                 span_m storage_mult);

template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type> knuth_sum(const eval_type &lhs,
                                                    const eval_type &rhs);
template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type> dekker_sum(const eval_type &lhs,
                                                     const eval_type &rhs);
template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type> dekker_sum2(const eval_type &lhs,
                                                      const eval_type &rhs);
template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_unchecked(const eval_type &lhs, const eval_type &rhs);

template <std::floating_point eval_type>
std::pair<eval_type, eval_type> exact_mult(const eval_type &lhs,
                                           const eval_type &rhs);

template <std::floating_point eval_type, typename E>
  requires expr_type<E> || arith_number<E>
consteval eval_type max_rel_error() {
  if constexpr (is_expr_v<E>) {
    using Op = typename E::Op;
    const eval_type max_left = max_rel_error<eval_type, typename E::LHS>();
    const eval_type max_right = max_rel_error<eval_type, typename E::RHS>();
    const eval_type eps = std::numeric_limits<eval_type>::epsilon();
    if constexpr (std::is_same_v<std::plus<>, Op> ||
                  std::is_same_v<std::minus<>, Op>) {
      // For plus and minus, this only applies to their adaptive implementation;
      // using the basic floating point plus and minus of course has unbounded
      // relative error
      if (max_left == 0 && max_right == 0) {
        return eps / 2;
      } else {
        return std::max(max_left, max_right) * 2.0;
      }
    } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
      if (max_left == 0 && max_right == 0) {
        return eps / 2;
      } else if (max_left == 0) {
        return max_right * 2;
      } else if (max_right == 0) {
        return max_left * 2;
      } else {
        return (eps / 2 + 2 * max_left + 2 * max_right + max_left * max_right);
      }
    } else {
      static_assert(std::is_same_v<std::plus<>, Op> ||
                    std::is_same_v<std::minus<>, Op> ||
                    std::is_same_v<std::multiplies<>, Op>);
      return std::numeric_limits<eval_type>::signaling_NaN();
    }
  } else {
    return 0;
  }
}

template <typename Op, arith_number eval_type>
constexpr std::pair<eval_type, eval_type>
eval_with_max_abs_err(const eval_type left, const eval_type left_abs_err,
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

template <std::floating_point eval_type, typename E_, std::ranges::range span_t>
  requires expr_type<E_> || arith_number<E_>
constexpr void exactfp_eval_impl(E_ &&e, span_t partial_results) noexcept {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr_v<E>) {
    constexpr std::size_t reserve_left =
        num_partials_for_exact<typename E::LHS>();
    const auto storage_left = partial_results.template first<reserve_left>();
    exactfp_eval_impl<eval_type>(e.lhs(), storage_left);
    constexpr std::size_t reserve_right =
        num_partials_for_exact<typename E::RHS>();
    const auto storage_right =
        partial_results.template subspan<reserve_left, reserve_right>();
    exactfp_eval_impl<eval_type>(e.rhs(), storage_right);
    using Op = typename E::Op;
    if constexpr (std::is_same_v<std::minus<>, Op>) {
      for (eval_type &v : storage_right) {
        v = -v;
      }
      merge_sum_linear(partial_results,
                       partial_results.begin() + storage_left.size());
    } else if constexpr (std::is_same_v<std::plus<>, Op>) {
      merge_sum_linear(partial_results,
                       partial_results.begin() + storage_left.size());
    } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
      const auto storage_mult = [partial_results]() {
        if constexpr (span_t::extent == std::dynamic_extent) {
          return partial_results.last(partial_results.size() - reserve_left -
                                      reserve_right);
        } else {
          return partial_results.template last<partial_results.size() -
                                               reserve_left - reserve_right>();
        }
      }();
      sparse_mult(storage_left, storage_right, storage_mult);
    }
  } else if constexpr (!std::is_same_v<additive_id, E>) {
    *partial_results.begin() = eval_type(e);
  }
}

auto merge_sum_linear_fast(
    std::ranges::range auto &&storage,
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type {
  using eval_type = typename std::remove_cvref_t<decltype(storage)>::value_type;
  if (storage.size() > 1) {
    std::ranges::inplace_merge(storage, midpoint, [](eval_type l, eval_type r) {
      return std::abs(l) > std::abs(r);
    });
    auto [Q, _] = dekker_sum(storage[0], storage[1]);
    for (auto g : storage | std::views::drop(2) |
                      std::views::filter([](const eval_type v) {
                        return v != eval_type{0};
                      })) {
      std::tie(Q, _) = dekker_sum(g, Q);
    }
    return Q;
  } else if (storage.size() == 1) {
    return storage[0];
  } else {
    return 0.0;
  }
}

auto merge_sum_linear(
    std::ranges::range auto &&storage,
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type {
  using eval_type = typename std::remove_cvref_t<decltype(storage)>::value_type;
  if (storage.size() > 1) {
    std::ranges::inplace_merge(
        storage, midpoint, [](const eval_type &left, const eval_type &right) {
          return std::abs(left) < std::abs(right);
        });
    auto [Q, q] = dekker_sum_unchecked(storage[1], storage[0]);
    for (auto g : storage | std::views::drop(2) |
                      std::views::filter([](const eval_type v) {
                        return v != eval_type{0};
                      })) {
      auto [R, _] = dekker_sum_unchecked(g, q);
      std::tie(Q, q) = knuth_sum(Q, R);
    }
    return Q;
  } else if (storage.size() == 1) {
    return storage[0];
  } else {
    return 0.0;
  }
}

auto merge_sum_append(auto begin, auto end, auto v) {
  using eval_type = decltype(v);
  auto out = begin;
  for (auto &e : std::span{begin, end}) {
    const auto [result, error] = dekker_sum(v, e);
    e = eval_type{0.0};
    v = result;
    if (error) {
      *out = error;
      ++out;
    }
  }
  return std::pair{out, v};
}

auto merge_sum_quadratic(std::ranges::range auto &&storage) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type {
  using eval_type = typename std::remove_cvref_t<decltype(storage)>::value_type;
  if (storage.size() > 1) {
    auto out = storage.begin();
    for (eval_type &inp : storage | std::views::filter([](const eval_type v) {
                            return v != eval_type{0};
                          })) {
      eval_type v = inp;
      inp = eval_type{0.0};
      auto [new_out, result] = merge_sum_append(storage.begin(), out, v);
      out = new_out;
      if (result) {
        *out = result;
        ++out;
      }
    }
    if (out == storage.begin()) {
      return eval_type{0};
    } else {
      return *(out - 1);
    }
  } else if (storage.size() == 1) {
    return storage[0];
  } else {
    return eval_type{0.0};
  }
}

// This performs multiplication in-place for a contiguous piece of memory
// starting at storage_left.begin() and ending at storage_mult.end()
//
// storage_mult is initially empty and written to first
// storage_right is overwritten second, each value in storage_left is finished
// and over-writable when its iteration of the outer loop finishes
// storage_left can be shown to only be is overwritten during the final
// iteration of the outer loop, the values in it are only overwritten after
// they've been multiplied If storage_left and storage_right are sorted by
// increasing magnitude before multiplying, the first element in the output is
// the least significant and the last element is the most significant
template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_m>
void sparse_mult(span_l storage_left, span_r storage_right,
                 span_m storage_mult) {
  if (storage_right.size() == 0 || storage_left.size() == 0) {
    return;
  }
  using eval_type = typename span_l::value_type;
  auto seq_begin = storage_mult.end();
  auto out_i = storage_mult.end() - 1;
  for (auto r : storage_right | std::views::reverse) {
    eval_type err;
    std::tie(*out_i, err) =
        exact_mult(r, (storage_left | std::views::reverse)[0]);
    out_i--;
    for (auto l : storage_left | std::views::reverse | std::views::drop(1)) {
      auto [upper, lower] = exact_mult(l, r);
      std::tie(*out_i, err) = dekker_sum(err, upper);
      --out_i;
      std::tie(*out_i, err) = dekker_sum_unchecked(err, lower);
      --out_i;
    }
    *out_i = err;
    out_i--;
    if (seq_begin != storage_mult.end()) {
      std::span one_mult{out_i + 1, storage_mult.end()};
      merge_sum_linear(one_mult, one_mult.begin() + (seq_begin - out_i - 1));
    }
    seq_begin = out_i + 1;
  }
}

template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type> dekker_sum(const eval_type &lhs,
                                                     const eval_type &rhs) {
  if (std::abs(lhs) >= std::abs(rhs)) {
    return dekker_sum_unchecked(lhs, rhs);
  } else {
    return dekker_sum_unchecked(rhs, lhs);
  }
}

template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type> dekker_sum2(const eval_type &lhs,
                                                      const eval_type &rhs) {
  if ((lhs > rhs) == (lhs > -rhs)) {
    return dekker_sum_unchecked(lhs, rhs);
  } else {
    return dekker_sum_unchecked(rhs, lhs);
  }
}

template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_unchecked(const eval_type &lhs, const eval_type &rhs) {
  const eval_type upper = lhs + rhs;
  const eval_type rounding_err = upper - lhs;
  return {upper, rhs - rounding_err};
}

template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type> knuth_sum(const eval_type &lhs,
                                                    const eval_type &rhs) {
  const eval_type upper = lhs + rhs;
  const eval_type rounding_err_rhs = upper - lhs;
  const eval_type rounding_err_lhs = upper - rounding_err_rhs;
  const eval_type lhs_roundoff = rhs - rounding_err_rhs;
  const eval_type rhs_roundoff = lhs - rounding_err_lhs;
  return {upper, lhs_roundoff + rhs_roundoff};
}

template <std::floating_point eval_type>
std::pair<eval_type, eval_type> exact_mult(const eval_type &lhs,
                                           const eval_type &rhs) {
  eval_type big = lhs * rhs;
  return {big, std::fma(lhs, rhs, -big)};
}

} // namespace _impl

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_IMPL_HPP

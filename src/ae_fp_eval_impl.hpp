
#ifndef ADAPTIVE_PREDICATES_AE_FP_EVAL_IMPL_HPP
#define ADAPTIVE_PREDICATES_AE_FP_EVAL_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>
#include <span>

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"

namespace adaptive_expr {

namespace _impl {

/* merge_sum_linear runs in-place in linear time, but requires the two sequences
 * in storage to be strongly non-overlapping.
 * That is, each sequence must be non-overlapping and elements which aren't
 * powers of two must be non-adjacent. Elements which are powers of two can be
 * adjacent to at most one other element in its sequence.
 * Elements a, b with abs(a) < abs(b) are adjacent if (a, b) is overlapping or
 * if (2 * a, b) is overlapping
 */
auto merge_sum_linear(
    std::ranges::range auto &&storage,
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type;
auto merge_sum_linear_fast(std::ranges::range auto &&storage,
                           const typename decltype(storage)::iterator midpoint)
    -> typename decltype(storage)::value_type;
auto merge_sum_quadratic(std::ranges::range auto &&storage) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type;
auto merge_sum_quadratic_keep_zeros(std::ranges::range auto &&storage) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type;

auto merge_sum(std::ranges::range auto storage) ->
    typename decltype(storage)::value_type {
  if constexpr (vector_type<typename decltype(storage)::value_type>) {
    return merge_sum_quadratic_keep_zeros(storage);
  } else {
    return merge_sum_quadratic(storage);
  }
}

template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_m>
void sparse_mult(span_l storage_left, span_r storage_right,
                 span_m storage_mult);

template <arith_number eval_type>
constexpr std::pair<eval_type, eval_type> knuth_sum(const eval_type &lhs,
                                                    const eval_type &rhs);
template <arith_number eval_type>
  requires scalar_type<eval_type>
constexpr std::pair<eval_type, eval_type> dekker_sum(const eval_type &lhs,
                                                     const eval_type &rhs);
template <vector_type eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_vector_1(const eval_type &lhs, const eval_type &rhs);
template <arith_number eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_vector_2(const eval_type &lhs, const eval_type &rhs);
template <arith_number eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_unchecked(const eval_type &lhs, const eval_type &rhs);

template <typename eval_type>
  requires std::floating_point<eval_type> || vector_type<eval_type>
constexpr std::pair<eval_type, eval_type> two_sum(const eval_type &lhs,
                                                  const eval_type &rhs) {
  if constexpr (vector_type<eval_type>) {
    return knuth_sum(lhs, rhs);
  } else {
    return dekker_sum(lhs, rhs);
  }
}

template <arith_number eval_type>
std::pair<eval_type, eval_type> exact_mult(const eval_type &lhs,
                                           const eval_type &rhs);

// This technically would work with a vector_type, but would waste cycles
template <typename eval_type, typename E>
  requires((expr_type<E> || arith_number<E>) && scalar_type<eval_type>)
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
        return std::max(max_left, max_right) + eps / (eval_type{2} - eps);
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
error_contributions(const eval_type left, const eval_type left_abs_err,
                    const eval_type right, const eval_type right_abs_err) {
  if constexpr (std::is_same_v<Op, std::plus<>> ||
                std::is_same_v<Op, std::minus<>>) {
    return {left_abs_err, right_abs_err};
  } else if constexpr (std::is_same_v<Op, std::multiplies<>>) {
    return {abs(right) * left_abs_err, abs(left) * right_abs_err};
  } else {
    if constexpr (scalar_type<eval_type>) {
      return {std::numeric_limits<eval_type>::signaling_NaN(),
              std::numeric_limits<eval_type>::signaling_NaN()};
    } else {
      return {eval_type{std::numeric_limits<double>::signaling_NaN()},
              eval_type{std::numeric_limits<double>::signaling_NaN()}};
    }
  }
}

template <typename Op, arith_number eval_type>
constexpr std::pair<eval_type, eval_type>
eval_with_max_abs_err(const eval_type left, const eval_type left_abs_err,
                      const eval_type right, const eval_type right_abs_err) {
  const auto [left_contrib, right_contrib] =
      error_contributions<Op>(left, left_abs_err, right, right_abs_err);
  const eval_type result = Op()(left, right);
  return {result,
          left_contrib + right_contrib +
              abs(result) * std::numeric_limits<eval_type>::epsilon() / 2};
}

constexpr auto error_overshoot(const arith_number auto result,
                               const arith_number auto max_abs_err) {
  return max_abs_err - abs(result);
}

template <arith_number eval_type>
  requires scalar_type<eval_type>
constexpr bool
error_overlaps(const eval_type left_result, const eval_type left_abs_err,
               const eval_type right_result, const eval_type right_abs_err) {
  if ((left_result - left_abs_err) > (right_result - right_abs_err)) {
    return error_overlaps(right_result, right_abs_err, left_result,
                          left_abs_err);
  }
  return (right_result - right_abs_err) < (left_result + left_abs_err);
}

template <vector_type eval_type>
constexpr auto
error_overlaps(const eval_type left_result, const eval_type left_abs_err,
               const eval_type right_result, const eval_type right_abs_err) {
  return (((left_result - left_abs_err) <= (right_result - right_abs_err)) &&
          ((right_result - right_abs_err) < (left_result + left_abs_err))) ||
         (((left_result - left_abs_err) > (right_result - right_abs_err)) &&
          ((right_result - right_abs_err) >= (left_result + left_abs_err)));
}

template <arith_number eval_type, typename E_, std::ranges::range span_t>
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
    partial_results[0] = eval_type(e);
  }
}

auto merge_sum_linear_fast(
    std::ranges::range auto &&storage,
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type {
  using eval_type = typename std::remove_cvref_t<decltype(storage)>::value_type;
  if (storage.size() > 1) {
    std::ranges::inplace_merge(
        storage, midpoint, [](const eval_type &left, const eval_type &right) {
          return abs(left) < abs(right);
        });
    auto nonzero_itr = storage.begin();
    for (; nonzero_itr != storage.end() && *nonzero_itr == eval_type{0};
         ++nonzero_itr) {
    }
    std::ranges::rotate(storage, nonzero_itr);

    auto [Q, q] = dekker_sum_unchecked(storage[1], storage[0]);
    auto out = storage.begin();
    *out = q;
    ++out;
    for (auto g : storage | std::views::drop(2)) {
      auto [Qnew, h] = two_sum(Q, g);
      Q = Qnew;
      *out = h;
      ++out;
    }

    *out = Q;
    ++out;
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
          return abs(left) < abs(right);
        });
    auto nonzero_itr = storage.begin();
    for (; nonzero_itr != storage.end() && *nonzero_itr == eval_type{0};
         ++nonzero_itr) {
    }
    std::ranges::rotate(storage, nonzero_itr);
    auto [Q, q] = dekker_sum_unchecked(storage[1], storage[0]);
    auto out = storage.begin();
    for (auto h : storage | std::views::drop(2)) {
      auto [R, g] = dekker_sum_unchecked(h, q);
      *out = g;
      ++out;
      std::tie(Q, q) = two_sum(Q, R);
    }

    *out = q;
    ++out;
    *out = Q;
    ++out;

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
    const auto [result, error] = two_sum(v, e);
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

auto merge_sum_append_keep_zeros(auto begin, auto end) {
  auto v = *end;
  for (auto &e : std::span{begin, end}) {
    const auto [result, error] = two_sum(v, e);
    e = error;
    v = result;
  }
  return v;
}

auto merge_sum_quadratic_keep_zeros(std::ranges::range auto &&storage) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type {
  using eval_type = typename std::remove_cvref_t<decltype(storage)>::value_type;
  if (storage.size() > 1) {
    for (auto inp = storage.begin(); inp != storage.end(); ++inp) {
      *inp = merge_sum_append_keep_zeros(storage.begin(), inp);
    }
    return std::reduce(storage.begin(), storage.end());
  } else if (storage.size() == 1) {
    return storage[0];
  } else {
    return eval_type{0.0};
  }
}

template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_m>
void sparse_mult(span_l storage_left, span_r storage_right,
                 span_m storage_mult) {
#ifndef __FMA__
  static_assert(!vector_type<typename span_l::value_type>,
                "Vectorization doesn't have a functional mul_sub method, "
                "cannot efficiently evaluate multiplications exactly");
#endif // __FMA__
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
  auto out_i = storage_mult.end() - 1;
  for (auto r : storage_right | std::views::reverse) {
    for (auto l : storage_left | std::views::reverse) {
      auto [upper, lower] = exact_mult(r, l);
      *out_i = upper;
      --out_i;
      *out_i = lower;
      --out_i;
    }
  }
}

template <arith_number eval_type>
  requires scalar_type<eval_type>
constexpr std::pair<eval_type, eval_type> dekker_sum(const eval_type &lhs,
                                                     const eval_type &rhs) {
  if (abs(lhs) >= abs(rhs)) {
    return dekker_sum_unchecked(lhs, rhs);
  } else {
    return dekker_sum_unchecked(rhs, lhs);
  }
}

template <vector_type eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_vector_1(const eval_type &lhs, const eval_type &rhs) {
  const auto swaps = abs(lhs) >= abs(rhs);
  const eval_type newLeft = select(swaps, lhs, rhs);
  const eval_type newRight = select(!swaps, lhs, rhs);
  return dekker_sum_unchecked(newLeft, newRight);
}

template <arith_number eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_vector_2(const eval_type &lhs, const eval_type &rhs) {
  const auto swaps = abs(lhs) >= abs(rhs);
  const eval_type newLeft =
      (eval_type{0} + swaps) * lhs + (eval_type{1} - swaps) * rhs;
  const eval_type newRight =
      (eval_type{1} - swaps) * lhs + (eval_type{0} + swaps) * rhs;
  return dekker_sum_unchecked(newLeft, newRight);
}

template <arith_number eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_unchecked(const eval_type &lhs, const eval_type &rhs) {
  const eval_type upper = lhs + rhs;
  const eval_type rounding_err = upper - lhs;
  return {upper, rhs - rounding_err};
}

template <arith_number eval_type>
constexpr std::pair<eval_type, eval_type> knuth_sum(const eval_type &lhs,
                                                    const eval_type &rhs) {
  const eval_type upper = lhs + rhs;
  const eval_type rounding_err_rhs = upper - lhs;
  const eval_type rounding_err_lhs = upper - rounding_err_rhs;
  const eval_type lhs_roundoff = rhs - rounding_err_rhs;
  const eval_type rhs_roundoff = lhs - rounding_err_lhs;
  return {upper, lhs_roundoff + rhs_roundoff};
}

template <arith_number eval_type>
std::pair<eval_type, eval_type> exact_mult(const eval_type &lhs,
                                           const eval_type &rhs) {
  eval_type big = lhs * rhs;
  return {big, mul_sub(lhs, rhs, big)};
}

} // namespace _impl

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_IMPL_HPP

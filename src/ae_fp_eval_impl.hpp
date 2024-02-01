
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
auto merge_sum_linear_fast(
    std::ranges::range auto &&storage,
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type;
constexpr auto merge_sum_quadratic(std::ranges::range auto &&storage) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type;
constexpr auto merge_sum_quadratic_keep_zeros(std::ranges::range auto &&storage)
    -> typename std::remove_cvref_t<decltype(storage)>::value_type;

constexpr auto merge_sum(std::ranges::range auto storage) ->
    typename decltype(storage)::value_type {
  if constexpr (vector_type<typename decltype(storage)::value_type>) {
    return merge_sum_quadratic_keep_zeros(storage);
  } else {
    return merge_sum_quadratic(storage);
  }
}

template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_m>
constexpr void sparse_mult(span_l storage_left, span_r storage_right,
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
constexpr std::pair<eval_type, eval_type> exact_mult(const eval_type &lhs,
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
    for (auto g : std::span{storage.begin() + 2, storage.end()}) {
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
    for (auto h : std::span{storage.begin() + 2, storage.end()}) {
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

constexpr auto merge_sum_append(auto begin, auto end, auto v) {
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

constexpr auto merge_sum_quadratic(std::ranges::range auto &&storage) ->
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

constexpr auto merge_sum_append_keep_zeros(auto begin, auto end) {
  auto v = *end;
  for (auto e = begin; e != end; ++e) {
    const auto [result, error] = two_sum(v, *e);
    *e = error;
    v = result;
  }
  return v;
}

constexpr auto merge_sum_quadratic_keep_zeros(std::ranges::range auto &&storage)
    -> typename std::remove_cvref_t<decltype(storage)>::value_type {
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
constexpr void sparse_mult(span_l storage_left, span_r storage_right,
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
  for (auto r_itr = storage_right.rbegin(); r_itr != storage_right.rend();
       ++r_itr) {
    const auto r = *r_itr;
    for (auto l_itr = storage_left.rbegin(); l_itr != storage_left.rend();
         ++l_itr) {
      const auto l = *l_itr;
      auto [upper, lower] = exact_mult(r, l);
      *out_i = upper;
      --out_i;
      *out_i = lower;
      --out_i;
    }
  }
}

template <std::ranges::range span_l, typename eval_type,
          std::ranges::range span_result>
constexpr void sparse_mult_merge_term(const span_l storage_left,
                                      const eval_type v, span_result result) {
  // h is the output list
  // We only need two values of Q at a time
  // T_i, t_i are transient
  //
  // (Q_2, h_1) <= exact_mult(l[0], v)
  // for i : 1 ... m
  //   (T_i, t_i) <= exact_mult(l[i], v)
  //   (Q_{2i - 1}, h_{2i - 2}) <= two_sum(Q_{2i - 2}, t_i)
  //   (Q_{2i}, h_{2i - 1}) <= fast_two_sum(T_i, Q_{2i - 1})
  // h_{2m} <= Q_{2m}
  const auto r0 = exact_mult(storage_left[0], v);
  eval_type q0{r0.first};
  eval_type q1{0};
  result[0] = r0.second;
  for (std::size_t i = 1; i < storage_left.size(); ++i) {
    const auto [T, t] = exact_mult(storage_left[i], v);
    const auto ri0 = two_sum(q0, t);
    q1 = ri0.first;
    result[2 * i - 1] = ri0.second;
    const auto ri1 = two_sum(T, q1);
    q0 = ri1.first;
    result[2 * i] = ri0.second;
  }
  result[result.size() - 1] = q0;
}

template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_result, typename allocator_type_>
constexpr void sparse_mult_merge(span_l storage_left, span_r storage_right,
                                 span_result result,
                                 allocator_type_ &&mem_pool) {
  using eval_type = std::remove_cvref_t<decltype(*storage_left.begin())>;
  using allocator_type = std::remove_cvref_t<allocator_type_>;
  std::vector<eval_type, allocator_type> left_terms{storage_left.size(),
                                                    mem_pool};
  std::ranges::copy(storage_left, left_terms.begin());
  std::vector<eval_type, allocator_type> right_terms{storage_right.size(),
                                                     mem_pool};
  std::ranges::copy(storage_right, right_terms.begin());
  if (left_terms.size() < right_terms.size()) {
    // We want to minimize the number of lists to merge at the end since merging
    // has a high constant cost
    std::swap(left_terms, right_terms);
  }
  const auto output_size = 2 * left_terms.size();
  // Multiply all left_terms by all right_terms
  for (const auto [i, v] : right_terms | std::views::enumerate) {
    const auto output = result.subspan(i * output_size, output_size);
    sparse_mult_merge_term(left_terms, v, output);
  }
  // We have |right_terms| strongly non-overlapping lists, we
  // need to merge them efficiently
  // Most efficient is to merge them in pairs of increasing size, giving
  // O(n log(n)) runtime, versus O(n^2)

  for (std::size_t merge_level = 1; 2 * merge_level <= right_terms.size();
       merge_level *= 2) {
    const std::size_t merge_size = merge_level * output_size;
    std::size_t start = 0;
    for (; start + 2 * merge_size <= result.size(); start += 2 * merge_size) {
      auto merge_span = result.subspan(start, 2 * merge_size);
      merge_sum_linear(merge_span, merge_span.begin() + merge_size);
    }
    if (start < result.size()) {
      if (start > 0) {
        const std::size_t prev_start = start - 2 * merge_size;
        const std::size_t span_len = result.size() - prev_start;
        auto merge_span = result.subspan(prev_start, span_len);
        merge_sum_linear(merge_span, merge_span.begin() + 2 * merge_size);
      } else {
        merge_sum_linear(result, result.begin() + 2 * merge_size);
      }
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
constexpr std::pair<eval_type, eval_type> exact_mult(const eval_type &lhs,
                                                     const eval_type &rhs) {
  eval_type big = lhs * rhs;
  return {big, mul_sub(lhs, rhs, big)};
}

} // namespace _impl

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_IMPL_HPP

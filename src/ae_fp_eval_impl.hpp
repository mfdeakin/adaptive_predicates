
#ifndef ADAPTIVE_PREDICATES_AE_FP_EVAL_IMPL_HPP
#define ADAPTIVE_PREDICATES_AE_FP_EVAL_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>
#include <span>

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"

#include <fmt/ranges.h>

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
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint)
    -> std::pair<typename std::remove_cvref_t<decltype(storage)>::value_type,
                 typename std::remove_cvref_t<decltype(storage.end())>>;
auto merge_sum_linear_fast(
    std::ranges::range auto &&storage,
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type;
constexpr auto merge_sum_quadratic(std::ranges::range auto &&storage)
    -> std::pair<typename std::remove_cvref_t<decltype(storage)>::value_type,
                 std::remove_cvref_t<decltype(storage.end())>>;
constexpr auto merge_sum_quadratic_keep_zeros(std::ranges::range auto &&storage)
    -> std::pair<typename std::remove_cvref_t<decltype(storage)>::value_type,
                 std::remove_cvref_t<decltype(storage.end())>>;

constexpr auto merge_sum(std::ranges::range auto storage)
    -> std::pair<typename std::remove_cvref_t<decltype(storage)>::value_type,
                 std::remove_cvref_t<decltype(storage.end())>> {
  if constexpr (vector_type<typename decltype(storage)::value_type>) {
    return merge_sum_quadratic_keep_zeros(storage);
  } else {
    return merge_sum_quadratic(storage);
  }
}

template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_m>
constexpr auto sparse_mult(span_l storage_left, span_r storage_right,
                           span_m storage_mult)
    -> std::remove_cvref_t<decltype(storage_mult.end())>;

template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_result, typename allocator_type_>
constexpr auto sparse_mult_merge(span_l left_terms, span_r right_terms,
                                 span_result result, allocator_type_ &&mem_pool)
    -> decltype(result.end());

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
      sparse_mult(storage_left, storage_right, partial_results);
    }
  } else if constexpr (!std::is_same_v<additive_id, E>) {
    partial_results[0] = eval_type(e);
  }
}

// Linear merge sum which requires the inputs be strongly non-overlapping
auto merge_sum_linear_fast(
    std::ranges::range auto &&storage,
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint) ->
    typename std::remove_cvref_t<decltype(storage)>::value_type {
  using eval_type = typename std::remove_cvref_t<decltype(storage)>::value_type;
  if (storage.size() > 1) {
    std::ranges::inplace_merge(
        storage, midpoint, [](const eval_type &left, const eval_type &right) {
          // Zero pruning ensures all of the zeros are at the ends of left and
          // right, so we need to ensure that zero is considered greater than
          // any non-zero number
          // This algorithm technically works regardless of where the zeros are,
          // but ensuring they remain at the end allows us to reduce the number
          // of computations we have to perform
          if (left == eval_type{0}) {
            return false;
          } else if (right == eval_type{0}) {
            return true;
          } else {
            return abs(left) < abs(right);
          }
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

// Linear merge sum without the strongly non-overlapping requirement
auto merge_sum_linear(
    std::ranges::range auto &&storage,
    const typename std::remove_cvref_t<decltype(storage)>::iterator midpoint)
    -> std::pair<typename std::remove_cvref_t<decltype(storage)>::value_type,
                 typename std::remove_cvref_t<decltype(storage.end())>> {
  using eval_type = typename std::remove_cvref_t<decltype(storage)>::value_type;
  if (storage.size() > 1) {
    std::ranges::inplace_merge(
        storage, midpoint, [](const eval_type &left, const eval_type &right) {
          // Zero pruning ensures all of the zeros are at the ends of left and
          // right, so we need to ensure that zero is considered greater than
          // any non-zero number
          // This algorithm technically works regardless of where the zeros are,
          // but ensuring they remain at the end allows us to reduce the number
          // of computations we have to perform
          if (left == eval_type{0}) {
            return false;
          } else if (right == eval_type{0}) {
            return true;
          } else {
            return abs(left) < abs(right);
          }
        });
    // Not all of the code puts filtered zeros at the end yet, so we have to use
    // an expensive filter
    auto nonzero = storage | std::views::filter(is_nonzero<eval_type>);
    std::size_t nonzero_size = 0;
    for (auto i = nonzero.begin(); i != nonzero.end(); ++i) {
      ++nonzero_size;
    }
    if (nonzero_size > 1) {
      auto &a = *nonzero.begin();
      auto &b = *(++nonzero.begin());
      auto [Q, q] = dekker_sum_unchecked(b, a);
      // Because we aren't overwriting the entire array, we need to clear
      // the values that we've processed
      a = 0;
      b = 0;
      auto out = storage.begin();
      for (auto &h : nonzero | std::views::drop(1)) {
        auto [R, g] = dekker_sum_unchecked(h, q);
        h = eval_type{0};
        if (g != eval_type{0}) {
          *out = g;
          ++out;
        }
        std::tie(Q, q) = two_sum(Q, R);
      }

      if (q != eval_type{0}) {
        *out = q;
        ++out;
      }
      if (Q != eval_type{0}) {
        *out = Q;
        ++out;
      }
      return std::pair{Q, out};
    } else if (nonzero_size == 1) {
      eval_type &v = *nonzero.begin();
      eval_type tmp = v;
      v = 0;
      storage[0] = tmp;
      return std::pair{tmp, storage.begin() + 1};
    } else {
      return std::pair{0, storage.begin()};
    }
  } else if (storage.size() == 1) {
    return std::pair{storage[0], storage.begin() + 1};
  } else {
    return std::pair{0.0, storage.begin()};
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

constexpr auto merge_sum_quadratic(std::ranges::range auto &&storage)
    -> std::pair<typename std::remove_cvref_t<decltype(storage)>::value_type,
                 std::remove_cvref_t<decltype(storage.end())>> {
  using eval_type = typename std::remove_cvref_t<decltype(storage)>::value_type;
  if (storage.size() > 1) {
    auto out = storage.begin();
    for (eval_type &inp : storage | std::views::filter(is_nonzero<eval_type>)) {
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
      return {eval_type{0}, out};
    } else {
      return {*(out - 1), out};
    }
  } else if (storage.size() == 1) {
    return {storage[0], storage.end()};
  } else {
    return {eval_type{0.0}, storage.end()};
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
    -> std::pair<typename std::remove_cvref_t<decltype(storage)>::value_type,
                 std::remove_cvref_t<decltype(storage.end())>> {
  using eval_type = typename std::remove_cvref_t<decltype(storage)>::value_type;
  if (storage.size() > 1) {
    for (auto inp = storage.begin(); inp != storage.end(); ++inp) {
      *inp = merge_sum_append_keep_zeros(storage.begin(), inp);
    }
    return {std::reduce(storage.begin(), storage.end()), storage.end()};
  } else if (storage.size() == 1) {
    return {storage[0], storage.end()};
  } else {
    return {eval_type{0.0}, storage.end()};
  }
}

template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_m>
constexpr auto sparse_mult(span_l storage_left, span_r storage_right,
                           span_m storage_mult)
    -> std::remove_cvref_t<decltype(storage_mult.end())> {
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
      out_i = zero_prune_store_dec(upper, out_i);
      out_i = zero_prune_store_dec(lower, out_i);
    }
  }
  if (out_i >= storage_mult.begin()) {
    // We want zeros at the end, so move the nonzero values to the beginning and
    // overwrite the end
    ++out_i;
    std::copy(out_i, storage_mult.end(), storage_mult.begin());
    const std::size_t num_terms =
        storage_mult.size() - (out_i - storage_mult.begin());
    out_i = storage_mult.begin() + num_terms;
    std::fill(out_i, storage_mult.end(), typename span_m::value_type{0});
    return out_i;
  } else {
    return storage_mult.end();
  }
}

template <std::ranges::range span_l, typename eval_type,
          std::ranges::range span_result>
constexpr auto sparse_mult_merge_term(const span_l storage_left,
                                      const eval_type v, span_result result)
    -> decltype(result.begin()) {
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
  auto out = result.begin();
  eval_type q0;
  std::tie(q0, *out) = exact_mult(storage_left[0], v);
  if (*out != eval_type{0}) {
    ++out;
  }
  eval_type q1{0};
  for (std::size_t i = 1; i < storage_left.size(); ++i) {
    const auto [T, t] = exact_mult(storage_left[i], v);
    std::tie(q1, *out) = two_sum(q0, t);
    if (*out != eval_type{0}) {
      ++out;
    }
    std::tie(q0, *out) = two_sum(T, q1);
    if (*out != eval_type{0}) {
      ++out;
    }
  }
  *out = q0;
  if (*out != eval_type{0}) {
    ++out;
  }
  std::fill(out, result.end(), eval_type{0});
  return out;
}

template <std::ranges::range span_l, std::ranges::range span_r,
          std::ranges::range span_result, typename allocator_type_>
constexpr auto sparse_mult_merge(span_l storage_left, span_r storage_right,
                                 span_result result, allocator_type_ &&mem_pool)
    -> decltype(result.end()) {
  using eval_type = std::remove_cvref_t<decltype(*storage_left.begin())>;
  using allocator_type = std::remove_cvref_t<allocator_type_>;
  const auto copy_nonzero = [&mem_pool](const std::ranges::range auto range) {
    auto nonzero_range = range | std::views::filter(is_nonzero<eval_type>);
    const std::size_t size =
        std::distance(nonzero_range.begin(), nonzero_range.end());
    std::vector<eval_type, allocator_type> terms{size, mem_pool};
    std::ranges::copy(nonzero_range, terms.begin());
    std::ranges::fill(range, eval_type{0});
    return terms;
  };
  auto left_terms = copy_nonzero(storage_left);
  auto right_terms = copy_nonzero(storage_right);
  if (left_terms.size() < right_terms.size()) {
    // We want to minimize the number of lists to merge at the end since merging
    // has a high constant cost
    std::swap(left_terms, right_terms);
  }
  const auto output_size = 2 * left_terms.size();
  // Multiply all left_terms by all right_terms
  for (std::size_t i = 0; i < right_terms.size(); ++i) {
    const auto output = result.subspan(i * output_size, output_size);
    sparse_mult_merge_term(left_terms, right_terms[i], output);
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
  return result.end();
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

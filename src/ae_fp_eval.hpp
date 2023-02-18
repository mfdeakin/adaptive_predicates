
#ifndef ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP
#define ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

#include <ranges>
#include <span>

#define FP_FAST_FMA
#define FP_FAST_FMAF
#define FP_FAST_FMAL

#include <cmath>

#include "ae_expr.hpp"
#include "ae_expr_utils.hpp"

namespace adaptive_expr {

template <std::floating_point eval_type, typename E_>
  requires expr_type<E_> || arith_number<E_>
constexpr eval_type fp_eval(E_ &&e) noexcept {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr<E>::value) {
    using Op = typename E::Op;
    return Op()(fp_eval<eval_type>(e.lhs()), fp_eval<eval_type>(e.rhs()));
  } else {
    return static_cast<eval_type>(e);
  }
}

// Exact, non-adaptive implementation; incredibly slow and meant for testing; 5
// point in-sphere test takes 2880 fp values to store the result...
template <std::floating_point eval_type, typename E>
  requires expr_type<E> || arith_number<E>
constexpr eval_type exactfp_eval(E &&e) noexcept {
  if constexpr (is_expr<std::remove_reference_t<E>>::value) {
    const std::size_t storage_needed =
        num_partials_for_exact(std::forward<E>(e));
    std::vector<eval_type> partial_results(storage_needed, eval_type(0));
    exactfp_eval_impl<eval_type>(
        std::forward<E>(e),
        std::span{partial_results.begin(), partial_results.end()});
    std::ranges::sort(partial_results);
    return std::accumulate(partial_results.begin(), partial_results.end(),
                           eval_type(0));
  } else {
    return static_cast<eval_type>(e);
  }
}

template <std::floating_point eval_type>
void sparse_mult(std::span<eval_type> storage_left,
                 std::span<eval_type> storage_right,
                 std::span<eval_type> storage_mult);

template <std::floating_point eval_type>
void merge_sum(std::span<eval_type> storage, std::span<eval_type> storage_left,
               std::span<eval_type> storage_right);

template <std::floating_point eval_type, typename E_>
  requires expr_type<E_> || arith_number<E_>
constexpr void
exactfp_eval_impl(E_ &&e, std::span<eval_type> partial_results) noexcept {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr<E>::value) {
    const std::size_t reserve_left = num_partials_for_exact(e.lhs());
    const auto storage_left = partial_results.first(reserve_left);
    exactfp_eval_impl<eval_type>(e.lhs(), storage_left);
    const std::size_t reserve_right = num_partials_for_exact(e.rhs());
    const auto storage_right =
        partial_results.subspan(reserve_left, reserve_right);
    exactfp_eval_impl<eval_type>(e.rhs(), storage_right);
    using Op = typename E::Op;
    if constexpr (std::is_same_v<std::plus<>, Op> ||
                  std::is_same_v<std::minus<>, Op>) {
      if constexpr (std::is_same_v<std::minus<>, Op>) {
        for (eval_type &v : storage_right) {
          v = -v;
        }
      }
      merge_sum(partial_results, storage_left, storage_right);
    } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
      const auto storage_mult = partial_results.last(
          partial_results.size() - reserve_left - reserve_right);
      sparse_mult(storage_left, storage_right, storage_mult);
    }
  } else if constexpr (!std::is_same_v<additive_id, E>) {
    *partial_results.begin() = eval_type(e);
  }
}

template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_unchecked(const eval_type &lhs, const eval_type &rhs);

template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type> knuth_sum(const eval_type &lhs,
                                                    const eval_type &rhs);

template <std::floating_point eval_type>
void merge_sum(std::span<eval_type> storage, std::span<eval_type> storage_left,
               std::span<eval_type> storage_right) {
  if (storage.size() > 1) {
    // std::ranges::inplace_merge(
    //         storage_left.begin(), storage_right.begin(),
    //         storage_right.end(),
    //         [](eval_type l, eval_type r) { return std::abs(l) <
    //         std::abs(r); });
    std::ranges::sort(storage, [](eval_type l, eval_type r) {
      return std::abs(l) < std::abs(r);
    });
    auto [Q, q] = dekker_sum_unchecked(storage[0], storage[1]);
    for (auto [g, t] :
         std::ranges::views::zip(storage | std::views::drop(2), storage)) {
      const auto [R, h] = dekker_sum_unchecked(g, q);
      t = h;
      std::tie(Q, q) = knuth_sum(Q, R);
    }
    *(storage.end() - 2) = q;
    *(storage.end() - 1) = Q;
  }
}

template <typename eval_type, typename I>
void overflow_write(const eval_type &v, I &itr1, const I &itr1_end, I &itr2);

template <std::floating_point eval_type>
std::pair<eval_type, eval_type> exact_mult(const eval_type &lhs,
                                           const eval_type &rhs);
template <std::floating_point eval_type>
void sparse_mult(std::span<eval_type> storage_left,
                 std::span<eval_type> storage_right,
                 std::span<eval_type> storage_mult) {
  auto unwritten_left = storage_left.begin();
  auto unwritten_right = storage_right.begin();
  auto unwritten_mult = storage_mult.begin();
  // This performs multiplication in-place
  // As soon as a value is no longer needed; it is overwritten
  //
  // Since the value in the outer loop is immediately copied and never returned
  // to, it is overwritten immediately
  //
  // The value in the inner loop can be shown to only be overwritten after it
  // was copied during the last iteration of the outer loop if storage_mult is
  // sized correctly (ie, as specified by num_partials_for_exact)
  for (auto l : storage_left) {
    bool overwrote_l = false;
    for (auto r : storage_right) {
      auto m = exact_mult(l, r);
      if (!overwrote_l) {
        *unwritten_left = m.first;
        unwritten_left++;
        overflow_write(m.second, unwritten_mult, storage_mult.end(),
                       unwritten_right);
        overwrote_l = true;
      } else {
        overflow_write(m.first, unwritten_mult, storage_mult.end(),
                       unwritten_right);
        overflow_write(m.second, unwritten_mult, storage_mult.end(),
                       unwritten_right);
      }
    }
  }
}

template <typename eval_type, typename I>
void overflow_write(const eval_type &v, I &itr1, const I &itr1_end, I &itr2) {
  // This writes v to either itr1 or failing that, itr2, so there must be enough
  // storage in one of them. Whichever it writes to, the iterator is incremented
  if (itr1 != itr1_end) {
    *itr1 = v;
    ++itr1;
  } else {
    *itr2 = v;
    ++itr2;
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

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

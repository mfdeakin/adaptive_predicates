
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

template <std::floating_point eval_type>
eval_type merge_sum1(std::span<eval_type> storage);
template <std::floating_point eval_type>
eval_type merge_sum2(std::span<eval_type> storage);
template <std::floating_point eval_type>
eval_type merge_sum3(std::span<eval_type> storage);
template <std::floating_point eval_type>
eval_type merge_sum4(std::span<eval_type> storage);
template <std::floating_point eval_type>
eval_type merge_sum5(std::span<eval_type> storage);

template <std::floating_point eval_type>
eval_type merge_sum(std::span<eval_type> storage) {
  return merge_sum5(storage);
}

template <std::floating_point eval_type, typename E>
  requires expr_type<E> || arith_number<E>
consteval eval_type max_rel_error() {
  if constexpr (is_expr<E>::value) {
    using Op = typename E::Op;
    const eval_type max_left = max_rel_error<eval_type, typename E::LHS>();
    const eval_type max_right = max_rel_error<eval_type, typename E::RHS>();
    const eval_type eps = std::numeric_limits<eval_type>::epsilon();
    if constexpr (std::is_same_v<std::plus<>, Op> ||
                  std::is_same_v<std::minus<>, Op>) {
      // For plus and minus, this only applies to their adaptive implementation;
      // using the basic floating point plus and minus of course has unbounded
      // relative error
      if(max_left == 0 && max_right == 0) {
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

// Exact, non-adaptive implementation; incredibly slow and meant for testing; 5
// point in-sphere test takes 2880 fp values to store the result...
template <std::floating_point eval_type, typename E>
  requires expr_type<E> || arith_number<E>
constexpr eval_type exactfp_eval(E &&e) noexcept {
  if constexpr (is_expr<std::remove_reference_t<E>>::value) {
    auto partial_results = []() {
      constexpr std::size_t max_stack_storage = 1024 / sizeof(eval_type);
      constexpr std::size_t storage_needed = num_partials_for_exact<E>();
      if constexpr (storage_needed > max_stack_storage) {
        return std::vector<eval_type>(storage_needed);
      } else {
        auto arr = std::array<eval_type, storage_needed>{};
        return arr;
      }
    }();
    exactfp_eval_impl<eval_type>(
        std::forward<E>(e),
        std::span{partial_results.begin(), partial_results.end()});
    return merge_sum(std::span<eval_type>{partial_results});
    // return std::accumulate(partial_results.begin(), partial_results.end(),
    // eval_type(0));
  } else {
    return static_cast<eval_type>(e);
  }
}

template <std::floating_point eval_type>
void sparse_mult(std::span<eval_type> storage_left,
                 std::span<eval_type> storage_right,
                 std::span<eval_type> storage_mult);

template <std::floating_point eval_type, typename E_>
  requires expr_type<E_> || arith_number<E_>
constexpr void
exactfp_eval_impl(E_ &&e, std::span<eval_type> partial_results) noexcept {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr<E>::value) {
    const std::size_t reserve_left = num_partials_for_exact<typename E::LHS>();
    const auto storage_left = partial_results.first(reserve_left);
    exactfp_eval_impl<eval_type>(e.lhs(), storage_left);
    const std::size_t reserve_right = num_partials_for_exact<typename E::RHS>();
    const auto storage_right =
        partial_results.subspan(reserve_left, reserve_right);
    exactfp_eval_impl<eval_type>(e.rhs(), storage_right);
    using Op = typename E::Op;
    if constexpr (std::is_same_v<std::minus<>, Op>) {
      for (eval_type &v : storage_right) {
        v = -v;
      }
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
constexpr std::pair<eval_type, eval_type> dekker_sum(const eval_type &lhs,
                                                     const eval_type &rhs);

// Requires |lhs| >= |rhs|
template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type>
dekker_sum_unchecked(const eval_type &lhs, const eval_type &rhs);

template <std::floating_point eval_type>
constexpr std::pair<eval_type, eval_type> knuth_sum(const eval_type &lhs,
                                                    const eval_type &rhs);

template <std::floating_point eval_type>
eval_type merge_sum1(std::span<eval_type> storage) {
  if (storage.size() > 1) {
    auto [Q, _] = dekker_sum(storage[0], storage[1]);
    for (auto g : storage | std::views::drop(2)) {
      std::tie(Q, _) = dekker_sum(g, Q);
    }
    return Q;
  } else if (storage.size() == 1) {
    return storage[0];
  } else {
    return 0.0;
  }
}

template <std::floating_point eval_type>
eval_type merge_sum2(std::span<eval_type> storage) {
  if (storage.size() > 1) {
    auto [Q, _] = dekker_sum(storage[0], storage[1]);
    for (auto g : storage | std::views::drop(2)) {
      std::tie(Q, _) = knuth_sum(g, Q);
    }
    return Q;
  } else if (storage.size() == 1) {
    return storage[0];
  } else {
    return 0.0;
  }
}

template <std::floating_point eval_type>
eval_type merge_sum3(std::span<eval_type> storage) {
  if (storage.size() > 1) {
    std::ranges::sort(storage, [](eval_type l, eval_type r) {
      return std::abs(l) < std::abs(r);
    });
    auto [Q, _] = dekker_sum_unchecked(storage[0], storage[1]);
    for (auto g : storage | std::views::drop(2)) {
      std::tie(Q, _) = dekker_sum_unchecked(g, Q);
    }
    return Q;
  } else if (storage.size() == 1) {
    return storage[0];
  } else {
    return 0.0;
  }
}

template <std::floating_point eval_type>
eval_type merge_sum4(std::span<eval_type> storage) {
  if (storage.size() > 1) {
    std::ranges::sort(storage, [](eval_type l, eval_type r) {
      return std::abs(l) > std::abs(r);
    });
    auto [Q, q] = dekker_sum_unchecked(storage[1], storage[0]);
    for (auto g : storage | std::views::drop(2)) {
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

template <std::floating_point eval_type>
eval_type merge_sum5_append(std::span<eval_type> storage, eval_type v) {
  for (auto &e : storage) {
    std::tie(v, e) = knuth_sum(v, e);
  }
  return v;
}

template <std::floating_point eval_type>
eval_type merge_sum5(std::span<eval_type> storage) {
  if (storage.size() > 1) {
    for (std::size_t i = 1, j = 1; i < storage.size(); ++i) {
      storage[i] = merge_sum5_append(storage.first(j), storage[i]);
      if (storage[i] != eval_type(0)) {
        if (i != j) {
          storage[j] = storage[i];
          storage[i] = eval_type(0);
        }
        ++j;
      }
    }
    return *(storage.end() - 1);
  } else if (storage.size() == 1) {
    return storage[0];
  } else {
    return 0.0;
  }
}

template <std::floating_point eval_type>
std::pair<eval_type, eval_type> exact_mult(const eval_type &lhs,
                                           const eval_type &rhs);
template <std::floating_point eval_type>
void sparse_mult(std::span<eval_type> storage_left,
                 std::span<eval_type> storage_right,
                 std::span<eval_type> storage_mult) {
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
  for (auto l : storage_right | std::views::reverse) {
    for (auto r : storage_left | std::views::reverse) {
      auto [upper, lower] = exact_mult(l, r);
      *out_i = upper;
      --out_i;
      *out_i = lower;
      --out_i;
    }
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

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_FP_EVAL_HPP

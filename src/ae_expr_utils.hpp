
#ifndef ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP
#define ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP

#include "ae_expr.hpp"

namespace adaptive_expr {

template <typename Op>
concept comparison_op = std::is_same_v<std::less<>, Op> ||
                        std::is_same_v<std::greater_equal<>, Op> ||
                        std::is_same_v<std::greater<>, Op> ||
                        std::is_same_v<std::less_equal<>, Op> ||
                        std::is_same_v<std::equal_to<>, Op>;

template <typename E>
concept predicate = expr_type<E> && comparison_op<typename E::Op>;

template <typename E> constexpr std::size_t num_ops(const E &&e) {
  if constexpr (is_expr<E>::value) {
    return num_ops(e.lhs()) + num_ops(e.rhs()) + 1;
  } else {
    return 0;
  }
}

template <typename E> constexpr std::size_t num_values(const E &&e) {
  if constexpr (is_expr<E>::value) {
    return num_ops(e.lhs()) + num_ops(e.rhs());
  } else {
    return 1;
  }
}

template <typename E_> consteval std::size_t num_partials_for_exact() {
  using E = std::remove_cvref_t<E_>;
  if constexpr (is_expr<E>::value) {
    using Op = typename E::Op;
    if constexpr (std::is_same_v<std::plus<>, Op> ||
                  std::is_same_v<std::minus<>, Op>) {
      return num_partials_for_exact<typename E::LHS>() +
             num_partials_for_exact<typename E::RHS>();
    } else if constexpr (std::is_same_v<std::multiplies<>, Op>) {
      auto num_left = num_partials_for_exact<typename E::LHS>();
      auto num_right = num_partials_for_exact<typename E::RHS>();
      return 2 * num_left * num_right;
    } else {
      // always triggers a static assert in a way that doesn't trip up the
      // compiler
      static_assert(!std::is_same_v<std::divides<>, Op>,
                    "No limit on number of partials for division!");
      static_assert(std::is_same_v<std::plus<>, Op> ||
                        std::is_same_v<std::multiplies<>, Op>,
                    "Unhandled operation!");
      return 0;
    }
  } else if constexpr (std::is_same_v<additive_id, E>) {
    return 0;
  } else {
    return 1;
  }
}

} // namespace adaptive_expr

template <typename Op, typename LHS, typename RHS>
struct fmt::formatter<adaptive_expr::arith_expr<Op, LHS, RHS>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const adaptive_expr::arith_expr<Op, LHS, RHS> &e, FormatContext &ctx) const {
    if constexpr (std::floating_point<LHS> && std::floating_point<RHS>) {
      return format_to(ctx.out(), "({: .17f} {} {: .17f})", e.lhs(), Op(), e.rhs());
    } else if constexpr (std::floating_point<LHS>) {
      return format_to(ctx.out(), "({: .17f} {} {})", e.lhs(), Op(), e.rhs());
    } else if constexpr (std::floating_point<RHS>) {
      return format_to(ctx.out(), "({} {} {: .17f})", e.lhs(), Op(), e.rhs());
    } else {
      return format_to(ctx.out(), "({} {} {})", e.lhs(), Op(), e.rhs());
    }
  }
};

template <>
struct fmt::formatter<adaptive_expr::additive_id> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const adaptive_expr::additive_id &, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "0");
  }
};

template <>
struct fmt::formatter<std::plus<>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::plus<> &, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "+");
  }
};

template <>
struct fmt::formatter<std::minus<>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::minus<> &, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "-");
  }
};

template <>
struct fmt::formatter<std::multiplies<>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::multiplies<> &, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "*");
  }
};

template <>
struct fmt::formatter<std::divides<>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::divides<> &, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "/");
  }
};

#endif // ADAPTIVE_PREDICATES_AE_EXPR_UTILS_HPP

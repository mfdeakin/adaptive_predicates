
#ifndef ADAPTIVE_PREDICATES_AE_EXPR_PRINT_HPP
#define ADAPTIVE_PREDICATES_AE_EXPR_PRINT_HPP

#include "ae_expr.hpp"

// To verify when I have an implementation of print
#include <print>

template <typename Op, typename LHS, typename RHS>
struct std::formatter<adaptive_expr::arith_expr<Op, LHS, RHS>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const adaptive_expr::arith_expr<Op, LHS, RHS> &e, FormatContext &ctx) const {
    if constexpr (std::floating_point<LHS> && std::floating_point<RHS>) {
      return std::format_to(ctx.out(), "({: .67f} {} {: .67f})", e.lhs(), Op(), e.rhs());
    } else if constexpr (std::floating_point<LHS>) {
      return std::format_to(ctx.out(), "({: .67f} {} {})", e.lhs(), Op(), e.rhs());
    } else if constexpr (std::floating_point<RHS>) {
      return std::format_to(ctx.out(), "({} {} {: .67f})", e.lhs(), Op(), e.rhs());
    } else {
      return std::format_to(ctx.out(), "({} {} {})", e.lhs(), Op(), e.rhs());
    }
  }
};

template <>
struct std::formatter<adaptive_expr::additive_id> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const adaptive_expr::additive_id &, FormatContext &ctx) const {
    return std::format_to(ctx.out(), "0");
  }
};

template <>
struct std::formatter<std::plus<>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::plus<> &, FormatContext &ctx) const {
    return std::format_to(ctx.out(), "+");
  }
};

template <>
struct std::formatter<std::minus<>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::minus<> &, FormatContext &ctx) const {
    return std::format_to(ctx.out(), "-");
  }
};

template <>
struct std::formatter<std::multiplies<>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::multiplies<> &, FormatContext &ctx) const {
    return std::format_to(ctx.out(), "*");
  }
};

template <>
struct std::formatter<std::divides<>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const std::divides<> &, FormatContext &ctx) const {
    return std::format_to(ctx.out(), "/");
  }
};

#endif // ADAPTIVE_PREDICATES_AE_EXPR_PRINT_HPP

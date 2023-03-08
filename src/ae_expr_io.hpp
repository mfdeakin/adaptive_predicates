
#ifndef ADAPTIVE_PREDICATES_AE_EXPR_IO_HPP
#define ADAPTIVE_PREDICATES_AE_EXPR_IO_HPP

#include "ae_expr.hpp"

namespace adaptive_expr {

//
// iostream operators
//

std::ostream &operator<<(std::ostream &os, const additive_id &e) {
  return (os << "0");
}

template <typename Op, typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os, const arith_expr<Op, LHS, RHS> &e);

template <typename RHS>
std::ostream &operator<<(std::ostream &os,
                         const arith_expr<std::minus<>, additive_id, RHS> &e) {
  return os << "-" << e.rhs();
}

template <typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os,
                         const arith_expr<std::plus<>, LHS, RHS> &e) {
  return os << "(" << e.lhs() << " + " << e.rhs() << ")";
}

template <typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os,
                         const arith_expr<std::minus<>, LHS, RHS> &e) {
  return os << "(" << e.lhs() << " - " << e.rhs() << ")";
}

template <typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os,
                         const arith_expr<std::multiplies<>, LHS, RHS> &e) {
  return os << "(" << e.lhs() << " * " << e.rhs() << ")";
}

template <typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os,
                         const arith_expr<std::divides<>, LHS, RHS> &e) {
  return os << "(" << e.lhs() << " / " << e.rhs() << ")";
}

template <typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os,
                         const arith_expr<std::less<>, LHS, RHS> &e) {
  return os << "(" << e.lhs() << " < " << e.rhs() << ")";
}

template <typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os,
                         const arith_expr<std::less_equal<>, LHS, RHS> &e) {
  return os << "(" << e.lhs() << " <= " << e.rhs() << ")";
}

template <typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os,
                         const arith_expr<std::equal_to<>, LHS, RHS> &e) {
  return os << "(" << e.lhs() << " == " << e.rhs() << ")";
}

}

//
// libfmt operators
//
#include <fmt/core.h>

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

#endif // ADAPTIVE_PREDICATES_AE_EXPR_IO_HPP

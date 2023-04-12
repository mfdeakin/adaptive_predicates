
#ifndef TESTING_UTILS_HPP
#define TESTING_UTILS_HPP

#include <cmath>

#include "ae_expr.hpp"

template <adaptive_expr::arith_number eval_type>
constexpr bool check_sign(eval_type correct, eval_type check) {
  if (correct == eval_type{0.0}) {
    return check == eval_type{0.0};
  } else if (check == eval_type{0.0}) {
    return false;
  } else {
    return std::signbit(correct) == std::signbit(check);
  }
}

template <adaptive_expr::arith_number eval_type>
constexpr auto
build_orient2d_case(const std::array<std::array<eval_type, 2>, 3> &points) {
  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  const auto cross_expr = [](const std::array<eval_type, 2> &lhs,
                             const std::array<eval_type, 2> &rhs) constexpr {
    return adaptive_expr::mult_expr(lhs[x], rhs[y]) - adaptive_expr::mult_expr(lhs[y], rhs[x]);
  };
  return cross_expr(points[1], points[2]) - cross_expr(points[0], points[2]) +
         cross_expr(points[0], points[1]);
}

#endif // TESTING_UTILS_HPP


#ifndef ADAPTIVE_PREDICATES_AE_EXPR_IOSTREAM_HPP
#define ADAPTIVE_PREDICATES_AE_EXPR_IOSTREAM_HPP

#include "ae_expr.hpp"

#include <iostream>

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

} // adaptive_predicates

#endif // ADAPTIVE_PREDICATES_AE_EXPR_IOSTREAM_HPP

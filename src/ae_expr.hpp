
#ifndef ADAPTIVE_PREDICATES_AE_EXPR_HPP
#define ADAPTIVE_PREDICATES_AE_EXPR_HPP

#include <compare>
#include <functional>
#include <type_traits>

namespace adaptive_expr {

template <typename Op_, typename LHS_, typename RHS_> class arith_expr {
public:
  using LHS = std::remove_cvref_t<LHS_>;
  using RHS = std::remove_cvref_t<RHS_>;
  using Op = std::remove_cvref_t<Op_>;

  arith_expr() : m_lhs(), m_rhs() {}

  arith_expr(const LHS_ &lhs, const RHS_ &rhs) : m_lhs(lhs), m_rhs(rhs) {}

  const auto &lhs() const noexcept { return m_lhs; }
  const auto &rhs() const noexcept { return m_rhs; }

  constexpr auto &operator()() { return *this; }

private:
  LHS m_lhs;
  RHS m_rhs;
};

class additive_id {
public:
  template <typename T>
    requires std::signed_integral<T> || std::floating_point<T>
  operator T() const noexcept {
    return T{0};
  }
};

arith_expr()->arith_expr<std::plus<>, additive_id, additive_id>;

template <typename E> struct is_expr_impl : std::false_type {};

template <typename Op, typename LHS, typename RHS>
struct is_expr_impl<arith_expr<Op, LHS, RHS>> : std::true_type {};

template <typename E> using is_expr = is_expr_impl<std::remove_cvref_t<E>>;

template <typename E>
concept expr_type =
    is_expr<E>::value || std::is_same_v<additive_id, std::remove_cvref_t<E>>;

template <typename T>
concept arith_number = std::signed_integral<std::remove_cvref_t<T>> ||
                       std::floating_point<std::remove_cvref_t<T>>;

template <typename LHS, typename RHS>
concept arith_expr_operands = (expr_type<LHS> &&
                               (expr_type<RHS> || arith_number<RHS>)) ||
                              (expr_type<RHS> && arith_number<LHS>);

template <expr_type E> constexpr auto operator-(const E &expr) {
  return arith_expr<std::minus<>, additive_id, E>(additive_id{}, expr);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator+(const LHS &lhs, const RHS &rhs) {
  return arith_expr<std::plus<>, LHS, RHS>(lhs, rhs);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator-(const LHS &lhs, const RHS &rhs) {
  return arith_expr<std::minus<>, LHS, RHS>(lhs, rhs);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator*(const LHS &lhs, const RHS &rhs) {
  return arith_expr<std::multiplies<>, LHS, RHS>(lhs, rhs);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator/(const LHS &lhs, const RHS &rhs) {
  return arith_expr<std::divides<>, LHS, RHS>(lhs, rhs);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator<(const LHS &lhs, const RHS &rhs) {
  return arith_expr<std::less<>, LHS, RHS>(lhs, rhs);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator>(const LHS &lhs, const RHS &rhs) {
  return rhs < lhs;
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator<=(const LHS &lhs, const RHS &rhs) {
  return arith_expr<std::less_equal<>, LHS, RHS>(lhs, rhs);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator>=(const LHS &lhs, const RHS &rhs) {
  return rhs <= lhs;
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator==(const LHS &lhs, const RHS &rhs) {
  return arith_expr<std::equal_to<>, LHS, RHS>(lhs, rhs);
}

//
// io-operators
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

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_EXPR_HPP

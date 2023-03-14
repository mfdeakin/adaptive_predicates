
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

  constexpr arith_expr() = default;

  constexpr arith_expr(const LHS &lhs, const RHS &rhs)
      : m_lhs(lhs), m_rhs(rhs) {}
  constexpr arith_expr(LHS &&lhs, RHS &&rhs)
      : m_lhs(std::move(lhs)), m_rhs(std::move(rhs)) {}

  constexpr auto &lhs() const noexcept { return m_lhs; }
  constexpr auto &rhs() const noexcept { return m_rhs; }

  constexpr auto &operator()() { return *this; }

private:
  [[no_unique_address]] LHS m_lhs;
  [[no_unique_address]] RHS m_rhs;
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
template <typename E> static constexpr bool is_expr_v = is_expr<E>::value;

template <typename E>
concept expr_type =
    is_expr_v<E> || std::is_same_v<additive_id, std::remove_cvref_t<E>>;

template <typename E>
concept negate_expr_type =
    is_expr_v<E> && std::is_same_v<typename E::Op, std::minus<>> &&
    std::is_same_v<typename E::LHS, additive_id>;

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
constexpr auto operator+(LHS &&lhs, RHS &&rhs) {
  return arith_expr<std::plus<>, std::remove_cvref_t<LHS>,
                    std::remove_cvref_t<RHS>>(std::forward<LHS>(lhs),
                                              std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator-(LHS &&lhs, RHS &&rhs) {
  return arith_expr<std::minus<>, std::remove_cvref_t<LHS>,
                    std::remove_cvref_t<RHS>>(std::forward<LHS>(lhs),
                                              std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator*(LHS &&lhs, RHS &&rhs) {
  return arith_expr<std::multiplies<>, std::remove_cvref_t<LHS>,
                    std::remove_cvref_t<RHS>>(std::forward<LHS>(lhs),
                                              std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator/(LHS &&lhs, RHS &&rhs) {
  return arith_expr<std::divides<>, std::remove_cvref_t<LHS>,
                    std::remove_cvref_t<RHS>>(std::forward<LHS>(lhs),
                                              std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator<(LHS &&lhs, RHS &&rhs) {
  return arith_expr<std::less<>, std::remove_cvref_t<LHS>,
                    std::remove_cvref_t<RHS>>(std::forward<LHS>(lhs),
                                              std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator>(LHS &&lhs, RHS &&rhs) {
  return std::forward<RHS>(rhs) < std::forward<LHS>(lhs);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator<=(LHS &&lhs, RHS &&rhs) {
  return arith_expr<std::less_equal<>, std::remove_cvref_t<LHS>,
                    std::remove_cvref_t<RHS>>(std::forward<LHS>(lhs),
                                              std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator>=(LHS &&lhs, RHS &&rhs) {
  return std::forward<RHS>(rhs) <= std::forward<LHS>(lhs);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator==(LHS &&lhs, RHS &&rhs) {
  return arith_expr<std::equal_to<>, std::remove_cvref_t<LHS>,
                    std::remove_cvref_t<RHS>>(std::forward<LHS>(lhs),
                                              std::forward<RHS>(rhs));
}

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_EXPR_HPP


#ifndef ADAPTIVE_PREDICATES_AE_EXPR_HPP
#define ADAPTIVE_PREDICATES_AE_EXPR_HPP

#include <cmath>
#include <compare>
#include <functional>
#include <type_traits>

namespace adaptive_expr {

template <typename Op_, typename LHS_, typename RHS_> class arith_expr final {
public:
  using LHS = std::remove_cvref_t<LHS_>;
  using RHS = std::remove_cvref_t<RHS_>;
  using Op = std::remove_cvref_t<Op_>;

  constexpr arith_expr() = default;
  constexpr ~arith_expr() = default;

  constexpr arith_expr(const arith_expr &e) = default;
  constexpr arith_expr(arith_expr &&e) = default;
  constexpr arith_expr &operator=(const arith_expr &e) = default;
  constexpr arith_expr &operator=(arith_expr &&e) = default;

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
  template <typename T> operator T() const noexcept { return T{0}; }
};

arith_expr() -> arith_expr<std::plus<>, additive_id, additive_id>;

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
  requires std::floating_point<T> || std::integral<T>
constexpr T mul_sub(const T a, const T b, const T c) {
  return std::fma(a, b, -c);
}

template <std::signed_integral T> constexpr T abs(const T a) { return std::abs(a); }
template <std::floating_point T> constexpr T abs(const T a) { return std::abs(a); }
template <std::unsigned_integral T> constexpr T abs(const T a) { return a; }

template <typename T>
concept arith_number = !expr_type<T> && requires {
  std::remove_cvref_t<T>{std::remove_cvref_t<T>{} + std::remove_cvref_t<T>{}};
  std::remove_cvref_t<T>{std::remove_cvref_t<T>{} - std::remove_cvref_t<T>{}};
  std::remove_cvref_t<T>{std::remove_cvref_t<T>{} * std::remove_cvref_t<T>{}};

  std::remove_cvref_t<T>{} >= std::remove_cvref_t<T>{};

  std::remove_cvref_t<T>{abs(std::remove_cvref_t<T>{})};
  std::remove_cvref_t<T>{mul_sub(std::remove_cvref_t<T>{},
                                 std::remove_cvref_t<T>{},
                                 std::remove_cvref_t<T>{})};
};

template <typename T>
concept not_scalar_type_ = requires { std::remove_cvref_t<T>{}[0]; };

template <typename T>
concept scalar_type = arith_number<T> && (!not_scalar_type_<T>);

template <typename T>
concept vector_type = arith_number<T> && requires {
  // vectors are indexable and have a 3 parameter select function which chooses
  // elements from the second element when the corresponding element in the
  // first element is true
  std::remove_cvref_t<T>{}[0];
  select(std::remove_cvref_t<T>{} >= std::remove_cvref_t<T>{},
         std::remove_cvref_t<T>{}, std::remove_cvref_t<T>{});
};

template <typename E>
concept evaluatable = expr_type<E> || arith_number<E>;

template <typename LHS, typename RHS>
concept arith_expr_operands =
    (expr_type<LHS> && (expr_type<RHS> || arith_number<RHS>)) ||
    (expr_type<RHS> && arith_number<LHS>);

template <typename Op, typename LHS_, typename RHS_>
  requires arith_expr_operands<LHS_, RHS_> ||
           (arith_number<LHS_> && arith_number<RHS_>)
constexpr auto make_expr(LHS_ &&lhs, RHS_ &&rhs) {
  using LHS = std::remove_cvref_t<LHS_>;
  using RHS = std::remove_cvref_t<RHS_>;
  return arith_expr<Op, LHS, RHS>(std::forward<LHS_>(lhs),
                                  std::forward<RHS_>(rhs));
}

template <typename LHS, typename RHS> constexpr auto plus_expr(LHS &&lhs, RHS &&rhs) {
  return make_expr<std::plus<>>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS> constexpr auto minus_expr(LHS &&lhs, RHS &&rhs) {
  return make_expr<std::minus<>>(std::forward<LHS>(lhs),
                                 std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS> constexpr auto mult_expr(LHS &&lhs, RHS &&rhs) {
  return make_expr<std::multiplies<>>(std::forward<LHS>(lhs),
                                      std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires(!std::is_same_v<RHS, additive_id>)
constexpr auto divide_expr(LHS &&lhs, RHS &&rhs) {
  return make_expr<std::divides<>>(std::forward<LHS>(lhs),
                                   std::forward<RHS>(rhs));
}

template <expr_type E> constexpr auto operator-(const E &expr) {
  return minus_expr(additive_id{}, expr);
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator+(LHS &&lhs, RHS &&rhs) {
  return plus_expr(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator-(LHS &&lhs, RHS &&rhs) {
  return minus_expr(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator*(LHS &&lhs, RHS &&rhs) {
  return mult_expr(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
}

template <typename LHS, typename RHS>
  requires arith_expr_operands<LHS, RHS>
constexpr auto operator/(LHS &&lhs, RHS &&rhs) {
  return divide_expr(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
}

} // namespace adaptive_expr

#endif // ADAPTIVE_PREDICATES_AE_EXPR_HPP

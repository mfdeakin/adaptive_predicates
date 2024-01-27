
#ifndef GENERATE_CASES_HPP
#define GENERATE_CASES_HPP

#include <array>
#include <random>

std::array<std::array<double, 2>, 3>
generate_orient2d_case(std::mt19937_64 &rng);
std::array<std::array<double, 3>, 4>
generate_orient3d_case(std::mt19937_64 &rng);
std::array<std::array<double, 2>, 4>
generate_incircle2d_case(std::mt19937_64 &rng);
std::array<std::array<double, 3>, 5>
generate_incircle3d_case(std::mt19937_64 &rng);

// Returns true if the signs match or both are zero
template <adaptive_expr::arith_number eval_type>
static constexpr bool check_sign(eval_type correct, eval_type check) {
  if (correct == eval_type{0.0}) {
    return check == eval_type{0.0};
  } else if (check == eval_type{0.0}) {
    return false;
  } else {
    return std::signbit(correct) == std::signbit(check);
  }
}

#endif // GENERATE_CASES_HPP

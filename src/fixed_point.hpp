
#ifndef FIXED_POINT_HPP
#define FIXED_POINT_HPP

#include <concepts>
#include <type_traits>

namespace FixedPoint {

template <typename T>
concept arithmetic_prim = std::is_arithmetic_v<T>;
template <arithmetic_prim orig_mantissa> struct upsize {};
template <arithmetic_prim orig_mantissa> struct downsize {};

template <> struct upsize<char> {
  using type = short;
};

template <> struct upsize<short> {
  using type = int;
};

template <> struct upsize<int> {
  using type = long;
};

template <> struct upsize<long> {
  using type = long long;
};

template <> struct upsize<unsigned char> {
  using type = unsigned short;
};

template <> struct upsize<unsigned short> {
  using type = unsigned int;
};

template <> struct upsize<unsigned int> {
  using type = unsigned long;
};

template <> struct upsize<unsigned long> {
  using type = unsigned long long;
};

template <> struct upsize<float> {
  using type = double;
};

template <> struct upsize<double> {
  using type = long double;
};

template <> struct downsize<short> {
  using type = char;
};

template <> struct downsize<int> {
  using type = short;
};

template <> struct downsize<long> {
  using type = int;
};

template <> struct downsize<long long> {
  using type = long;
};

template <> struct downsize<unsigned short> {
  using type = unsigned char;
};

template <> struct downsize<unsigned int> {
  using type = unsigned short;
};

template <> struct downsize<unsigned long> {
  using type = unsigned int;
};

template <> struct downsize<unsigned long long> {
  using type = unsigned long;
};

template <> struct downsize<double> {
  using type = float;
};

template <> struct downsize<long double> {
  using type = double;
};

template <int exponent, arithmetic_prim mantissa_t> class fixed_point;

template <typename T> struct is_fixed_point : public std::false_type {};
template <int exponent, typename mantissa_t>
struct is_fixed_point<fixed_point<exponent, mantissa_t>>
    : public std::true_type {};

template <typename T>
constexpr bool is_fixed_point_v = is_fixed_point<T>::value;

template <typename T> class numeric_limits {
public:
  static constexpr bool is_specialized = false;
};

template <int exponent, typename mantissa_t_>
class numeric_limits<fixed_point<exponent, mantissa_t_>> {
public:
  using fixed_pt = fixed_point<exponent, mantissa_t>;
  using mantissa_t = std::remove_cvref_t<mantissa_t_>;

  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = std::numeric_limits<mantissa_t>::is_signed;
  static constexpr bool is_integer =
      std::numeric_limits<mantissa_t>::is_integer;
  static constexpr bool is_exact = std::numeric_limits<mantissa_t>::is_exact;
  static constexpr bool has_infinity =
      std::numeric_limits<mantissa_t>::has_infinity;
  static constexpr bool has_quiet_NaN =
      std::numeric_limits<mantissa_t>::has_quiet_NaN;
  static constexpr bool has_signaling_NaN =
      std::numeric_limits<mantissa_t>::has_signaling_NaN;
  static constexpr bool has_denorm =
      std::numeric_limits<mantissa_t>::has_denorm;
  static constexpr bool has_denorm_loss =
      std::numeric_limits<mantissa_t>::has_denorm_loss;
  static constexpr std::float_round_style round_style =
      std::numeric_limits<mantissa_t>::round_style;
  static constexpr bool is_iec559 = std::numeric_limits<mantissa_t>::is_iec559;
  static constexpr bool is_bounded =
      std::numeric_limits<mantissa_t>::is_bounded;
  static constexpr bool is_modulo = std::numeric_limits<mantissa_t>::is_modulo;
  static constexpr int digits = std::numeric_limits<mantissa_t>::digits;
  static constexpr int digits10 = std::numeric_limits<mantissa_t>::digits10;
  static constexpr int max_digits10 =
      std::numeric_limits<mantissa_t>::max_digits10;
  static constexpr int radix = std::numeric_limits<mantissa_t>::radix;
  static constexpr int min_exponent = exponent;
  static constexpr int min_exponent10 = int(log10(2) * min_exponent);
  static constexpr int max_exponent = exponent;
  static constexpr int max_exponent10 = int(log10(2) * max_exponent);
  static constexpr bool traps = std::numeric_limits<mantissa_t>::traps;
  static constexpr bool tinyness_before =
      std::numeric_limits<mantissa_t>::tinyness_before;

  // returns the smallest finite value of the given type
  static constexpr fixed_pt min() { return fixed::min_value(); }
  // returns the lowest finite value of the given type
  static constexpr fixed_pt lowest() { return fixed_pt::lowest_value(); }
  // returns the largest finite value of the given type
  static constexpr fixed_pt max() { return fixed_pt::max_positive_value(); }
  // returns the difference between 1.0 and the next representable value of the
  // given floating-point type
  static constexpr fixed_pt epsilon() { return fixed_pt::epsilon(); }
  // returns the maximum rounding error of the given floating-point type
  static constexpr fixed_pt round_error() { return fixed_pt::zero(); }
  // returns the positive infinity value of the given floating-point type
  static constexpr fixed_pt infinity() { return fixed_pt::infinity(); }
  // returns a quiet NaN value of the given floating-point type
  static constexpr fixed_pt quiet_NaN() { return fixed_pt::quiet_NaN(); }
  // returns a signaling NaN value of the given floating-point type
  static constexpr fixed_pt signaling_NaN() {
    return fixed_pt::signaling_NaN();
  }
  // returns the smallest positive subnormal value of the given floating-point
  // type
  static constexpr fixed_pt denorm_min() { return fixed_pt::denorm_min(); }
};

template <int exponent_, arithmetic_prim mantissa_t_ = float>
class fixed_point {
public:
  using mantissa_t = mantissa_t_;
  static constexpr int exponent = exponent_;

  fixed_point() = default;
  fixed_point(const mantissa_t &v) : mantissa(v) {}
  fixed_point(mantissa_t &&v) : mantissa(std::move(v)) {}

  fixed_point &operator+=(auto &&rhs) {
    constexpr int exp_diff = exponent - rhs.exponent;
    if constexpr (exp_diff >= 0) {
      if constexpr (exp_diff >= sizeof(rhs.mantissa) * 8 + 1) {
        return *this;
      } else {
        const auto [rhs_mantissa, rounding_bit, trailing_bits] =
            split_mantissa<exp_diff>(rhs.mantissa);
        auto new_mantissa = mantissa + rhs_mantissa;

        if (rounding_bit) {
          if (new_mantissa & 1) {
            if (rhs.mantissa > 0) {
              new_mantissa++;
            } else {
              new_mantissa--;
            }
          } else {
          }
        }
        mantissa = new_mantissa;
      }
    } else {
      mantissa += rhs.mantissa << exp_diff;
    }
    return *this;
  }

  fixed_point &operator-=(auto &&rhs) {
    constexpr int exp_diff = rhs.exponent - exponent;
    if constexpr (exp_diff <= 0) {
      mantissa -= rhs.mantissa >> -exp_diff;
    } else {
      mantissa -= rhs.mantissa << exp_diff;
    }
    return *this;
  }

  fixed_point &operator*=(const fixed_point<0, mantissa_t> &rhs) {
    mantissa *= rhs.mantissa;
    return *this;
  }

  fixed_point &operator*=(fixed_point<0, mantissa_t> &&rhs) {
    mantissa *= rhs.mantissa;
    return *this;
  }

  fixed_point &operator/=(const fixed_point<0, mantissa_t> &rhs) {
    mantissa /= rhs.mantissa;
    return *this;
  }

  fixed_point &operator/=(fixed_point<0, mantissa_t> &&rhs) {
    mantissa /= rhs.mantissa;
    return *this;
  }

private:
  mantissa_t mantissa;
};

} // namespace FixedPoint

#endif // FIXED_POINT_HPP

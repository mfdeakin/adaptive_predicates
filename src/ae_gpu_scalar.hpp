
#ifndef AE_GPU_SCALAR_HPP
#define AE_GPU_SCALAR_HPP

#include <cmath>
#include <compare>

namespace adaptive_expr {

// A convenience class for when you want to ensure the vector exact arithmetic
// implementations are being used, generally when targetting a GPU
template <typename real_type> struct GPUVec {
  constexpr GPUVec() : val(0) {}
  constexpr GPUVec(const real_type v) : val(v) {}
  constexpr GPUVec(const GPUVec &g) = default;
  constexpr GPUVec(GPUVec &&g) = default;
  constexpr GPUVec &operator=(const GPUVec &g) = default;
  constexpr GPUVec &operator=(GPUVec &&g) = default;

  constexpr operator real_type() const { return val; }
  constexpr real_type &operator[](int) { return val; }
  constexpr real_type operator[](int) const { return val; }
  constexpr GPUVec operator-() const { return -val; }
  constexpr GPUVec operator+(const real_type &r) const { return val + r; }
  constexpr GPUVec operator-(const real_type &r) const { return val - r; }
  constexpr GPUVec operator*(const real_type &r) const { return val * r; }
  constexpr GPUVec operator/(const real_type &r) const { return val / r; }
  constexpr auto operator<=>(const real_type &r) const { return val <=> r; }
  constexpr auto operator<=>(const int &r) const { return val <=> r; }
  constexpr GPUVec operator+(const GPUVec &r) const { return val + r.val; }
  constexpr GPUVec operator-(const GPUVec &r) const { return val - r.val; }
  constexpr GPUVec operator*(const GPUVec &r) const { return val * r.val; }
  constexpr GPUVec operator/(const GPUVec &r) const { return val / r.val; }
  constexpr auto operator<=>(const GPUVec &r) const { return val <=> r.val; }
  constexpr GPUVec operator+(const real_type &&r) const { return val + r; }
  constexpr GPUVec operator-(const real_type &&r) const { return val - r; }
  constexpr GPUVec operator*(const real_type &&r) const { return val * r; }
  constexpr GPUVec operator/(const real_type &&r) const { return val / r; }
  constexpr auto operator<=>(const real_type &&r) const { return val <=> r; }
  constexpr auto operator<=>(const int &&r) const { return val <=> r; }
  constexpr GPUVec operator+(const GPUVec &&r) const { return val + r.val; }
  constexpr GPUVec operator-(const GPUVec &&r) const { return val - r.val; }
  constexpr GPUVec operator*(const GPUVec &&r) const { return val * r.val; }
  constexpr GPUVec operator/(const GPUVec &&r) const { return val / r.val; }
  constexpr auto operator<=>(const GPUVec &&r) const { return val <=> r.val; }

  real_type val;
};

using std::abs;
using std::fma;

template <typename real>
constexpr GPUVec<real> select(bool p, GPUVec<real> v1, GPUVec<real> v2) {
  if (p) {
    return v1;
  } else {
    return v2;
  }
}

template <typename real>
constexpr GPUVec<real> mul_sub(GPUVec<real> a, GPUVec<real> b, GPUVec<real> c) {
  return fma(a[0], b[0], -c[0]);
}

template <typename real> constexpr GPUVec<real> abs(GPUVec<real> a) {
  return abs(a.val);
}

}; // namespace adaptive_expr

#endif // AE_GPU_SCALAR_HPP

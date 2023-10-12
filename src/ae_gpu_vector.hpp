
#ifndef AE_GPU_SCALAR_HPP
#define AE_GPU_SCALAR_HPP

#include <cmath>
#include <compare>

#ifdef __CUDA_ARCH__
#define CUDA_SHARED __shared__
#else // __CUDA_ARCH__
#define CUDA_SHARED
#endif // __CUDA_ARCH__

namespace adaptive_expr {

// A GPU capable vector class
template <typename real_type, size_t num_threads = 1> struct GPUVec final {
  static constexpr size_t size() { return num_threads; }

  constexpr real_type &operator[](int i) { return vals[i]; }
  constexpr real_type operator[](int i) const { return vals[i]; }
  constexpr GPUVec insert(int i, real_type r) {
    vals[i] = r;
    return *this;
  }
  constexpr auto begin() { return vals.begin(); }
  constexpr auto end() { return vals.end(); }

#ifndef __CUDA_ARCH__
  constexpr GPUVec() : vals() {}
  constexpr GPUVec(const real_type v) : vals() {
    for (real_type &r : vals) {
      r = v;
    }
  }
  constexpr GPUVec(const std::array<real_type, num_threads> vals_)
      : vals(vals_) {}
  constexpr GPUVec(const GPUVec &g) = default;
  constexpr GPUVec(GPUVec &&g) = default;
  constexpr GPUVec &operator=(const GPUVec &g) = default;
  constexpr GPUVec &operator=(GPUVec &&g) = default;

  constexpr GPUVec operator-() const {
    GPUVec new_vals(vals);
    for (real_type &v : new_vals.vals) {
      v = -v;
    }
    return new_vals;
  }

  constexpr GPUVec operator+=(const real_type &r) {
    for (real_type &v : vals) {
      v += r;
    }
    return *this;
  }
  constexpr GPUVec operator-=(const real_type &r) {
    for (real_type &v : vals) {
      v -= r;
    }
    return *this;
  }
  constexpr GPUVec operator*=(const real_type &r) {
    for (real_type &v : vals) {
      v *= r;
    }
    return *this;
  }
  constexpr GPUVec operator/=(const real_type &r) {
    for (real_type &v : vals) {
      v /= r;
    }
    return *this;
  }
  constexpr auto operator<(const real_type &r) const {
    GPUVec<bool, num_threads> pred;
    for (size_t i = 0; i < num_threads; ++i) {
      pred[i] = vals[i] < r;
    }
    return pred;
  }
  constexpr auto operator<=(const real_type &r) const {
    GPUVec<bool, num_threads> pred;
    for (size_t i = 0; i < num_threads; ++i) {
      pred[i] = vals[i] <= r;
    }
    return pred;
  }
  constexpr auto operator>(const real_type &r) const {
    GPUVec<bool, num_threads> pred;
    for (size_t i = 0; i < num_threads; ++i) {
      pred[i] = vals[i] > r;
    }
    return pred;
  }
  constexpr auto operator>=(const real_type &r) const {
    GPUVec<bool, num_threads> pred;
    for (size_t i = 0; i < num_threads; ++i) {
      pred[i] = vals[i] >= r;
    }
    return pred;
  }
  constexpr auto operator==(const real_type &r) const {
    GPUVec<bool, num_threads> pred;
    for (size_t i = 0; i < num_threads; ++i) {
      pred[i] = vals[i] == r;
    }
    return pred;
  }

  constexpr GPUVec operator+=(const GPUVec &right) {
    for (size_t i = 0; i < num_threads; ++i) {
      vals[i] += right[i];
    }
    return *this;
  }
  constexpr GPUVec operator-=(const GPUVec &right) {
    for (size_t i = 0; i < num_threads; ++i) {
      vals[i] -= right[i];
    }
    return *this;
  }
  constexpr GPUVec operator*=(const GPUVec &right) {
    for (size_t i = 0; i < num_threads; ++i) {
      vals[i] *= right[i];
    }
    return *this;
  }
  constexpr GPUVec operator/=(const GPUVec &right) {
    for (size_t i = 0; i < num_threads; ++i) {
      vals[i] /= right[i];
    }
    return *this;
  }
  constexpr auto operator<(const GPUVec &right) const {
    GPUVec<bool, num_threads> pred;
    for (size_t i = 0; i < num_threads; ++i) {
      pred[i] = vals[i] < right[i];
    }
    return pred;
  }
  constexpr auto operator<=(const GPUVec &right) const {
    GPUVec<bool, num_threads> pred;
    for (size_t i = 0; i < num_threads; ++i) {
      pred[i] = vals[i] <= right[i];
    }
    return pred;
  }
  constexpr auto operator==(const GPUVec &right) const {
    GPUVec<bool, num_threads> pred;
    for (size_t i = 0; i < num_threads; ++i) {
      pred[i] = vals[i] == right[i];
    }
    return pred;
  }

#else  // __CUDA_ARCH__
  static constexpr int thread_index() {
    return (threadIdx.x * blockDim.y + threadIdx.y) * blockDim.z + threadIdx.z;
  }
  constexpr GPUVec() : vals() {}
  constexpr GPUVec(const real_type v) : vals() { vals[thread_index()] = v; }
  constexpr GPUVec(const std::array<real_type, num_threads> vals_) : vals() {
    vals[thread_index()] = vals_[thread_index()];
  }
  constexpr GPUVec(const GPUVec &g) : vals() {
    vals[thread_index()] = g[thread_index()];
  }
  constexpr GPUVec(GPUVec &&g) : vals() {
    vals[thread_index()] = g[thread_index()];
  }
  constexpr GPUVec &operator=(const GPUVec &g) {
    vals[thread_index()] = g[thread_index()];
    return *this;
  }
  constexpr GPUVec &operator=(GPUVec &&g) {
    vals[thread_index()] = g[thread_index()];
    return *this;
  }

  constexpr GPUVec operator-() const {
    CUDA_SHARED GPUVec new_vals;
    new_vals[thread_index()] = -vals[thread_index()];
    return new_vals;
  }

  constexpr GPUVec operator+=(const real_type &r) {
    vals[thread_index()] += r;
    return *this;
  }
  constexpr GPUVec operator-=(const real_type &r) {
    vals[thread_index()] -= r;
    return *this;
  }
  constexpr GPUVec operator*=(const real_type &r) {
    vals[thread_index()] *= r;
    return *this;
  }
  constexpr GPUVec operator/=(const real_type &r) {
    vals[thread_index()] /= r;
    return *this;
  }
  constexpr auto operator<(const real_type &r) const {
    CUDA_SHARED GPUVec<bool, num_threads> pred;
    pred[thread_index()] = vals[thread_index()] < r;
    return pred;
  }
  constexpr auto operator<=(const real_type &r) const {
    CUDA_SHARED GPUVec<bool, num_threads> pred;
    pred[thread_index()] = vals[thread_index()] <= r;
    return pred;
  }
  constexpr auto operator>(const real_type &r) const {
    CUDA_SHARED GPUVec<bool, num_threads> pred;
    pred[thread_index()] = vals[thread_index()] > r;
    return pred;
  }
  constexpr auto operator>=(const real_type &r) const {
    CUDA_SHARED GPUVec<bool, num_threads> pred;
    pred[thread_index()] = vals[thread_index()] >= r;
    return pred;
  }
  constexpr auto operator==(const real_type &r) const {
    CUDA_SHARED GPUVec<bool, num_threads> pred;
    pred[thread_index()] = vals[thread_index()] == r;
    return pred;
  }

  constexpr GPUVec operator+=(const GPUVec &right) {
    vals[thread_index()] += right[thread_index()];
    return *this;
  }
  constexpr GPUVec operator-=(const GPUVec &right) {
    vals[thread_index()] -= right[thread_index()];
    return *this;
  }
  constexpr GPUVec operator*=(const GPUVec &right) {
    vals[thread_index()] *= right[thread_index()];
    return *this;
  }
  constexpr GPUVec operator/=(const GPUVec &right) {
    vals[thread_index()] /= right[thread_index()];
    return *this;
  }
  constexpr auto operator<(const GPUVec &right) const {
    CUDA_SHARED GPUVec<bool, num_threads> pred;
    pred[thread_index()] = vals[thread_index()] < right[thread_index()];
    return pred;
  }
  constexpr auto operator<=(const GPUVec &right) const {
    CUDA_SHARED GPUVec<bool, num_threads> pred;
    pred[thread_index()] = vals[thread_index()] <= right[thread_index()];
    return pred;
  }
  constexpr auto operator==(const GPUVec &right) const {
    CUDA_SHARED GPUVec<bool, num_threads> pred;
    pred[thread_index()] = vals[thread_index()] == right[thread_index()];
    return pred;
  }
#endif // __CUDA_ARCH__

  constexpr GPUVec operator+(const real_type &r) const {
    CUDA_SHARED GPUVec new_vals; new_vals = *this;
    new_vals += r;
    return new_vals;
  }
  constexpr GPUVec operator-(const real_type &r) const {
    CUDA_SHARED GPUVec new_vals; new_vals = *this;
    new_vals -= r;
    return new_vals;
  }
  constexpr GPUVec operator*(const real_type &r) const {
    CUDA_SHARED GPUVec new_vals; new_vals = *this;
    new_vals *= r;
    return new_vals;
  }
  constexpr GPUVec operator/(const real_type &r) const {
    CUDA_SHARED GPUVec new_vals; new_vals = *this;
    new_vals /= r;
    return new_vals;
  }
  constexpr GPUVec operator+(const GPUVec &r) const {
    CUDA_SHARED GPUVec new_vals; new_vals = *this;
    new_vals += r;
    return new_vals;
  }
  constexpr GPUVec operator-(const GPUVec &r) const {
    CUDA_SHARED GPUVec new_vals; new_vals = *this;
    new_vals -= r;
    return new_vals;
  }
  constexpr GPUVec operator*(const GPUVec &r) const {
    CUDA_SHARED GPUVec new_vals; new_vals = *this;
    new_vals *= r;
    return new_vals;
  }
  constexpr GPUVec operator/(const GPUVec &r) const {
    CUDA_SHARED GPUVec new_vals; new_vals = *this;
    new_vals /= r;
    return new_vals;
  }

  constexpr auto operator>(const GPUVec &right) const { return right < *this; }
  constexpr auto operator>=(const GPUVec &right) const {
    return right <= *this;
  }

  constexpr GPUVec operator+(const real_type &&r) const { return *this + r; }
  constexpr GPUVec operator-(const real_type &&r) const { return *this - r; }
  constexpr GPUVec operator*(const real_type &&r) const { return *this * r; }
  constexpr GPUVec operator/(const real_type &&r) const { return *this / r; }
  constexpr auto operator<(const real_type &&r) const { return *this < r; }
  constexpr auto operator<=(const real_type &&r) const { return *this <= r; }
  constexpr auto operator>(const real_type &&r) const { return *this > r; }
  constexpr auto operator>=(const real_type &&r) const { return *this >= r; }
  constexpr auto operator==(const real_type &&r) const { return *this == r; }
  // TODO: Make CPU/GPU specific versions of these in case we can reuse the
  // memory of the values in r
  // (though ideally you won't use types with dynamic memory as values)
  constexpr GPUVec operator+(const GPUVec &&r) const { return *this + r; }
  constexpr GPUVec operator-(const GPUVec &&r) const { return *this - r; }
  constexpr GPUVec operator*(const GPUVec &&r) const { return *this * r; }
  constexpr GPUVec operator/(const GPUVec &&r) const { return *this / r; }
  constexpr auto operator<(const GPUVec &&r) const { return *this < r; }
  constexpr auto operator<=(const GPUVec &&r) const { return *this <= r; }
  constexpr auto operator>(const GPUVec &&r) const { return *this > r; }
  constexpr auto operator>=(const GPUVec &&r) const { return *this >= r; }
  constexpr auto operator==(const GPUVec &&r) const { return *this == r; }

protected:
  std::array<real_type, num_threads> vals;
};

using std::abs;
using std::fma;

#ifndef __CUDA_ARCH__
template <typename PredicateVec, typename real, size_t num_threads>
constexpr GPUVec<real, num_threads> select(PredicateVec p,
                                           GPUVec<real, num_threads> v1,
                                           GPUVec<real, num_threads> v2) {
  GPUVec<real, num_threads> r;
  for (size_t i = 0; i < num_threads; ++i) {
    if (p[i]) {
      r[i] = v1[i];
    } else {
      r[i] = v2[i];
    }
  }
  return r;
}

template <typename real, size_t num_threads>
constexpr GPUVec<real, num_threads> mul_sub(GPUVec<real, num_threads> a,
                                            GPUVec<real, num_threads> b,
                                            GPUVec<real, num_threads> c) {
  GPUVec<real, num_threads> r;
  for (size_t i = 0; i < num_threads; ++i) {
    r[i] = fma(a[i], b[i], -c[i]);
  }
  return r;
}

template <typename real, size_t num_threads>
constexpr GPUVec<real, num_threads> abs(GPUVec<real, num_threads> a) {
  GPUVec<real, num_threads> r;
  for (size_t i = 0; i < num_threads; ++i) {
    r[i] = abs(a[i]);
  }
  return r;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads> operator&&(GPUVec<bool, num_threads> &l,
                                               const bool r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] && r;
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads> operator&&(GPUVec<bool, num_threads> &&l,
                                               const bool r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] && r;
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator&&(const GPUVec<bool, num_threads> &l,
           const GPUVec<bool, num_threads> &r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] && r[i];
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator&&(const GPUVec<bool, num_threads> &&l,
           const GPUVec<bool, num_threads> &r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] && r[i];
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator&&(const GPUVec<bool, num_threads> &l,
           const GPUVec<bool, num_threads> &&r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] && r[i];
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator&&(const GPUVec<bool, num_threads> &&l,
           const GPUVec<bool, num_threads> &&r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] && r[i];
  }
  return new_vals;
}
template <size_t num_threads>
constexpr GPUVec<bool, num_threads> operator||(GPUVec<bool, num_threads> &l,
                                               const bool r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] || r;
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads> operator||(GPUVec<bool, num_threads> &&l,
                                               const bool r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] || r;
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator||(const GPUVec<bool, num_threads> &l,
           const GPUVec<bool, num_threads> &r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] || r[i];
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator||(const GPUVec<bool, num_threads> &&l,
           const GPUVec<bool, num_threads> &r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] || r[i];
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator||(const GPUVec<bool, num_threads> &l,
           const GPUVec<bool, num_threads> &&r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] || r[i];
  }
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator||(const GPUVec<bool, num_threads> &&l,
           const GPUVec<bool, num_threads> &&r) {
  GPUVec<bool, num_threads> new_vals;
  for (size_t i = 0; i < num_threads; ++i) {
    new_vals[i] = l[i] || r[i];
  }
  return new_vals;
}

#else  // __CUDA_ARCH__

template <typename PredicateVec, typename real, size_t num_threads>
constexpr GPUVec<real, num_threads> select(PredicateVec p,
                                           GPUVec<real, num_threads> v1,
                                           GPUVec<real, num_threads> v2) {
  CUDA_SHARED GPUVec<real, num_threads> r;
  const size_t i = r.thread_index();
  if (p[i]) {
    r[i] = v1[i];
  } else {
    r[i] = v2[i];
  }
  return r;
}

template <typename real, size_t num_threads>
constexpr GPUVec<real, num_threads> mul_sub(GPUVec<real, num_threads> a,
                                            GPUVec<real, num_threads> b,
                                            GPUVec<real, num_threads> c) {
  CUDA_SHARED GPUVec<real, num_threads> r;
  const size_t i = r.thread_index();
  r[i] = fma(a[i], b[i], -c[i]);
  return r;
}

template <typename real, size_t num_threads>
constexpr GPUVec<real, num_threads> abs(GPUVec<real, num_threads> a) {
  CUDA_SHARED GPUVec<real, num_threads> r;
  const size_t i = r.thread_index();
  r[i] = abs(a[i]);
  return r;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads> operator&&(GPUVec<bool, num_threads> &l,
                                               const bool r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = l.thread_index();
  new_vals[i] = l[i] && r;
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads> operator&&(GPUVec<bool, num_threads> &&l,
                                               const bool r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = l.thread_index();
  new_vals[i] = l[i] && r;
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator&&(const GPUVec<bool, num_threads> &l,
           const GPUVec<bool, num_threads> &r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = r.thread_index();
  new_vals[i] = l[i] && r[i];
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator&&(const GPUVec<bool, num_threads> &&l,
           const GPUVec<bool, num_threads> &r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = r.thread_index();
  new_vals[i] = l[i] && r[i];
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator&&(const GPUVec<bool, num_threads> &l,
           const GPUVec<bool, num_threads> &&r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = r.thread_index();
  new_vals[i] = l[i] && r[i];
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator&&(const GPUVec<bool, num_threads> &&l,
           const GPUVec<bool, num_threads> &&r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = r.thread_index();
  new_vals[i] = l[i] && r[i];
  return new_vals;
}
template <size_t num_threads>
constexpr GPUVec<bool, num_threads> operator||(GPUVec<bool, num_threads> &l,
                                               const bool r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = l.thread_index();
  new_vals[i] = l[i] || r;
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads> operator||(GPUVec<bool, num_threads> &&l,
                                               const bool r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = l.thread_index();
  new_vals[i] = l[i] || r;
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator||(const GPUVec<bool, num_threads> &l,
           const GPUVec<bool, num_threads> &r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = r.thread_index();
  new_vals[i] = l[i] || r[i];
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator||(const GPUVec<bool, num_threads> &&l,
           const GPUVec<bool, num_threads> &r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = r.thread_index();
  new_vals[i] = l[i] || r[i];
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator||(const GPUVec<bool, num_threads> &l,
           const GPUVec<bool, num_threads> &&r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = r.thread_index();
  new_vals[i] = l[i] || r[i];
  return new_vals;
}

template <size_t num_threads>
constexpr GPUVec<bool, num_threads>
operator||(const GPUVec<bool, num_threads> &&l,
           const GPUVec<bool, num_threads> &&r) {
  CUDA_SHARED GPUVec<bool, num_threads> new_vals;
  const size_t i = r.thread_index();
  new_vals[i] = l[i] || r[i];
  return new_vals;
}
#endif // __CUDA_ARCH__

}; // namespace adaptive_expr

#endif // AE_GPU_SCALAR_HPP

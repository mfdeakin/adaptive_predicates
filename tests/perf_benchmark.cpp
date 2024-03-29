
#include <fmt/format.h>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"

#include "testing_utils.hpp"
#include "float_testing_data.hpp"

#include <random>

int main() {
  double final_result = 0.0;
  for (int i = 0; i < 10000000; ++i) {
    for (auto [points, _] : orient2d_cases) {
      auto expr = adaptive_expr::pt_orient_expr(points);
      final_result += adaptive_eval<real>(expr);
    }
  }
  fmt::print("final_result: {}\n", final_result);
  return 0;
}

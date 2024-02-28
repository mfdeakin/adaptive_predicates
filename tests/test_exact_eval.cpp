
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <ranges>

#include "ae_expr.hpp"
#include "ae_fp_eval.hpp"

#include "testing_utils.hpp"

using namespace adaptive_expr;
using namespace _impl;

using real = double;

auto mult_test_case() {
  const std::vector<real> left_terms{
      9.2730153767185534643293381538690987834189119468944775061e-69,
      -1.4836824602749685542926941046190558053470259115031164010e-67,
      -8.0712325838958289353522559291276635810878209585769532213e-65,
      -3.7146931176476335494342239740309342440743131677144559041e-50,
      1.8816263559049057348829376728746432426858986035939542834e-48,
      -2.2999754552016361580557916276085068746722863910872779169e-34,
      -1.0014835710813626435891085301441622056398128570720018615e-32,
      -5.7128445182814558007931378008513521845906262593833835339e-17,
      8.8817841970012523233890533447265625000000000000000000000e-16,
      2.6181623585046946089960329118184745311737060546875000000e+01};
  const std::vector<real> right_terms{
      3.4159996254284212719523204452351641690112422793523625658e-53,
      -1.2829270608442539101474575042327113018312945024504233750e-49,
      1.5826728910280977138562999662532483368295423737368913138e-33,
      -5.3571703870932964415335228169487282410437732177140371381e-17};
  std::vector<real> results(2 * left_terms.size() * right_terms.size());
  return std::tuple{std::move(results), std::move(left_terms),
                    std::move(right_terms)};
}

TEST_CASE("sparse_mult eval", "[sparse_mult]") {
  auto [results, left, right] = mult_test_case();
  std::vector<real> high_terms;
  high_terms.reserve(left.size() * right.size());
  std::vector<real> low_terms;
  low_terms.reserve(left.size() * right.size());
  for (const auto l : left) {
    for (const auto r : right) {
      const auto high = l * r;
      high_terms.push_back(high);
      const auto low = std::fma(l, r, -high);
      low_terms.push_back(low);
    }
  }
  std::span result_span{results};
  const auto result_last =
      sparse_mult(std::span{left}, std::span{right}, result_span);
  for (const auto v : std::span{result_span.begin(), result_last}) {
    // zero pruning check
    CHECK(v != real{0});
  }
  // check that all of the expected values are in the result
  for (const auto v : high_terms) {
    if (v != real{0}) {
      REQUIRE(std::find(result_span.begin(), result_last, v) != result_last);
    }
  }
  for (const auto v : low_terms) {
    if (v != real{0}) {
      REQUIRE(std::find(result_span.begin(), result_last, v) != result_last);
    }
  }
  for (const auto v : std::span{result_span.begin(), result_last}) {
    // Ensure that v is in either in_high or in_low, but not both
    const bool in_high = std::ranges::find(high_terms, v) != high_terms.end();
    const bool in_low = std::ranges::find(low_terms, v) != low_terms.end();
    REQUIRE(in_high != in_low);
  }
}

TEST_CASE("sparse_mult_merge eval", "[sparse_mult_merge]") {
  auto [results, left, right] = mult_test_case();
  REQUIRE(is_nonoverlapping(left));
  REQUIRE(is_nonoverlapping(right));
  std::vector expected_results_vec = results;
  std::span result_span{results};
  auto result_last =
      sparse_mult_merge(left, right, result_span, std::allocator<real>());

  result_span = std::span{result_span.begin(), result_last};
  REQUIRE(is_nonoverlapping(result_span));
  std::vector<real> nonzero_results;
  for (const auto v : result_span) {
    // zero-pruning check
    REQUIRE(v != real{0});
    nonzero_results.push_back(v);
  }
  // Check that the result is correct by subtracting values that sum to the same
  // thing that a correct implementation produces
  std::span expected_results{expected_results_vec};
  auto expected_last = sparse_mult(left, right, expected_results);
  expected_results = std::span{expected_results.begin(), expected_last};
  for (const auto v : expected_results) {
    nonzero_results.push_back(-v);
  }
  result_span = std::span{nonzero_results};
  const auto merge_result = merge_sum(result_span);

  REQUIRE(merge_result.first == real{0});
  REQUIRE(merge_result.second == result_span.begin());
  for (const auto v : nonzero_results) {
    REQUIRE(v == real{0});
  }
}

auto mult_inplace_test_case() {
  auto [results, left, right] = mult_test_case();
  auto out = results.end();
  for (const auto v : right) {
    --out;
    *out = v;
  }
  const auto right_begin = out;
  for (const auto v : left) {
    --out;
    *out = v;
  }
  const auto left_begin = out;
  return std::tuple{std::move(results), std::span{left_begin, right_begin},
                    std::span{right_begin, results.end()}};
}

TEST_CASE("sparse_mult inplace eval", "[sparse_mult_inplace]") {
  auto [results, left, right] = mult_inplace_test_case();
  std::vector<real> high_terms;
  high_terms.reserve(2 * left.size() * right.size());
  std::vector<real> low_terms;
  low_terms.reserve(left.size() * right.size());
  for (const auto l : left) {
    for (const auto r : right) {
      const auto high = l * r;
      high_terms.push_back(high);
      const auto low = std::fma(l, r, -high);
      low_terms.push_back(low);
    }
  }
  std::span result_span{results};
  const auto result_last =
      sparse_mult(std::span{left}, std::span{right}, result_span);
  for (const auto v : std::span{std::span{results}.begin(), result_last}) {
    // zero pruning check
    CHECK(v != real{0});
  }
  // check that all of the expected values are in the result
  for (const auto v : high_terms) {
    if (v != real{0}) {
      REQUIRE(std::find(result_span.begin(), result_last, v) != result_last);
    }
  }
  for (const auto v : low_terms) {
    if (v != real{0}) {
      REQUIRE(std::find(result_span.begin(), result_last, v) != result_last);
    }
  }
  for (const auto v : std::span{result_span.begin(), result_last}) {
    // Ensure that v is in either in_high or in_low
    const bool in_high = std::ranges::find(high_terms, v) != high_terms.end();
    const bool in_low = std::ranges::find(low_terms, v) != low_terms.end();
    REQUIRE(in_high != in_low);
  }
}

TEST_CASE("sparse_mult_merge inplace eval", "[sparse_mult_merge_inplace]") {
  auto [results, left, right] = mult_test_case();
  REQUIRE(is_nonoverlapping(left));
  REQUIRE(is_nonoverlapping(right));
  std::vector expected_results = results;
  sparse_mult(left, right, std::span{expected_results});
  std::span result_span{results};
  auto result_last =
      sparse_mult_merge(left, right, result_span, std::allocator<real>());
  REQUIRE(is_nonoverlapping(std::span{result_span.begin(), result_last}));
  std::vector<real> nonzero_results;
  for (const auto v : std::span{result_span.begin(), result_last}) {
    // zero pruning check
    REQUIRE(v != real{0});
    nonzero_results.push_back(v);
  }
  // Check that the result is correct by subtracting values that sum to the same
  // thing that a correct implementation produces
  for (const auto v : expected_results) {
    nonzero_results.push_back(-v);
  }
  result_span = std::span{nonzero_results};
  const auto merge_result = merge_sum(result_span);
  REQUIRE(merge_result.first == real{0});
  REQUIRE(merge_result.second == result_span.begin());
  for (const auto v : nonzero_results) {
    REQUIRE(v == real{0});
  }
}

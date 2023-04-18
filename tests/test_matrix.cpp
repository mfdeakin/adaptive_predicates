
#include <array>
#include <span>
#include <tuple>
#include <type_traits>

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "ae_adaptive_predicate_eval.hpp"

#include "testing_utils.hpp"

using namespace adaptive_expr;

TEST_CASE("determinant_sign", "[matrix]") {
  REQUIRE(determinant<double>(mdspan<double, extents<std::size_t, 2, 2>>{
              std::array{5.0, 2.0, -1.0, 0.0}.data()}) == 2.0);
  REQUIRE(determinant<double>(mdspan<double, extents<std::size_t, 2, 2>>{
              std::array{0.0, 1.0, 1.0, 0.0}.data()}) == -1.0);
  REQUIRE(determinant<double>(mdspan<double, extents<std::size_t, 2, 2>>{
              std::array{1.0, 0.0, 1.0, 0.0}.data()}) == 0.0);
  REQUIRE(
      determinant<double>(mdspan<double, extents<std::size_t, 3, 3>>{
          std::array{0.0, 5.0, 2.0, 0.0, 1.0, 0.0, 1.0, 86.0, 35.0}.data()}) ==
      -2.0);
  REQUIRE(
      determinant<double>(mdspan<double, extents<std::size_t, 3, 3>>{
          std::array{0.0, 5.1, 2.0, 3.0, 1.0, 0.0, 6.0, 7.1, 2.0}.data()}) ==
      0.0);
}

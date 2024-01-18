
#include "ae_expr.hpp"
#include "ae_geom_exprs.hpp"

#include <tuple>
#include <type_traits>

using namespace adaptive_expr;

static_assert(std::is_same_v<std::tuple<>,
                             decltype(_impl::submatrix_rm_row(
                                 std::tuple{std::integral_constant<int, 0>{}},
                                 std::make_index_sequence<0>(),
                                 std::make_index_sequence<0>()))>);

static_assert(std::is_same_v<std::tuple<std::integral_constant<int, 0>>,
                             decltype(_impl::submatrix_rm_row(
                                 std::tuple{std::integral_constant<int, 0>{},
                                            std::integral_constant<int, 0>{}},
                                 std::make_index_sequence<0>(),
                                 std::make_index_sequence<1>()))>);

static_assert(
    std::is_same_v<std::remove_cvref_t<decltype(std::get<0>(std::tuple{
                       std::tuple{std::integral_constant<int, 0>{},
                                  std::integral_constant<int, 1>{}},
                       std::tuple{std::integral_constant<int, 2>{},
                                  std::integral_constant<int, 3>{}}}))>,
                   std::tuple<std::integral_constant<int, 0>,
                              std::integral_constant<int, 1>>>);

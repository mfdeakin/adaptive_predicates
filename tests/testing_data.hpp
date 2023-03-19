
#ifndef TESTING_DATA_HPP
#define TESTING_DATA_HPP

using namespace adaptive_expr;

template <arith_number eval_type>
constexpr bool check_sign(eval_type correct, eval_type check) {
  if (correct == eval_type{0.0}) {
    return check == eval_type{0.0};
  } else if (check == eval_type{0.0}) {
    return false;
  } else {
    return std::signbit(correct) == std::signbit(check);
  }
}

constexpr auto
build_orient2d_case(const std::array<std::array<real, 2>, 3> &points) {
  constexpr std::size_t x = 0;
  constexpr std::size_t y = 1;
  using mult_expr = arith_expr<std::multiplies<>, real, real>;
  const auto cross_expr = [](const std::array<real, 2> &lhs,
                             const std::array<real, 2> &rhs) constexpr {
    return mult_expr{lhs[x], rhs[y]} - mult_expr{lhs[y], rhs[x]};
  };
  return cross_expr(points[1], points[2]) - cross_expr(points[0], points[2]) +
         cross_expr(points[0], points[1]);
}

// An array of test cases; each case contains an array of points to check the
// orientation of, and the exact value from the orientation test
constexpr auto orient2d_cases = std::array{
    std::pair{
        std::array{
            std::array<real, 2>{-0.257641255855560303, 0.282396793365478516},
            std::array<real, 2>{-0.734969973564147949, 0.716774165630340576},
            std::array<real, 2>{0.48675835132598877, -0.395019501447677612}},
        real{-9.392445044e-8}},
    std::pair{
        std::array{std::array<real, 2>{-6.9985747337e-01, 5.3128957748e-02},
                   std::array<real, 2>{-9.0628862381e-02, 6.3606452942e-01},
                   std::array<real, 2>{-1.2271618843e-01, 6.0536205769e-01}},
        real{1.9546234849e-08}},
    std::pair{
        std::array{std::array<real, 2>{8.7057256699e-01, -5.6040859222e-01},
                   std::array<real, 2>{-4.2754793167e-01, -7.5050014257e-01},
                   std::array<real, 2>{-9.5216739178e-01, -1.3943153620e-01}},
        real{-8.9296641054e-01}},
    std::pair{
        std::array{std::array<real, 2>{-6.8958437443e-01, 2.2038877010e-01},
                   std::array<real, 2>{-2.7992570400e-01, -9.8853969574e-01},
                   std::array<real, 2>{-4.6557217836e-01, -4.4068527222e-01}},
        real{7.2881434170e-09}},
    std::pair{
        std::array{std::array<real, 2>{8.8695299625e-01, 8.9612734318e-01},
                   std::array<real, 2>{8.5273647308e-01, 5.3343832493e-01},
                   std::array<real, 2>{8.3510673046e-01, 3.4657371044e-01}},
        real{-2.5663237579e-07}},
    std::pair{
        std::array{std::array<real, 2>{-7.0179581642e-02, -4.2452341318e-01},
                   std::array<real, 2>{-9.8506367207e-01, 6.8608045578e-01},
                   std::array<real, 2>{7.4601817131e-01, -5.1958918571e-01}},
        real{-8.1949821943e-01}},
    std::pair{
        std::array{std::array<real, 2>{8.1790435314e-01, 8.5958325863e-01},
                   std::array<real, 2>{-4.9467927217e-01, -9.8880779743e-01},
                   std::array<real, 2>{-2.8476309776e-01, -6.9320201874e-01}},
        real{-1.2535769399e-07}},
    std::pair{
        std::array{std::array<real, 2>{-2.1472203732e-01, -1.9780474901e-01},
                   std::array<real, 2>{-7.1168404818e-01, -6.5561270714e-01},
                   std::array<real, 2>{3.7637495995e-01, -2.7921193838e-01}},
        real{3.1106518990e-01}},
    std::pair{
        std::array{std::array<real, 2>{-8.3859920502e-01, -9.5567250252e-01},
                   std::array<real, 2>{1.0471141338e-01, -2.6719611883e-01},
                   std::array<real, 2>{4.6125137806e-01, 3.3664488792e-01}},
        real{3.2414028798e-01}},
    std::pair{
        std::array{std::array<real, 2>{-3.0781310797e-01, -3.4863823652e-01},
                   std::array<real, 2>{-4.6442991495e-01, -5.2602708340e-01},
                   std::array<real, 2>{6.0313379765e-01, 8.7965607643e-01}},
        real{-3.0779712181e-02}},
    std::pair{
        std::array{std::array<real, 2>{-4.4732582569e-01, 1.2238073349e-01},
                   std::array<real, 2>{5.4575634003e-01, -1.5447413921e-01},
                   std::array<real, 2>{9.3076920509e-01, -2.6345044374e-01}},
        real{-1.6297367685e-03}},
    std::pair{
        std::array{std::array<real, 2>{6.1338424683e-02, -6.7303591967e-01},
                   std::array<real, 2>{-1.1495411396e-01, -6.4643704891e-01},
                   std::array<real, 2>{-5.1052320004e-01, -5.8675348759e-01}},
        real{-7.5546367385e-08}},
    std::pair{
        std::array{std::array<real, 2>{-8.0545777082e-01, 8.7238311768e-01},
                   std::array<real, 2>{-9.5276731253e-01, 9.0831506252e-01},
                   std::array<real, 2>{9.4272518158e-01, -8.9874148369e-01}},
        real{1.9808793990e-01}},
    std::pair{
        std::array{std::array<real, 2>{4.9767935276e-01, -8.0622249842e-01},
                   std::array<real, 2>{2.2120451927e-01, -7.6843857765e-02},
                   std::array<real, 2>{-2.3564928770e-01, 3.8174331188e-01}},
        real{2.0643159734e-01}},
    std::pair{
        std::array{std::array<real, 2>{7.0970463753e-01, -2.9361897707e-01},
                   std::array<real, 2>{-4.6404141188e-01, 2.5502932072e-01},
                   std::array<real, 2>{-7.4774158001e-01, 3.4951949120e-01}},
        real{4.4744150021e-02}},
    std::pair{
        std::array{std::array<real, 2>{-1.1045682430e-01, -1.9889712334e-01},
                   std::array<real, 2>{2.2073578835e-01, -7.6588892937e-01},
                   std::array<real, 2>{-7.7728575468e-01, 9.4269382954e-01}},
        real{-4.9290818541e-08}},
    std::pair{
        std::array{std::array<real, 2>{-8.5646986961e-01, -4.9995595217e-01},
                   std::array<real, 2>{-6.5672194958e-01, -2.4796634912e-01},
                   std::array<real, 2>{6.6050350666e-01, 8.3325016499e-01}},
        real{-1.1595637003e-01}},
    std::pair{
        std::array{std::array<real, 2>{8.4089565277e-01, -1.0077524185e-01},
                   std::array<real, 2>{-3.9864897728e-02, 5.5077266693e-01},
                   std::array<real, 2>{6.3252210617e-01, 5.3370475769e-02}},
        real{-1.1857565597e-07}}};

#endif // TESTING_DATA_HPP

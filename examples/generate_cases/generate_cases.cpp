
#include <numbers>
#include <span>
#include <unordered_map>

#include "ae_adaptive_predicate_eval.hpp"
#include "ae_expr_fmt.hpp"
#include "ae_fp_eval.hpp"

#include "shewchuk.h"

#include <fmt/format.h>

#include <argparse/argparse.hpp>

#include "generate_cases.hpp"

using namespace adaptive_expr;

enum class case_type {
  ORIENT2D,
  ORIENT3D,
  INCIRCLE2D,
  INCIRCLE3D,
};

template <typename case_gen_type>
void generate_cases(const int num_cases, std::mt19937_64 &rng,
                    case_gen_type case_gen) {
  for (int i = 0; i < num_cases; ++i) {
    const auto points = case_gen(rng);

    fmt::println("std::array{{");
    for (const auto pt : points) {
      fmt::println("  std::array{{ ");
      for (const auto coord : pt) {
        fmt::print("{: .55e}, ", coord);
      }
      fmt::println("  }},");
    }
    fmt::println("}}");
  }
}

int main(int argc, char **argv) {
  const std::unordered_map<case_type, std::pair<std::string, std::string>>
      expr_type{{case_type::ORIENT2D,
                 std::pair{"--orient-2d",
                           "Generate points that test the 2D Orientation "
                           "Determinant expression evaluation"}},
                {case_type::ORIENT3D,
                 std::pair{"--orient-3d",
                           "Generate points that test the 3D Orientation "
                           "Determinant expression evaluation"}},
                {case_type::INCIRCLE2D,
                 std::pair{"--in-circle-2d", "Generate points that test the 2D "
                                             "In-Circle Determinant expression "
                                             "evaluation"}},
                {case_type::INCIRCLE3D,
                 std::pair{"--in-circle-3d", "Generate points that test the 3D "
                                             "In-Circle Determinant expression "
                                             "evaluation"}}};

  argparse::ArgumentParser program(argv[0]);
  program.add_argument("-c", "--cases")
      .help("Number of cases to generate")
      .default_value(20)
      .scan<'i', int>();
  for (const auto &[_, cmd_line] : expr_type) {
    const auto [option, help] = cmd_line;
    program.add_argument(option).help(help).default_value(false).implicit_value(
        true);
  }

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  Shewchuk::exactinit();

  const auto seed = std::random_device()();
  fmt::print("Random seed: {}\n", seed);
  std::mt19937_64 rng(seed);

  const int num_cases = program.get<int>("cases");
  int num_chosen = 0;
  for (const auto &[c, cmdline] : expr_type) {
    const auto [option, _] = cmdline;
    if (program[option] == true) {
      fmt::println("Generating {} {} cases", num_cases, option);
      if (c == case_type::ORIENT2D) {
        generate_cases(num_cases, rng, generate_orient2d_case);
      } else if (c == case_type::ORIENT3D) {
        generate_cases(num_cases, rng, generate_orient3d_case);
      } else if (c == case_type::INCIRCLE2D) {
        generate_cases(num_cases, rng, generate_incircle2d_case);
      } else if (c == case_type::INCIRCLE3D) {
        generate_cases(num_cases, rng, generate_incircle3d_case);
      }
      num_chosen++;
    }
  }
  if (num_chosen == 0) {
    fmt::println("No type of case specified, not generating any");
  }

  return 0;
}

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "utils.h"

namespace {
std::vector<float> make_matrix(std::size_t n) {
  throw std::runtime_error("make_matrix not implemented");
}

std::vector<float> run_sequential(const std::vector<float>& matrix,
                                  std::size_t n) {
  throw std::runtime_error("Sequential method not implemented");
}

std::vector<float> run_openmp(const std::vector<float>& matrix, std::size_t n) {
  throw std::runtime_error("OpenMP method not implemented");
}

std::vector<float> run_simd(const std::vector<float>& matrix, std::size_t n) {
  throw std::runtime_error("SIMD method not implemented");
}

std::vector<float> run_openmp_simd(const std::vector<float>& matrix,
                                   std::size_t n) {
  throw std::runtime_error("OpenMP + SIMD method not implemented");
}
}  // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n>\n";
      return EXIT_FAILURE;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (n == 0) {
      throw std::invalid_argument("Matrix size must be positive");
    }

    const auto input = make_matrix(n);

    std::vector<float> sequential_result;
    const double sequential_seconds = utils::measure_seconds(
        [&]() { return run_sequential(input, n); }, sequential_result);

    auto omp_res = utils::run_test_case([&] { return run_openmp(input, n); },
                                        sequential_result, "OpenMP");
    auto simd_res = utils::run_test_case([&] { return run_simd(input, n); },
                                         sequential_result, "SIMD");
    auto omp_simd_res =
        utils::run_test_case([&] { return run_openmp_simd(input, n); },
                             sequential_result, "OpenMP + SIMD");

    std::cout << "Sequential: " << utils::format_time(sequential_seconds)
              << " sec\n";
    utils::print_report("OpenMP", omp_res);
    utils::print_report("SIMD", simd_res);
    utils::print_report("OpenMP + SIMD", omp_simd_res);

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.h"

namespace {
std::vector<__half> make_input_matrix(std::size_t n) {
  throw std::runtime_error("make_input_matrix not implemented");
}

std::vector<float> run_openmp_reference(const std::vector<__half> &matrix,
                                        std::size_t n) {
  throw std::runtime_error("OpenMP reference not implemented");
}

void warmup_wmma(const std::vector<__half> &matrix, std::size_t n) {
  throw std::runtime_error("WMMA warm-up not implemented");
}

std::vector<float> run_wmma(const std::vector<__half> &matrix, std::size_t n) {
  throw std::runtime_error("WMMA method not implemented");
}

void warmup_cutlass(const std::vector<__half> &matrix, std::size_t n) {
  throw std::runtime_error("CUTLASS warm-up not implemented");
}

std::vector<float> run_cutlass(const std::vector<__half> &matrix,
                               std::size_t n) {
  throw std::runtime_error("CUTLASS method not implemented");
}
}  // namespace

int main(int argc, char *argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n>\n";
      return EXIT_FAILURE;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (n == 0) {
      throw std::invalid_argument("Matrix size must be positive");
    }

    const auto input = make_input_matrix(n);
    std::vector<float> openmp_result;
    const double openmp_seconds = utils::measure_seconds(
        [&]() { return run_openmp_reference(input, n); }, openmp_result);

    utils::RunResult wmma_res;
    try {
      warmup_wmma(input, n);
      wmma_res.seconds = utils::measure_seconds(
          [&]() { return run_wmma(input, n); }, wmma_res.result);
      wmma_res.diff = utils::max_abs_diff(openmp_result, wmma_res.result);
      wmma_res.success = true;
    } catch (const std::exception &ex) {
      std::cerr << "WMMA method failed: " << ex.what() << '\n';
    }

    utils::RunResult cutlass_res;
    try {
      warmup_cutlass(input, n);
      cutlass_res.seconds = utils::measure_seconds(
          [&]() { return run_cutlass(input, n); }, cutlass_res.result);
      cutlass_res.diff = utils::max_abs_diff(openmp_result, cutlass_res.result);
      cutlass_res.success = true;
    } catch (const std::exception &ex) {
      std::cerr << "CUTLASS method failed: " << ex.what() << '\n';
    }

    std::cout << "OpenMP: " << utils::format_time(openmp_seconds) << " sec\n";
    utils::print_report("WMMA", wmma_res);
    utils::print_report("CUTLASS", cutlass_res);

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}

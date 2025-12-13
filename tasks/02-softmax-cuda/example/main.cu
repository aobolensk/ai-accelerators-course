#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.h"

namespace {
std::vector<float> make_matrix(std::size_t n) {
  throw std::runtime_error("make_matrix not implemented");
}

std::vector<float> run_sequential(const std::vector<float> &matrix,
                                  std::size_t n) {
  throw std::runtime_error("Sequential method not implemented");
}

void launch_softmax_kernel(const float *d_input, float *d_output, std::size_t n,
                           cudaStream_t stream) {
  throw std::runtime_error("CUDA kernel launch not implemented");
}

void warmup_cuda(const std::vector<float> &matrix, std::size_t n) {
  throw std::runtime_error("CUDA warm-up not implemented");
}

std::vector<float> run_cuda_simt(const std::vector<float> &matrix,
                                 std::size_t n) {
  throw std::runtime_error("CUDA SIMT method not implemented");
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

    const auto input = make_matrix(n);
    std::vector<float> sequential_result;
    const double sequential_seconds = utils::measure_seconds(
        [&]() { return run_sequential(input, n); }, sequential_result);

    utils::RunResult simt_res;
    try {
      warmup_cuda(input, n);
      simt_res.seconds = utils::measure_seconds(
          [&]() { return run_cuda_simt(input, n); }, simt_res.result);
      simt_res.diff = utils::max_abs_diff(sequential_result, simt_res.result);
      simt_res.success = true;
    } catch (const std::exception &ex) {
      std::cerr << "CUDA SIMT method failed: " << ex.what() << '\n';
    }

    std::cout << "Sequential: " << utils::format_time(sequential_seconds)
              << " sec\n";
    utils::print_report("SIMT", simt_res);

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}

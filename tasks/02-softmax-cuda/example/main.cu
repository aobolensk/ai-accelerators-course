#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

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

double measure_seconds(std::function<std::vector<float>()> work,
                       std::vector<float> &result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> duration = stop - start;
  return duration.count();
}

float max_abs_diff(const std::vector<float> &baseline,
                   const std::vector<float> &candidate) {
  if (baseline.size() != candidate.size()) {
    throw std::runtime_error(
        "Result size mismatch while validating correctness");
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(baseline[i] - candidate[i]));
  }
  return max_diff;
}

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

    const auto format_time = [](double seconds) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << seconds;
      return oss.str();
    };

    const auto format_diff = [](float diff) {
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(2) << diff;
      return oss.str();
    };

    std::vector<float> sequential_result;
    const double sequential_seconds = measure_seconds(
        [&]() { return run_sequential(input, n); }, sequential_result);

    std::vector<float> simt_result;
    double simt_seconds = 0.0;
    float simt_diff = 0.0f;
    bool simt_success = false;
    try {
      warmup_cuda(input, n);
      simt_seconds = measure_seconds([&]() { return run_cuda_simt(input, n); },
                                     simt_result);
      simt_diff = max_abs_diff(sequential_result, simt_result);
      simt_success = true;
      // TODO: Compare simt_seconds with the OpenMP+AVX2 timing from practice
      // #1.
    } catch (const std::exception &ex) {
      std::cerr << "CUDA SIMT method failed: " << ex.what() << '\n';
    }

    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";
    if (simt_success) {
      std::cout << "SIMT: " << format_time(simt_seconds)
                << " sec (diff: " << format_diff(simt_diff) << ")\n";
    } else {
      std::cout << "SIMT: n/a (diff: n/a)\n";
    }

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}

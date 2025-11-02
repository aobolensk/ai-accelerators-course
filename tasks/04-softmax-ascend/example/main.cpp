#include <acl/acl.h>

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

#include "softmax_kernel.h"

struct AscendResources {
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  uint32_t block_dim = 0;
};

std::vector<float> make_matrix(std::size_t n) {
  throw std::runtime_error("make_matrix not implemented");
}

std::vector<float> run_sequential(const std::vector<float> &matrix,
                                  std::size_t n) {
  throw std::runtime_error("Sequential method not implemented");
}

AscendResources prepare_ascend_resources(std::size_t n) {
  (void)n;
  throw std::runtime_error("Ascend resource preparation not implemented");
}

void release_ascend_resources(AscendResources &resources) {
  (void)resources;
  throw std::runtime_error("Ascend resource teardown not implemented");
}

void warmup_ascend(const std::vector<float> &matrix, std::size_t n,
                   AscendResources &resources) {
  (void)matrix;
  (void)n;
  (void)resources;
  throw std::runtime_error("Ascend warm-up not implemented");
}

std::vector<float> run_ascend_simd(const std::vector<float> &matrix,
                                   std::size_t n,
                                   AscendResources &resources) {
  (void)matrix;
  (void)n;
  (void)resources;
  throw std::runtime_error("Ascend SIMD method not implemented");
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

    AscendResources resources{};
    bool ascend_ready = false;
    try {
      resources = prepare_ascend_resources(n);
      ascend_ready = true;
    } catch (const std::exception &ex) {
      std::cerr << "Ascend resource preparation failed: " << ex.what() << '\n';
    }

    std::vector<float> simd_result;
    double simd_seconds = 0.0;
    float simd_diff = 0.0f;
    bool simd_success = false;
    if (ascend_ready) {
      try {
        warmup_ascend(input, n, resources);
        simd_seconds = measure_seconds(
            [&]() { return run_ascend_simd(input, n, resources); },
            simd_result);
        simd_diff = max_abs_diff(sequential_result, simd_result);
        simd_success = true;
      } catch (const std::exception &ex) {
        std::cerr << "Ascend SIMD method failed: " << ex.what() << '\n';
      }
    }

    if (ascend_ready) {
      try {
        release_ascend_resources(resources);
      } catch (const std::exception &ex) {
        std::cerr << "Ascend resource teardown failed: " << ex.what() << '\n';
      }
    }

    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";
    if (simd_success) {
      std::cout << "Ascend SIMD: " << format_time(simd_seconds)
                << " sec (diff: " << format_diff(simd_diff) << ")\n";
    } else {
      std::cout << "Ascend SIMD: n/a (diff: n/a)\n";
    }

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}

#pragma once

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
#include <string_view>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace utils {

inline double measure_seconds(const std::function<std::vector<float>()>& work,
                              std::vector<float>& result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

inline float max_abs_diff(const std::vector<float>& baseline,
                          const std::vector<float>& candidate) {
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

struct RunResult {
  std::vector<float> result;
  double seconds = 0.0;
  float diff = 0.0f;
  bool success = false;
  explicit operator bool() const noexcept { return success; }
};

inline std::string format_time(double seconds) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << seconds;
  return oss.str();
}

inline std::string format_diff(float diff) {
  std::ostringstream oss;
  oss << std::defaultfloat << std::setprecision(1) << diff;
  return oss.str();
}

inline void print_report(std::string_view test_name, const RunResult& result) {
  if (result) {
    std::cout << test_name << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << test_name << ": n/a (diff: n/a)\n";
  }
}

inline RunResult run_test_case(
    const std::function<std::vector<float>()>& runner,
    const std::vector<float>& baseline, std::string_view method_name) {
  RunResult result;
  try {
    result.seconds = measure_seconds(runner, result.result);
    result.diff = max_abs_diff(baseline, result.result);
    result.success = true;
  } catch (const std::exception& ex) {
    std::cerr << method_name << " method failed: " << ex.what() << '\n';
  }
  return result;
}

#ifdef __CUDACC__
[[noreturn]] inline void cuda_fail(cudaError_t code_error, const char* file,
                                   int line) {
  std::cerr << "\033[1;31merror\033[0m: " << cudaGetErrorString(code_error)
            << '\n';
  std::cerr << "code error: " << static_cast<int>(code_error) << '\n';
  std::cerr << "loc: " << file << '(' << line << ")\n";
  std::exit(static_cast<int>(code_error));
}

inline void check_cuda(cudaError_t code_error, const char* file, int line) {
  if (code_error != cudaSuccess) {
    cuda_fail(code_error, file, line);
  }
}
#endif

}  // namespace utils

#ifdef __CUDACC__
#define CHECK_CUDA_ERROR(callable) \
  ::utils::check_cuda((callable), __FILE__, __LINE__)
#endif

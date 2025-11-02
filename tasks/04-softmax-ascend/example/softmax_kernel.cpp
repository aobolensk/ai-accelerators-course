#include "kernel_operator.h"

#include <cstdio>

#include "softmax_kernel.h"
extern "C" uint32_t aclrtlaunch_softmax_kernel(uint32_t block_dim, void* stream,
                                               void* input, void* output,
                                               uint32_t rows, uint32_t cols);

extern "C" __global__ __aicore__ void softmax_kernel(__gm__ float* input,
                                                     __gm__ float* output,
                                                     uint32_t rows,
                                                     uint32_t cols) {
  (void)input;
  (void)output;
  (void)rows;
  (void)cols;
  AscendC::printf("softmax_kernel is not implemented.\n");
}

extern "C" void softmax_do(uint32_t block_dim, void* stream, void* input,
                           void* output, uint32_t rows, uint32_t cols) {
  const uint32_t launch_block_dim = block_dim == 0U ? 1U : block_dim;
  const uint32_t status = aclrtlaunch_softmax_kernel(
      launch_block_dim, stream, input, output, rows, cols);
  if (status != 0U) {
    std::fprintf(stderr,
                 "softmax_do: aclrtlaunch_softmax_kernel failed with status "
                 "%u.\n",
                 status);
  }
}

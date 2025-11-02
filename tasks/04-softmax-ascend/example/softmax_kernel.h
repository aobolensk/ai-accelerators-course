#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void softmax_do(uint32_t block_dim, void* stream, void* input, void* output,
                uint32_t rows, uint32_t cols);

#ifdef __cplusplus
}
#endif

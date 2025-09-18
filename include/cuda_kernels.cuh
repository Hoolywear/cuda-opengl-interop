#pragma once

#include <cuda_runtime.h>

// Launch a dummy CUDA kernel (defined in .cu) to verify toolchain.
// Returns 0 on success, non-zero on failure.
int launch_dummy_kernel();

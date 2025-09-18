#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// Launch a dummy CUDA kernel (defined in .cu) to verify toolchain later.
// Returns 0 on success, non-zero on failure (when ENABLE_CUDA is on).
int launch_dummy_kernel();

#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstdlib>

// Core macro for checking CUDA runtime API calls.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _cuda_err = (call); \
        if (_cuda_err != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(_cuda_err), static_cast<int>(_cuda_err)); \
            std::abort(); \
        } \
    } while (0)

// Compatibility macro if existing code used checkCudaErrors (common in samples)
#ifndef checkCudaErrors
#define checkCudaErrors(call) CUDA_CHECK(call)
#endif

// Helper to check last kernel launch (similar to sample macros)
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

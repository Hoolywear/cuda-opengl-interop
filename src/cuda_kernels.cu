#include "cuda_kernels.cuh"
#include <cstdio>

#ifdef __CUDACC__
__global__ void dummy_kernel(int *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 42;
    }
}
#endif

int launch_dummy_kernel() {
#ifdef __CUDACC__
    int *d_out = nullptr;
    int h_out = 0;
    cudaError_t err = cudaMalloc(&d_out, sizeof(int));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    dummy_kernel<<<1, 32>>>(d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 2;
    }
    err = cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 3;
    }
    cudaFree(d_out);
    // Silent success for now; could print h_out if needed
    return h_out == 42 ? 0 : 4;
#else
    return 0; // CUDA disabled; treat as success stub.
#endif
}

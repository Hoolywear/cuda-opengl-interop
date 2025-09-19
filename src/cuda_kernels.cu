#include "cuda_kernels.cuh"
#include <cstdio>
__global__ void setup_kernel(float3 *pos, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // calculate uv coordinates
    float u = ( x + 0.5f ) / (float)width;
    float v = ( y + 0.5f ) / (float)height;
    
    u       = u * 2.0f - 1.0f;
    v       = v * 2.0f - 1.0f;
    
    // write output vertex
    pos[y * width + x] = make_float3(u, v, 1.0f);
}

__global__ void update_kernel(float3 *pos, unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;


    pos[idx].x = pos[idx].x > 1.0f ? -1.0f : pos[idx].x + 0.01f;
}

void launch_kernel(float3 *pos, unsigned int w, unsigned int h, bool setup) {
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    if (setup) {
        setup_kernel<<<grid,block>>>(pos,w,h);
    } else {
        update_kernel<<<grid,block>>>(pos,w,h);
    }
}
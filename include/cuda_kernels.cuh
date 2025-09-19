#pragma once

#include <cuda_runtime.h>

void launch_kernel(float3 *pos, unsigned int width, unsigned int height, bool setup = false);
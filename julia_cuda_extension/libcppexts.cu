/*

This is meant as an illustrative example; the code is not optimized. 

*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#ifdef DEBUG
#define DEBUG_LOG(...) printf(__VA_ARGS__)
#else
#define DEBUG_LOG(...)
#endif




struct LaunchConfig {
    int nWarps;
    int idWarp;
    int sizeWarp;
    int nThreads;
    int threadId;
    int lane;
};

// host function to compute warp size
__host__ int gpu_warp_size() {
    // get device properties for warpsize
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    int warp_size = prop.warpSize; 
    return warp_size;
}

// warp-level reduction using shuffle down
__device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// device function for dot product within a warp
__device__ void wrap_dot_product(float* C, const float* A, const float* B, int lane, int sizeWarp, int dim) {
    float partial = 0.0f;
    for (int i = lane; i < dim; i += sizeWarp) {
        partial += A[i] * B[i];
    }
    partial = warp_reduce_sum(partial);
    if (lane == 0) {
        C[0] = partial;
    }
}

// kernel for batched dot product with slice ranges
__global__ void batched_dot_product_warp_kernel_slices(float* C, const float* A, const float* B,
                                                       int N, const int* slice_starts, const int* slice_stops, int warp_size) {

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadId % warp_size;
    int idWarp = threadId / warp_size;
    int nThreads = blockDim.x * gridDim.x;
    int nWarps = nThreads / warp_size;
    
    DEBUG_LOG("%d\n", threadId);

    LaunchConfig config;
    config.nWarps = nWarps;
    config.idWarp = idWarp;
    config.sizeWarp = warp_size;
    config.nThreads = nThreads;
    config.threadId = threadId;
    config.lane = lane;

    int start = config.idWarp;
    int skip = config.nWarps;

    DEBUG_LOG("warp:%d, lane:%d, thread:%d, skip:%d\n", config.idWarp, config.lane, config.threadId, skip);
    
    for (int i = start; i < N; i += skip) {
        int dim = slice_stops[i] - slice_starts[i] + 1; // start/stop indices are inclusive
        int startpos = slice_starts[i] - 1;  // adjust for julia being 1 based
        wrap_dot_product(&C[idWarp], &A[startpos], &B[startpos], lane, warp_size, dim); // __device__ function
    }
}


// host function callable from Julia
extern "C" void launch_batched_dot_product(float* C, const float* A, const float* B,
                                           int N, int dim,
                                           const int* slice_starts, const int* slice_stops,
                                           int threadsPerBlock, int numBlocks) {
    int warp_size = gpu_warp_size();
    std::cout << "(cfg:" <<  warp_size << "," << numBlocks << "," << threadsPerBlock << ")" << std::endl;
    std::cout << N << "," << dim << std::endl; 

    batched_dot_product_warp_kernel_slices<<<numBlocks, threadsPerBlock>>>(C, A, B, N, slice_starts, slice_stops, warp_size);
    cudaDeviceSynchronize();
    std::cout << "ok";
}

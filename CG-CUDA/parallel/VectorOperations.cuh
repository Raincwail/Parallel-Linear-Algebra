#include "../Base.h"

#define NB_ELEM_MAT 32
#define BLOCK_SIZE_MAT 32

#define BLOCK_DIM_VEC 32

__global__ void matDotVec(float *A, float *b, float *res) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        float tmp = 0.0;
        for (int i = 0; i < SIZE; ++i)
            tmp += b[i] * A[SIZE * idx + i];
        res[idx] = tmp;
    }
}

__global__ void vecDotVec(float *a, float *b, float *res) {
    __shared__ float shared_tmp[BLOCK_DIM_VEC];

    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        *res = 0.0;
    }

    if (idx < SIZE) {
        shared_tmp[threadIdx.x] = a[idx] * b[idx];
    } else {
        shared_tmp[threadIdx.x] = 0.0;
    }

    for (int i = blockDim.x / 2; i >= 1; i = i / 2) {
        __syncthreads();
        if (threadIdx.x < i) {
            shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + i];
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(res, shared_tmp[0]);
    }
}

__global__ void scalDotVec(float *a, float *b, float *res) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        res[idx] = *a * b[idx];
    }
}

__global__ void vecPlusVec(float *a, float *b, float *res) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        res[idx] = b[idx] + a[idx];
    }
}

__global__ void vecMinVec(float *a, float *b, float *res) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        res[idx] = a[idx] - b[idx];
    }
}

__global__ void div(float *num, float *den, float *out) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x == 0) {
        *out = *num / *den;
    }
}

__global__ void vecCpy(float *a, float *b) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        b[idx] = a[idx];
    }
}

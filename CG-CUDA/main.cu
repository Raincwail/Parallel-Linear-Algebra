#include <iostream>

#define SIZE 3

#define NB_ELEM_MAT 32
#define BLOCK_SIZE_MAT 32

#define BLOCK_DIM_VEC 32

#define MAX_ITER 1
#define EPS 1e-4

#define A(row, col) (A[(row) * SIZE + (col)])

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

void solveCG(float *A, float *b, float *x, float *p, float *r, float *tmp, float *tmp_scal, float *alpha, float *beta, float *r_norm, float *r_norm_old, float *h_r_norm) {
    dim3 vec_block_dim(BLOCK_DIM_VEC);
    dim3 vec_grid_dim((SIZE + BLOCK_DIM_VEC - 1) / BLOCK_DIM_VEC);

    dim3 mat_grid_dim((SIZE + NB_ELEM_MAT - 1) / NB_ELEM_MAT, (SIZE + BLOCK_SIZE_MAT - 1) / BLOCK_SIZE_MAT);
    dim3 mat_block_dim(BLOCK_SIZE_MAT);

    vecDotVec<<<vec_grid_dim, vec_block_dim>>>(r, r, r_norm_old);
    int it = 0;
    while ((it < MAX_ITER) && (*h_r_norm > EPS)) {
        // Get Ap (tmp)
        matDotVec<<<mat_grid_dim, mat_block_dim>>>(A, p, tmp);

        // Get alpha_k
        vecDotVec<<<vec_grid_dim, vec_block_dim>>>(p, tmp, tmp_scal);
        div<<<1, 1>>>(r_norm_old, tmp_scal, alpha);

        // Get r_{k + 1}
        scalDotVec<<<vec_grid_dim, vec_block_dim>>>(alpha, tmp, tmp);
        vecMinVec<<<vec_grid_dim, vec_block_dim>>>(r, tmp, r);

        // Get x_{k + 1}
        scalDotVec<<<vec_grid_dim, vec_block_dim>>>(alpha, p, tmp);
        vecPlusVec<<<vec_grid_dim, vec_block_dim>>>(x, tmp, x);

        // r_{k + 1} is small??

        // Get beta_{k}
        vecDotVec<<<vec_grid_dim, vec_block_dim>>>(r, r, r_norm);
        div<<<1, 1>>>(r_norm, r_norm_old, beta);

        // Get p_{k + 1}
        scalDotVec<<<vec_grid_dim, vec_block_dim>>>(beta, p, tmp);
        vecPlusVec<<<vec_grid_dim, vec_block_dim>>>(r, tmp, p);

        vecCpy<<<1, 1>>>(r_norm, r_norm_old);

        cudaMemcpy(h_r_norm, r_norm, sizeof(float), cudaMemcpyDeviceToHost);
        it++;
    }
}

int main() {
    float *A = (float *)malloc(SIZE * SIZE * sizeof(float));
    float *b = (float *)malloc(SIZE * sizeof(float));
    float *x = (float *)malloc(SIZE * sizeof(float));
    float *h_r_norm = (float *)malloc(sizeof(float));
    *h_r_norm = 1.0;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                A(i, j) = i + SIZE;
            } else {
                A(i, j) = i;
                A(j, i) = i;
            }
        }
    }

    for (int i = 0; i < SIZE; i++) {
        b[i] = i;
    }

    float *dev_A, *dev_b, *dev_x, *dev_p, *dev_r, *dev_tmp;
    cudaMalloc((void **)&dev_A, SIZE * SIZE * sizeof(float));
    cudaMalloc((void **)&dev_b, SIZE * sizeof(float));
    cudaMalloc((void **)&dev_x, SIZE * sizeof(float));
    cudaMalloc((void **)&dev_p, SIZE * sizeof(float));
    cudaMalloc((void **)&dev_r, SIZE * sizeof(float));
    cudaMalloc((void **)&dev_tmp, SIZE * sizeof(float));

    cudaMemcpy(dev_A, A, SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_p, b, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r, b, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    float *dev_alpha, *dev_beta, *dev_r_norm, *dev_r_norm_old, *dev_tmp_scal;
    cudaMalloc((void **)&dev_alpha, sizeof(float));
    cudaMalloc((void **)&dev_beta, sizeof(float));
    cudaMalloc((void **)&dev_r_norm, sizeof(float));
    cudaMalloc((void **)&dev_r_norm_old, sizeof(float));
    cudaMalloc((void **)&dev_tmp_scal, sizeof(float));

    for (int i = 0; i < SIZE; ++i) {
        std::cout << x[i] << std::endl;
    }

    solveCG(dev_A, dev_b, dev_x, dev_p, dev_r, dev_tmp, dev_tmp_scal, dev_alpha, dev_beta, dev_r_norm, dev_r_norm_old, h_r_norm);

    cudaDeviceSynchronize();

    cudaMemcpy(x, dev_x, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; ++i) {
        std::cout << x[i] << std::endl;
    }

    free(A);
    free(b);
    free(x);
    free(h_r_norm);

    cudaFree(dev_A);
    cudaFree(dev_b);
    cudaFree(dev_x);
    cudaFree(dev_p);
    cudaFree(dev_r);
    cudaFree(dev_tmp);

    cudaFree(dev_alpha);
    cudaFree(dev_beta);
    cudaFree(dev_r_norm);
    cudaFree(dev_r_norm_old);
    cudaFree(dev_tmp_scal);
}
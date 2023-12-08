#include <iostream>

#include "GpuTimer.cuh"
#include "VectorOperations.cuh"

#define MAX_ITER 1000
#define EPS 1e-4

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

    fillA(A);
    fillb(b);

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

    GpuTimer timing;

    timing.Start();
    solveCG(dev_A, dev_b, dev_x, dev_p, dev_r, dev_tmp, dev_tmp_scal, dev_alpha, dev_beta, dev_r_norm, dev_r_norm_old, h_r_norm);

    cudaDeviceSynchronize();
    timing.Stop();

    double res = timing.GetTime();

    std::cout << "Elapsed time: " << res << std::endl;

    cudaMemcpy(x, dev_x, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    print1DVec(x);

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
#include <iostream>

#include "Timer.h"
#include "VectorOperations.h"

#define MAX_ITER 1000
#define EPS 1e-4

void solveCG(float *A, float *b, float *x, float *p, float *r, float *tmp, float tmp_scal, float alpha, float beta, float r_norm, float r_norm_old, float h_r_norm) {
    vecDotVec(r, r, &r_norm_old);
    int it = 0;
    while ((it < MAX_ITER) && (h_r_norm > EPS)) {
        // Get Ap (tmp)
        matDotVec(A, p, tmp);

        // Get alpha_k
        vecDotVec(p, tmp, &tmp_scal);
        alpha = r_norm_old / tmp_scal;

        // Get r_{k + 1}
        scalDotVec(alpha, tmp, tmp);
        vecMinVec(r, tmp, r);

        // Get x_{k + 1}
        scalDotVec(alpha, p, tmp);
        vecPlusVec(x, tmp, x);

        // r_{k + 1} is small??

        // Get beta_{k}
        vecDotVec(r, r, &r_norm);
        beta = r_norm / r_norm_old;

        // Get p_{k + 1}
        scalDotVec(beta, p, tmp);

        vecPlusVec(r, tmp, p);

        r_norm_old = r_norm;
        h_r_norm = r_norm;

        it++;
    }
}

int main() {
    float *A = (float *)malloc(SIZE * SIZE * sizeof(float));
    float *b = (float *)malloc(SIZE * sizeof(float));
    float *x = (float *)malloc(SIZE * sizeof(float));
    float h_r_norm = 1.0;

    fillA(A);
    fillb(b);

    for (int i = 0; i < SIZE; ++i) {
        x[i] = 0;
    }

    float *p = (float *)malloc(SIZE * sizeof(float));
    float *r = (float *)malloc(SIZE * sizeof(float));
    float *tmp = (float *)malloc(SIZE * sizeof(float));

    fillb(p);
    fillb(r);

    float dev_alpha, dev_beta, dev_r_norm, dev_r_norm_old, dev_tmp_scal;

    Timer timing;

    timing.Start();
    solveCG(A, b, x, p, r, tmp, dev_tmp_scal, dev_alpha, dev_beta, dev_r_norm, dev_r_norm_old, h_r_norm);
    timing.Stop();

    std::cout << "Elapsed time: " << timing.GetTime() << std::endl;

    print1DVec(x);
}
#include "../Base.h"

void matDotVec(float *A, float *b, float *res) {
    for (int i = 0; i < SIZE; ++i) {
        res[i] = 0;
        for (int j = 0; j < SIZE; ++j) {
            res[i] += b[j] * A(i, j);
        }
    }
}

void vecDotVec(float *a, float *b, float *res) {
    *res = 0;
    for (int i = 0; i < SIZE; ++i) {
        *res += a[i] * b[i];
    }
};

void scalDotVec(float a, float *b, float *res) {
    for (int i = 0; i < SIZE; ++i) {
        res[i] = a * b[i];
    }
}

void vecPlusVec(float *a, float *b, float *res) {
    for (int i = 0; i < SIZE; ++i) {
        res[i] = a[i] + b[i];
    }
}

void vecMinVec(float *a, float *b, float *res) {
    for (int i = 0; i < SIZE; ++i) {
        res[i] = a[i] - b[i];
    }
}

void div(float *a, float *b, float *res) {
    for (int i = 0; i < SIZE; ++i) {
        res[i] = a[i] / b[i];
    }
}
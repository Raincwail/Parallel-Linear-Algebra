#ifndef MPI_TEST_COMMON_H
#define MPI_TEST_COMMON_H

#define EPS 1e-2

void printMatrix(const float* matrix, int rows, int cols);

void gemvRef(const float* matrix, const float* vector, float* result, int rows, int cols);

bool checkResult(const float* matrix, const float* vector, const float* result, int rows, int cols);

void clearVector(float* vector, int size);

void generateData(float* matrix, float* vector, float* res, int rows, int cols, int vecSize);

void printMaxTime(int rank, double& duration);

void execUnitTest(const float* matrix, const float* vector, const float* res, int rows, int cols);

void printDebugInfo(const float* matrix, const float* vector, const float* res, int rows, int cols, int vecSize);

#endif //MPI_TEST_COMMON_H

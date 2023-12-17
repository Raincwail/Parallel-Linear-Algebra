#include <iostream>
#include <random>
#include <mpi.h>
#include <fstream>
#include "common.h"

void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

void gemvRef(const float* matrix, const float* vector, float* result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        result[i] = 0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

bool checkResult(const float* matrix, const float* vector, const float* result, int rows, int cols) {
    auto* resultRef = new float[rows];
    gemvRef(matrix, vector, resultRef, rows, cols);
    for (int i = 0; i < rows; ++i) {
        if (std::abs(result[i] - resultRef[i]) >= EPS) {
            std::cout << result[i] << " != " << resultRef[i] << "\n";
            return false;
        }
    }
    return true;
}

void clearVector(float* vector, int size) {
    for (int i = 0; i < size; ++i) {
        vector[i] = 0;
    }
}

void generateData(float* matrix, float* vector, float* res, const int rows, int cols, int vecSize) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1, 1);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(gen);
    }
    for (int i = 0; i < vecSize; ++i) {
        vector[i] = dist(gen);
    }
    std::fill(res, res + rows, 0.0f);
}

void printMaxTime(int rank, double& duration) {
    double maxTime = 0;
    MPI_Reduce(&duration, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    maxTime *= 1e3;
    if (rank == 0) {
        std::cout << "Max Time: " << maxTime << " ms\n";
        duration = maxTime;
    }
}

void execUnitTest(const float* matrix, const float* vector, const float* res, int rows, int cols) {
    if (checkResult(matrix, vector, res, rows, cols)) {
        std::cout << "Test PASSED\n";
    } else {
        std::cout << "(!) Test FAILED\n";
    }
}

void printDebugInfo(const float* matrix, const float* vector, const float* res, int rows, int cols, int vecSize) {
    std::cout << "Matrix:" << "\n";
    printMatrix(matrix, rows, cols);
    std::cout << "Vector:" << "\n";
    printMatrix(vector, 1, vecSize);
    std::cout << "Result:" << "\n";
    printMatrix(res, 1, rows);
}

std::ofstream getFileStream(int argc, char* const* argv, int nProc) {
    std::ofstream fs;
    if (argc == 4) {
        fs.open(argv[3], std::ios_base::app);
        if (nProc == 1) {
            fs.close();
            fs.open(argv[3]);
        }
    }
    return fs;
}

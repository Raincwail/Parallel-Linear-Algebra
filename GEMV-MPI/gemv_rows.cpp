#include <iostream>
#include <mpi.h>
#include <string>
#include "common.h"

void gemvByRows(const float* localMatrix, const float* localVector, float* localResult, size_t localRows, size_t cols) {
    for (size_t i = 0; i < localRows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            localResult[i] += localMatrix[i * cols + j] * localVector[j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nProc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double startTime;
    double endTime;
    double duration = 0;
    int rows = 5000;
    int cols = 5000;
    if (argc == 3) {
        rows = std::stoi(argv[1]);
        cols = std::stoi(argv[2]);
    } else {
        std::cout << "WARNING: matrix sizes are not provided. Use default 5000x5000\n";
    }
    std::cout << "Matrix: " << rows << "x" << cols << "\n";
    int vecSize = cols;

    int localRows = rows / nProc;
    int localElements = localRows * cols;

    float* matrix = nullptr;
    auto* vector = new float[vecSize];
    auto* localMatrix = new float[localElements];
    auto* localResult = new float[localRows];
    float* globalResult = nullptr;

    for (int i = 0; i < localRows; ++i) {
        localResult[i] = 0;
    }

    if (rank == 0) {
        matrix = new float[rows * cols];
        globalResult = new float[rows];
        generateData(matrix, vector, globalResult, rows, cols, vecSize);
    }

    MPI_Scatter(matrix, localElements, MPI_FLOAT, localMatrix, localElements, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, vecSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Test performance
    size_t iters = 100;
    for (size_t i = 0; i < iters; ++i) {
        startTime = MPI_Wtime();
        gemvByRows(localMatrix, vector, localResult, localRows, cols);
        endTime = MPI_Wtime();
        duration += endTime - startTime;
        clearVector(localResult, localRows);
    }
    duration /= static_cast<double>(iters);

    // Actual run
    gemvByRows(localMatrix, vector, localResult, localRows, cols);

    // Aggregate the results of all processes
    startTime = MPI_Wtime();
    MPI_Gather(localResult, localRows, MPI_FLOAT, globalResult, localRows, MPI_FLOAT, 0, MPI_COMM_WORLD);
    endTime = MPI_Wtime();
    duration += endTime - startTime;

    // Postprocess remaining rows
    if (rank == 0) {
        size_t remainRows = rows % nProc;
        size_t offset = rows - remainRows;
        startTime = MPI_Wtime();
        gemvByRows(matrix + cols * offset, vector, globalResult + offset, remainRows, cols);
        endTime = MPI_Wtime();
        duration += endTime - startTime;
    }

    printMaxTime(rank, duration);
    if (rank == 0) {
//        printDebugInfo(matrix, vector, globalResult, rows, cols, rank, vecSize);
        execUnitTest(matrix, vector, globalResult, rows, cols);
    }

    delete[] matrix;
    delete[] globalResult;
    delete[] vector;
    delete[] localMatrix;
    delete[] localResult;

    MPI_Finalize();

    return 0;
}

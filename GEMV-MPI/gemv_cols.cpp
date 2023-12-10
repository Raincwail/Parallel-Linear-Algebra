#include <iostream>
#include <mpi.h>
#include "common.h"

void
gemvByCols(const float* matrix, const float* vector, float* result, size_t numRows, size_t numCols, size_t startCol,
           size_t endCol) {
    for (size_t i = startCol; i < endCol; ++i) {
        for (size_t j = 0; j < numRows; ++j) {
            result[j] += matrix[j * numCols + i] * vector[i];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nProc, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);

    double startTime;
    double endTime;
    double duration = 0;
    int rows = 5000;
    int cols = 5000;
    int vecSize = cols;

    auto* matrix = new float[rows * cols];
    auto* vector = new float[vecSize];
    auto* localResult = new float[rows];
    float* globalResult = nullptr;

    for (size_t i = 0; i < rows; ++i) {
        localResult[i] = 0;
    }

    if (rank == 0) {
        globalResult = new float[rows];
        generateData(matrix, vector, globalResult, rows, cols, vecSize);
    }

    MPI_Bcast(vector, vecSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix, rows * cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    size_t myCols = 1;
    size_t startCol = 0;
    size_t endCol = 0;
    if (rank < cols) {
        if (nProc <= cols) {
            myCols = cols / nProc;
        }
        startCol = rank * myCols;
        endCol = startCol + myCols;
        if (rank == nProc - 1) {
            endCol = cols;
        }
    }

    // Test performance
    size_t iters = 100;
    for (size_t i = 0; i < iters; ++i) {
        if (rank < cols) {
            startTime = MPI_Wtime();
            gemvByCols(matrix, vector, localResult, rows, cols, startCol, endCol);
            endTime = MPI_Wtime();
            duration += endTime - startTime;
            clearVector(localResult, rows);
        }
    }
    duration /= static_cast<double>(iters);

    // Actual run
    if (rank < cols) {
        gemvByCols(matrix, vector, localResult, rows, cols, startCol, endCol);
    }

    // Aggregate the results of all processes
    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();
    MPI_Reduce(localResult, globalResult, rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    endTime = MPI_Wtime();
    duration += endTime - startTime;

    printMaxTime(rank, duration);
    if (rank == 0) {
//        printDebugInfo(matrix, vector, globalResult, rows, cols, rank, vecSize);
        execUnitTest(matrix, vector, globalResult, rows, cols);
    }

    delete[] matrix;
    delete[] vector;
    delete[] localResult;
    delete[] globalResult;

    MPI_Finalize();

    return 0;
}

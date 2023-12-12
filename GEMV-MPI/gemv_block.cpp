#include <iostream>
#include <cmath>
#include <filesystem>
#include <mpi.h>
#include <string>
#include "common.h"
#include <fstream>

void
gemvByBlocks(const float* matrix, const float* vector, float* result, size_t numCols, size_t startRow, size_t endRow,
             size_t startCol, size_t endCol) {
    for (size_t i = startRow; i < endRow; ++i) {
        for (size_t j = startCol; j < endCol; ++j) {
            result[i] += matrix[i * numCols + j] * vector[j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nProc, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);

    std::ofstream fs = getFileStream(argc, argv, nProc);

    double startTime;
    double endTime;
    double duration = 0;
    int rows = 5000;
    int cols = 5000;

    if (argc == 4) {
        rows = std::stoi(argv[2]);
        cols = std::stoi(argv[3]);
    } else {
        if (rank == 0) {
            std::cout << "WARNING: matrix sizes are not provided. Use default 5000x5000\n";
        }
    }
    if (rank == 0) {
        std::cout << "Number of processes: " << nProc << "\n";
        std::cout << "Matrix: " << rows << "x" << cols << "\n";
    }

    int vecSize = cols;
    auto blocks1d = static_cast<int>(std::sqrt(nProc));
    int totalBlocks = blocks1d * blocks1d;
    int blockRows = rows / blocks1d;
    int blockCols = cols / blocks1d;

    auto* matrix = new float[rows * cols];
    auto* vector = new float[vecSize];
    auto* localResult = new float[rows];
    float* globalResult = nullptr;

    for (int i = 0; i < rows; ++i) {
        localResult[i] = 0;
    }

    if (rank == 0) {
        globalResult = new float[rows];
        generateData(matrix, vector, globalResult, rows, cols, vecSize);
    }

    MPI_Bcast(vector, cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix, rows * cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    size_t startRow = (rank / blocks1d) * blockRows;
    size_t endRow = startRow + blockRows;
    size_t startCol = (rank % blocks1d) * blockCols;
    size_t endCol = startCol + blockCols;

    if (rank < totalBlocks) {
        if (rank == totalBlocks - 1) {
            endRow = rows;
            endCol = cols;
        } else if ((rank + 1) % blocks1d == 0) {
            endCol = cols;
        } else if ((rank + 1) > totalBlocks - blocks1d) {
            endRow = rows;
        }
    }

    // Test performance
    size_t iters = 100;
    for (size_t i = 0; i < iters; ++i) {
        startTime = MPI_Wtime();
        if (rank < totalBlocks) {
            gemvByBlocks(matrix, vector, localResult, cols, startRow, endRow, startCol, endCol);
        }
        endTime = MPI_Wtime();
        duration += endTime - startTime;
        clearVector(localResult, rows);
    }
    duration /= static_cast<double>(iters);

    // Actual run
    if (rank < totalBlocks) {
        gemvByBlocks(matrix, vector, localResult, cols, startRow, endRow, startCol, endCol);
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
        fs << duration << " ";
        std::cout << "\n";
    }

    delete[] matrix;
    delete[] vector;
    delete[] localResult;

    MPI_Finalize();

    return 0;
}

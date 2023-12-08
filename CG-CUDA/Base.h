#include <iostream>

#define SIZE 1024
#define A(row, col) (A[(row) * SIZE + (col)])

void fillA(float* A) {
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
}

void fillb(float* b) {
    for (int i = 0; i < SIZE; ++i) {
        b[i] = i;
    }
}

void print2DVec(float* A) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void print1DVec(float* b) {
    for (int i = 0; i < SIZE; ++i) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;
}

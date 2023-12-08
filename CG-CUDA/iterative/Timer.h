#include <time.h>

struct Timer {
    clock_t start;
    clock_t stop;

    void Start() {
        start = clock();
    }

    void Stop() {
        stop = clock();
    }

    float GetTime() {
        return stop - start;
    }
};
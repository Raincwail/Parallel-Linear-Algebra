struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    };

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    void Start() {
        cudaEventRecord(start, 0);
    };

    void Stop() {
        cudaEventRecord(stop, 0);
    };

    float GetTime() {
        float time;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }
};
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

#include "errors.hpp"
#include "kernels.cuh"

int main(int argc, char *argv[]) {

    std::ifstream data("dataset.dat");

    float tmpvar = 0.0f;

    std::vector<float> data_vector;

    if (!data) {
        std::cout << "Could not open the data file\n";
        return 1;
    }

    std::cout << "Reading the data file...\n" << std::flush;
    while(data >> tmpvar) {
        data_vector.push_back(tmpvar);
    }

    std::cout << "Read the data file with " << data_vector.size() << " elements (" << data_vector.size() / 1024.0f / 1024.0f << "MiB)\n";
    std::cout << "STD min and max...\n" << std::flush;

    float stdmin = 0.0f;
    float stdmax = 0.0f;
    std::chrono::time_point<std::chrono::steady_clock> stdstart;
    std::chrono::time_point<std::chrono::steady_clock> stdend;

    stdstart = std::chrono::steady_clock::now();
    stdmin = *(std::min_element(data_vector.begin(), data_vector.end()));
    stdmax = *(std::max_element(data_vector.begin(), data_vector.end()));
    stdend = std::chrono::steady_clock::now();

    float stdelapsed = std::chrono::duration<float>(stdend - stdstart).count();
    std::cout << "Took " << stdelapsed << "s to obtain min and max using std\n";
    std::cout << "STD min: " << stdmin << ", STD max: " << stdmax << "\n" << std::flush;

    float *devicedata;
    cudaCheckError(cudaMalloc((void**)&devicedata, data_vector.size() * sizeof(float)));

    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);

    float gpumin = 0.0f;
    float gpumax = 0.0f;
    std::chrono::time_point<std::chrono::steady_clock> naivestart;
    std::chrono::time_point<std::chrono::steady_clock> naiveend;

    naivestart = std::chrono::steady_clock::now();
    naive_kernel<<<grid, block>>>(devicedata, gpumin, gpumax);
    cudaDeviceSynchronize();
    cudaCheckError(cudaGetLastError());
    naiveend = std::chrono::steady_clock::now();

    float naiveelapsed = std::chrono::duration<float>(naiveend - naivestart).count();
    std::cout << "Took " << naiveelapsed << "s to obtain min and max using simple GPU kernel\n";
    std::cout << "GPU (simple) min: " << gpumin << ", GPU (simple) max: " << gpumax << "\n" << std::flush;

    cudaFree(devicedata);

    return 0;

}
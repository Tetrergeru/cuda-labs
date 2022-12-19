#include <iostream>
#include <vector>
#include <chrono>

#include "matrices.h"
#include "../cuda_main.h"

#define EPS 0.000001;

__global__ void check_if_shifted_hierarchical(float *mat, float *shifted_mat, int width, int height, uint shift, unsigned int *number_of_errors)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int shifted_idx = y * width + (x + shift) % width;

    int error = abs(mat[idx] - shifted_mat[shifted_idx]) >= EPS;

    atomicAdd(number_of_errors, error);
}

void run_for_matrices_hierarchical(std::vector<float> &mat, std::vector<float> &shifted_mat, uint width, uint height, uint shift)
{
    auto block_width = 32u;
    auto block_height = 1024u / block_width;

    auto grid_width = (width + block_width - 1u) / block_width;
    auto grid_height = (height + block_height - 1u) / block_height;

    std::cout << "grid_width: " << grid_width << std::endl
              << "grid_height: " << grid_height << std::endl;

    auto result_mat = std::vector<unsigned int>(1);

    float *host_mat = mat.data();
    float *host_shifted_mat = shifted_mat.data();
    unsigned int *host_result_mat = result_mat.data();

    float *device_mat;
    float *device_shifted_mat;
    unsigned int *device_result_mat;

    std::cout << std::endl
              << "Calculating hierarchical..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    CHECK(cudaMalloc(&device_mat, width * height * sizeof(float)));
    CHECK(cudaMalloc(&device_shifted_mat, width * height * sizeof(float)));
    CHECK(cudaMalloc(&device_result_mat, 1 * sizeof(unsigned int)));

    CHECK(cudaMemcpy(device_mat, host_mat, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_shifted_mat, host_shifted_mat, width * height * sizeof(float), cudaMemcpyHostToDevice));

    // Call kernell, measure time

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaEventRecord(startCUDA, 0);

    check_if_shifted_hierarchical<<<{grid_width, grid_height}, {block_width, block_height}>>>(device_mat, device_shifted_mat, width, height, shift, device_result_mat);

    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    // ==========================

    CHECK(cudaMemcpy(host_result_mat, device_result_mat, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(device_mat));
    CHECK(cudaFree(device_shifted_mat));
    CHECK(cudaFree(device_result_mat));

    // auto check = check_matrix(result_mat, grid_width, grid_height);
    auto check = result_mat[0] == 0;

    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration_total = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // print_matrix(result_mat, grid_width, grid_height);

    std::cout << std::endl
              << "result: " << (check ? "true" : "false") << std::endl
              << "CUDA time: " << elapsedTimeCUDA * 0.001 << " seconds" << std::endl
              << "CUDA memory throughput: "
              << ((sizeof(float) * 2) * width * height) / elapsedTimeCUDA / 1024 / 1024 / 1.024
              << " Gb/s" << std::endl
              << "Total time: " << (double)duration_total.count() << " seconds" << std::endl;
}

void lab3_hiersrchical(int width = 256u, int height = 256u * 1024u)
{
    auto shift = width / 2;

    std::cout << "Filling matrices..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto mat = generate_matrix(width, height);

    auto stop_1 = std::chrono::high_resolution_clock::now();

    auto shifted_mat = generate_matrix(width, height);//cyclic_shift(mat, width, height, shift);

    auto stop_2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_1 - start);
    std::chrono::duration<float> duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start);

    std::cout << "mat: " << duration_1.count() << " seconds" << std::endl
              << "shifted_mat: " << duration_2.count() << " seconds" << std::endl;

    run_for_matrices_hierarchical(mat, shifted_mat, width, height, shift);
}
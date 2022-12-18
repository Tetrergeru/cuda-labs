#include <iostream>
#include <vector>
#include <chrono>

#include "matrices.h"
#include "../cuda_main.h"

#define EPS 0.000001;

__global__ void check_if_shifted(float *mat, float *shifted_mat, int width, int height, uint shift, unsigned char *result_mat)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int idx = y * width + x;

    if (x >= width || y >= height)
        return;

    int shifted_idx = y * width + (x + shift) % width;
    bool value = abs(mat[idx] - shifted_mat[shifted_idx]) < EPS;

    result_mat[idx] = value;
}

void run_for_matrices(std::vector<float> const &mat, std::vector<float> const &shifted_mat, uint width, uint height, uint shift)
{
    auto result_mat = std::vector<unsigned char>(width * height);

    const float *host_mat = mat.data();
    const float *host_shifted_mat = shifted_mat.data();
    unsigned char *host_result_mat = result_mat.data();

    float *device_mat;
    float *device_shifted_mat;
    unsigned char *device_result_mat;

    std::cout << std::endl
              << "Calculating..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    CHECK(cudaMalloc(&device_mat, width * height * sizeof(float)));
    CHECK(cudaMalloc(&device_shifted_mat, width * height * sizeof(float)));
    CHECK(cudaMalloc(&device_result_mat, width * height * sizeof(unsigned char)));

    CHECK(cudaMemcpy(device_mat, host_mat, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_shifted_mat, host_shifted_mat, width * height * sizeof(float), cudaMemcpyHostToDevice));

    auto block_width = 16u;
    auto block_height = 16u;

    auto grid_width = (width + block_width - 1u) / block_width;
    auto grid_height = (height + block_height - 1u) / block_height;

    // Call kernell, measure time

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaEventRecord(startCUDA, 0);

    check_if_shifted<<<{grid_width, grid_height}, {block_width, block_height}>>>(device_mat, device_shifted_mat, width, height, shift, device_result_mat);

    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    // ==========================

    CHECK(cudaMemcpy(host_result_mat, device_result_mat, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(device_mat));
    CHECK(cudaFree(device_shifted_mat));
    CHECK(cudaFree(device_result_mat));

    auto check = check_matrix(result_mat, width, height);

    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration_total = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // print_matrix(result_mat, width, height);

    std::cout << std::endl
              << "result: " << (check ? "true" : "false") << std::endl
              << "CUDA time: " << elapsedTimeCUDA * 0.001 << " seconds" << std::endl
              << "CUDA memory throughput: "
              << ((sizeof(float) * 2 + sizeof(unsigned char)) * width * height) / elapsedTimeCUDA / 1024 / 1024 / 1.024
              << " Gb/s" << std::endl
              << "Total time: " << (double)duration_total.count() << " seconds" << std::endl;
}

void lab_3(int width = 256u, int height = 256u * 1024u)
{
    auto shift = width / 2u;

    std::cout << "Filling matrices..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto mat = generate_matrix(width, height);

    auto stop_1 = std::chrono::high_resolution_clock::now();

    auto shifted_mat = cyclic_shift(mat, width, height, shift);

    auto stop_2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_1 - start);
    std::chrono::duration<float> duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start);

    std::cout << "mat: " << duration_1.count() << " seconds" << std::endl
              << "shifted_mat: " << duration_2.count() << " seconds" << std::endl;

    run_for_matrices(mat, shifted_mat, width, height, shift);
}
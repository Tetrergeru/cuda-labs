#include <iostream>
#include <vector>
#include <chrono>

#include "matrices.h"
#include "../main.h"

#define EPS 0.000001;

__global__ void check_if_shifted(float *mat, float *shifted_mat, int width, int height, uint shift, unsigned char *result_mat)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int idx = y * width + x;

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

    auto start = std::chrono::high_resolution_clock::now();

    CHECK(cudaMalloc(&device_mat, width * height * sizeof(float)));
    CHECK(cudaMalloc(&device_shifted_mat, width * height * sizeof(float)));
    CHECK(cudaMalloc(&device_result_mat, width * height * sizeof(unsigned char)));

    CHECK(cudaMemcpy(device_mat, host_mat, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_shifted_mat, host_shifted_mat, width * height * sizeof(float), cudaMemcpyHostToDevice));

    check_if_shifted<<<{width, height}, 1>>>(device_mat, device_shifted_mat, width, height, shift, device_result_mat);

    CHECK(cudaMemcpy(host_result_mat, device_result_mat, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(device_mat));
    CHECK(cudaFree(device_shifted_mat));
    CHECK(cudaFree(device_result_mat));

    auto check = check_matrix(result_mat, width, height);

    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::chrono::duration<float> duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << check << std::endl;
    std::cout << duration.count() << " seconds" << std::endl;
    std::cout << duration_ms.count() << " microseconds" << std::endl;
}

void lab_3()
{
    auto width = 300u;
    auto height = 30000u;
    auto shift = 100u;

    auto mat = generate_matrix(width, height);
    auto shifted_mat = cyclic_shift(mat, width, height, shift);

    run_for_matrices(mat, shifted_mat, width, height, shift);
}
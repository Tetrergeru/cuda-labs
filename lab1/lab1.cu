#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "../cuda_main.h"

using namespace cv;
using namespace std;

__global__ void swap_simple(uchar *in, uchar *out, int width, int height)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int x = idx % width;
    int y = idx / width;

    if (x >= width || y >= height)
        return;

    int i = (y * width + x) * 3;
    int j = (x * height + y) * 3;

    for (int di = 0; di < 3; di++)
    {
        out[j + di] = in[i + di];
    }
}

void swap_cpu(uchar *in, uchar *out, int width, int height)
{
    for (auto idx = 0; idx < width * height; idx++)
    {
        int i = idx * 3;

        int column = idx % width;
        int row = idx / width;

        int j = (column * height + row) * 3;

        for (auto di = 0; di < 3; di++)
        {
            out[j + di] = in[i + di];
        }
    }
}

Mat LoadImage(char *fname)
{
    Mat image;
    image = imread(fname, IMREAD_COLOR);
    if (!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        throw -1;
    }
    return image;
}

void lab_1()
{
    auto image = LoadImage("pic.jpg");
    auto width = image.cols;
    auto height = image.rows;

    auto result = Mat(width, height, CV_8UC3);

    uchar *host_in = image.ptr();
    uchar *host_out = result.ptr();

    uchar *dev_in;
    uchar *dev_out;

    CHECK(cudaMalloc(&dev_in, width * height * 3))
    CHECK(cudaMalloc(&dev_out, width * height * 3))

    // Call kernell, measure time

    CHECK(cudaMemcpy(dev_in, host_in, width * height * 3, cudaMemcpyHostToDevice));

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaEventRecord(startCUDA, 0);

    auto block_size = 256;
    auto grid_size = (width * height + block_size - 1) / block_size;

    swap_simple<<<grid_size, block_size>>>(dev_in, dev_out, width, height);

    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    CHECK(cudaMemcpy(host_out, dev_out, width * height * 3, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dev_in));
    CHECK(cudaFree(dev_out));

    // swap_cpu(host_in, host_out, width, height);

    imwrite("pic2.jpg", result);

    std::cout << std::endl
              << "CUDA time: " << elapsedTimeCUDA * 0.001 << " seconds" << std::endl
              << "CUDA memory throughput: "
              << (2 * 3 * sizeof(unsigned char) * width * height) / elapsedTimeCUDA / 1024 / 1024 / 1.024
              << " Gb/s" << std::endl;
}

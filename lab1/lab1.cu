#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "../main.h"

using namespace cv;
using namespace std;

__global__ void swap_simple(uchar * in, uchar * out, int width, int height)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i = idx * 3;

    int column = idx % width;
    int row = idx / width;

    int j = (column * height + row) * 3;


    if (i < width * height * 3) {
        for (auto di = 0; di < 3; di++) {
            out[j + di] = in[i + di];
        }
    }
}

void swap_cpu(uchar * in, uchar * out, int width, int height)
{
    for (auto idx = 0; idx < width * height; idx++) {
        int i = idx * 3;

        int column = idx % width;
        int row = idx / width;

        int j = (column * height + row) * 3;

        for (auto di = 0; di < 3; di++) {
            out[j + di] = in[i + di];
        }
    }
}

Mat LoadImage(char * fname)
{
    Mat image;
    image = imread(fname, IMREAD_COLOR);
    if(! image.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
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

    uchar * host_in = image.ptr();
    uchar * host_out = result.ptr();

    uchar * dev_in;
    uchar * dev_out;

    CHECK( cudaMalloc(&dev_in, width * height * 3) )
    CHECK( cudaMalloc(&dev_out, width * height * 3) )

    CHECK( cudaMemcpy(dev_in, host_in, width * height * 3, cudaMemcpyHostToDevice) );

    swap_simple<<<(width * height * 3 + 511) / 512, 512>>>(dev_in, dev_out, width, height);

    CHECK( cudaMemcpy(host_out, dev_out, width * height * 3, cudaMemcpyDeviceToHost) );

    CHECK( cudaFree(dev_in) );
    CHECK( cudaFree(dev_out) );

    // swap_cpu(host_in, host_out, width, height);

    imwrite("pic2.jpg", result);

    cout << width << "  " << height << " -||- " << result.cols << "  " << result.rows << endl;
}


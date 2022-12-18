#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD

#include <CL/opencl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lab6.h"
#include "../cl_main.h"

#include <iostream>
#include <time.h>
#include <string.h>

using namespace cl;
using namespace std;

cv::Mat LoadImage(std::string const &fname)
{
    auto image = cv::imread(fname, cv::IMREAD_COLOR);
    if (!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        throw -1;
    }
    return image;
}

std::string source_string_swaps = R"(
__kernel void swap(__global uchar *in, __global uchar *out, int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int i = (y * width + x) * 3;
    int j = (x * height + y) * 3;

    for (int di = 0; di < 3; di++)
    {
        out[j + di] = in[i + di];
    }
})";

void lab_6_run_for_pointers(unsigned char *host_in, unsigned char *host_out, int width, int height)
{
    cl::size_type buffer_size = width * height * sizeof(unsigned char) * 3;

    auto ctx = cl_ctx();
    auto program = ctx.create_program(source_string_swaps);

    // Buffers

    cl_int errcode;
    auto dev_in = cl::Buffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer_size, host_in, &errcode);
    auto dev_out = cl::Buffer(ctx.context, CL_MEM_READ_WRITE, buffer_size, NULL, &errcode);

    // Args

    auto sum = cl::Kernel(program, "swap");
    sum.setArg(0, dev_in);
    sum.setArg(1, dev_out);
    sum.setArg(2, width);
    sum.setArg(3, height);

    // Running with event
    auto run_event = Event();
    clock_t t0 = clock();

    auto local = 16;

    ctx.queue.enqueueNDRangeKernel(sum, NullRange, NDRange(width, height), NDRange(local, local), NULL, &run_event);
    run_event.wait();

    clock_t t1 = clock();

    auto retrieve_event = Event();
    ctx.queue.enqueueReadBuffer(dev_out, true, 0, buffer_size, host_out, NULL, &retrieve_event);
    retrieve_event.wait();

    // Get profiling info

    cl_ulong time_start, time_end;
    errcode = run_event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start);
    errcode = run_event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end);

    double elapsedTimeGPU;
    if (errcode == CL_PROFILING_INFO_NOT_AVAILABLE)
    {
        cout << "Profiling info is not available" << endl;
        elapsedTimeGPU = (double)(t1 - t0) / CLOCKS_PER_SEC;
    }
    else
    {
        elapsedTimeGPU = (double)(time_end - time_start) / 1e9;
    }

    // Calc throughput

    cout << "GPU swap time = " << elapsedTimeGPU * 1000 << " ms\n";
    cout << "GPU memory throughput = " << buffer_size / elapsedTimeGPU / 1024 / 1024 / 1024 << " Gb/s\n";
}

void lab_6(std::string const &fname)
{
    auto image = LoadImage(fname);
    auto width = image.cols;
    auto height = image.rows;

    auto result = cv::Mat(width, height, CV_8UC3);

    unsigned char *host_in = image.ptr();
    unsigned char *host_out = result.ptr();

    lab_6_run_for_pointers(host_in, host_out, width, height);

    cv::imwrite("pic2.jpg", result);
}
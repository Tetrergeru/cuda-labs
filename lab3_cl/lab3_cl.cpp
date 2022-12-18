#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD

#include <CL/opencl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lab3_cl.h"
#include "../cl_main.h"
#include "matrices.h"

#include <iostream>
#include <time.h>
#include <string>
#include <vector>

using namespace cl;
using namespace std;

std::string source_string_shifts = R"(
__kernel void check_if_shifted(__global float *mat, __global float *shifted_mat, int width, int height, int shift, __global uchar *result_mat)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    int shifted_idx = y * width + (x + shift) % width;
    bool value = fabs(mat[idx] - shifted_mat[shifted_idx]) < 0.0001;

    result_mat[idx] = (uchar)value;
})";

void lab_3_cl_for_matrices(std::vector<float> &mat, std::vector<float> &shifted_mat, uint width, uint height, uint shift)
{
    cl::size_type area = width * height;
    cl::size_type area_float = area * sizeof(float);
    cl::size_type area_char = area * sizeof(unsigned char);
    auto result_mat = std::vector<unsigned char>(area);

    auto ctx = cl_ctx();
    auto program = ctx.create_program(source_string_shifts);

    // Buffers

    cl_int errcode;

    auto dev_mat = cl::Buffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, area_float, mat.data(), &errcode);
    auto dev_shifted_mat = cl::Buffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, area_float, shifted_mat.data(), &errcode);
    auto dev_result_mat = cl::Buffer(ctx.context, CL_MEM_READ_WRITE, area_char, NULL, &errcode);

    // Args

    auto sum = cl::Kernel(program, "check_if_shifted");
    sum.setArg(0, dev_mat);
    sum.setArg(1, dev_shifted_mat);
    sum.setArg(2, width);
    sum.setArg(3, height);
    sum.setArg(4, shift);
    sum.setArg(5, dev_result_mat);

    // Running with event
    auto run_event = Event();
    clock_t t0 = clock();

    auto local_width = 16;
    auto local_height = 16;

    ctx.queue.enqueueNDRangeKernel(sum, NullRange, NDRange(width, height), NDRange(local_width, local_height), NULL, &run_event);
    run_event.wait();

    clock_t t1 = clock();

    ctx.queue.enqueueReadBuffer(dev_result_mat, true, 0, area_char, result_mat.data());

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

    // print_matrix(result_mat, width, height);

    auto check = check_matrix(result_mat, width, height);

    cout << endl
         << "check: " << (check ? "true" : "false") << endl
         << "GPU check shift time = " << elapsedTimeGPU * 1000 << " ms\n"
         << "GPU memory throughput = " << (sizeof(float) * 2 + sizeof(unsigned char)) * area / elapsedTimeGPU / 1024 / 1024 / 1024 << " Gb/s\n";
}

void lab_3_cl(int width, int height)
{
    auto shift = width / 2u;

    std::cout << "Filling matrices..." << std::endl;

    auto mat = generate_matrix(width, height);
    auto shifted_mat = cyclic_shift(mat, width, height, shift);

    lab_3_cl_for_matrices(mat, shifted_mat, width, height, shift);
}
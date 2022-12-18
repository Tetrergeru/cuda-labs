#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD

#include <CL/cl2.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lab6.h"

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

class cl_ctx
{
public:
    std::vector<Device> devices;
    cl::CommandQueue queue;
    cl::Context context;

    int device_index = 0;

    cl_ctx()
    {
        cl_int errcode;
        std::vector<Platform> platform;

        cout << "Searching for platforms..." << endl;
        errcode = Platform::get(&platform);

        cout << "OpenCL platforms found: " << platform.size() << endl;
        cout << "Platform[0] is : " << platform[0].getInfo<CL_PLATFORM_VENDOR>()
             << " ver. " << platform[0].getInfo<CL_PLATFORM_VERSION>() << endl;

        this->devices = std::vector<Device>();
        platform[0].getDevices(CL_DEVICE_TYPE_GPU, &this->devices);

        cout << "GPGPU devices found: " << devices.size() << endl;
        if (devices.size() == 0)
        {
            cout << "Warning: YOU DON'T HAVE GPGPU. Then CPU will be used instead." << endl;
            errcode = platform[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);

            cout << "CPU devices found: " << devices.size() << endl;
            if (devices.size() == 0)
            {
                cout << "Error: CPU devices not found\n";
                exit(-1);
            }
        }
        cout << "Use device " << device_index << ": " << devices[device_index].getInfo<CL_DEVICE_NAME>() << endl;

        cout << "Creating context..." << endl;
        this->context = Context(devices, NULL, NULL, NULL, &errcode);

        cout << "Creating comand queue..." << endl;
        this->queue = CommandQueue(context, devices[device_index], CL_QUEUE_PROFILING_ENABLE, &errcode);
    }

    Program create_program(std::string &src)
    {
        try
        {
            cl_int errcode;

            cout << "Creating program..." << endl;
            auto program = Program(this->context, src, false, &errcode);

            cout << "Compiling program..." << endl;
            errcode = program.build(this->devices, "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable");

            if (errcode != CL_SUCCESS)
            {
                cout << "There were errors during build kernel code. Please, check program code." << endl
                     << "Errcode = " << errcode << endl
                     << "BUILD LOG: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_index]) << endl;
                exit(1);
            }

            return program;
        }
        catch (cl::BuildError e)
        {
            cout << "Build error" << endl
                 << "log:" << endl;

            BuildLogType log = e.getBuildLog();
            for (auto line : log)
                cout << "> " << line.second << endl;

            throw e;
        }
    }
};

std::string sourceString = R"(
__kernel void swap(__global uchar *in, __global uchar *out, int width, int height)
{
    int idx = get_global_id(0);

    int x = idx % width;
    int y = idx / width;

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
    auto program = ctx.create_program(sourceString);

    // Buffers

    cl_int errcode;
    auto dev_in = cl::Buffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer_size, host_in, &errcode);
    auto dev_out = cl::Buffer(ctx.context, CL_MEM_READ_WRITE, buffer_size, NULL, &errcode);

    // Args

    cout << "Creating kernel..." << endl;
    auto sum = cl::Kernel(program, "swap");
    sum.setArg(0, dev_in);
    sum.setArg(1, dev_out);
    sum.setArg(2, width);
    sum.setArg(3, height);

    // Running with event
    cout << "Running..." << endl;
    auto run_event = Event();
    clock_t t0 = clock();

    auto local = 4;

    try
    {
        ctx.queue.enqueueNDRangeKernel(sum, NullRange, NDRange(width * height), NDRange(local), NULL, &run_event);
    }
    catch (cl::Error e)
    {
        cout << "Error id: " << e.err() << endl;
        throw e;
    }
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
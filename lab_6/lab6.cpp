#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD

#include <CL/opencl.hpp>

#include "lab6.h"

#include <iostream>
#include <time.h>
#include <string.h>

using namespace cl;
using namespace std;

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

    void create_buffer()
    {
    }
};

std::string sourceString = R"(
    __kernel void sum(__global float *a, __global float *b, __global float *c, int N)
    {
        int  id = get_global_id(0);
        int threadsNum = get_global_size(0);
        for (int i = id; i < N; i += threadsNum)
            c[i] = a[i]+b[i];
    }
)";

void lab_6_run_for_pointers(float *host_a, float *host_b, float *host_c, int N)
{
    auto ctx = cl_ctx();
    auto program = ctx.create_program(sourceString);

    // Buffers

    cl_int errcode;
    auto dev_a = Buffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(float), host_a, &errcode);
    auto dev_b = Buffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(float), host_b, &errcode);
    auto dev_c = Buffer(ctx.context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &errcode);

    // Args

    auto sum = cl::Kernel(program, "sum");
    sum.setArg(0, dev_a);
    sum.setArg(1, dev_b);
    sum.setArg(2, dev_c);
    sum.setArg(3, N);

    // Running with event
    auto run_event = Event();
    clock_t t0 = clock();

    ctx.queue.enqueueNDRangeKernel(sum, NullRange, NDRange(12 * 1024), NullRange, NULL, &run_event);
    run_event.wait();

    clock_t t1 = clock();

    auto retrieve_event = Event();
    ctx.queue.enqueueReadBuffer(dev_c, true, 0, N * sizeof(float), host_c, NULL, &retrieve_event);
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

    cout << "GPU sum time = " << elapsedTimeGPU * 1000 << " ms\n";
    cout << "GPU memory throughput = " << 3 * N * sizeof(float) / elapsedTimeGPU / 1024 / 1024 / 1024 << " Gb/s\n";
}

void lab_6(std::string const &fname)
{
    int N = 10 * 1000 * 1000;

    auto host_a = new float[N];
    auto host_b = new float[N];
    auto host_c = new float[N];

    for (int i = 0; i < N; i++)
    {
        host_a[i] = i;
        host_b[i] = 2 * i;
    }

    lab_6_run_for_pointers(host_a, host_b, host_c, N);
}
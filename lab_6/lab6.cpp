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
#include <queue>

using namespace cl;
using namespace std;

#define checkError(func)                                                \
    if (errcode != CL_SUCCESS)                                          \
    {                                                                   \
        cout << "Error in " #func "\nError code = " << errcode << "\n"; \
        exit(1);                                                        \
    }

#define checkErrorEx(command) \
    command;                  \
    checkError(command);



void lab_6(std::string const &fname)
{
    int device_index = 0;
    cl_int errcode;

    int N = 10 * 1000 * 1000;

    auto host_a = new float[N];
    auto host_b = new float[N];
    auto host_c = new float[N];
    auto host_c_check = new float[N];

    for (int i = 0; i < N; i++)
    {
        host_a[i] = i;
        host_b[i] = 2 * i;
    }

    auto startCPU = clock();

    for (int i = 0; i < N; i++)
        host_c_check[i] = host_a[i] + host_b[i];

    auto elapsedTimeCPU = (double)(clock() - startCPU) / CLOCKS_PER_SEC;

    // код kernel-функции
    string sourceString = R"(
    __kernel void sum(__global float *a, __global float *b, __global float *c, int N)
    {
        int  id = get_global_id(0);
        int threadsNum = get_global_size(0);
        for (int i = id; i < N; i += threadsNum)
            c[i] = a[i]+b[i];
    })";

    std::vector<Platform> platform;

    cout << "Searching for platforms..." << endl;
    checkErrorEx(errcode = Platform::get(&platform));

    cout << "OpenCL platforms found: " << platform.size() << "\n";
    cout << "Platform[0] is : " << platform[0].getInfo<CL_PLATFORM_VENDOR>() << " ver. " << platform[0].getInfo<CL_PLATFORM_VERSION>() << "\n";

    // в полученном списке платформ находим устройство GPU (видеокарту)
    std::vector<Device> devices;
    platform[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cout << "GPGPU devices found: " << devices.size() << "\n";
    if (devices.size() == 0)
    {
        cout << "Warning: YOU DON'T HAVE GPGPU. Then CPU will be used instead.\n";
        checkErrorEx(errcode = platform[0].getDevices(CL_DEVICE_TYPE_CPU, &devices));
        cout << "CPU devices found: " << devices.size() << "\n";
        if (devices.size() == 0)
        {
            cout << "Error: CPU devices not found\n";
            exit(-1);
        }
    }
    cout << "Use device N " << device_index << ": " << devices[device_index].getInfo<CL_DEVICE_NAME>() << "\n";

    cout << "Creating context..." << endl;
    // создаем контекст на видеокарте
    checkErrorEx(Context context(devices, NULL, NULL, NULL, &errcode));

    cout << "Creating comand queue..." << endl;
    // создаем очередь задач для контекста
    checkErrorEx(CommandQueue queue(context, devices[device_index], CL_QUEUE_PROFILING_ENABLE, &errcode)); // третий параметр - свойства

    cout << "Creating program..." << endl;
    // создаем обьект-программу с заданным текстом программы
    checkErrorEx(auto program = Program(context, sourceString, false, &errcode));
    cout << "Compiling program..." << endl;
    // компилируем и линкуем программу для видеокарты

    try
    {
        errcode = program.build(devices, "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable");
    }
    catch (cl::BuildError e)
    {
        cout << "Build log:" << endl;
        BuildLogType log = e.getBuildLog();
        for (auto line : log)
        {
            cout << "> " << line.second << endl;
        }
    }

    if (errcode != CL_SUCCESS)
    {
        cout << "There were error during build kernel code. Please, check program code. Errcode = " << errcode << "\n";
        cout << "BUILD LOG: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_index]) << "\n";
        return;
    }

    cout << "Creating buffers..." << endl;
    // создаем буфферы в видеопамяти
    checkErrorEx(Buffer dev_a = Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(float), host_a, &errcode));
    checkErrorEx(Buffer dev_b = Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(float), host_b, &errcode));
    checkErrorEx(Buffer dev_c = Buffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &errcode));

    // создаем объект - точку входа GPU-программы
    auto sum = cl::Kernel(program, "sum");
    sum.setArg(0, dev_a);
    sum.setArg(1, dev_b);
    sum.setArg(2, dev_c);
    sum.setArg(3, N);

    Event event;
    // запускаем и ждем
    clock_t t0 = clock();
    queue.enqueueNDRangeKernel(sum, NullRange, NDRange(12 * 1024), NullRange, NULL, &event);
    checkErrorEx(errcode = event.wait());
    clock_t t1 = clock();

    // считаем время
    cl_ulong time_start, time_end;
    errcode = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start);
    errcode = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end);
    double elapsedTimeGPU;
    if (errcode == CL_PROFILING_INFO_NOT_AVAILABLE)
        elapsedTimeGPU = (double)(t1 - t0) / CLOCKS_PER_SEC;
    else
    {
        checkError(event.getEventProfilingInfo);
        elapsedTimeGPU = (double)(time_end - time_start) / 1e9;
    }
    checkErrorEx(errcode = queue.enqueueReadBuffer(dev_c, true, 0, N * sizeof(float), host_c, NULL, NULL));
    // check
    for (int i = 0; i < N; i++)
        if (::abs(host_c[i] - host_c_check[i]) > 1e-6)
        {
            cout << "Error in element N " << i << ": c[i] = " << host_c[i] << " c_check[i] = " << host_c_check[i] << "\n";
            exit(1);
        }
    cout << "CPU sum time = " << elapsedTimeCPU * 1000 << " ms\n";
    cout << "CPU memory throughput = " << 3 * N * sizeof(float) / elapsedTimeCPU / 1024 / 1024 / 1024 << " Gb/s\n";
    cout << "GPU sum time = " << elapsedTimeGPU * 1000 << " ms\n";
    cout << "GPU memory throughput = " << 3 * N * sizeof(float) / elapsedTimeGPU / 1024 / 1024 / 1024 << " Gb/s\n";
}
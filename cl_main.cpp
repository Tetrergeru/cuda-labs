#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>

#include "lab_6/lab6.h"
#include "lab3_cl/lab3_cl.h"
#include "cl_main.h"

int main()
{
    std::cout << "Lab 6: (opencl)" << std::endl;
    lab_6(std::string("pic.jpg"));

    std::cout << std::endl
              << "Lab 3: (opencl)" << std::endl;
    auto width = 2u * 256u;
    auto height = 256u * 1024u;
    lab_3_cl(width, height);

    return 0;
}

cl_ctx::cl_ctx()
{
    cl_int errcode;
    std::vector<cl::Platform> platform;

    std::cout << "Searching for platforms..." << std::endl;
    errcode = cl::Platform::get(&platform);

    std::cout << "OpenCL platforms found: " << platform.size() << std::endl;
    std::cout << "Platform[0] is : " << platform[0].getInfo<CL_PLATFORM_VENDOR>()
              << " ver. " << platform[0].getInfo<CL_PLATFORM_VERSION>() << std::endl;

    this->devices = std::vector<cl::Device>();
    platform[0].getDevices(CL_DEVICE_TYPE_GPU, &this->devices);

    std::cout << "GPGPU devices found: " << devices.size() << std::endl;
    if (devices.size() == 0)
    {
        std::cout << "Warning: YOU DON'T HAVE GPGPU. Then CPU will be used instead." << std::endl;
        errcode = platform[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);

        std::cout << "CPU devices found: " << devices.size() << std::endl;
        if (devices.size() == 0)
        {
            std::cout << "Error: CPU devices not found\n";
            exit(-1);
        }
    }
    std::cout << "Use device " << device_index << ": " << devices[device_index].getInfo<CL_DEVICE_NAME>() << std::endl;

    std::cout << "Creating context..." << std::endl;
    this->context = cl::Context(devices, NULL, NULL, NULL, &errcode);

    std::cout << "Creating comand queue..." << std::endl;
    this->queue = cl::CommandQueue(context, devices[device_index], CL_QUEUE_PROFILING_ENABLE, &errcode);
}

cl::Program cl_ctx::create_program(std::string &src)
{
    try
    {
        cl_int errcode;

        std::cout << "Creating program..." << std::endl;
        auto program = cl::Program(this->context, src, false, &errcode);

        std::cout << "Compiling program..." << std::endl;
        errcode = program.build(this->devices, "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable");

        if (errcode != CL_SUCCESS)
        {
            std::cout << "There were errors during build kernel code. Please, check program code." << std::endl
                      << "Errcode = " << errcode << std::endl
                      << "BUILD LOG: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_index]) << std::endl;
            exit(1);
        }

        return program;
    }
    catch (cl::BuildError e)
    {
        std::cout << "Build error" << std::endl
                  << "log:" << std::endl;

        cl::BuildLogType log = e.getBuildLog();
        for (auto line : log)
            std::cout << "> " << line.second << std::endl;

        throw e;
    }
}

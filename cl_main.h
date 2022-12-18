#ifndef CL_MAIN_H
#define CL_MAIN_H

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD

#include <CL/opencl.hpp>

class cl_ctx
{
public:
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;
    cl::Context context;

    int device_index = 0;

    cl_ctx();

    cl::Program create_program(std::string &src);
};

#endif
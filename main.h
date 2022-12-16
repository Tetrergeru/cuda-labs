#include <iostream>

#define CHECK(value)                                                             \
    {                                                                            \
        cudaError_t _m_cudaStat = value;                                         \
        if (_m_cudaStat != cudaSuccess) {                                        \
            std::cout << "Error:" << cudaGetErrorString(_m_cudaStat)                  \
                 << " at line " << __LINE__ << " in file " << __FILE__ << "\n";  \
            exit(1);                                                             \
        }                                                                        \
    }

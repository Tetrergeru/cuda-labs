#ifndef MATRICES_H
#define MATRICES_H

#include <vector>

std::vector<float> generate_matrix(int width, int height);

std::vector<float> cyclic_shift(std::vector<float> const &vec, int width, int height, unsigned int shift);

bool check_matrix(std::vector<unsigned char> const &vec, int width, int height) ;

template <typename T>
void print_matrix(std::vector<T> const &vec, int width, int height)
{
    for (auto i = 0; i < height; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            std::cout << vec[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

void print_matrix(std::vector<unsigned char> const &vec, int width, int height);

#endif
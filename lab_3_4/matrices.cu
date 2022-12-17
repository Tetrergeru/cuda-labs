#include <vector>
#include <random>
#include <iostream>

std::vector<float> generate_matrix(int width, int height)
{
    std::default_random_engine generator;
    auto vec = std::vector<float>(width * height);

    auto dis = std::uniform_real_distribution<float>();

    for (auto i = 0; i < width * height; i++)
    {
        vec[i] = dis(generator);
    }

    return vec;
}

int positive_mod(int value, int mod)
{
    while (value < 0)
        value += mod;
    return value % mod;
}

std::vector<float> cyclic_shift(std::vector<float> const &vec, int width, int height, unsigned int shift)
{
    auto shifted_vec = std::vector<float>(width * height);
    shift = positive_mod(shift, width);

    for (auto i = 0; i < height; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            shifted_vec[i * width + (j + shift) % width] = vec[i * width + j];
        }
    };

    return shifted_vec;
}

bool check_matrix(std::vector<unsigned char> const &vec, int width, int height)
{
    for (auto i = 0; i < height; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            if (vec[i * width + j] != 1)
                return false;
        }
    }
    return true;
}
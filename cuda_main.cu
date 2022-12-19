#include <iostream>

#include "lab1/lab1.h"
#include "lab_3_4/lab3.h"

int main()
{
    std::cout << "Lab 1: (cuda)" << std::endl;
    lab_1();

    std::cout << "Lab 3: (cuda)" << std::endl;
    auto width = 2u * 256u;
    auto height = 256u * 1024u;

    lab_3(width, height);
    std::cout << std::endl
              << "Lab 3 hierarchical: (cuda)" << std::endl;
    lab3_hiersrchical(width, height);
    return 0;
}
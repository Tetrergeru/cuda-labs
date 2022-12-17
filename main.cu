#include <iostream>

#include "lab1/lab1.h"
#include "lab_3_4/lab3.h"

int main()
{
    auto width = 2u * 256u;
    auto height = 256u * 1024u;

    lab_3(width, height);
    std::cout << std::endl;
    lab3_hiersrchical(width, height);
    return 0;
}
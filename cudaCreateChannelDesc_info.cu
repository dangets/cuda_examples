#include <iostream>
#include <string>

#include <cuda.h>


void printDesc(const std::string &label, const cudaChannelFormatDesc &desc) {
    std::cout << label << " -----------" << std::endl;
    std::cout << "x: " << desc.x << std::endl;
    std::cout << "y: " << desc.y << std::endl;
    std::cout << "z: " << desc.z << std::endl;
    std::cout << "w: " << desc.w << std::endl;
}

int main(int argc, char const *argv[])
{
    cudaChannelFormatDesc desc;

    desc = cudaCreateChannelDesc<float4>();
    printDesc("float4", desc);

    desc = cudaCreateChannelDesc<float>();
    printDesc("float", desc);

    desc = cudaCreateChannelDesc<int>();
    printDesc("int", desc);

    desc = cudaCreateChannelDesc<char>();
    printDesc("char", desc);

    desc = cudaCreateChannelDesc<char4>();
    printDesc("char4", desc);

    return 0;
}

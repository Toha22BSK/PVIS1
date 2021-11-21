#include "pch.h"

DeviceContext DeviceContext::getDefaultGPU()
{
    auto platform = cl::Platform::getDefault();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty())
    {
        throw std::runtime_error("Default gpu device not found");
    }

    std::array<cl_context_properties, 3> props
        = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0 };

    cl::Context context(devices[0], props.data());

    return { context, devices[0] };
}

std::string deviceTypeToStr(cl_device_type type)
{
    std::string res;

    if (type & CL_DEVICE_TYPE_GPU)
    {
        res += "GPU";
    }

    if (type & CL_DEVICE_TYPE_CPU)
    {
        res += "CPU";
    }

    if (res.empty())
    {
        res = std::to_string(type);
    }

    return res;
}

void printOpenCL()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (size_t i = 0; i < platforms.size(); ++i)
    {
        auto name   = platforms[i].getInfo<CL_PLATFORM_NAME>();
        auto vendor = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
        auto ver    = platforms[i].getInfo<CL_PLATFORM_VERSION>();

        std::cout << i << ":" << std::endl;
        std::cout << "  name: " << name << std::endl;
        std::cout << "  vendor: " << vendor << std::endl;
        std::cout << "  version: " << ver << std::endl;

        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        std::cout << "  devices:" << std::endl;
        for (size_t j = 0; j < devices.size(); ++j)
        {
            auto dev_name = devices[j].getInfo<CL_DEVICE_NAME>();
            auto dev_type = devices[j].getInfo<CL_DEVICE_TYPE>();
            auto dev_ver  = devices[j].getInfo<CL_DEVICE_VERSION>();

            std::cout << "  " << j << ":" << std::endl;
            std::cout << "    name: " << dev_name << std::endl;
            std::cout << "    type: " << deviceTypeToStr(dev_type) << std::endl;
            std::cout << "    version: " << dev_ver << std::endl;
        }
    }
}

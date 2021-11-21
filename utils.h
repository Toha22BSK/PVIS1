//
// Created by lev23 on 03.10.2021.
//

#ifndef INC_1_UTILS_H
#define INC_1_UTILS_H

struct DeviceContext
{
    cl::Context context;
    cl::Device device;

    static DeviceContext getDefaultGPU();
};

std::string deviceTypeToStr(cl_device_type type);

void printOpenCL();

#endif //INC_1_UTILS_H

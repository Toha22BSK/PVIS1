#include "pch.h"
#include <string>
#include <sstream>

double spentCpu = 0;
double spentGpu = 0;

std::vector<int> cpuWork(int width, int height, bitmap_image & myImage1, bitmap_image & myImage2)
{
    std::vector<unsigned  char> intensityA(width * height);
    std::vector<unsigned  char> intensityB(width * height);
    std::vector<int> intensityResult (width * height);

    auto startCPUWork = clock();
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            unsigned char R, G, B;
            myImage1.get_pixel(x, y, R, G, B);
            int sum = R;
            sum += G;
            sum += B;
            intensityA[y * width + x] = sum / 3;
        }
    }

    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            unsigned char R, G, B;
            myImage2.get_pixel(x, y, R, G, B);
            int sum = R;
            sum += G;
            sum += B;
            intensityB[y * width + x] = sum / 3;
        }
    }

    for(int y = 1;y < height - 1; ++y)
        for(int x = 1; x < width - 1; ++x)
            intensityResult[y * width + x] = intensityA[y * width + x] + intensityB[y * width + x];

    for (int x = 1; x < width; ++x)
    {
        for (int y = 1; y < height; ++y)
        {
            intensityResult[y * width + x] = intensityResult[y * width + x] * 255 / 510;
        }
    }

    auto endCPUWork = clock();
    spentCpu = ((double) (endCPUWork - startCPUWork)) / CLOCKS_PER_SEC;
    std::cout << "CPU rendering time: " << spentCpu << std::endl;

    return intensityResult;
}

std::vector<int> openCLWork(int width, int height, bitmap_image & myImage1, bitmap_image & myImage2)
{
    auto [context, device] = DeviceContext::getDefaultGPU();

    size_t groupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    size_t imageInputSize  = width * height * 3;

    size_t pixelCount      = groupSize * ((width * height + groupSize - 1) / groupSize);

    std::vector<int> imageOutputData (pixelCount);
    size_t imageOutputSize = imageOutputData.size() * sizeof (int);

    auto buffRgbInfoOne = cl::Buffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  imageInputSize,
                                  (void *)myImage1.data());

    auto buffRgbInfoTwo = cl::Buffer(context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     imageInputSize,
                                     (void *)myImage2.data());

    auto buffIntensityOne = cl::Buffer(context,
                                    CL_MEM_READ_WRITE,
                                    pixelCount);

    auto buffIntensityTwo = cl::Buffer(context,
                                    CL_MEM_READ_WRITE,
                                    pixelCount);

    auto buffIntensityResult = cl::Buffer(context,
                                       CL_MEM_READ_WRITE,
                                       pixelCount);

    auto buffResult = cl::Buffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 imageOutputSize,
                                 imageOutputData.data());


    cl::Program program (context, sobelProgram);
    program.build({ device });

    cl::CommandQueue queue(context, device, 0);

    auto startGPUWork = clock();
    cl::Kernel kernelIntensityOne(program, "calculateIntensityA");
    kernelIntensityOne.setArg(0, buffRgbInfoOne);
    kernelIntensityOne.setArg(1, buffIntensityOne);
    queue.enqueueNDRangeKernel(kernelIntensityOne, cl::NullRange, cl::NDRange(width, height));


    cl::Kernel kernelIntensityTwo(program, "calculateIntensityB");
    kernelIntensityTwo.setArg(0, buffRgbInfoTwo);
    kernelIntensityTwo.setArg(1, buffIntensityTwo);
    queue.enqueueNDRangeKernel(kernelIntensityTwo, cl::NullRange, cl::NDRange(width, height));


    cl::Kernel kernelIntensityOut(program, "calculateIntensityOut");
    kernelIntensityOut.setArg(0, buffIntensityOne);
    kernelIntensityOut.setArg(1, buffIntensityTwo);
    kernelIntensityOut.setArg(2, buffIntensityResult);
    queue.enqueueNDRangeKernel(kernelIntensityOut, cl::NullRange, cl::NDRange(width, height));

    cl::Kernel kernelOverlayHalftone(program, "overlayHalftoneImages");
    kernelOverlayHalftone.setArg(0, buffIntensityResult);
    kernelOverlayHalftone.setArg(1, buffResult);
    queue.enqueueNDRangeKernel(kernelOverlayHalftone, cl::NullRange, cl::NDRange(width, height));

    queue.finish();

    auto endGPUWork = clock();
    spentGpu = ((double) (endGPUWork - startGPUWork)) / CLOCKS_PER_SEC;
    std::cout << "GPU rendering time: " << spentGpu << std::endl;

    queue.enqueueReadBuffer(buffResult, true, 0, imageOutputSize, imageOutputData.data());

    return imageOutputData;
}

void saveImage (int width, int height, std::vector<int> & MR, const std::string & fileName)
{
    bitmap_image outImg(width, height);
    auto * data = const_cast<unsigned char *>(outImg.data());
    for(int i = 0; i < width * height; ++i)
    {
        data[i * 3 + 0] = MR[i];
        data[i * 3 + 1] = MR[i];
        data[i * 3 + 2] = MR[i];
    }

    outImg.save_image(fileName);
}

int main()
{
    const int countImage = 7;
    const int countTrying = 10;
    std::string mapSize[countImage] = {
            "1024", "1280", "2048", "3200", "4000", "6400", "7680"
    };
    spentCpu = 0;
    spentGpu = 0;
    std::string rootPath("./");
    double summCPU[countTrying][countImage] = {};
    double summGPU[countTrying][countImage] = {};

    for(int i = 0; i < countTrying; ++i)
    {
        std::cout << "Program execution: " << i+1 << std::endl;
        for (int j = 0; j < countImage; ++j) {
            bitmap_image myImage1(rootPath + "testImage/imageOne"+mapSize[j]+".bmp");
            bitmap_image myImage2(rootPath + "testImage/imageTwo"+mapSize[j]+".bmp");
            int width = myImage1.width();
            int height = myImage1.height();
            std::cout << "image size: " << width << "px" << " x " << height << "px" << std::endl;

            try {
                auto MR = cpuWork(width, height, myImage1, myImage2);
                saveImage(width, height, MR, rootPath + "resultImage/resultCPU_" + std::to_string(i+1)+ "_" + mapSize[j] +".bmp");
                summCPU[i][j] = spentCpu;

                auto MR2 = openCLWork(width, height, myImage1, myImage2);
                saveImage(width, height, MR2, rootPath + "resultImage/resultGPU_" + std::to_string(i+1)+ "_" + mapSize[j] +".bmp");
                summGPU[i][j] = spentGpu;
            }
            catch (cl::BuildError const &e) {
                std::cerr << "Build OpenCl error: " << e.what() << std::endl;
                for (auto &[dev, log]: e.getBuildLog()) {
                    std::cerr << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
                    std::cerr << log << std::endl;
                }
                return -1;
            }
            catch (cl::Error const &e) {
                std::cerr << "OpenCl exception with code = " << e.err() << " :" << e.what() << std::endl;
                return -1;
            }
            catch (std::exception const &e) {
                std::cerr << e.what() << std::endl;
                return -1;
            }
            std::cout << std::string(30, '-') << "\n" << std::endl;
        }
    }
    for(int i = 0; i < countImage; ++i)
    {
        double meanSpentCPU = 0.0;
        double meanSpentGPU = 0.0;
        for(int j = 0; j < countTrying; ++j)
        {
            meanSpentCPU+= summCPU[j][i];
            meanSpentGPU+= summGPU[j][i];
        }
        std::cout << "Mean spent time of CPU for image size "<< mapSize[i] << ": " << meanSpentCPU / 10 <<std::endl;
        std::cout << "Mean spent time of GPU for image size "<< mapSize[i] <<": " << meanSpentGPU / 10 <<std::endl;
    }
    return 0;
}
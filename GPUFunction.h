//
// Created by lev23 on 09.10.2021.
//

#ifndef INC_1_GPUFUNCTION_H
#define INC_1_GPUFUNCTION_H

#define TOSTRING(...) #__VA_ARGS__

static constexpr const char *sobelProgram = TOSTRING(
        kernel void calculateIntensityA (global unsigned char *image,
                                        global unsigned char *intensity)
        {
            size_t x = get_global_id(0);
            size_t y = get_global_id(1);
            size_t width = get_global_size(0);

            size_t index = y * (width * 3) + x * 3;
            int sum = image[index + 0];
            sum += image[index + 1];
            sum += image[index + 2];

            intensity[y * width + x] = sum / 3;
        }

        kernel void calculateIntensityB (global unsigned char *image,
                                         global unsigned char *intensity)
        {
            size_t x = get_global_id(0);
            size_t y = get_global_id(1);
            size_t width = get_global_size(0);

            size_t index = y * (width * 3) + x * 3;
            int sum = image[index + 0];
            sum += image[index + 1];
            sum += image[index + 2];

            intensity[y * width + x] = sum / 3;
        }

        kernel void calculateIntensityOut (global unsigned char *intensityA, global unsigned char *intensityB,
                                 global int *intensityResult)
        {
            size_t x = get_global_id(0);
            size_t y = get_global_id(1);
            size_t width = get_global_size(0);

            intensityResult[y * width + x] = intensityA[y * width + x] + intensityB[y * width + x];
        }

        kernel void overlayHalftoneImages (global int *intensityResult, global int *buffResult)
        {
            size_t x = get_global_id(0);
            size_t y = get_global_id(1);
            size_t width = get_global_size(0);

            buffResult[y * width + x] = intensityResult[y * width + x] * 255 / 510;

        });

#endif //INC_1_GPUFUNCTION_H

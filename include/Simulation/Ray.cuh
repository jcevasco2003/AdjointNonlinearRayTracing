#pragma once

#include <cuda_runtime.h>

struct Ray {
    double3 position;
    double3 direction;
};
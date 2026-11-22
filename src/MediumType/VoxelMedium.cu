
#include "MediumType/VoxelMedium.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <new>
#include <string>

// ============================================================
// CUDA error handling
// ============================================================

namespace {

void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

constexpr unsigned int BLOCK_SIZE = 256;


// ============================================================
// Device helpers
// ============================================================

struct Cell {
    unsigned int i0, i1;
    unsigned int j0, j1;
    unsigned int k0, k1;

    double u, v, w;
};

struct Weights {
    double value[2][2][2];
    double3 gradient[2][2][2];

    // xx, yy, zz, xy, xz, yz
    double hessian[2][2][2][6];
};

__device__ __forceinline__
std::size_t flatten(
    unsigned int i,
    unsigned int j,
    unsigned int k,
    const uint3& resolution
)
{
    return static_cast<std::size_t>(i) +
        static_cast<std::size_t>(resolution.x) *
        (
            static_cast<std::size_t>(j) +
            static_cast<std::size_t>(resolution.y) * k
        );
}

__device__ __forceinline__
double clampValue(double x, double lo, double hi)
{
    return fmin(fmax(x, lo), hi);
}

__device__ __forceinline__
int zCornerCount(const uint3& resolution)
{
    return resolution.z == 1 ? 1 : 2;
}

__device__ __forceinline__
Cell locateCell(
    const double3& position,
    const double3& origin,
    const double3& size,
    const uint3& resolution
)
{
    const double hx =
        size.x / static_cast<double>(resolution.x - 1);

    const double hy =
        size.y / static_cast<double>(resolution.y - 1);

    const double gx = clampValue(
        (position.x - origin.x) / hx,
        0.0,
        static_cast<double>(resolution.x - 1)
    );

    const double gy = clampValue(
        (position.y - origin.y) / hy,
        0.0,
        static_cast<double>(resolution.y - 1)
    );

    Cell cell{};

    cell.i0 = min(
        static_cast<unsigned int>(floor(gx)),
        resolution.x - 2
    );

    cell.j0 = min(
        static_cast<unsigned int>(floor(gy)),
        resolution.y - 2
    );

    cell.i1 = cell.i0 + 1;
    cell.j1 = cell.j0 + 1;

    cell.u = gx - cell.i0;
    cell.v = gy - cell.j0;

    if (resolution.z == 1) {
        cell.k0 = 0;
        cell.k1 = 0;
        cell.w = 0.0;
    }
    else {
        const double hz =
            size.z / static_cast<double>(resolution.z - 1);

        const double gz = clampValue(
            (position.z - origin.z) / hz,
            0.0,
            static_cast<double>(resolution.z - 1)
        );

        cell.k0 = min(
            static_cast<unsigned int>(floor(gz)),
            resolution.z - 2
        );

        cell.k1 = cell.k0 + 1;
        cell.w = gz - cell.k0;
    }

    return cell;
}

__device__ __forceinline__
Weights computeWeights(
    const Cell& cell,
    const double3& size,
    const uint3& resolution
)
{
    const double hx =
        size.x / static_cast<double>(resolution.x - 1);

    const double hy =
        size.y / static_cast<double>(resolution.y - 1);

    const bool is2D = resolution.z == 1;

    const double hz = is2D
        ? 1.0
        : size.z / static_cast<double>(resolution.z - 1);

    const double U[2] = {
        1.0 - cell.u,
        cell.u
    };

    const double V[2] = {
        1.0 - cell.v,
        cell.v
    };

    const double W[2] = {
        is2D ? 1.0 : 1.0 - cell.w,
        is2D ? 0.0 : cell.w
    };

    const double dU[2] = {
        -1.0 / hx,
         1.0 / hx
    };

    const double dV[2] = {
        -1.0 / hy,
         1.0 / hy
    };

    const double dW[2] = {
        is2D ? 0.0 : -1.0 / hz,
        is2D ? 0.0 :  1.0 / hz
    };

    Weights result{};

    const int nz = zCornerCount(resolution);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                result.value[a][b][c] =
                    U[a] * V[b] * W[c];

                result.gradient[a][b][c] = make_double3(
                    dU[a] * V[b] * W[c],
                    U[a] * dV[b] * W[c],
                    U[a] * V[b] * dW[c]
                );

                // Pure second derivatives vanish inside
                // a bilinear/trilinear interpolation cell.
                result.hessian[a][b][c][0] = 0.0;
                result.hessian[a][b][c][1] = 0.0;
                result.hessian[a][b][c][2] = 0.0;

                result.hessian[a][b][c][3] =
                    dU[a] * dV[b] * W[c];

                result.hessian[a][b][c][4] =
                    dU[a] * V[b] * dW[c];

                result.hessian[a][b][c][5] =
                    U[a] * dV[b] * dW[c];
            }
        }
    }

    return result;
}

__device__ __forceinline__
double dot3(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__
double3 add3(const double3& a, const double3& b)
{
    return make_double3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__device__ __forceinline__
double3 mul3(double s, const double3& v)
{
    return make_double3(
        s * v.x,
        s * v.y,
        s * v.z
    );
}

} // namespace
// ============================================================
// Device construction
// ============================================================

__device__
VoxelMedium::VoxelMedium(
    const double3& origin,
    const double3& size,
    const uint3& resolution,
    double* voxels
)
    : deviceOrigin_(origin),
      deviceSize_(size),
      deviceResolution_(resolution),
      d_voxels_(voxels)
{
}


// ============================================================
// Device interface: Medium
// ============================================================

__device__
bool VoxelMedium::isInside(const double3& x) const
{
    const double lx = x.x - deviceOrigin_.x;
    const double ly = x.y - deviceOrigin_.y;

    if (lx < 0.0 || lx > deviceSize_.x ||
        ly < 0.0 || ly > deviceSize_.y) {
        return false;
    }

    // A 2D medium has no restriction along Z.
    if (deviceResolution_.z == 1) {
        return true;
    }

    const double lz = x.z - deviceOrigin_.z;

    return lz >= 0.0 && lz <= deviceSize_.z;
}

__device__
double VoxelMedium::refractiveIndex(const double3& x) const
{
    if (!isInside(x)) {
        return 1.0;
    }

    const Cell cell = locateCell(
        x,
        deviceOrigin_,
        deviceSize_,
        deviceResolution_
    );

    const Weights weights = computeWeights(
        cell,
        deviceSize_,
        deviceResolution_
    );

    double result = 0.0;

    const int nz = zCornerCount(deviceResolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const unsigned int i = a ? cell.i1 : cell.i0;
                const unsigned int j = b ? cell.j1 : cell.j0;
                const unsigned int k = c ? cell.k1 : cell.k0;

                const std::size_t idx =
                    flatten(i, j, k, deviceResolution_);

                result += weights.value[a][b][c] * d_voxels_[idx];
            }
        }
    }

    return result;
}

__device__
double3 VoxelMedium::gradient(const double3& x) const
{
    const Cell cell = locateCell(
        x,
        deviceOrigin_,
        deviceSize_,
        deviceResolution_
    );

    const Weights weights = computeWeights(
        cell,
        deviceSize_,
        deviceResolution_
    );

    double3 result = make_double3(0.0, 0.0, 0.0);

    const int nz = zCornerCount(deviceResolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const unsigned int i = a ? cell.i1 : cell.i0;
                const unsigned int j = b ? cell.j1 : cell.j0;
                const unsigned int k = c ? cell.k1 : cell.k0;

                const std::size_t idx =
                    flatten(i, j, k, deviceResolution_);

                result = add3(
                    result,
                    mul3(
                        d_voxels_[idx],
                        weights.gradient[a][b][c]
                    )
                );
            }
        }
    }

    return result;
}


__device__
void VoxelMedium::hessian(
    const double3& x,
    double* H
) const
{
    // Row-major:
    // H[0] = xx, H[1] = xy, H[2] = xz
    // H[3] = yx, H[4] = yy, H[5] = yz
    // H[6] = zx, H[7] = zy, H[8] = zz

    for (int i = 0; i < 9; ++i) {
        H[i] = 0.0;
    }

    const Cell cell = locateCell(
        x,
        deviceOrigin_,
        deviceSize_,
        deviceResolution_
    );

    const Weights weights = computeWeights(
        cell,
        deviceSize_,
        deviceResolution_
    );

    const int nz = zCornerCount(deviceResolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const unsigned int i = a ? cell.i1 : cell.i0;
                const unsigned int j = b ? cell.j1 : cell.j0;
                const unsigned int k = c ? cell.k1 : cell.k0;

                const std::size_t idx =
                    flatten(i, j, k, deviceResolution_);

                const double theta = d_voxels_[idx];

                const double dxy =
                    weights.hessian[a][b][c][3] * theta;

                const double dxz =
                    weights.hessian[a][b][c][4] * theta;

                const double dyz =
                    weights.hessian[a][b][c][5] * theta;

                H[1] += dxy;
                H[3] += dxy;

                H[2] += dxz;
                H[6] += dxz;

                H[5] += dyz;
                H[7] += dyz;
            }
        }
    }
}


// ============================================================
// Device interface: OptimizableMedium
// ============================================================

double* VoxelMedium::deviceParameters()
{
    return d_voxels_;
}

__device__
std::size_t VoxelMedium::parameterCount() const
{
    return
        static_cast<std::size_t>(deviceResolution_.x) *
        deviceResolution_.y *
        deviceResolution_.z;
}


__device__
double VoxelMedium::indexParameterGradient(
    const double3& x,
    std::size_t parameterIndex
) const
{
    if (parameterIndex >= parameterCount()) {
        return 0.0;
    }

    const Cell cell = locateCell(
        x,
        deviceOrigin_,
        deviceSize_,
        deviceResolution_
    );

    const Weights weights = computeWeights(
        cell,
        deviceSize_,
        deviceResolution_
    );

    const int nz = zCornerCount(deviceResolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const unsigned int i = a ? cell.i1 : cell.i0;
                const unsigned int j = b ? cell.j1 : cell.j0;
                const unsigned int k = c ? cell.k1 : cell.k0;

                const std::size_t idx =
                    flatten(i, j, k, deviceResolution_);

                if (idx == parameterIndex) {
                    return weights.value[a][b][c];
                }
            }
        }
    }

    return 0.0;
}


__device__
double3 VoxelMedium::gradientParameterGradient(
    const double3& x,
    std::size_t parameterIndex
) const
{
    if (parameterIndex >= parameterCount()) {
        return make_double3(0.0, 0.0, 0.0);
    }

    const Cell cell = locateCell(
        x,
        deviceOrigin_,
        deviceSize_,
        deviceResolution_
    );

    const Weights weights = computeWeights(
        cell,
        deviceSize_,
        deviceResolution_
    );

    const int nz = zCornerCount(deviceResolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const unsigned int i = a ? cell.i1 : cell.i0;
                const unsigned int j = b ? cell.j1 : cell.j0;
                const unsigned int k = c ? cell.k1 : cell.k0;

                const std::size_t idx =
                    flatten(i, j, k, deviceResolution_);

                if (idx == parameterIndex) {
                    return weights.gradient[a][b][c];
                }
            }
        }
    }

    return make_double3(0.0, 0.0, 0.0);
}


// ============================================================
// Optimized parameter gradient accumulation
// ============================================================

__device__
void VoxelMedium::accumulateParameterGradient(
    const double3& x,
    double eta,
    const double3& gradEta,
    const double3& mu,
    double dt,
    double* parameterGradient
) const
{
    const Cell cell = locateCell(
        x,
        deviceOrigin_,
        deviceSize_,
        deviceResolution_
    );

    const Weights weights = computeWeights(
        cell,
        deviceSize_,
        deviceResolution_
    );

    const double muDotGradEta = dot3(mu, gradEta);

    const int nz = zCornerCount(deviceResolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const unsigned int i = a ? cell.i1 : cell.i0;
                const unsigned int j = b ? cell.j1 : cell.j0;
                const unsigned int k = c ? cell.k1 : cell.k0;

                const std::size_t idx =
                    flatten(i, j, k, deviceResolution_);

                const double dEta = weights.value[a][b][c];

                const double3 dGradEta =
                    weights.gradient[a][b][c];

                const double contribution = dt * (
                    muDotGradEta * dEta +
                    eta * dot3(mu, dGradEta)
                );

                atomicAdd(
                    &parameterGradient[idx],
                    contribution
                );
            }
        }
    }
}


// ============================================================
// Device object construction kernel
// ============================================================

__global__
void constructVoxelMediumKernel(
    VoxelMedium* object,
    double3 origin,
    double3 size,
    uint3 resolution,
    double* voxels
)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (object) VoxelMedium(
            origin,
            size,
            resolution,
            voxels
        );
    }
}


// ============================================================
// Grid refinement kernel
// ============================================================

__global__
void refineVoxelKernel(
    const double* oldVoxels,
    uint3 oldResolution,
    double3 origin,
    double3 size,
    double* newVoxels,
    uint3 newResolution
)
{
    const std::size_t id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::size_t newCount =
        static_cast<std::size_t>(newResolution.x) *
        newResolution.y *
        newResolution.z;

    if (id >= newCount) {
        return;
    }

    const unsigned int i =
        id % newResolution.x;

    const unsigned int j =
        (id / newResolution.x) % newResolution.y;

    const unsigned int k =
        id / (
            static_cast<std::size_t>(newResolution.x) *
            newResolution.y
        );

    const double3 position = make_double3(
        origin.x +
            static_cast<double>(i) * size.x /
            static_cast<double>(newResolution.x - 1),

        origin.y +
            static_cast<double>(j) * size.y /
            static_cast<double>(newResolution.y - 1),

        newResolution.z == 1
            ? origin.z
            : origin.z +
                static_cast<double>(k) * size.z /
                static_cast<double>(newResolution.z - 1)
    );

    const Cell cell = locateCell(
        position,
        origin,
        size,
        oldResolution
    );

    const Weights weights = computeWeights(
        cell,
        size,
        oldResolution
    );

    double value = 0.0;

    const int nz = zCornerCount(oldResolution);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const unsigned int oi = a ? cell.i1 : cell.i0;
                const unsigned int oj = b ? cell.j1 : cell.j0;
                const unsigned int ok = c ? cell.k1 : cell.k0;

                const std::size_t oldIndex =
                    flatten(oi, oj, ok, oldResolution);

                value +=
                    weights.value[a][b][c] *
                    oldVoxels[oldIndex];
            }
        }
    }

    newVoxels[id] = value;
}



// ============================================================
// Host construction / destruction
// ============================================================

__global__ void initializeVoxelKernel(
    double* voxels,
    std::size_t count,
    double initialEta
);

VoxelMedium::VoxelMedium(
    const glm::dvec3& origin,
    const glm::dvec3& size,
    const glm::uvec3& resolution,
    double initialEta
)
    : origin_(origin),
      size_(size),
      resolution_(resolution)
{
    if (resolution_.x < 2 ||
        resolution_.y < 2 ||
        resolution_.z < 1) {
        throw std::invalid_argument(
            "VoxelMedium requires Nx >= 2, Ny >= 2, Nz >= 1"
        );
    }

    if (size_.x <= 0.0 || size_.y <= 0.0) {
        throw std::invalid_argument(
            "VoxelMedium X and Y sizes must be positive"
        );
    }

    if (resolution_.z >= 2 && size_.z <= 0.0) {
        throw std::invalid_argument(
            "VoxelMedium Z size must be positive in 3D"
        );
    }

    if (resolution_.z == 1 && size_.z < 0.0) {
        throw std::invalid_argument(
            "VoxelMedium Z size cannot be negative in 2D"
        );
    }

    try {
        allocateDeviceMemory();

        const std::size_t count = parameterCountHost();

        // Initialize the voxel field directly on GPU.
        checkCuda(
            cudaMemset(d_voxels_, 0, count * sizeof(double)),
            "cudaMemset VoxelMedium"
        );

        // Fill with initialEta.
        // A small initialization kernel is used below.
        initializeVoxelKernel<<<
            static_cast<unsigned int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE),
            BLOCK_SIZE
        >>>(d_voxels_, count, initialEta);

        checkCuda(cudaGetLastError(), "initializeVoxelKernel");
        checkCuda(cudaDeviceSynchronize(), "initializeVoxelKernel sync");

        createDeviceObject();
    }
    catch (...) {
        releaseDeviceMemory();
        throw;
    }
}


// Initializes all voxel parameters.
__global__
void initializeVoxelKernel(
    double* voxels,
    std::size_t count,
    double initialEta
)
{
    const std::size_t id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (id < count) {
        voxels[id] = initialEta;
    }
}


VoxelMedium::~VoxelMedium()
{
    destroyDeviceObject();
    releaseDeviceMemory();
}


// ============================================================
// Move semantics
// ============================================================

VoxelMedium::VoxelMedium(VoxelMedium&& other) noexcept
    : origin_(other.origin_),
      size_(other.size_),
      resolution_(other.resolution_),
      d_voxels_(other.d_voxels_),
      d_self_(other.d_self_)
{
    other.d_voxels_ = nullptr;
    other.d_self_ = nullptr;
}


VoxelMedium& VoxelMedium::operator=(VoxelMedium&& other) noexcept
{
    if (this != &other) {
        destroyDeviceObject();
        releaseDeviceMemory();

        origin_ = other.origin_;
        size_ = other.size_;
        resolution_ = other.resolution_;

        d_voxels_ = other.d_voxels_;
        d_self_ = other.d_self_;

        other.d_voxels_ = nullptr;
        other.d_self_ = nullptr;
    }

    return *this;
}


// ============================================================
// Host memory management
// ============================================================

void VoxelMedium::allocateDeviceMemory()
{
    const std::size_t count = parameterCountHost();

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_voxels_),
            count * sizeof(double)
        ),
        "cudaMalloc voxel parameters"
    );
}


void VoxelMedium::releaseDeviceMemory()
{
    if (d_voxels_) {
        cudaFree(d_voxels_);
        d_voxels_ = nullptr;
    }
}


void VoxelMedium::createDeviceObject()
{
    if (d_self_) {
        destroyDeviceObject();
    }

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_self_),
            sizeof(VoxelMedium)
        ),
        "cudaMalloc VoxelMedium object"
    );

    const double3 origin = make_double3(
        origin_.x, origin_.y, origin_.z
    );

    const double3 size = make_double3(
        size_.x, size_.y, size_.z
    );

    const uint3 resolution = make_uint3(
        resolution_.x, resolution_.y, resolution_.z
    );

    constructVoxelMediumKernel<<<1, 1>>>(
        d_self_,
        origin,
        size,
        resolution,
        d_voxels_
    );

    checkCuda(cudaGetLastError(), "constructVoxelMediumKernel");
    checkCuda(cudaDeviceSynchronize(), "constructVoxelMediumKernel sync");
}


void VoxelMedium::destroyDeviceObject()
{
    if (d_self_) {
        // The device object owns no allocations.
        // Its storage can be released directly.
        cudaFree(d_self_);
        d_self_ = nullptr;
    }
}


VoxelMedium* VoxelMedium::deviceObject() const
{
    return d_self_;
}


// ============================================================
// Host parameter transfers
// ============================================================

std::vector<double> VoxelMedium::downloadParameters() const
{
    const std::size_t count = parameterCountHost();

    std::vector<double> result(count);

    checkCuda(
        cudaMemcpy(
            result.data(),
            d_voxels_,
            count * sizeof(double),
            cudaMemcpyDeviceToHost
        ),
        "downloadParameters"
    );

    return result;
}


void VoxelMedium::uploadParameters(
    const std::vector<double>& parameters
)
{
    if (parameters.size() != parameterCountHost()) {
        throw std::invalid_argument(
            "VoxelMedium::uploadParameters: incorrect parameter count"
        );
    }

    checkCuda(
        cudaMemcpy(
            d_voxels_,
            parameters.data(),
            parameters.size() * sizeof(double),
            cudaMemcpyHostToDevice
        ),
        "uploadParameters"
    );
}


// ============================================================
// Grid management
// ============================================================

glm::uvec3 VoxelMedium::resolution() const
{
    return resolution_;
}


glm::dvec3 VoxelMedium::voxelSize() const
{
    const double hz = resolution_.z == 1
        ? 0.0
        : size_.z / static_cast<double>(resolution_.z - 1);

    return glm::dvec3(
        size_.x / static_cast<double>(resolution_.x - 1),
        size_.y / static_cast<double>(resolution_.y - 1),
        hz
    );
}


std::size_t VoxelMedium::parameterCountHost() const
{
    return
        static_cast<std::size_t>(resolution_.x) *
        resolution_.y *
        resolution_.z;
}


void VoxelMedium::refine(
    const glm::uvec3& newResolution
)
{
    if (newResolution.x < 2 ||
        newResolution.y < 2 ||
        newResolution.z < 1) {
        throw std::invalid_argument(
            "VoxelMedium::refine: invalid resolution"
        );
    }

    if (newResolution.x < resolution_.x ||
        newResolution.y < resolution_.y ||
        newResolution.z < resolution_.z) {
        throw std::invalid_argument(
            "VoxelMedium::refine: resolution cannot decrease"
        );
    }

    if ((resolution_.z == 1) != (newResolution.z == 1)) {
        throw std::invalid_argument(
            "VoxelMedium::refine: cannot change dimensionality"
        );
    }

    if (newResolution == resolution_) {
        return;
    }

    const uint3 oldRes = make_uint3(
        resolution_.x, resolution_.y, resolution_.z
    );

    const uint3 newRes = make_uint3(
        newResolution.x, newResolution.y, newResolution.z
    );

    const double3 origin = make_double3(
        origin_.x, origin_.y, origin_.z
    );

    const double3 size = make_double3(
        size_.x, size_.y, size_.z
    );

    const std::size_t newCount =
        static_cast<std::size_t>(newResolution.x) *
        newResolution.y *
        newResolution.z;

    double* newVoxels = nullptr;

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&newVoxels),
            newCount * sizeof(double)
        ),
        "cudaMalloc refined voxels"
    );

    try {
        refineVoxelKernel<<<
            static_cast<unsigned int>((newCount + BLOCK_SIZE - 1) / BLOCK_SIZE),
            BLOCK_SIZE
        >>>(
            d_voxels_,
            oldRes,
            origin,
            size,
            newVoxels,
            newRes
        );

        checkCuda(cudaGetLastError(), "refineVoxelKernel");
        checkCuda(cudaDeviceSynchronize(), "refineVoxelKernel sync");
    }
    catch (...) {
        cudaFree(newVoxels);
        throw;
    }

    // Replace the old GPU field.
    destroyDeviceObject();

    checkCuda(cudaFree(d_voxels_), "free old voxel parameters");

    d_voxels_ = newVoxels;
    resolution_ = newResolution;

    createDeviceObject();
}

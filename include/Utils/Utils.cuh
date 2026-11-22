// Utils.cuh
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <glm/glm.hpp>

// ============================================================
// CUDA error handling
// ============================================================

inline void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

inline void checkKernel(const char* operation)
{
    checkCuda(cudaGetLastError(), operation);
}

// ============================================================
// Constants and general helpers
// ============================================================

constexpr unsigned int BLOCK_SIZE = 256;

inline unsigned int blocksFor(std::size_t count)
{
    return static_cast<unsigned int>(
        (count + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
}

inline double3 toDouble3(const glm::dvec3& v)
{
    return make_double3(v.x, v.y, v.z);
}

inline glm::dvec3 toGlm(const double3& v)
{
    return glm::dvec3(v.x, v.y, v.z);
}

inline double dot3(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ============================================================
// Small CUDA buffer
// ============================================================

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count)
    {
        allocate(count);
    }

    ~DeviceBuffer()
    {
        if (ptr_)
            cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            if (ptr_)
                cudaFree(ptr_);

            ptr_ = other.ptr_;
            count_ = other.count_;

            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        return *this;
    }

    void allocate(std::size_t count)
    {
        if (ptr_)
            checkCuda(cudaFree(ptr_), "cudaFree");

        ptr_ = nullptr;
        count_ = count;

        if (count > 0) {
            checkCuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&ptr_),
                    count * sizeof(T)
                ),
                "cudaMalloc"
            );
        }
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

    std::size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

// ============================================================
// Geometry helpers
// ============================================================

__host__ __device__
inline double3 add3(const double3& a, const double3& b)
{
    return make_double3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__host__ __device__
inline double3 sub3(const double3& a, const double3& b)
{
    return make_double3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

__host__ __device__
inline double3 mul3(const double3& a, double s)
{
    return make_double3(
        a.x * s,
        a.y * s,
        a.z * s
    );
}

__host__ __device__
inline double normSquared3(const double3& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}
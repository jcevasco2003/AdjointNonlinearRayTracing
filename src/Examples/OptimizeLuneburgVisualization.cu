
#include "MediumType/VoxelMedium.cuh"
#include "Simulation/Medium.cuh"
#include "MediumType/LuneburgLens.h"
#include "Simulation/Ray.cuh"
#include "Simulation/Tracer.cuh"
#include "Utils/Utils.cuh"

#include <cuda_runtime.h>

#include <matplot/matplot.h>

#include <filesystem>
#include <vector>
#include <string>
#include <cstddef>

// ============================================================
// Kernel: sample XY refractive-index field
// ============================================================


#include "MediumType/VoxelMedium.cuh"
#include "MediumType/LuneburgLens.h"
#include "Utils/Utils.cuh"

#include <matplot/matplot.h>

#include <filesystem>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstddef>

const std::vector<double> visualizationOffsets = {
    -2.0 / 3.0,
    -1.0 / 3.0,
     0.0,
     1.0 / 3.0,
     2.0 / 3.0
};

// ============================================================
// CPU voxel interpolation
// ============================================================

static double sampleVoxelMedium(
    const std::vector<double>& parameters,
    const glm::dvec3& origin,
    const glm::dvec3& size,
    const glm::uvec3& resolution,
    const glm::dvec3& position
)
{
    const unsigned int nx = resolution.x;
    const unsigned int ny = resolution.y;
    const unsigned int nz = resolution.z;

    if (nx < 2 || ny < 2 || nz < 2)
        throw std::invalid_argument("Voxel resolution must be >= 2");

    // Outside the voxel domain.
    if (
        position.x < origin.x ||
        position.y < origin.y ||
        position.z < origin.z ||
        position.x > origin.x + size.x ||
        position.y > origin.y + size.y ||
        position.z > origin.z + size.z
    ) {
        return 1.0;
    }

    // Grid spacing.
    const double hx = size.x / static_cast<double>(nx - 1);
    const double hy = size.y / static_cast<double>(ny - 1);
    const double hz = size.z / static_cast<double>(nz - 1);

    // Continuous grid coordinates.
    const double gx = (position.x - origin.x) / hx;
    const double gy = (position.y - origin.y) / hy;
    const double gz = (position.z - origin.z) / hz;

    // Lower voxel indices.
    const unsigned int i0 = std::min(
        static_cast<unsigned int>(std::floor(gx)),
        nx - 2
    );

    const unsigned int j0 = std::min(
        static_cast<unsigned int>(std::floor(gy)),
        ny - 2
    );

    const unsigned int k0 = std::min(
        static_cast<unsigned int>(std::floor(gz)),
        nz - 2
    );

    const unsigned int i1 = i0 + 1;
    const unsigned int j1 = j0 + 1;
    const unsigned int k1 = k0 + 1;

    // Local interpolation coordinates.
    const double tx = gx - static_cast<double>(i0);
    const double ty = gy - static_cast<double>(j0);
    const double tz = gz - static_cast<double>(k0);

    // Linear storage: x is the fastest-changing coordinate.
    const auto index = [nx, ny](
        unsigned int i,
        unsigned int j,
        unsigned int k
    ) -> std::size_t {
        return static_cast<std::size_t>(i)
             + static_cast<std::size_t>(nx) * j
             + static_cast<std::size_t>(nx) * ny * k;
    };

    // Eight surrounding voxel values.
    const double c000 = parameters[index(i0, j0, k0)];
    const double c100 = parameters[index(i1, j0, k0)];
    const double c010 = parameters[index(i0, j1, k0)];
    const double c110 = parameters[index(i1, j1, k0)];

    const double c001 = parameters[index(i0, j0, k1)];
    const double c101 = parameters[index(i1, j0, k1)];
    const double c011 = parameters[index(i0, j1, k1)];
    const double c111 = parameters[index(i1, j1, k1)];

    // Trilinear interpolation.
    const double c00 = c000 * (1.0 - tx) + c100 * tx;
    const double c10 = c010 * (1.0 - tx) + c110 * tx;

    const double c01 = c001 * (1.0 - tx) + c101 * tx;
    const double c11 = c011 * (1.0 - tx) + c111 * tx;

    const double c0 = c00 * (1.0 - ty) + c10 * ty;
    const double c1 = c01 * (1.0 - ty) + c11 * ty;

    return c0 * (1.0 - tz) + c1 * tz;
}

// ============================================================
// Save forward visualization (CPU)
// ============================================================

void saveForwardVisualization(
    const VoxelMedium& medium,
    const LuneburgLens& idealLens,
    const Tracer& tracer,
    const std::filesystem::path& outputPath,
    int iteration,
    int fieldResolution,
    cudaStream_t stream
)
{
    constexpr int LINE_RESOLUTION = 300;
    constexpr int CIRCLE_RESOLUTION = 300;

    constexpr double radius = 1.0;

    const double xMin = -1.0;
    const double xMax =  1.0;
    const double yMin = -1.0;
    const double yMax =  1.0;

    constexpr double z = 0.0;
    constexpr double rayStartEpsilon = 1e-5;

    if (fieldResolution < 2)
        throw std::invalid_argument(
            "fieldResolution must be >= 2"
        );

    // --------------------------------------------------------
    // Download optimized parameters
    // --------------------------------------------------------

    const std::vector<double> parameters =
        medium.downloadParameters();

    const glm::uvec3 resolution =
        medium.resolution();

    const glm::dvec3 voxelSpacing =
        medium.voxelSize();

    // voxelSize() is the spacing between samples.
    const glm::dvec3 size(
        voxelSpacing.x * (resolution.x - 1),
        voxelSpacing.y * (resolution.y - 1),
        voxelSpacing.z * (resolution.z - 1)
    );

    // Grid centered at the origin
    const glm::dvec3 origin = -0.5 * size;

    const std::size_t expectedCount =
        static_cast<std::size_t>(resolution.x) *
        resolution.y *
        resolution.z;

    if (parameters.size() != expectedCount)
    {
        throw std::runtime_error(
            "Voxel parameter count does not match grid resolution"
        );
    }

    // --------------------------------------------------------
    // Sample XY field
    // --------------------------------------------------------

    std::vector<double> field(
        static_cast<std::size_t>(fieldResolution) *
        fieldResolution
    );

    for (int j = 0; j < fieldResolution; ++j)
    {
        const double ty =
            static_cast<double>(j) /
            (fieldResolution - 1);

        const double y = yMin + ty * (yMax - yMin);

        for (int i = 0; i < fieldResolution; ++i)
        {
            const double tx =
                static_cast<double>(i) /
                (fieldResolution - 1);

            const double x = xMin + tx * (xMax - xMin);

            field[
                static_cast<std::size_t>(j) *
                fieldResolution + i
            ] = sampleVoxelMedium(
                parameters,
                origin,
                size,
                resolution,
                glm::dvec3(x, y, z)
            );
        }
    }

    // --------------------------------------------------------
    // Sample centerline
    // --------------------------------------------------------

    std::vector<double> xs(LINE_RESOLUTION);
    std::vector<double> optimizedLine(LINE_RESOLUTION);
    std::vector<double> idealLine(LINE_RESOLUTION);

    for (int i = 0; i < LINE_RESOLUTION; ++i)
    {
        const double t =
            static_cast<double>(i) /
            (LINE_RESOLUTION - 1);

        const double x = xMin + t * (xMax - xMin);

        xs[i] = x;

        optimizedLine[i] = sampleVoxelMedium(
            parameters,
            origin,
            size,
            resolution,
            glm::dvec3(x, 0.0, 0.0)
        );

        idealLine[i] = idealLens.refractiveIndex(
            glm::dvec3(x, 0.0, 0.0)
        );
    }

    // --------------------------------------------------------
    // Convert field to matrix for Matplot++
    // --------------------------------------------------------

    std::vector<std::vector<double>> image(
        fieldResolution,
        std::vector<double>(fieldResolution)
    );

    for (int j = 0; j < fieldResolution; ++j)
    {
        for (int i = 0; i < fieldResolution; ++i)
        {
            image[j][i] =
                field[
                    static_cast<std::size_t>(j) *
                    fieldResolution + i
                ];
        }
    }

    // --------------------------------------------------------
    // Generate five visualization rays
    // --------------------------------------------------------

    const std::vector<double> visualizationOffsets = {
        -2.0 / 3.0,
        -1.0 / 3.0,
         0.0,
         1.0 / 3.0,
         2.0 / 3.0
    };

    constexpr std::size_t visualizationRayCount = 5;

    std::vector<Ray> raysHost;
    raysHost.reserve(visualizationRayCount);

    for (double offset : visualizationOffsets)
    {
        const double y = offset * radius;

        raysHost.push_back({
            make_double3(
                xMin + rayStartEpsilon,
                y,
                0.0
            ),
            make_double3(1.0, 0.0, 0.0)
        });
    }

    // --------------------------------------------------------
    // Allocate GPU trajectory buffers
    // --------------------------------------------------------

    const std::size_t stride =
        tracer.trajectoryStride();

    const std::size_t trajectoryCount =
        visualizationRayCount * stride;

    Ray* d_rays = nullptr;
    DeviceTraceResult* d_results = nullptr;
    double3* d_trajectories = nullptr;

    auto checkCuda = [](cudaError_t error, const char* operation)
    {
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(operation) + ": " +
                cudaGetErrorString(error)
            );
        }
    };

    std::vector<double3> trajectoriesHost;
    std::vector<DeviceTraceResult> resultsHost;

    try
    {
        checkCuda(
            cudaMalloc(
                &d_rays,
                visualizationRayCount * sizeof(Ray)
            ),
            "cudaMalloc d_rays"
        );

        checkCuda(
            cudaMalloc(
                &d_results,
                visualizationRayCount *
                    sizeof(DeviceTraceResult)
            ),
            "cudaMalloc d_results"
        );

        checkCuda(
            cudaMalloc(
                &d_trajectories,
                trajectoryCount * sizeof(double3)
            ),
            "cudaMalloc d_trajectories"
        );

        checkCuda(
            cudaMemcpyAsync(
                d_rays,
                raysHost.data(),
                visualizationRayCount * sizeof(Ray),
                cudaMemcpyHostToDevice,
                stream
            ),
            "Copy visualization rays to device"
        );

        // ----------------------------------------------------
        // Forward with trajectory
        // ----------------------------------------------------

        const Medium* d_medium =
            static_cast<const Medium*>(
                medium.deviceObject()
            );

        tracer.forwardWithTrajectory(
            d_rays,
            d_results,
            d_trajectories,
            d_medium,
            visualizationRayCount,
            stream
        );

        // ----------------------------------------------------
        // Download trajectories
        // ----------------------------------------------------

        trajectoriesHost.resize(trajectoryCount);

        resultsHost.resize(visualizationRayCount);

        checkCuda(
            cudaMemcpyAsync(
                trajectoriesHost.data(),
                d_trajectories,
                trajectoryCount * sizeof(double3),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "Download trajectories"
        );

        checkCuda(
            cudaMemcpyAsync(
                resultsHost.data(),
                d_results,
                visualizationRayCount *
                    sizeof(DeviceTraceResult),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "Download trace results"
        );

        checkCuda(
            cudaStreamSynchronize(stream),
            "Synchronize visualization tracing"
        );
    }
    catch (...)
    {
        if (d_rays)
            cudaFree(d_rays);

        if (d_results)
            cudaFree(d_results);

        if (d_trajectories)
            cudaFree(d_trajectories);

        throw;
    }

    // --------------------------------------------------------
    // Free GPU memory
    // --------------------------------------------------------

    cudaFree(d_rays);
    cudaFree(d_results);
    cudaFree(d_trajectories);

    // --------------------------------------------------------
    // Circle: intersection of sphere with z = 0
    // --------------------------------------------------------

    std::vector<double> circleX(CIRCLE_RESOLUTION + 1);
    std::vector<double> circleY(CIRCLE_RESOLUTION + 1);

    for (int i = 0; i <= CIRCLE_RESOLUTION; ++i)
    {
        const double theta =
            2.0 * 3.14159265358979323846 *
            static_cast<double>(i) /
            CIRCLE_RESOLUTION;

        circleX[i] = radius * std::cos(theta);
        circleY[i] = radius * std::sin(theta);
    }

    // --------------------------------------------------------
    // Plot
    // --------------------------------------------------------

    using namespace matplot;

    auto fig = figure(true);
    fig->size(1500, 650);

    // ========================================================
    // Subplot 1: Refractive-index field + ray trajectories
    // ========================================================

    subplot(1, 2, 1);

    imagesc(
        xMin,
        xMax,
        yMin,
        yMax,
        image
    );

    hold(on);

    colorbar();

    title("Refractive index field, z = 0");

    xlabel("x");
    ylabel("y");

    // Lens boundary
    plot(
        circleX,
        circleY,
        "k-"
    )->line_width(2);

    // Five traced rays
    for (std::size_t r = 0; r < visualizationRayCount; ++r)
    {
        const std::size_t offset = r * stride;

        const std::size_t pointCount =
            static_cast<std::size_t>(
                resultsHost[r].stepsTaken
            ) + 1;

        std::vector<double> pathX;
        std::vector<double> pathY;

        pathX.reserve(pointCount + 1);
        pathY.reserve(pointCount + 1);

        // Draw ray from the edge of the plotting domain
        pathX.push_back(xMin);
        pathY.push_back(raysHost[r].position.y);

        for (std::size_t s = 0; s < pointCount; ++s)
        {
            const double3 p =
                trajectoriesHost[offset + s];

            pathX.push_back(p.x);
            pathY.push_back(p.y);
        }

        plot(
            pathX,
            pathY,
            "r-"
        )->line_width(1.5);
    }

    xlim({xMin, xMax});
    ylim({yMin, yMax});

    axis(equal);
    grid(on);

    hold(off);

    // ========================================================
    // Subplot 2: Centerline refractive-index profile
    // ========================================================

    subplot(1, 2, 2);

    plot(
        xs,
        optimizedLine,
        "b-"
    )->line_width(2);

    hold(on);

    plot(
        xs,
        idealLine,
        "r--"
    )->line_width(2);

    hold(off);

    title("Centerline refractive-index profile");

    xlabel("x");
    ylabel("Refractive index");

    legend({
        "Optimized medium",
        "Ideal Luneburg profile"
    });

    grid(on);

    // --------------------------------------------------------
    // Figure title and save
    // --------------------------------------------------------

    fig->title(
        "Luneburg optimization - iteration " +
        std::to_string(iteration)
    );

    fig->save(outputPath.string());
}

/*
void saveForwardVisualization(
    const VoxelMedium& medium,
    const LuneburgLens& idealLens,
    const std::filesystem::path& outputPath,
    int iteration,
    int fieldResolution,
    cudaStream_t stream
)
{
    (void)stream; // No CUDA kernels are launched here.

    constexpr int LINE_RESOLUTION = 300;

    const double xMin = -1.0;
    const double xMax =  1.0;
    const double yMin = -1.0;
    const double yMax =  1.0;
    constexpr double z = 0.0;

    if (fieldResolution < 2)
        throw std::invalid_argument(
            "fieldResolution must be >= 2"
        );

    // --------------------------------------------------------
    // Download optimized parameters
    // --------------------------------------------------------

    const std::vector<double> parameters =
        medium.downloadParameters();

    const glm::uvec3 resolution = medium.resolution();

    const glm::dvec3 voxelSize = medium.voxelSize();

    // NOTE:
    // This assumes that the voxel grid is centered at the origin.
    // Adjust origin if your VoxelMedium uses a different convention.
    const glm::dvec3 origin = -0.5 * voxelSize;

    const std::size_t expectedCount =
        static_cast<std::size_t>(resolution.x) *
        resolution.y *
        resolution.z;

    if (parameters.size() != expectedCount) {
        throw std::runtime_error(
            "Voxel parameter count does not match grid resolution"
        );
    }

    // --------------------------------------------------------
    // Sample XY field
    // --------------------------------------------------------

    std::vector<double> field(
        static_cast<std::size_t>(fieldResolution) *
        fieldResolution
    );

    for (int j = 0; j < fieldResolution; ++j) {
        const double ty =
            static_cast<double>(j) / (fieldResolution - 1);

        const double y = yMin + ty * (yMax - yMin);

        for (int i = 0; i < fieldResolution; ++i) {
            const double tx =
                static_cast<double>(i) / (fieldResolution - 1);

            const double x = xMin + tx * (xMax - xMin);

            field[
                static_cast<std::size_t>(j) *
                fieldResolution + i
            ] = sampleVoxelMedium(
                parameters,
                origin,
                voxelSize,
                resolution,
                glm::dvec3(x, y, z)
            );
        }
    }

    // --------------------------------------------------------
    // Sample centerline
    // --------------------------------------------------------

    std::vector<double> xs(LINE_RESOLUTION);
    std::vector<double> optimizedLine(LINE_RESOLUTION);
    std::vector<double> idealLine(LINE_RESOLUTION);

    for (int i = 0; i < LINE_RESOLUTION; ++i) {
        const double t =
            static_cast<double>(i) /
            (LINE_RESOLUTION - 1);

        const double x = xMin + t * (xMax - xMin);

        xs[i] = x;

        optimizedLine[i] = sampleVoxelMedium(
            parameters,
            origin,
            voxelSize,
            resolution,
            glm::dvec3(x, 0.0, 0.0)
        );

        idealLine[i] = idealLens.refractiveIndex(
            glm::dvec3(x, 0.0, 0.0)
        );
    }

    // --------------------------------------------------------
    // Convert field to matrix
    // --------------------------------------------------------

    std::vector<std::vector<double>> image(
        fieldResolution,
        std::vector<double>(fieldResolution)
    );

    for (int j = 0; j < fieldResolution; ++j) {
        for (int i = 0; i < fieldResolution; ++i) {
            image[j][i] =
                field[
                    static_cast<std::size_t>(j) *
                    fieldResolution + i
                ];
        }
    }

    // --------------------------------------------------------
    // Plot
    // --------------------------------------------------------

    using namespace matplot;

    auto fig = figure(true);
    fig->size(1500, 650);

    subplot(1, 2, 1);

    imagesc(
        xMin,
        xMax,
        yMin,
        yMax,
        image
    );

    colorbar();

    title("Refractive index field, z = 0");

    xlabel("x");
    ylabel("y");

    axis(equal);

    subplot(1, 2, 2);

    plot(
        xs,
        optimizedLine,
        "b-"
    )->line_width(2);

    hold(on);

    plot(
        xs,
        idealLine,
        "r--"
    )->line_width(2);

    hold(off);

    title("Centerline refractive-index profile");

    xlabel("x");
    ylabel("Refractive index");

    legend({
        "Optimized medium",
        "Ideal Luneburg profile"
    });

    grid(on);

    fig->title(
        "Luneburg optimization - iteration " +
        std::to_string(iteration)
    );

    fig->save(outputPath.string());
}
*/


/*
__global__
void sampleXYKernel(
    const Medium* medium,
    double xMin,
    double xMax,
    double yMin,
    double yMax,
    double z,
    int resolution,
    double* output)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = resolution * resolution;

    if (idx >= count)
        return;

    const int i = idx % resolution;
    const int j = idx / resolution;

    const double x = xMin + (xMax - xMin) *
        static_cast<double>(i) / (resolution - 1);

    const double y = yMin + (yMax - yMin) *
        static_cast<double>(j) / (resolution - 1);

    const double3 p = make_double3(x, y, z);

    if (!medium->isInside(p)) {
        output[idx] = 1.0;
        return;
    }

    output[idx] = medium->refractiveIndex(p);
}

// ============================================================
// Kernel: sample centerline eta(x, 0, 0)
// ============================================================

__global__
void sampleCenterLineKernel(
    const Medium* medium,
    double xMin,
    double xMax,
    int sampleCount,
    double* output
)
{
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= sampleCount)
        return;

    const double t =
        static_cast<double>(idx) / (sampleCount - 1);

    const double x = xMin + t * (xMax - xMin);

    const double3 p = make_double3(x, 0.0, 0.0);

    if (!medium->isInside(p)) {
        output[idx] = 1.0;
        return;
    }

    output[idx] = medium->refractiveIndex(p);
}

// ============================================================
// Save forward visualization
// ============================================================

void saveForwardVisualization(
    const VoxelMedium& medium,
    const LuneburgLens& idealLens,
    const std::filesystem::path& outputPath,
    int iteration,
    int fieldResolution,
    cudaStream_t stream
)
{
    constexpr int LINE_RESOLUTION = 300;
    constexpr int THREADS = 256;

    const double xMin = -1.0;
    const double xMax =  1.0;
    const double yMin = -1.0;
    const double yMax =  1.0;

    if (fieldResolution < 2)
    throw std::invalid_argument("fieldResolution must be >= 2");

    const std::size_t fieldCount =
        static_cast<std::size_t>(fieldResolution) *
        fieldResolution;

    const int fieldBlocks =
        static_cast<int>((fieldCount + THREADS - 1) / THREADS);

    const int lineBlocks =
        (LINE_RESOLUTION + THREADS - 1) / THREADS;

    constexpr double z = 0.0;

    DeviceBuffer<double> d_field(fieldCount);
    DeviceBuffer<double> d_line(LINE_RESOLUTION);

    // --------------------------------------------------------
    // Sample XY field
    // --------------------------------------------------------

    sampleXYKernel<<<fieldBlocks, THREADS, 0, stream>>>(
        medium.deviceObject(),
        xMin,
        xMax,
        yMin,
        yMax,
        z,
        fieldResolution,
        d_field.get()
    );

    checkCuda(cudaGetLastError(), "sampleXYKernel launch");

    checkCuda(
        cudaStreamSynchronize(stream),
        "execute sampleXYKernel"
    );

    // --------------------------------------------------------
    // Sample centerline
    // --------------------------------------------------------

    sampleCenterLineKernel<<<lineBlocks, THREADS, 0, stream>>>(
        medium.deviceObject(),
        xMin,
        xMax,
        LINE_RESOLUTION,
        d_line.get()
    );

    checkCuda(cudaGetLastError(), "sampleCenterLineKernel launch");

    checkCuda(
        cudaStreamSynchronize(stream),
        "execute sampleCenterLineKernel"
    );

    // --------------------------------------------------------
    // Copy results to host
    // --------------------------------------------------------

    std::vector<double> field(fieldCount);
    std::vector<double> optimizedLine(LINE_RESOLUTION);

    checkCuda(
        cudaMemcpyAsync(
            field.data(),
            d_field.get(),
            fieldCount * sizeof(double),
            cudaMemcpyDeviceToHost,
            stream
        ),
        "copy XY field"
    );

    checkCuda(
        cudaMemcpyAsync(
            optimizedLine.data(),
            d_line.get(),
            LINE_RESOLUTION * sizeof(double),
            cudaMemcpyDeviceToHost,
            stream
        ),
        "copy centerline"
    );

    checkCuda(
        cudaStreamSynchronize(stream),
        "synchronize visualization"
    );

    // --------------------------------------------------------
    // Ideal Luneburg profile (CPU)
    // --------------------------------------------------------

    std::vector<double> xs(LINE_RESOLUTION);
    std::vector<double> idealLine(LINE_RESOLUTION);

    for (int i = 0; i < LINE_RESOLUTION; ++i) {

        const double t =
            static_cast<double>(i) /
            (LINE_RESOLUTION - 1);

        const double x = xMin + t * (xMax - xMin);

        xs[i] = x;

        idealLine[i] = idealLens.refractiveIndex(
            glm::dvec3(x, 0.0, 0.0)
        );
    }

    // --------------------------------------------------------
    // Convert field to matrix
    // --------------------------------------------------------

    std::vector<std::vector<double>> image(
        fieldResolution,
        std::vector<double>(fieldResolution)
    );

    for (int j = 0; j < fieldResolution; ++j) {
        for (int i = 0; i < fieldResolution; ++i) {
            image[j][i] =
                field[
                    static_cast<std::size_t>(j) *
                    fieldResolution + i
                ];
        }
    }

    // --------------------------------------------------------
    // Plot
    // --------------------------------------------------------

    using namespace matplot;

    auto fig = figure(true);
    fig->size(1500, 650);

    subplot(1, 2, 1);

    imagesc(
        xMin,
        xMax,
        yMin,
        yMax,
        image
    );

    colorbar();

    title("Refractive index field, z = 0");

    xlabel("x");
    ylabel("y");

    axis(equal);

    subplot(1, 2, 2);

    plot(
        xs,
        optimizedLine,
        "b-"
    )->line_width(2);

    hold(on);

    plot(
        xs,
        idealLine,
        "r--"
    )->line_width(2);

    hold(off);

    title("Centerline refractive-index profile");

    xlabel("x");
    ylabel("Refractive index");

    legend({
        "Optimized medium",
        "Ideal Luneburg profile"
    });

    grid(on);

    fig->title(
        "Luneburg optimization - iteration " +
        std::to_string(iteration)
    );

    fig->save(outputPath.string());
}*/

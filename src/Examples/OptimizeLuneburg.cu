#include "MediumType/VoxelMedium.cuh"
#include "MediumType/LuneburgLens.h"

#include "Simulation/Ray.cuh"
#include "Simulation/Tracer.cuh"
#include "Simulation/OptimizableMedium.cuh"

#include "Optimization/LossType/LuneburgLoss.cuh"
#include "Optimization/AdjointStateSolver.cuh"
#include "Optimization/OptimizationTypes.cuh"

#include "Utils/Utils.cuh"

#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/constants.hpp>

#include <matplot/matplot.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// CUDA utilities
// ============================================================

void saveForwardVisualization(
    const VoxelMedium& medium,
    const LuneburgLens& idealLens,
    const Tracer& tracer,
    const std::filesystem::path& outputPath,
    int iteration,
    int fieldResolution,
    cudaStream_t stream
);

namespace {


// ============================================================
// Kernel: project parameters outside the lens
// ============================================================

__global__
void projectOutsideLensKernel(
    double* parameters,
    uint3 resolution,
    double3 origin,
    double3 voxelSize,
    double3 sphereCenter,
    double sphereRadius,
    std::size_t count
)
{
    const std::size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count)
        return;

    const std::size_t i = idx % resolution.x;
    const std::size_t j = (idx / resolution.x) % resolution.y;
    const std::size_t k = idx / (resolution.x * resolution.y);

    const double3 p = make_double3(
        origin.x + static_cast<double>(i) * voxelSize.x,
        origin.y + static_cast<double>(j) * voxelSize.y,
        origin.z + static_cast<double>(k) * voxelSize.z
    );

    const double3 relative = sub3(p, sphereCenter);

    if (normSquared3(relative) > sphereRadius * sphereRadius)
        parameters[idx] = 1.0;
}

// ============================================================
// Kernel: mask gradient outside the lens
// ============================================================

__global__
void maskOutsideGradientKernel(
    double* gradient,
    uint3 resolution,
    double3 origin,
    double3 voxelSize,
    double3 sphereCenter,
    double sphereRadius,
    std::size_t count
)
{
    const std::size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count)
        return;

    const std::size_t i = idx % resolution.x;
    const std::size_t j = (idx / resolution.x) % resolution.y;
    const std::size_t k = idx / (resolution.x * resolution.y);

    const double3 p = make_double3(
        origin.x + static_cast<double>(i) * voxelSize.x,
        origin.y + static_cast<double>(j) * voxelSize.y,
        origin.z + static_cast<double>(k) * voxelSize.z
    );

    const double3 relative = sub3(p, sphereCenter);

    if (normSquared3(relative) > sphereRadius * sphereRadius)
        gradient[idx] = 0.0;
}

// ============================================================
// Kernel: extract terminal states from DeviceTraceResult
// ============================================================

__global__
void extractTerminalStatesKernel(
    const DeviceTraceResult* results,
    double3* finalPositions,
    double3* finalDirections,
    std::size_t rayCount
)
{
    const std::size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= rayCount)
        return;

    finalPositions[idx] = results[idx].finalPosition;
    finalDirections[idx] = results[idx].finalDirection;
}

// ============================================================
// Kernel: count incomplete rays
// ============================================================

__global__
void countIncompleteRaysKernel(
    const DeviceTraceResult* results,
    unsigned int* invalidCount,
    std::size_t rayCount
)
{
    const std::size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= rayCount)
        return;

    if (!results[idx].completed)
        atomicAdd(invalidCount, 1u);
}


// ============================================================
// Host-side projection wrappers
// ============================================================

void projectOutsideLens(
    VoxelMedium& medium,
    const double3& center,
    double radius,
    cudaStream_t stream
)
{
    const glm::uvec3 res = medium.resolution();
    const glm::dvec3 voxelSize = medium.voxelSize();

    const glm::dvec3 origin(-1.0, -1.0, -1.0);

    const std::size_t count = medium.parameterCountHost();

    const uint3 deviceResolution = make_uint3(
        res.x, res.y, res.z
    );

    const double3 deviceOrigin = toDouble3(origin);
    const double3 deviceVoxelSize = toDouble3(voxelSize);

    projectOutsideLensKernel<<<
        blocksFor(count),
        BLOCK_SIZE,
        0,
        stream
    >>>(
        medium.deviceParameters(),
        deviceResolution,
        deviceOrigin,
        deviceVoxelSize,
        center,
        radius,
        count
    );

    checkKernel("projectOutsideLensKernel");
}

void maskOutsideGradient(
    VoxelMedium& medium,
    double* deviceGradient,
    const double3& center,
    double radius,
    cudaStream_t stream
)
{
    const glm::uvec3 res = medium.resolution();
    const glm::dvec3 voxelSize = medium.voxelSize();

    const glm::dvec3 origin(-1.0, -1.0, -1.0);

    const std::size_t count = medium.parameterCountHost();

    maskOutsideGradientKernel<<<
        blocksFor(count),
        BLOCK_SIZE,
        0,
        stream
    >>>(
        deviceGradient,
        make_uint3(res.x, res.y, res.z),
        toDouble3(origin),
        toDouble3(voxelSize),
        center,
        radius,
        count
    );

    checkKernel("maskOutsideGradientKernel");
}

// ============================================================
// Ray generation
// ============================================================

struct TrainingRay {
    Ray ray;
    double3 incomingDirection;
};

glm::dvec3 randomUnitVector(std::mt19937_64& rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);

    glm::dvec3 v(
        normal(rng),
        normal(rng),
        normal(rng)
    );

    return glm::normalize(v);
}

std::vector<TrainingRay> generateTrainingRays(
    double radius,
    int raysPerDirection,
    double startEpsilon,
    std::uint64_t seed
)
{
    constexpr double GOLDEN_ANGLE = 2.399963229728653;
    constexpr double maxRadial = 0.98;
    constexpr double edgeBias = 0.65; // > 0.5 da algo más de peso al borde que al centro

    const glm::dvec3 principalDirections[6] = {
        glm::dvec3( 1.0,  0.0,  0.0),
        glm::dvec3(-1.0,  0.0,  0.0),
        glm::dvec3( 0.0,  1.0,  0.0),
        glm::dvec3( 0.0, -1.0,  0.0),
        glm::dvec3( 0.0,  0.0,  1.0),
        glm::dvec3( 0.0,  0.0, -1.0)
    };

    std::mt19937_64 rng(seed);

    const glm::dvec3 randomAxis = randomUnitVector(rng);

    std::uniform_real_distribution<double> angleDistribution(
        0.0,
        2.0 * glm::pi<double>()
    );

    const double rotationAngle = angleDistribution(rng);

    const glm::dquat rotation = glm::angleAxis(
        rotationAngle,
        randomAxis
    );

    std::vector<TrainingRay> rays;
    rays.reserve(static_cast<std::size_t>(6 * raysPerDirection));

    for (const glm::dvec3& principal : principalDirections) {

        const glm::dvec3 incoming =
            glm::normalize(rotation * principal);

        const glm::dvec3 helper =
            std::abs(incoming.x) < 0.9
                ? glm::dvec3(1.0, 0.0, 0.0)
                : glm::dvec3(0.0, 1.0, 0.0);

        const glm::dvec3 tangent =
            glm::normalize(glm::cross(incoming, helper));

        const glm::dvec3 bitangent =
            glm::normalize(glm::cross(incoming, tangent));

        for (int i = 0; i < raysPerDirection; ++i) {

            double radial = 0.0;
            double angle = 0.0;

            if (i > 0) {
                const int outerCount = std::max(1, raysPerDirection - 1);
                const double s =
                    (static_cast<double>(i) - 0.5) /
                    static_cast<double>(outerCount);

                // Smoothstep concentra cerca de s=0 y s=1; la potencia en s
                // permite inclinar el balance entre centro y borde.
                const double biasedS = std::pow(s, 1.0 + (1.0 - edgeBias));
                const double smooth = biasedS * biasedS * (3.0 - 2.0 * biasedS);

                radial = maxRadial * smooth;
                angle = GOLDEN_ANGLE * static_cast<double>(i);
            }

            const double u = radial * std::cos(angle);
            const double v = radial * std::sin(angle);

            const glm::dvec3 offset =
                radius * (u * tangent + v * bitangent);

            double tEnter = -std::numeric_limits<double>::infinity();
            double tExit  =  std::numeric_limits<double>::infinity();

            for (int axis = 0; axis < 3; ++axis) {

                const double o = offset[axis];
                const double d = incoming[axis];

                if (std::abs(d) < 1e-12) {
                    if (o < -1.0 || o > 1.0) {
                        throw std::runtime_error(
                            "Ray does not intersect the medium cube"
                        );
                    }

                    continue;
                }

                double t0 = (-1.0 - o) / d;
                double t1 = ( 1.0 - o) / d;

                if (t0 > t1)
                    std::swap(t0, t1);

                tEnter = std::max(tEnter, t0);
                tExit  = std::min(tExit, t1);
            }

            if (tEnter > tExit || tExit < 0.0) {
                throw std::runtime_error(
                    "Invalid ray-cube intersection"
                );
            }

            const glm::dvec3 direction = incoming;

            const glm::dvec3 position =
                offset +
                (tEnter + startEpsilon) * direction;

            TrainingRay trainingRay;

            trainingRay.ray.position = toDouble3(position);
            trainingRay.ray.direction = toDouble3(direction);
            trainingRay.incomingDirection = toDouble3(incoming);

            rays.push_back(trainingRay);
        }
    }

    return rays;
}

// ============================================================
// Save loss history
// ============================================================

void saveLossHistory(
    const std::vector<double>& losses,
    const fs::path& outputDirectory
)
{
    const fs::path csvPath =
        outputDirectory / "loss_history.csv";

    std::ofstream csv(csvPath);

    if (!csv)
        throw std::runtime_error("Could not create loss_history.csv");

    csv << "iteration,loss\n";

    for (std::size_t i = 0; i < losses.size(); ++i)
        csv << i << "," << std::setprecision(16) << losses[i] << "\n";

    csv.close();

    using namespace matplot;

    std::vector<double> iterations(losses.size());

    for (std::size_t i = 0; i < losses.size(); ++i)
        iterations[i] = static_cast<double>(i);

    auto fig = figure(true);

    plot(iterations, losses)->line_width(2);

    xlabel("Iteration");
    ylabel("Mean loss");
    title("Optimization loss");
    grid(on);

    fig->save(
        (outputDirectory / "loss_history.png").string()
    );
}

} // anonymous namespace

// ============================================================
// Main optimization routine
// ============================================================

int main()
{
    try {

        // ----------------------------------------------------
        // Configuration
        // ----------------------------------------------------

        constexpr double radius = 1.0;

        int raysPerDirection = 32;
        constexpr int rayBundleCount = 6;

        constexpr int maxIterations = 2000;
        constexpr int saveEvery = 10;
        constexpr int refineEvery = 400;

        constexpr double learningRate = 1e-3;
        constexpr double convergenceTolerance = 1e-9;
        constexpr int convergencePatience = 30;

        constexpr int fieldResolution = 150;

        constexpr double rayStartEpsilon = 1e-5;

        constexpr double xMin = -1.0;
        constexpr double xMax =  1.0;

        const glm::dvec3 mediumOrigin(
            xMin, xMin, xMin
        );

        const glm::dvec3 mediumSize(
            xMax - xMin,
            xMax - xMin,
            xMax - xMin
        );

        const glm::uvec3 mediumResolution(
            8, 8, 8
        );

        const double3 lensCenter =
            make_double3(0.0, 0.0, 0.0);

        std::size_t rayCount =
            static_cast<std::size_t>(raysPerDirection) * rayBundleCount;

        // ----------------------------------------------------
        // CUDA device
        // ----------------------------------------------------

        int deviceCount = 0;

        checkCuda(
            cudaGetDeviceCount(&deviceCount),
            "cudaGetDeviceCount"
        );

        if (deviceCount == 0)
            throw std::runtime_error("No CUDA device found");

        int device = 0;

        checkCuda(
            cudaSetDevice(device),
            "cudaSetDevice"
        );

        cudaDeviceProp properties{};

        checkCuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties"
        );

        std::cout
            << "Using CUDA device: "
            << properties.name
            << "\n";

        std::cout
            << "Initial ray count: "
            << rayCount
            << "\n";

        // ----------------------------------------------------
        // Output directory
        // ----------------------------------------------------

        const fs::path outputDirectory =
            "output/luneburg_optimization";

        fs::create_directories(outputDirectory);

        // ----------------------------------------------------
        // CUDA stream
        // ----------------------------------------------------

        cudaStream_t stream = nullptr;

        checkCuda(
            cudaStreamCreateWithFlags(
                &stream,
                cudaStreamNonBlocking
            ),
            "cudaStreamCreateWithFlags"
        );

        // ----------------------------------------------------
        // Medium
        // ----------------------------------------------------

        VoxelMedium medium(
            mediumOrigin,
            mediumSize,
            mediumResolution,
            1.0
        );

        std::size_t parameterCount =
            medium.parameterCountHost();

        std::cout
            << "Voxel parameters: "
            << parameterCount
            << "\n";

        // Initial projection
        projectOutsideLens(
            medium,
            lensCenter,
            radius,
            stream
        );

        // ----------------------------------------------------
        // Host & device buffers
        // ----------------------------------------------------

        std::vector<Ray> hostRays(rayCount);
        std::vector<double3> hostInitialDirections(rayCount);

        DeviceBuffer<Ray> d_rays(rayCount);
        DeviceBuffer<double3> d_initialDirections(rayCount);
        DeviceBuffer<DeviceTraceResult> d_traceResults(rayCount);

        DeviceBuffer<double3> d_finalPositions(rayCount);
        DeviceBuffer<double3> d_finalDirections(rayCount);

        DeviceBuffer<double> d_losses(rayCount);

        DeviceBuffer<double3> d_deltaX(rayCount);
        DeviceBuffer<double3> d_deltaV(rayCount);

        DeviceBuffer<double> d_parameterGradient(parameterCount);

        DeviceBuffer<double> d_meanLoss(1);
        DeviceBuffer<unsigned int> d_invalidCount(1);

        // ----------------------------------------------------
        // Components
        // ----------------------------------------------------

        TracerConfig tracerConfig;

        tracerConfig.stepSize = 1e-3;
        tracerConfig.maxSteps = 10000;

        Tracer tracer(tracerConfig);

        SolverConfig adjointConfig;

        adjointConfig.stepSize = tracerConfig.stepSize;
        adjointConfig.maxSteps = 100000;

        AdjointStateSolver adjointSolver(adjointConfig);

        LuneburgLoss lossFunction(
            lensCenter,
            radius
        );

        AdamConfig adamConfig;

        adamConfig.learningRate = learningRate;
        adamConfig.beta1 = 0.9;
        adamConfig.beta2 = 0.999;
        adamConfig.epsilon = 1e-8;

        Adam optimizer(adamConfig);
        optimizer.initialize(parameterCount);

        LuneburgLens idealLens(
            radius,
            glm::dvec3(0.0, 0.0, 0.0)
        );

        // ----------------------------------------------------
        // Optimization loop
        // ----------------------------------------------------

        std::vector<double> lossHistory;

        lossHistory.reserve(maxIterations);

        int stableIterations = 0;

        double previousLoss =
            std::numeric_limits<double>::infinity();

        const Medium* deviceMedium =
            static_cast<const Medium*>(medium.deviceObject());

        const OptimizableMedium* deviceOptimizableMedium =
            static_cast<const OptimizableMedium*>(
                medium.deviceObject()
            );

        for (int iteration = 0;
             iteration < maxIterations;
             ++iteration) {

            // ------------------------------------------------
            // Refinamiento cada N iteraciones
            // ------------------------------------------------

            if (iteration > 0 && iteration % refineEvery == 0) {

                checkCuda(
                    cudaStreamSynchronize(stream),
                    "synchronize before medium refinement"
                );

                const glm::uvec3 currentRes = medium.resolution();
                const glm::uvec3 nextRes = currentRes * 2u;

                medium.refine(nextRes);

                parameterCount = medium.parameterCountHost();
                deviceMedium = static_cast<const Medium*>(medium.deviceObject());
                deviceOptimizableMedium =
                    static_cast<const OptimizableMedium*>(medium.deviceObject());

                d_parameterGradient = DeviceBuffer<double>(parameterCount);

                // Reducir learning rate al refinar para evitar picos en grillas finas
                adamConfig.learningRate *= 0.5;
                optimizer.setLearningRate(adamConfig.learningRate);
                optimizer.initialize(parameterCount);

                // Actualizar dt y maxSteps proporcionalmente al tamaño de celda
                tracerConfig.stepSize *= 0.5;
                tracerConfig.maxSteps *= 2;
                tracer = Tracer(tracerConfig);

                adjointConfig.stepSize = tracerConfig.stepSize;
                adjointConfig.maxSteps *= 2;
                adjointSolver = AdjointStateSolver(adjointConfig);

                projectOutsideLens(
                    medium,
                    lensCenter,
                    radius,
                    stream
                );

                stableIterations = 0;

                raysPerDirection = raysPerDirection * 2;
                const std::size_t newRayCount =
                    static_cast<std::size_t>(raysPerDirection) * rayBundleCount;

                if (newRayCount != rayCount) {
                    rayCount = newRayCount;
                    d_rays = DeviceBuffer<Ray>(rayCount);
                    d_initialDirections = DeviceBuffer<double3>(rayCount);
                    d_traceResults = DeviceBuffer<DeviceTraceResult>(rayCount);
                    d_finalPositions = DeviceBuffer<double3>(rayCount);
                    d_finalDirections = DeviceBuffer<double3>(rayCount);
                    d_losses = DeviceBuffer<double>(rayCount);
                    d_deltaX = DeviceBuffer<double3>(rayCount);
                    d_deltaV = DeviceBuffer<double3>(rayCount);
                    hostRays.resize(rayCount);
                    hostInitialDirections.resize(rayCount);
                }

                std::cout
                    << "Refined voxel grid at step " << iteration
                    << " -> resolution: "
                    << nextRes.x << "x" << nextRes.y << "x" << nextRes.z
                    << " (" << parameterCount << " parameters)"
                    << ", rays=" << rayCount
                    << ", dt=" << tracerConfig.stepSize
                    << ", lr=" << adamConfig.learningRate << "\n";
            }

            // ------------------------------------------------
            // Generar y subir rayos en cada iteración
            // ------------------------------------------------

            checkCuda(
                cudaStreamSynchronize(stream),
                "synchronize before uploading new iteration rays"
            );

            const std::vector<TrainingRay> trainingRays =
                generateTrainingRays(
                    radius,
                    raysPerDirection,
                    rayStartEpsilon,
                    42ULL + static_cast<std::uint64_t>(iteration)
                );

            for (std::size_t i = 0; i < rayCount; ++i) {
                hostRays[i] = trainingRays[i].ray;
                hostInitialDirections[i] =
                    trainingRays[i].incomingDirection;
            }

            checkCuda(
                cudaMemcpyAsync(
                    d_rays.get(),
                    hostRays.data(),
                    rayCount * sizeof(Ray),
                    cudaMemcpyHostToDevice,
                    stream
                ),
                "upload rays"
            );

            checkCuda(
                cudaMemcpyAsync(
                    d_initialDirections.get(),
                    hostInitialDirections.data(),
                    rayCount * sizeof(double3),
                    cudaMemcpyHostToDevice,
                    stream
                ),
                "upload initial directions"
            );

            // ------------------------------------------------
            // Forward tracing
            // ------------------------------------------------

            tracer.forward(
                d_rays.get(),
                d_traceResults.get(),
                deviceMedium,
                rayCount,
                stream
            );

            // ------------------------------------------------
            // Validate forward trajectories
            // ------------------------------------------------

            checkCuda(
                cudaMemsetAsync(
                    d_invalidCount.get(),
                    0,
                    sizeof(unsigned int),
                    stream
                ),
                "reset invalid ray counter"
            );

            countIncompleteRaysKernel<<<
                blocksFor(rayCount),
                BLOCK_SIZE,
                0,
                stream
            >>>(
                d_traceResults.get(),
                d_invalidCount.get(),
                rayCount
            );

            checkKernel("countIncompleteRaysKernel");

            // ------------------------------------------------
            // Extract final states
            // ------------------------------------------------

            extractTerminalStatesKernel<<<
                blocksFor(rayCount),
                BLOCK_SIZE,
                0,
                stream
            >>>(
                d_traceResults.get(),
                d_finalPositions.get(),
                d_finalDirections.get(),
                rayCount
            );

            checkKernel("extractTerminalStatesKernel");

            // ------------------------------------------------
            // Evaluate terminal loss
            // ------------------------------------------------

            lossFunction.evaluate(
                d_initialDirections.get(),
                d_finalPositions.get(),
                d_finalDirections.get(),
                rayCount,
                d_losses.get(),
                d_deltaX.get(),
                d_deltaV.get(),
                d_meanLoss.get(),
                stream
            );

            // ------------------------------------------------
            // Read scalar diagnostics
            // ------------------------------------------------

            double meanLoss = 0.0;
            unsigned int invalidCount = 0;

            checkCuda(
                cudaMemcpyAsync(
                    &meanLoss,
                    d_meanLoss.get(),
                    sizeof(double),
                    cudaMemcpyDeviceToHost,
                    stream
                ),
                "download mean loss"
            );

            checkCuda(
                cudaMemcpyAsync(
                    &invalidCount,
                    d_invalidCount.get(),
                    sizeof(unsigned int),
                    cudaMemcpyDeviceToHost,
                    stream
                ),
                "download invalid ray count"
            );

            checkCuda(
                cudaStreamSynchronize(stream),
                "synchronize loss evaluation"
            );

            if (invalidCount > 0) {
                throw std::runtime_error(
                    "Forward tracing failed for " +
                    std::to_string(invalidCount) +
                    " rays"
                );
            }

            if (!std::isfinite(meanLoss)) {
                throw std::runtime_error(
                    "Non-finite loss encountered"
                );
            }

            lossHistory.push_back(meanLoss);

            std::cout
                << "Iteration "
                << std::setw(5) << iteration
                << " | loss = "
                << std::setprecision(12)
                << meanLoss
                << "\n";

            // ------------------------------------------------
            // Visualization
            // ------------------------------------------------

            if (iteration % saveEvery == 0) {

                const fs::path imagePath =
                    outputDirectory /
                    (
                        "forward_" +
                        std::to_string(iteration) +
                        ".png"
                    );

                saveForwardVisualization(
                    medium,
                    idealLens,
                    tracer,
                    imagePath,
                    iteration,
                    fieldResolution,
                    stream
                );
            }

            // ------------------------------------------------
            // Convergence
            // ------------------------------------------------

            if (std::abs(previousLoss - meanLoss) <
                convergenceTolerance) {

                ++stableIterations;

            } else {

                stableIterations = 0;
            }

            previousLoss = meanLoss;

            if (stableIterations >= convergencePatience) {

                std::cout
                    << "Converged after "
                    << iteration
                    << " iterations.\n";

                break;
            }

            // ------------------------------------------------
            // Parameter gradient initialization
            // ------------------------------------------------

            checkCuda(
                cudaMemsetAsync(
                    d_parameterGradient.get(),
                    0,
                    parameterCount * sizeof(double),
                    stream
                ),
                "zero parameter gradient"
            );

            // ------------------------------------------------
            // Adjoint backward pass
            // ------------------------------------------------

            adjointSolver.backward(
                d_traceResults.get(),
                d_deltaX.get(),
                d_deltaV.get(),
                d_parameterGradient.get(),
                deviceOptimizableMedium,
                rayCount,
                stream
            );

            // ------------------------------------------------
            // Mask gradient outside the lens
            // ------------------------------------------------

            maskOutsideGradient(
                medium,
                d_parameterGradient.get(),
                lensCenter,
                radius,
                stream
            );

            // ------------------------------------------------
            // Adam update
            // ------------------------------------------------

            if (iteration >= refineEvery - 2 && iteration <= refineEvery + 5) {
                const auto params = medium.downloadParameters();
                double minEta = params[0], maxEta = params[0];
                for (double v : params) {
                    minEta = std::min(minEta, v);
                    maxEta = std::max(maxEta, v);
                }
                std::cout
                    << "  [debug] iter=" << iteration
                    << " res=" << medium.resolution().x
                    << " dLoss=" << std::abs(previousLoss - meanLoss)
                    << " eta=[" << minEta << ", " << maxEta << "]\n";
            }

            optimizer.step(
                medium.deviceParameters(),
                d_parameterGradient.get(),
                parameterCount,
                stream
            );

            // ------------------------------------------------
            // Project parameters outside the lens
            // ------------------------------------------------

            projectOutsideLens(
                medium,
                lensCenter,
                radius,
                stream
            );
        }

        // ----------------------------------------------------
        // Final synchronization
        // ----------------------------------------------------

        checkCuda(
            cudaStreamSynchronize(stream),
            "final optimization synchronization"
        );

        // ----------------------------------------------------
        // Final visualization
        // ----------------------------------------------------

        saveForwardVisualization(
            medium,
            idealLens,
            tracer,
            outputDirectory / "forward_final.png",
            static_cast<int>(lossHistory.size()),
            fieldResolution,
            stream
        );

        // ----------------------------------------------------
        // Loss history
        // ----------------------------------------------------

        saveLossHistory(
            lossHistory,
            outputDirectory
        );

        // ----------------------------------------------------
        // Save optimized parameters
        // ----------------------------------------------------

        const std::vector<double> optimizedParameters =
            medium.downloadParameters();

        std::ofstream parametersFile(
            outputDirectory / "optimized_parameters.csv"
        );

        if (!parametersFile)
            throw std::runtime_error(
                "Could not create optimized_parameters.csv"
            );

        parametersFile << "index,eta\n";

        for (std::size_t i = 0;
             i < optimizedParameters.size();
             ++i) {

            parametersFile
                << i << ","
                << std::setprecision(16)
                << optimizedParameters[i]
                << "\n";
        }

        parametersFile.close();

        // ----------------------------------------------------
        // Cleanup
        // ----------------------------------------------------

        checkCuda(
            cudaStreamSynchronize(stream),
            "synchronize before cleanup"
        );

        checkCuda(
            cudaStreamDestroy(stream),
            "cudaStreamDestroy"
        );

        std::cout
            << "\nOptimization completed.\n"
            << "Output directory: "
            << outputDirectory
            << "\n";

        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "Optimization error: "
            << e.what()
            << "\n";

        return 1;
    }
}
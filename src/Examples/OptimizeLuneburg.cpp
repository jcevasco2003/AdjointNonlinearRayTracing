
#include "Simulation/Tracer.h"
#include "Simulation/Ray.h"

#include "MediumType/VoxelMedium.h"
#include "MediumType/LuneburgLens.h"

#include "Optimization/OptimizationTypes.h"
#include "Optimization/AdjointStateSolver.h"

#include <matplot/matplot.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace matplot;

// ============================================================
// Configuration
// ============================================================

constexpr double radius = 1.0;

constexpr double xMin = -1.0;
constexpr double xMax =  1.0;
constexpr double yMin = -1.0;
constexpr double yMax =  1.0;
constexpr double zMin = -1.0;
constexpr double zMax =  1.0;

constexpr std::size_t initialResolution = 8;

constexpr std::size_t raysPerDirection = 17;

constexpr int maxIterations = 2000;
constexpr int saveEvery = 10;

constexpr double learningRate = 1e-3;
constexpr double convergenceTolerance = 1e-9;
constexpr int convergencePatience = 30;

constexpr int fieldResolution = 150;
constexpr int circleResolution = 300;

constexpr double rayStartEpsilon = 1e-5;

// ============================================================
// Random generator
// ============================================================

std::mt19937_64 rng(std::random_device{}());

std::uniform_real_distribution<double> uniform01(0.0, 1.0);

// ============================================================
// Geometry
// ============================================================

glm::dvec3 lensCenter()
{
    return glm::dvec3(0.0);
}

glm::dvec3 mediumOrigin()
{
    return glm::dvec3(xMin, yMin, zMin);
}

glm::dvec3 mediumSize()
{
    return glm::dvec3(
        xMax - xMin,
        yMax - yMin,
        zMax - zMin
    );
}

bool insideLens(const glm::dvec3& p)
{
    const glm::dvec3 d = p - lensCenter();

    return glm::dot(d, d) <= radius * radius;
}

// ============================================================
// Parameter projection
// ============================================================

std::size_t voxelIndex(
    const glm::uvec3& res,
    std::size_t i,
    std::size_t j,
    std::size_t k
)
{
    return i + res.x * (j + res.y * k);
}

void projectOutsideLens(VoxelMedium& medium)
{
    const glm::uvec3 res = medium.resolution();
    const glm::dvec3 origin = mediumOrigin();
    const glm::dvec3 spacing = medium.voxelSize();

    std::vector<double> parameters = medium.getParameters();

    for (std::size_t k = 0; k < res.z; ++k) {
        for (std::size_t j = 0; j < res.y; ++j) {
            for (std::size_t i = 0; i < res.x; ++i) {

                const glm::dvec3 p(
                    origin.x + static_cast<double>(i) * spacing.x,
                    origin.y + static_cast<double>(j) * spacing.y,
                    origin.z + static_cast<double>(k) * spacing.z
                );

                if (!insideLens(p)) {
                    parameters.at(voxelIndex(res, i, j, k)) = 1.0;
                }
            }
        }
    }

    medium.setParameters(parameters);
}

void maskOutsideGradient(
    const VoxelMedium& medium,
    std::vector<double>& gradient
)
{
    const glm::uvec3 res = medium.resolution();
    const glm::dvec3 origin = mediumOrigin();
    const glm::dvec3 spacing = medium.voxelSize();

    for (std::size_t k = 0; k < res.z; ++k) {
        for (std::size_t j = 0; j < res.y; ++j) {
            for (std::size_t i = 0; i < res.x; ++i) {

                const glm::dvec3 p(
                    origin.x + static_cast<double>(i) * spacing.x,
                    origin.y + static_cast<double>(j) * spacing.y,
                    origin.z + static_cast<double>(k) * spacing.z
                );

                if (!insideLens(p)) {
                    gradient.at(voxelIndex(res, i, j, k)) = 0.0;
                }
            }
        }
    }
}

// ============================================================
// Random SO(3) rotation
// ============================================================

glm::dmat3 randomRotation()
{
    const double u1 = uniform01(rng);
    const double u2 = uniform01(rng);
    const double u3 = uniform01(rng);

    const double x =
        std::sqrt(1.0 - u1) *
        std::sin(2.0 * glm::pi<double>() * u2);

    const double y =
        std::sqrt(1.0 - u1) *
        std::cos(2.0 * glm::pi<double>() * u2);

    const double z =
        std::sqrt(u1) *
        std::sin(2.0 * glm::pi<double>() * u3);

    const double w =
        std::sqrt(u1) *
        std::cos(2.0 * glm::pi<double>() * u3);

    const glm::dquat q(w, x, y, z);

    return glm::mat3_cast(q);
}

// ============================================================
// Perpendicular basis
// ============================================================

void perpendicularBasis(
    const glm::dvec3& direction,
    glm::dvec3& u,
    glm::dvec3& v
)
{
    const glm::dvec3 reference =
        std::abs(direction.z) < 0.9
        ? glm::dvec3(0.0, 0.0, 1.0)
        : glm::dvec3(0.0, 1.0, 0.0);

    u = glm::normalize(glm::cross(direction, reference));
    v = glm::normalize(glm::cross(direction, u));
}

// ============================================================
// Ray generation
// ============================================================

struct TrainingRay {
    Ray ray;
    glm::dvec3 incomingDirection;
};

std::vector<TrainingRay> generateTrainingRays()
{
    std::vector<TrainingRay> rays;

    rays.reserve(6 * raysPerDirection);

    // One common random rotation for all six bundles.
    const glm::dmat3 rotation = randomRotation();

    const std::array<glm::dvec3, 6> principalDirections = {
        glm::dvec3( 1.0,  0.0,  0.0),
        glm::dvec3(-1.0,  0.0,  0.0),
        glm::dvec3( 0.0,  1.0,  0.0),
        glm::dvec3( 0.0, -1.0,  0.0),
        glm::dvec3( 0.0,  0.0,  1.0),
        glm::dvec3( 0.0,  0.0, -1.0)
    };

    constexpr double offsetRadius = 0.9 * radius;

    for (const glm::dvec3& baseDirection : principalDirections) {

        const glm::dvec3 direction =
            glm::normalize(rotation * baseDirection);

        glm::dvec3 u, v;
        perpendicularBasis(direction, u, v);

        for (std::size_t i = 0; i < raysPerDirection; ++i) {

            const double radial =
                offsetRadius *
                std::sqrt(
                    (static_cast<double>(i) + 0.5) /
                    static_cast<double>(raysPerDirection)
                );

            // Golden-angle distribution.
            const double angle =
                static_cast<double>(i) * 2.399963229728653;

            const glm::dvec3 offset =
                radial * (
                    std::cos(angle) * u +
                    std::sin(angle) * v
                );

            // Find the entry point on the cube.
            double entryDistance =
                std::numeric_limits<double>::infinity();

            for (int axis = 0; axis < 3; ++axis) {

                const double component = direction[axis];

                if (std::abs(component) < 1e-12)
                    continue;

                const double distance =
                    (1.0 - std::abs(offset[axis])) /
                    std::abs(component);

                entryDistance =
                    std::min(entryDistance, distance);
            }

            const glm::dvec3 entryPoint =
                offset - entryDistance * direction;

            const glm::dvec3 start =
                entryPoint + rayStartEpsilon * direction;

            rays.push_back({
                Ray(start, direction),
                direction
            });
        }
    }

    return rays;
}

// ============================================================
// Plane focal loss
// ============================================================

struct PlaneFocusLossResult {
    double value;
    glm::dvec3 gradientPosition;
    glm::dvec3 gradientVelocity;
};

PlaneFocusLossResult evaluatePlaneFocusLoss(
    const glm::dvec3& finalPosition,
    const glm::dvec3& finalVelocity,
    const glm::dvec3& incomingDirection
)
{
    const glm::dvec3 target =
        radius * incomingDirection;

    const double denominator =
        glm::dot(incomingDirection, finalVelocity);

    if (std::abs(denominator) < 1e-10) {
        throw std::runtime_error(
            "Ray is parallel to its focal plane"
        );
    }

    const double t =
        glm::dot(
            incomingDirection,
            target - finalPosition
        ) / denominator;
    /*
    if (t < 0.0) {
        throw std::runtime_error(
            "Focal plane intersection is behind the ray"
        );
    }
    */

    const glm::dvec3 intersection =
        finalPosition + t * finalVelocity;

    const glm::dvec3 error =
        intersection - target;

    const double loss =
        glm::dot(error, error);

    const glm::dvec3 a =
        error -
        incomingDirection *
        (
            glm::dot(finalVelocity, error) /
            denominator
        );

    return {
        loss,
        2.0 * a,
        2.0 * t * a
    };
}

// ============================================================
// Optimization evaluation
// ============================================================

struct OptimizationEvaluation {
    double loss = 0.0;
    std::vector<double> gradient;
};

OptimizationEvaluation evaluateMedium(
    const VoxelMedium& medium,
    const Tracer& tracer,
    const AdjointStateSolver& adjoint,
    const std::vector<TrainingRay>& rays
)
{
    OptimizationEvaluation evaluation;

    evaluation.gradient.assign(
        medium.parameterCount(),
        0.0
    );

    for (const TrainingRay& trainingRay : rays) {

        const TraceResult trace =
            tracer.forward(trainingRay.ray, medium);

        if (!trace.completed) {
            throw std::runtime_error(
                "A training ray did not leave the voxel medium"
            );
        }

        const PlaneFocusLossResult terminalLoss =
            evaluatePlaneFocusLoss(
                trace.finalState.position,
                trace.finalState.direction,
                trainingRay.incomingDirection
            );

        evaluation.loss += terminalLoss.value;

        ForwardResult forwardResult{
            trace.finalState.position,
            trace.finalState.direction,
            trace.stepsTaken
        };

        const std::vector<double> rayGradient =
            adjoint.backward(
                forwardResult,
                terminalLoss.gradientPosition,
                terminalLoss.gradientVelocity,
                medium
            );

        if (rayGradient.size() != evaluation.gradient.size()) {
            throw std::runtime_error(
                "Adjoint returned an invalid gradient dimension"
            );
        }

        for (std::size_t j = 0; j < rayGradient.size(); ++j) {
            evaluation.gradient[j] += rayGradient[j];
        }
    }

    const double inverseRayCount =
        1.0 / static_cast<double>(rays.size());

    evaluation.loss *= inverseRayCount;

    for (double& g : evaluation.gradient) {
        g *= inverseRayCount;
    }

    return evaluation;
}

// ============================================================
// Visualization: XY slice at z = 0
// ============================================================

std::vector<double> visualizationOffsets()
{
    return {
        0.0,
        radius / 3.0,
        -radius / 3.0,
        2.0 * radius / 3.0,
        -2.0 * radius / 3.0
    };
}

void saveForwardVisualization(
    const VoxelMedium& medium,
    const Tracer& tracer,
    const fs::path& outputPath,
    int iteration
)
{
    std::vector<double> xs(fieldResolution);
    std::vector<double> ys(fieldResolution);

    for (int i = 0; i < fieldResolution; ++i) {

        const double t =
            static_cast<double>(i) /
            static_cast<double>(fieldResolution - 1);

        xs[i] = xMin + (xMax - xMin) * t;
        ys[i] = yMin + (yMax - yMin) * t;
    }

    // etaField[j][i] = eta(xs[i], ys[j], 0)
    std::vector<std::vector<double>> etaField(
        fieldResolution,
        std::vector<double>(fieldResolution, 1.0)
    );

    for (int j = 0; j < fieldResolution; ++j) {
        for (int i = 0; i < fieldResolution; ++i) {

            const glm::dvec3 p(
                xs[i],
                ys[j],
                0.0
            );

            etaField[j][i] =
                medium.refractiveIndex(p);
        }
    }

    auto fig = figure(true);
    hold(on);

    imagesc(xMin, xMax, yMin, yMax, etaField);
    colorbar();

    // Circular intersection of the sphere with z = 0.
    std::vector<double> circleX(circleResolution + 1);
    std::vector<double> circleY(circleResolution + 1);

    for (int i = 0; i <= circleResolution; ++i) {

        const double theta =
            2.0 * glm::pi<double>() *
            static_cast<double>(i) /
            static_cast<double>(circleResolution);

        circleX[i] = radius * std::cos(theta);
        circleY[i] = radius * std::sin(theta);
    }

    plot(circleX, circleY, "k-");

    // Five characteristic visualization rays.
    // They remain parallel to +X and lie in z = 0.
    for (double offset : visualizationOffsets()) {

        const double y = offset;

        Ray ray(
            glm::dvec3(xMin + rayStartEpsilon, y, 0.0),
            glm::dvec3(1.0, 0.0, 0.0)
        );

        std::vector<double> pathX;
        std::vector<double> pathY;

        pathX.push_back(xMin);
        pathY.push_back(y);

        TraceResult result = tracer.forward(
            ray,
            medium,
            [&](const glm::dvec3& p) {
                pathX.push_back(p.x);
                pathY.push_back(p.y);
            }
        );

        if (result.completed) {
            pathX.push_back(result.finalState.position.x);
            pathY.push_back(result.finalState.position.y);
        }

        plot(pathX, pathY, "r-");
    }

    xlim({xMin, xMax});
    ylim({yMin, yMax});

    axis(equal);
    grid(on);

    xlabel("x");
    ylabel("y");

    title(
        "3D Luneburg Lens - XY slice z=0 - Step " +
        std::to_string(iteration)
    );

    save(outputPath.string());
}

// ============================================================
// Final plots
// ============================================================

void saveErrorPlot(
    const std::vector<double>& errors,
    const fs::path& outputPath
)
{
    std::vector<double> steps(errors.size());

    for (std::size_t i = 0; i < errors.size(); ++i) {
        steps[i] = static_cast<double>(i);
    }

    auto fig = figure(true);
    hold(on);

    plot(steps, errors, "b-");

    xlabel("Optimization step");
    ylabel("Mean squared focal distance");
    title("3D Luneburg optimization error");

    grid(on);

    save(outputPath.string());
}

void saveEtaComparison(
    const VoxelMedium& medium,
    const fs::path& outputPath
)
{
    constexpr int samples = 400;

    std::vector<double> xs(samples);
    std::vector<double> etaOptimized(samples);
    std::vector<double> etaAnalytical(samples);

    LuneburgLens analyticalLens(
        radius,
        lensCenter()
    );

    for (int i = 0; i < samples; ++i) {

        const double t =
            static_cast<double>(i) /
            static_cast<double>(samples - 1);

        const double x =
            xMin + (xMax - xMin) * t;

        const glm::dvec3 p(x, 0.0, 0.0);

        xs[i] = x;

        etaOptimized[i] =
            medium.refractiveIndex(p);

        etaAnalytical[i] =
            analyticalLens.refractiveIndex(p);
    }

    auto fig = figure(true);
    hold(on);

    plot(xs, etaOptimized, "b-");
    plot(xs, etaAnalytical, "r--");

    xlabel("x");
    ylabel("Refractive index eta");

    title("Optimized vs analytical Luneburg lens");

    legend({
        "Optimized voxel medium",
        "Analytical Luneburg lens"
    });

    grid(on);

    save(outputPath.string());
}

// ============================================================
// CSV
// ============================================================

void saveErrorsCSV(
    const std::vector<double>& errors,
    const fs::path& outputPath
)
{
    std::ofstream file(outputPath);

    if (!file) {
        throw std::runtime_error(
            "Could not create error CSV file"
        );
    }

    file << "step,error\n";
    file << std::setprecision(16);

    for (std::size_t i = 0; i < errors.size(); ++i) {
        file << i << "," << errors[i] << "\n";
    }
}

} // namespace

// ============================================================
// Main
// ============================================================

int main()
{
    try {

        // ----------------------------------------------------
        // 1. Output directory
        // ----------------------------------------------------

        const fs::path outputDirectory =
            "Examples/OptimizeLuneburg3D";

        fs::create_directories(outputDirectory);

        // ----------------------------------------------------
        // 2. Create 3D voxel medium
        // ----------------------------------------------------

        const glm::dvec3 origin = mediumOrigin();
        const glm::dvec3 size = mediumSize();

        const glm::uvec3 resolution(
            initialResolution,
            initialResolution,
            initialResolution
        );

        VoxelMedium medium(
            origin,
            size,
            resolution,
            1.0
        );

        projectOutsideLens(medium);

        // ----------------------------------------------------
        // 3. Configure tracer
        // ----------------------------------------------------

        TracerConfig tracerConfig;

        tracerConfig.stepSize = 0.001f;
        tracerConfig.maxSteps = 10000;

        Tracer tracer(tracerConfig);

        // ----------------------------------------------------
        // 4. Configure adjoint
        // ----------------------------------------------------

        SolverConfig solverConfig;

        solverConfig.stepSize =
            static_cast<double>(tracerConfig.stepSize);

        solverConfig.maxSteps =
            tracerConfig.maxSteps;

        AdjointStateSolver adjoint(solverConfig);

        // ----------------------------------------------------
        // 5. Configure Adam
        // ----------------------------------------------------

        AdamConfig adamConfig;

        adamConfig.learningRate = learningRate;
        adamConfig.beta1 = 0.9;
        adamConfig.beta2 = 0.999;
        adamConfig.epsilon = 1e-8;

        Adam optimizer(adamConfig);

        // ----------------------------------------------------
        // 6. Generate training rays
        // ----------------------------------------------------

        std::vector<TrainingRay> trainingRays =
            generateTrainingRays();

        std::cout
            << "Training rays: "
            << trainingRays.size()
            << "\n";

        std::cout
            << "Voxel parameters: "
            << medium.parameterCount()
            << "\n";

        // ----------------------------------------------------
        // 7. Optimization loop
        // ----------------------------------------------------

        std::vector<double> errors;

        double previousLoss =
            std::numeric_limits<double>::infinity();

        int stableIterations = 0;

        for (int iteration = 0;
             iteration < maxIterations;
             ++iteration)
        {
            // Resample the common random rotation each iteration.
            trainingRays = generateTrainingRays();

            OptimizationEvaluation evaluation =
                evaluateMedium(
                    medium,
                    tracer,
                    adjoint,
                    trainingRays
                );

            errors.push_back(evaluation.loss);

            std::cout
                << "Step "
                << std::setw(5) << iteration
                << " | loss = "
                << std::setprecision(10)
                << evaluation.loss
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
                    tracer,
                    imagePath,
                    iteration
                );
            }

            // ------------------------------------------------
            // Convergence criterion
            // ------------------------------------------------

            if (std::isfinite(previousLoss)) {

                const double lossChange =
                    std::abs(previousLoss - evaluation.loss);

                if (lossChange < convergenceTolerance) {
                    ++stableIterations;
                } else {
                    stableIterations = 0;
                }
            }

            if (stableIterations >= convergencePatience) {

                std::cout
                    << "Converged after "
                    << iteration + 1
                    << " iterations.\n";

                break;
            }

            previousLoss = evaluation.loss;

            // ------------------------------------------------
            // Mask and update
            // ------------------------------------------------

            maskOutsideGradient(
                medium,
                evaluation.gradient
            );

            const std::vector<double> parameters =
                medium.getParameters();

            const std::vector<double> updatedParameters =
                optimizer.step(
                    parameters,
                    evaluation.gradient
                );

            medium.setParameters(updatedParameters);

            projectOutsideLens(medium);
        }

        // ----------------------------------------------------
        // 8. Save final outputs
        // ----------------------------------------------------

        saveErrorsCSV(
            errors,
            outputDirectory / "errors.csv"
        );

        saveErrorPlot(
            errors,
            outputDirectory / "error.png"
        );

        saveEtaComparison(
            medium,
            outputDirectory / "eta_comparison.png"
        );

        saveForwardVisualization(
            medium,
            tracer,
            outputDirectory / "forward_final.png",
            static_cast<int>(errors.size())
        );

        std::cout
            << "\nOptimization finished.\n"
            << "Results saved in: "
            << outputDirectory.string()
            << "\n";
    }
    catch (const std::exception& e) {

        std::cerr
            << "Optimization failed: "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}



/*

#include "Simulation/Tracer.h"
#include "Simulation/Ray.h"

#include "MediumType/VoxelMedium.h"
#include "MediumType/LuneburgLens.h"

#include "Optimization/LossType/LuneburgLoss.h"
#include "Optimization/OptimizationTypes.h"
#include "Optimization/AdjointStateSolver.h"

#include <matplot/matplot.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace matplot;

// ------------------------------------------------------------
// Configuration
// ------------------------------------------------------------

constexpr double radius = 1.0;

constexpr double xMin = -1.0;
constexpr double xMax =  1.0;
constexpr double yMin = -1.0;
constexpr double yMax =  1.0;

constexpr std::size_t initialResolution = 32;

constexpr std::size_t trainingRayCount = 17;

constexpr int maxIterations = 2000;
constexpr int saveEvery = 10;

constexpr double learningRate = 1e-3;
constexpr double convergenceTolerance = 1e-9;
constexpr int convergencePatience = 30;

// Refinement is prepared but disabled.
constexpr bool enableRefinement = false;
constexpr int refinementInterval = 500;
constexpr unsigned int refinementFactor = 2;

// Visualization
constexpr int fieldResolution = 150;
constexpr int circleResolution = 300;

// ------------------------------------------------------------
// Geometry helpers
// ------------------------------------------------------------

glm::dvec3 lensCenter()
{
    return glm::dvec3(0.0, 0.0, 0.0);
}

glm::dvec3 targetPoint()
{
    return glm::dvec3(xMax, 0.0, 0.0);
}

glm::dvec3 mediumOrigin()
{
    return glm::dvec3(xMin, yMin, -5e-7);
}

glm::dvec3 mediumSize()
{
    return glm::dvec3(
        xMax - xMin,
        yMax - yMin,
        1e-6
    );
}

bool insideLens(const glm::dvec3& p)
{
    const glm::dvec3 d = p - lensCenter();

    return glm::dot(d, d) <= radius * radius;
}

// ------------------------------------------------------------
// Parameter projection
// ------------------------------------------------------------

// Force eta = 1 outside the circular lens.
//
// Parameter ordering assumed:
// index(i,j,k) = (i * resolution.y + j) * resolution.z + k.

void projectOutsideLens(VoxelMedium& medium)
{
    const glm::uvec3 res = medium.resolution();
    const glm::dvec3 origin = mediumOrigin();
    const glm::dvec3 spacing = medium.voxelSize();

    std::vector<double> parameters = medium.getParameters();

    for (std::size_t i = 0; i < res.x; ++i) {
        for (std::size_t j = 0; j < res.y; ++j) {
            for (std::size_t k = 0; k < res.z; ++k) {

                const glm::dvec3 p(
                    origin.x + static_cast<double>(i) * spacing.x,
                    origin.y + static_cast<double>(j) * spacing.y,
                    0.0
                );

                if (!insideLens(p)) {
                    const std::size_t idx =
                        i + res.x * (j + res.y * k);

                    parameters.at(idx) = 1.0;
                }
            }
        }
    }

    medium.setParameters(parameters);
}

// Set gradient to zero for fixed parameters.
// This prevents Adam from accumulating updates for those parameters.

void maskOutsideGradient(
    const VoxelMedium& medium,
    std::vector<double>& gradient
)
{
    const glm::uvec3 res = medium.resolution();
    const glm::dvec3 origin = mediumOrigin();
    const glm::dvec3 spacing = medium.voxelSize();

    for (std::size_t i = 0; i < res.x; ++i) {
        for (std::size_t j = 0; j < res.y; ++j) {
            for (std::size_t k = 0; k < res.z; ++k) {

                const glm::dvec3 p(
                    origin.x + static_cast<double>(i) * spacing.x,
                    origin.y + static_cast<double>(j) * spacing.y,
                    0.0
                );

                if (!insideLens(p)) {
                    const std::size_t idx =
                        i + res.x * (j + res.y * k);

                    gradient.at(idx) = 0.0;
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Ray generation
// ------------------------------------------------------------

std::vector<Ray> generateTrainingRays()
{
    std::vector<Ray> rays;
    rays.reserve(trainingRayCount);

    const double epsilon = 1e-6;

    for (std::size_t i = 0; i < trainingRayCount; ++i) {

        const double t =
            static_cast<double>(i) /
            static_cast<double>(trainingRayCount - 1);

        const double y =
            -0.9 * radius + 1.8 * radius * t;

        rays.emplace_back(
            glm::dvec3(xMin + epsilon, y, 0.0),
            glm::dvec3(1.0, 0.0, 0.0)
        );
    }

    return rays;
}

std::vector<double> visualizationOffsets()
{
    return {
        0.0,
        radius / 3.0,
        -radius / 3.0,
        2.0 * radius / 3.0,
        -2.0 * radius / 3.0
    };
}

// ------------------------------------------------------------
// Forward tracing and adjoint accumulation
// ------------------------------------------------------------

struct OptimizationEvaluation {
    double loss = 0.0;
    std::vector<double> gradient;
};

OptimizationEvaluation evaluateMedium(
    const VoxelMedium& medium,
    const Tracer& tracer,
    const AdjointStateSolver& adjoint,
    const LuneburgLoss& lossFunction,
    const std::vector<Ray>& rays
)
{
    OptimizationEvaluation evaluation;

    evaluation.gradient.assign(
        medium.parameterCount(),
        0.0
    );

    for (const Ray& ray : rays) {

        const TraceResult trace =
            tracer.forward(ray, medium);

        if (!trace.completed) {
            throw std::runtime_error(
                "A training ray did not leave the voxel medium"
            );
        }

        const TerminalLossResult terminalLoss =
            lossFunction.evaluate(
                trace.finalState.position,
                trace.finalState.direction
            );

        evaluation.loss += terminalLoss.value;

        ForwardResult forwardResult{
            trace.finalState.position,
            trace.finalState.direction,
            trace.stepsTaken
        };

        const std::vector<double> rayGradient =
            adjoint.backward(
                forwardResult,
                terminalLoss.gradientPosition,
                terminalLoss.gradientVelocity,
                medium
            );

        if (rayGradient.size() != evaluation.gradient.size()) {
            throw std::runtime_error(
                "Adjoint returned an invalid gradient dimension"
            );
        }

        for (std::size_t j = 0; j < rayGradient.size(); ++j) {
            evaluation.gradient[j] += rayGradient[j];
        }
    }

    const double inverseRayCount =
        1.0 / static_cast<double>(rays.size());

    evaluation.loss *= inverseRayCount;

    for (double& g : evaluation.gradient) {
        g *= inverseRayCount;
    }

    return evaluation;
}

// ------------------------------------------------------------
// Visualization
// ------------------------------------------------------------

void saveForwardVisualization(
    const VoxelMedium& medium,
    const Tracer& tracer,
    const fs::path& outputPath,
    int iteration
)
{
    std::vector<double> xs(fieldResolution);
    std::vector<double> ys(fieldResolution);

    for (int i = 0; i < fieldResolution; ++i) {
        const double t =
            static_cast<double>(i) /
            static_cast<double>(fieldResolution - 1);

        xs[i] = xMin + (xMax - xMin) * t;
        ys[i] = yMin + (yMax - yMin) * t;
    }

    // etaField[j][i] = eta(xs[i], ys[j])
    std::vector<std::vector<double>> etaField(
        fieldResolution,
        std::vector<double>(fieldResolution, 1.0)
    );

    for (int j = 0; j < fieldResolution; ++j) {
        for (int i = 0; i < fieldResolution; ++i) {

            const glm::dvec3 p(
                xs[i],
                ys[j],
                0.0
            );

            etaField[j][i] =
                medium.refractiveIndex(p);
        }
    }

    auto fig = figure(true);
    hold(on);

    imagesc(xMin, xMax, yMin, yMax, etaField);
    colorbar();

    // Lens boundary
    std::vector<double> circleX(circleResolution + 1);
    std::vector<double> circleY(circleResolution + 1);

    for (int i = 0; i <= circleResolution; ++i) {

        const double theta =
            2.0 * glm::pi<double>() *
            static_cast<double>(i) /
            static_cast<double>(circleResolution);

        circleX[i] = radius * std::cos(theta);
        circleY[i] = radius * std::sin(theta);
    }

    plot(circleX, circleY, "k-");

    // Five visualization rays
    const std::vector<double> offsets =
        visualizationOffsets();

    for (double offset : offsets) {

        const double y = offset;

        Ray ray(
            glm::dvec3(xMin + 1e-6, y, 0.0),
            glm::dvec3(1.0, 0.0, 0.0)
        );

        std::vector<double> pathX;
        std::vector<double> pathY;

        pathX.push_back(xMin);
        pathY.push_back(y);

        TraceResult result = tracer.forward(
            ray,
            medium,
            [&](const glm::dvec3& p) {
                pathX.push_back(p.x);
                pathY.push_back(p.y);
            }
        );

        if (result.completed) {
            pathX.push_back(xMax);
            pathY.push_back(result.finalState.position.y);
        }

        plot(pathX, pathY, "r-");
    }

    xlim({xMin, xMax});
    ylim({yMin, yMax});

    axis(equal);
    grid(on);

    xlabel("x");
    ylabel("y");

    title(
        "Optimized Luneburg Lens - Step " +
        std::to_string(iteration)
    );

    save(outputPath.string());
}

// ------------------------------------------------------------
// Final plots
// ------------------------------------------------------------

void saveErrorPlot(
    const std::vector<double>& errors,
    const fs::path& outputPath
)
{
    std::vector<double> steps(errors.size());

    for (std::size_t i = 0; i < errors.size(); ++i) {
        steps[i] = static_cast<double>(i);
    }

    auto fig = figure(true);
    hold(on);

    plot(steps, errors, "b-");

    xlabel("Optimization step");
    ylabel("Mean terminal distance");
    title("Luneburg optimization error");

    grid(on);

    save(outputPath.string());
}

void saveEtaComparison(
    const VoxelMedium& medium,
    const fs::path& outputPath
)
{
    constexpr int samples = 400;

    std::vector<double> xs(samples);
    std::vector<double> etaOptimized(samples);
    std::vector<double> etaAnalytical(samples);

    LuneburgLens analyticalLens(
        radius,
        lensCenter()
    );

    for (int i = 0; i < samples; ++i) {

        const double t =
            static_cast<double>(i) /
            static_cast<double>(samples - 1);

        const double x =
            xMin + (xMax - xMin) * t;

        const glm::dvec3 p(x, 0.0, 0.0);

        xs[i] = x;

        etaOptimized[i] =
            medium.refractiveIndex(p);

        etaAnalytical[i] =
            analyticalLens.refractiveIndex(p);
    }

    auto fig = figure(true);
    hold(on);

    plot(xs, etaOptimized, "b-");
    plot(xs, etaAnalytical, "r--");

    xlabel("x");
    ylabel("Refractive index eta");

    title("Optimized vs analytical Luneburg lens");

    legend({
        "Optimized voxel medium",
        "Analytical Luneburg lens"
    });

    grid(on);

    save(outputPath.string());
}

// ------------------------------------------------------------
// CSV
// ------------------------------------------------------------

void saveErrorsCSV(
    const std::vector<double>& errors,
    const fs::path& outputPath
)
{
    std::ofstream file(outputPath);

    if (!file) {
        throw std::runtime_error(
            "Could not create error CSV file"
        );
    }

    file << "step,error\n";
    file << std::setprecision(16);

    for (std::size_t i = 0; i < errors.size(); ++i) {
        file << i << "," << errors[i] << "\n";
    }
}

} // namespace

// ============================================================
// Main
// ============================================================

int main()
{
    try {

        // ----------------------------------------------------
        // 1. Output directory
        // ----------------------------------------------------

        const fs::path outputDirectory =
            "Examples/OptimizeLuneburg";

        fs::create_directories(outputDirectory);

        // ----------------------------------------------------
        // 2. Create voxel medium
        // ----------------------------------------------------

        const glm::dvec3 origin = mediumOrigin();
        const glm::dvec3 size = mediumSize();

        glm::uvec3 resolution(
            initialResolution,
            initialResolution,
            2
        );

        VoxelMedium medium(
            origin,
            size,
            resolution,
            1.0
        );

        projectOutsideLens(medium);

        // ----------------------------------------------------
        // 3. Configure tracer
        // ----------------------------------------------------

        TracerConfig tracerConfig;

        tracerConfig.stepSize = 0.001f;
        tracerConfig.maxSteps = 10000;

        Tracer tracer(tracerConfig);

        // ----------------------------------------------------
        // 4. Configure adjoint solver
        // ----------------------------------------------------

        SolverConfig solverConfig;

        solverConfig.stepSize =
            static_cast<double>(tracerConfig.stepSize);

        solverConfig.maxSteps = tracerConfig.maxSteps;

        AdjointStateSolver adjoint(solverConfig);

        // ----------------------------------------------------
        // 5. Loss and optimizer
        // ----------------------------------------------------

        LuneburgLoss lossFunction(targetPoint());

        AdamConfig adamConfig;

        adamConfig.learningRate = learningRate;
        adamConfig.beta1 = 0.9;
        adamConfig.beta2 = 0.999;
        adamConfig.epsilon = 1e-8;

        Adam optimizer(adamConfig);

        // ----------------------------------------------------
        // 6. Generate training rays
        // ----------------------------------------------------

        const std::vector<Ray> trainingRays =
            generateTrainingRays();

        std::cout
            << "Training rays: "
            << trainingRays.size()
            << "\n";

        std::cout
            << "Voxel parameters: "
            << medium.parameterCount()
            << "\n";

        // ----------------------------------------------------
        // 7. Optimization loop
        // ----------------------------------------------------

        std::vector<double> errors;

        double previousLoss =
            std::numeric_limits<double>::infinity();

        int stableIterations = 0;

        for (int iteration = 0;
             iteration < maxIterations;
             ++iteration)
        {
            // Forward + terminal loss + adjoint gradient
            OptimizationEvaluation evaluation =
                evaluateMedium(
                    medium,
                    tracer,
                    adjoint,
                    lossFunction,
                    trainingRays
                );

            errors.push_back(evaluation.loss);

            std::cout
                << "Step "
                << std::setw(5) << iteration
                << " | loss = "
                << std::setprecision(10)
                << evaluation.loss
                << "\n";

            // Save visualization every N steps.
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
                    tracer,
                    imagePath,
                    iteration
                );
            }

            // Convergence criterion
            if (std::isfinite(previousLoss)) {

                const double lossChange =
                    std::abs(previousLoss - evaluation.loss);

                if (lossChange < convergenceTolerance) {
                    ++stableIterations;
                } else {
                    stableIterations = 0;
                }
            }

            if (stableIterations >= convergencePatience) {
                std::cout
                    << "Converged after "
                    << iteration + 1
                    << " iterations.\n";

                break;
            }

            previousLoss = evaluation.loss;

            // Fixed parameters outside the lens.
            maskOutsideGradient(
                medium,
                evaluation.gradient
            );

            // Adam update.
            const std::vector<double> parameters =
                medium.getParameters();

            const std::vector<double> updatedParameters =
                optimizer.step(
                    parameters,
                    evaluation.gradient
                );

            medium.setParameters(updatedParameters);

            // Project the exterior back to eta = 1.
            projectOutsideLens(medium);

            // ------------------------------------------------
            // Optional grid refinement
            // ------------------------------------------------

            if constexpr (enableRefinement) {

                if (iteration > 0 &&
                    iteration % refinementInterval == 0)
                {
                    const glm::uvec3 current =
                        medium.resolution();

                    const glm::uvec3 refined(
                        current.x * refinementFactor,
                        current.y * refinementFactor,
                        current.z
                    );

                    medium.refine(refined);

                    projectOutsideLens(medium);

                    // Adam's internal moments have the old dimension.
                    optimizer.reset();
                }
            }
        }

        // ----------------------------------------------------
        // 8. Save final outputs
        // ----------------------------------------------------

        saveErrorsCSV(
            errors,
            outputDirectory / "errors.csv"
        );

        saveErrorPlot(
            errors,
            outputDirectory / "error.png"
        );

        saveEtaComparison(
            medium,
            outputDirectory / "eta_comparison.png"
        );

        // Save final forward visualization too.
        saveForwardVisualization(
            medium,
            tracer,
            outputDirectory / "forward_final.png",
            static_cast<int>(errors.size())
        );

        std::cout
            << "\nOptimization finished.\n"
            << "Results saved in: "
            << outputDirectory.string()
            << "\n";

    }
    catch (const std::exception& e) {

        std::cerr
            << "Optimization failed: "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}
*/

#include "Simulation/Tracer.h"
#include "Simulation/Ray.h"
#include "MediumType/LuneburgLens.h"

#include <matplot/matplot.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// ------------------------------------------------------------
// Analytical Luneburg lens visualization
// ------------------------------------------------------------

int main()
{
    using namespace matplot;

    // --------------------------------------------------------
    // 1. Lens configuration
    // --------------------------------------------------------

    const double radius = 1.0;
    const glm::dvec3 center(0.0, 0.0, 0.0);

    LuneburgLens lens(radius, center);

    // --------------------------------------------------------
    // 2. Tracer configuration
    // --------------------------------------------------------

    TracerConfig config;
    config.stepSize = 0.001;
    config.maxSteps = 10000;

    Tracer tracer(config);

    // --------------------------------------------------------
    // 3. Generate eta field over the XY plane
    // --------------------------------------------------------

    constexpr int resolution = 300;

    const double xMin = center.x - 1.5 * radius;
    const double xMax = center.x + 1.5 * radius;
    const double yMin = center.y - 1.5 * radius;
    const double yMax = center.y + 1.5 * radius;

    std::vector<double> xs(resolution);
    std::vector<double> ys(resolution);

    for (int i = 0; i < resolution; ++i)
    {
        xs[i] = xMin + (xMax - xMin) * i / (resolution - 1);
        ys[i] = yMin + (yMax - yMin) * i / (resolution - 1);
    }

    // Z[j][i] = eta(xs[i], ys[j])
    std::vector<std::vector<double>> etaField(
        resolution,
        std::vector<double>(resolution, 1.0)
    );

    for (int j = 0; j < resolution; ++j)
    {
        for (int i = 0; i < resolution; ++i)
        {
            glm::dvec3 position(xs[i], ys[j], center.z);
            etaField[j][i] = lens.refractiveIndex(position);
        }
    }

    // --------------------------------------------------------
    // 4. Generate five parallel incident rays
    // --------------------------------------------------------

    // Offsets relative to the lens center and radius:
    // center, +R/3, +2R/3, -R/3, -2R/3.
    const std::vector<double> offsets = {
        0.0,
        radius / 3.0,
        2.0 * radius / 3.0,
        -radius / 3.0,
        -2.0 * radius / 3.0
    };

    const double rayStartX = xMin;

    std::vector<std::vector<double>> rayPathsX;
    std::vector<std::vector<double>> rayPathsY;

    for (double offset : offsets)
    {
        const double rayY = center.y + offset;

        // Analytic intersection with the left side of the sphere:
        //
        // (x - cx)^2 + (y - cy)^2 = R^2
        //
        // xEntry = cx - sqrt(R^2 - (y - cy)^2)

        const double discriminant =
            radius * radius - offset * offset;

        if (discriminant < 0.0)
            continue;

        const double entryX =
            center.x - std::sqrt(discriminant);

        glm::dvec3 entryPosition(
            entryX,
            rayY,
            center.z
        );

        // Initial ray direction: parallel to the positive X axis.
        glm::dvec3 direction(1.0, 0.0, 0.0);

        Ray ray(entryPosition, direction);

        // Store the complete path, including the external segment.
        std::vector<double> pathX;
        std::vector<double> pathY;

        pathX.push_back(rayStartX);
        pathY.push_back(rayY);

        // Entry point.
        pathX.push_back(entryPosition.x);
        pathY.push_back(entryPosition.y);

        // ----------------------------------------------------
        // 5. Integrate the ray inside the lens
        // ----------------------------------------------------

        TraceResult result = tracer.forward(
            ray,
            lens,
            [&](const glm::dvec3& position)
            {
                pathX.push_back(position.x);
                pathY.push_back(position.y);
            }
        );

        if (!result.completed)
        {
            std::cerr
                << "Warning: ray did not exit the lens within maxSteps\n";
        }

        // The last integrated position is already outside the lens
        // (the integration step may slightly overshoot the boundary).
        // Extend the ray horizontally to the right side of the plot.
        const glm::dvec3 exitPosition = result.finalState.position;

        pathX.push_back(xMax);
        pathY.push_back(exitPosition.y);

        rayPathsX.push_back(std::move(pathX));
        rayPathsY.push_back(std::move(pathY));
    }

    // --------------------------------------------------------
    // 6. Plot eta field, lens boundary and ray paths
    // --------------------------------------------------------

    auto fig = figure(true);
    hold(on);

    // Refractive index field.
    imagesc(xMin, xMax, yMin, yMax, etaField);
    colorbar();

    // Lens boundary.
    constexpr int circleResolution = 300;

    std::vector<double> circleX(circleResolution + 1);
    std::vector<double> circleY(circleResolution + 1);

    for (int i = 0; i <= circleResolution; ++i)
    {
        const double theta =
            2.0 * glm::pi<double>() * i / circleResolution;

        circleX[i] =
            center.x + radius * std::cos(theta);

        circleY[i] =
            center.y + radius * std::sin(theta);
    }

    plot(circleX, circleY, "k-");

    // Complete ray paths.
    for (std::size_t i = 0; i < rayPathsX.size(); ++i)
    {
        plot(rayPathsX[i], rayPathsY[i], "r-");
    }

    // Plot formatting.
    xlim({xMin, xMax});
    ylim({yMin, yMax});

    axis(equal);
    grid(on);

    xlabel("x");
    ylabel("y");
    title("Analytical Luneburg Lens - Ray Tracing");

    show();

    return 0;
}

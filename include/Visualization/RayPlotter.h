#pragma once

#include <vector>
#include <glm/glm.hpp>

class RayPlotter {
public:
    using RayPath = std::vector<glm::dvec3>;

    void plot2D(
        const std::vector<RayPath>& paths,
        const glm::dvec3& lensCenter,
        double lensRadius
    ) const;
};
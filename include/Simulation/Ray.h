#pragma once

#include <glm/glm.hpp>
#include <vector>

struct Ray {
    glm::dvec3 position;
    glm::dvec3 direction;

    Ray(
        const glm::dvec3& position,
        const glm::dvec3& direction
    )
        : position(position),
          direction(direction)
    {}
};

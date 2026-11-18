#pragma once

#include <glm/glm.hpp>

class SignedDistanceField {
public:
    virtual ~SignedDistanceField() = default;

    virtual double evaluate(
        const glm::dvec3& position
    ) const = 0;

    virtual glm::dvec3 gradient(
        const glm::dvec3& position
    ) const = 0;
};
#pragma once

#include "Simulation/Medium.h"
#include <glm/glm.hpp>
#include <cmath>
#include <stdexcept>

class LuneburgLens : public Medium {
public:

    LuneburgLens(double radius, const glm::dvec3& center)
    {
        if (radius <= 0.0)
            throw std::invalid_argument("Lens radius must be positive");

        radius_ = radius;
        center_ = center;
    }

    double refractiveIndex(const glm::dvec3& position) const override
    {
        glm::dvec3 displacement = position - center_;
        double r2 = glm::dot(displacement, displacement);

        if (r2 > radius_ * radius_)
            return 1.0;

        return std::sqrt(2.0 - r2 / (radius_ * radius_));
    }

    glm::dvec3 gradient(const glm::dvec3& position) const override
    {
        glm::dvec3 displacement = position - center_;
        double r2 = glm::dot(displacement, displacement);

        if (r2 >= radius_ * radius_)
            return glm::dvec3(0.0);

        double n = std::sqrt(2.0 - r2 / (radius_ * radius_));

        return -displacement / (radius_ * radius_ * n);
    }

    double getRadius() const { return radius_; }

    glm::dvec3 getCenter() const { return center_; }

    bool isInside(const glm::dvec3& position) const override
    {
        glm::dvec3 d = position - center_;

        return glm::dot(d, d) <= radius_ * radius_;
    }

private:

    double radius_;
    glm::dvec3 center_;
};
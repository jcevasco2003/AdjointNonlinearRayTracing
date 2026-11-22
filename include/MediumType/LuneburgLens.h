#pragma once

#include <glm/glm.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstddef>

class LuneburgLens {
public:

    LuneburgLens(double radius, const glm::dvec3& center)
    {
        if (radius <= 0.0)
            throw std::invalid_argument("Lens radius must be positive");

        radius_ = radius;
        center_ = center;
    }

    double refractiveIndex(const glm::dvec3& position) const
    {
        glm::dvec3 displacement = position - center_;
        double r2 = glm::dot(displacement, displacement);

        if (r2 > radius_ * radius_)
            return 1.0;

        return std::sqrt(2.0 - r2 / (radius_ * radius_));
    }

    glm::dvec3 gradient(const glm::dvec3& position) const
    {
        glm::dvec3 displacement = position - center_;
        double r2 = glm::dot(displacement, displacement);

        if (r2 >= radius_ * radius_)
            return glm::dvec3(0.0);

        double n = std::sqrt(2.0 - r2 / (radius_ * radius_));

        return -displacement / (radius_ * radius_ * n);
    }

    glm::dmat3 hessian(const glm::dvec3& position) const
    {
        glm::dvec3 d = position - center_;
        double r2 = glm::dot(d, d);

        if (r2 >= radius_ * radius_)
            return glm::dmat3(0.0);

        double R2 = radius_ * radius_;
        double n = std::sqrt(2.0 - r2 / R2);

        glm::dmat3 identity(1.0);
        glm::dmat3 outer = glm::outerProduct(d, d);

        return -identity / (R2 * n)
            -outer / (R2 * R2 * n * n * n);
    }

    double getRadius() const { return radius_; }

    glm::dvec3 getCenter() const { return center_; }

    bool isInside(const glm::dvec3& position) const
    {
        glm::dvec3 d = position - center_;

        return glm::dot(d, d) <= radius_ * radius_;
    }

    std::vector<double> indexParameterGradient(
        const glm::dvec3& position
    ) const
    {
        return {};
    }

    std::vector<glm::dvec3> gradientParameterGradient(
        const glm::dvec3& position
    ) const
    {
        return {};
    }

    std::size_t parameterCount() const
    {
        return 0;
    }

private:

    double radius_;
    glm::dvec3 center_;
};
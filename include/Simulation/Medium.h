#pragma once

#include <glm/glm.hpp>
#include <vector>

class Medium {
public:
    virtual ~Medium() = default;

    virtual bool isInside(const glm::dvec3& x) const = 0;

    virtual double refractiveIndex(
        const glm::dvec3& x
    ) const = 0;

    virtual glm::dvec3 gradient(
        const glm::dvec3& x
    ) const = 0;

    virtual glm::dmat3 hessian(
        const glm::dvec3& x
    ) const = 0;

    virtual std::vector<double> indexParameterGradient(
        const glm::dvec3& x
    ) const = 0;

    virtual std::vector<glm::dvec3> gradientParameterGradient(
        const glm::dvec3& x
    ) const = 0;
    
    virtual std::size_t parameterCount() const = 0;
};
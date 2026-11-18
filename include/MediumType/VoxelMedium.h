#pragma once

#include "Simulation/Medium.h"

#include <glm/glm.hpp>
#include <vector>
#include <cstddef>

class VoxelMedium : public Medium {
public:
    VoxelMedium(
        const glm::dvec3& origin,
        const glm::dvec3& size,
        const glm::uvec3& resolution,
        double initialEta = 1.0
    );

    // Medium interface
    double refractiveIndex(
        const glm::dvec3& position
    ) const override;

    glm::dvec3 gradient(
        const glm::dvec3& position
    ) const override;

    glm::dmat3 hessian(
        const glm::dvec3& position
    ) const override;

    std::vector<double> indexParameterGradient(
        const glm::dvec3& position
    ) const override;

    std::vector<glm::dvec3> gradientParameterGradient(
        const glm::dvec3& position
    ) const override;

    bool isInside(const glm::dvec3& position) const override;

    std::size_t parameterCount() const override;

    // Parameters
    std::vector<double> getParameters() const;

    void setParameters(
        const std::vector<double>& parameters
    );

    // Grid refinement
    void refine(const glm::uvec3& newResolution);

    // Grid information
    glm::uvec3 resolution() const;
    glm::dvec3 voxelSize() const;

    double& voxel(
        std::size_t i,
        std::size_t j,
        std::size_t k
    );

    double voxel(
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) const;

private:
    std::size_t index(
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) const;

    double interpolateScalar(
        const glm::dvec3& position,
        const std::vector<double>& field
    ) const;

    glm::dvec3 interpolateVector(
        const glm::dvec3& position,
        const std::vector<glm::dvec3>& field
    ) const;

    glm::dvec3 origin_;
    glm::dvec3 size_;
    glm::uvec3 resolution_;

    std::vector<double> voxels_;
};

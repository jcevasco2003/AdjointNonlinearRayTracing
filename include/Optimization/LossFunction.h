
#pragma once

#include <glm/glm.hpp>

struct TerminalLossResult {
    // L(x_f, v_f)
    double value = 0.0;

    // Partial derivatives of L with respect to the terminal state
    glm::dvec3 gradientPosition{0.0};
    glm::dvec3 gradientVelocity{0.0};
};

class LossFunction {
public:
    virtual ~LossFunction() = default;

    virtual TerminalLossResult evaluate(
        const glm::dvec3& finalPosition,
        const glm::dvec3& finalVelocity
    ) const = 0;
};

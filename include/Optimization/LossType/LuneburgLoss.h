
#pragma once

#include "Optimization/LossFunction.h"

#include <glm/glm.hpp>

class LuneburgLoss : public LossFunction {
public:
    explicit LuneburgLoss(
        const glm::dvec3& targetPoint
    );

    TerminalLossResult evaluate(
        const glm::dvec3& finalPosition,
        const glm::dvec3& finalVelocity
    ) const override;

private:
    glm::dvec3 targetPoint_;
};

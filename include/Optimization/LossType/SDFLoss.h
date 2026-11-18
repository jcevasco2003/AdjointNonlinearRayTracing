
#pragma once

#include "Optimization/LossFunction.h"
#include "Geometry/SignedDistanceField.h"

class SDFLoss : public LossFunction {
public:
    SDFLoss(
        const SignedDistanceField& sdf,
        double multiplier
    );

    TerminalLossResult evaluate(
        const glm::dvec3& finalPosition,
        const glm::dvec3& finalVelocity
    ) const override;

private:
    const SignedDistanceField& sdf_;
    double multiplier_;
};

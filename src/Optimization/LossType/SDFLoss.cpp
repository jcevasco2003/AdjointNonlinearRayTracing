
#include "Optimization/LossType/SDFLoss.h"

SDFLoss::SDFLoss(
    const SignedDistanceField& sdf,
    double multiplier
)
    : sdf_(sdf),
      multiplier_(multiplier)
{
}

TerminalLossResult SDFLoss::evaluate(
    const glm::dvec3& finalPosition,
    const glm::dvec3& finalVelocity
) const
{
    TerminalLossResult result;

    // L = -p * G(x_f)
    result.value = -multiplier_ * sdf_.evaluate(finalPosition);

    // dL/dx_f = -p * grad(G(x_f))
    result.gradientPosition =
        -multiplier_ * sdf_.gradient(finalPosition);

    // The loss does not depend on final velocity.
    result.gradientVelocity = glm::dvec3(0.0);

    return result;
}

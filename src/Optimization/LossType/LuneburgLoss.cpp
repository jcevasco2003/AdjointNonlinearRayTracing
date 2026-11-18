
#include "Optimization/LossType/LuneburgLoss.h"

#include <cmath>

LuneburgLoss::LuneburgLoss(
    const glm::dvec3& targetPoint
)
    : targetPoint_(targetPoint)
{}

TerminalLossResult LuneburgLoss::evaluate(
    const glm::dvec3& finalPosition,
    const glm::dvec3& /* finalVelocity */
) const
{
    TerminalLossResult result;

    const glm::dvec3 difference =
        finalPosition - targetPoint_;

    const double distanceSquared =
        glm::dot(difference, difference);

    const double distance = std::sqrt(distanceSquared);

    result.value = distance;

    // Subgradient at zero: choose the zero vector.
    if (distance > 1e-12) {
        result.gradientPosition = difference / distance;
    } else {
        result.gradientPosition = glm::dvec3(0.0);
    }

    result.gradientVelocity = glm::dvec3(0.0);

    return result;
}

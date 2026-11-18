#include "Optimization/AdjointStateSolver.h"

#include <stdexcept>
#include <algorithm>


std::vector<double> AdjointStateSolver::backward(
    const ForwardResult& forwardResult,
    const glm::dvec3& deltaX,
    const glm::dvec3& deltaV,
    const Medium& medium
) const
{
    glm::dvec3 x = forwardResult.finalPosition;
    glm::dvec3 v = forwardResult.finalVelocity;

    const double dt = config_.stepSize;

    // Terminal conditions for the adjoint variables
    glm::dvec3 mu = deltaV;
    glm::dvec3 lambda = deltaX + deltaV * dt;

    // Number of optimizable medium parameters
    const std::size_t parameterCount = medium.parameterCount();

    std::vector<double> deltaTheta(parameterCount, 0.0);

    for (std::size_t step = 0;
         step < forwardResult.steps;
         ++step)
    {
        // Reverse position integration
        x -= v * dt;

        const double eta = medium.refractiveIndex(x);
        const glm::dvec3 gradEta = medium.gradient(x);
        const glm::dmat3 hessianEta = medium.hessian(x);

        // Reverse velocity integration
        v -= eta * gradEta * dt;

        // Jacobian contribution of eta * grad(eta)
        const glm::dmat3 A =
            glm::outerProduct(gradEta, gradEta)
            + eta * hessianEta;

        // Adjoint integration
        lambda += A * mu * dt;
        mu += lambda * dt;

        // Derivatives with respect to medium parameters
        const std::vector<double> dEta_dTheta =
            medium.indexParameterGradient(x);

        const std::vector<glm::dvec3> dGradEta_dTheta =
            medium.gradientParameterGradient(x);

        if (dEta_dTheta.size() != parameterCount ||
            dGradEta_dTheta.size() != parameterCount)
        {
            throw std::runtime_error(
                "Medium returned inconsistent parameter dimensions"
            );
        }

        for (std::size_t i = 0; i < parameterCount; ++i)
        {
            deltaTheta[i] += dt * (
                glm::dot(mu, gradEta) * dEta_dTheta[i]
                + eta * glm::dot(mu, dGradEta_dTheta[i])
            );
        }
    }

    return deltaTheta;
}
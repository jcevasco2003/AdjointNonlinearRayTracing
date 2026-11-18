
#include "MediumType/VoxelMedium.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

struct Cell {
    std::size_t i0, i1;
    std::size_t j0, j1;
    std::size_t k0, k1;

    double u, v, w;
};

struct InterpolationWeights {
    double value[2][2][2]{};
    glm::dvec3 gradient[2][2][2]{};

    // [0] = d²w / dx²
    // [1] = d²w / dy²
    // [2] = d²w / dz²
    // [3] = d²w / dxdy
    // [4] = d²w / dxdz
    // [5] = d²w / dydz
    double hessian[2][2][2][6]{};
};

std::size_t flatten(
    std::size_t i,
    std::size_t j,
    std::size_t k,
    const glm::uvec3& resolution
)
{
    return i +
        static_cast<std::size_t>(resolution.x) *
        (j + static_cast<std::size_t>(resolution.y) * k);
}

// En 2D hay una sola contribución en Z.
// En 3D hay dos.
int zCornerCount(const glm::uvec3& resolution)
{
    return resolution.z == 1 ? 1 : 2;
}

Cell locateCell(
    const glm::dvec3& position,
    const glm::dvec3& origin,
    const glm::dvec3& size,
    const glm::uvec3& resolution
)
{
    const double hx =
        size.x / static_cast<double>(resolution.x - 1);

    const double hy =
        size.y / static_cast<double>(resolution.y - 1);

    const double gx = std::clamp(
        (position.x - origin.x) / hx,
        0.0,
        static_cast<double>(resolution.x - 1)
    );

    const double gy = std::clamp(
        (position.y - origin.y) / hy,
        0.0,
        static_cast<double>(resolution.y - 1)
    );

    Cell cell{};

    cell.i0 = std::min(
        static_cast<std::size_t>(std::floor(gx)),
        static_cast<std::size_t>(resolution.x - 2)
    );

    cell.j0 = std::min(
        static_cast<std::size_t>(std::floor(gy)),
        static_cast<std::size_t>(resolution.y - 2)
    );

    cell.i1 = cell.i0 + 1;
    cell.j1 = cell.j0 + 1;

    cell.u = gx - static_cast<double>(cell.i0);
    cell.v = gy - static_cast<double>(cell.j0);

    if (resolution.z == 1) {
        // Medio 2D: una única capa.
        cell.k0 = 0;
        cell.k1 = 0;
        cell.w = 0.0;
    } else {
        // Medio 3D.
        const double hz =
            size.z / static_cast<double>(resolution.z - 1);

        const double gz = std::clamp(
            (position.z - origin.z) / hz,
            0.0,
            static_cast<double>(resolution.z - 1)
        );

        cell.k0 = std::min(
            static_cast<std::size_t>(std::floor(gz)),
            static_cast<std::size_t>(resolution.z - 2)
        );

        cell.k1 = cell.k0 + 1;
        cell.w = gz - static_cast<double>(cell.k0);
    }

    return cell;
}

InterpolationWeights computeWeights(
    const Cell& cell,
    const glm::dvec3& size,
    const glm::uvec3& resolution
)
{
    const double hx =
        size.x / static_cast<double>(resolution.x - 1);

    const double hy =
        size.y / static_cast<double>(resolution.y - 1);

    const bool is2D = resolution.z == 1;

    const double hz = is2D
        ? 1.0
        : size.z / static_cast<double>(resolution.z - 1);

    const double U[2] = {
        1.0 - cell.u,
        cell.u
    };

    const double V[2] = {
        1.0 - cell.v,
        cell.v
    };

    const double W[2] = {
        is2D ? 1.0 : 1.0 - cell.w,
        is2D ? 0.0 : cell.w
    };

    const double dU[2] = {
        -1.0 / hx,
         1.0 / hx
    };

    const double dV[2] = {
        -1.0 / hy,
         1.0 / hy
    };

    const double dW[2] = {
        is2D ? 0.0 : -1.0 / hz,
        is2D ? 0.0 :  1.0 / hz
    };

    InterpolationWeights result{};

    const int nz = zCornerCount(resolution);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                result.value[a][b][c] =
                    U[a] * V[b] * W[c];

                result.gradient[a][b][c] = glm::dvec3(
                    dU[a] * V[b] * W[c],
                    U[a] * dV[b] * W[c],
                    U[a] * V[b] * dW[c]
                );

                // Las segundas derivadas puras son cero
                // dentro de una celda bilineal/trilineal.
                result.hessian[a][b][c][0] = 0.0;
                result.hessian[a][b][c][1] = 0.0;
                result.hessian[a][b][c][2] = 0.0;

                result.hessian[a][b][c][3] =
                    dU[a] * dV[b] * W[c];

                result.hessian[a][b][c][4] =
                    dU[a] * V[b] * dW[c];

                result.hessian[a][b][c][5] =
                    U[a] * dV[b] * dW[c];
            }
        }
    }

    return result;
}

} // namespace

// ------------------------------------------------------------
// Construction
// ------------------------------------------------------------

VoxelMedium::VoxelMedium(
    const glm::dvec3& origin,
    const glm::dvec3& size,
    const glm::uvec3& resolution,
    double initialEta
)
    : origin_(origin),
      size_(size),
      resolution_(resolution)
{
    if (resolution_.x < 2 ||
        resolution_.y < 2 ||
        resolution_.z < 1) {
        throw std::invalid_argument(
            "VoxelMedium requires resolution.x >= 2, "
            "resolution.y >= 2 and resolution.z >= 1"
        );
    }

    if (size_.x <= 0.0 || size_.y <= 0.0) {
        throw std::invalid_argument(
            "VoxelMedium X and Y sizes must be positive"
        );
    }

    if (resolution_.z >= 2 && size_.z <= 0.0) {
        throw std::invalid_argument(
            "VoxelMedium Z size must be positive in 3D"
        );
    }

    if (resolution_.z == 1 && size_.z < 0.0) {
        throw std::invalid_argument(
            "VoxelMedium Z size cannot be negative in 2D"
        );
    }

    const std::size_t count =
        static_cast<std::size_t>(resolution_.x) *
        resolution_.y *
        resolution_.z;

    voxels_.assign(count, initialEta);
}

// ------------------------------------------------------------
// Grid indexing and access
// ------------------------------------------------------------

std::size_t VoxelMedium::index(
    std::size_t i,
    std::size_t j,
    std::size_t k
) const
{
    if (i >= resolution_.x ||
        j >= resolution_.y ||
        k >= resolution_.z) {
        throw std::out_of_range(
            "VoxelMedium grid index out of range"
        );
    }

    return flatten(i, j, k, resolution_);
}

glm::dvec3 VoxelMedium::voxelSize() const
{
    const double hz = resolution_.z == 1
        ? 0.0
        : size_.z / static_cast<double>(resolution_.z - 1);

    return glm::dvec3(
        size_.x / static_cast<double>(resolution_.x - 1),
        size_.y / static_cast<double>(resolution_.y - 1),
        hz
    );
}

glm::uvec3 VoxelMedium::resolution() const
{
    return resolution_;
}

double& VoxelMedium::voxel(
    std::size_t i,
    std::size_t j,
    std::size_t k
)
{
    return voxels_[index(i, j, k)];
}

double VoxelMedium::voxel(
    std::size_t i,
    std::size_t j,
    std::size_t k
) const
{
    return voxels_[index(i, j, k)];
}

// ------------------------------------------------------------
// Scalar interpolation
// ------------------------------------------------------------

double VoxelMedium::interpolateScalar(
    const glm::dvec3& position,
    const std::vector<double>& field
) const
{
    const Cell cell =
        locateCell(position, origin_, size_, resolution_);

    const InterpolationWeights weights =
        computeWeights(cell, size_, resolution_);

    double result = 0.0;

    const int nz = zCornerCount(resolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const std::size_t i = a ? cell.i1 : cell.i0;
                const std::size_t j = b ? cell.j1 : cell.j0;
                const std::size_t k = c ? cell.k1 : cell.k0;

                result += weights.value[a][b][c] *
                    field[index(i, j, k)];
            }
        }
    }

    return result;
}

// ------------------------------------------------------------
// Medium interface
// ------------------------------------------------------------

double VoxelMedium::refractiveIndex(
    const glm::dvec3& position
) const
{
    return interpolateScalar(position, voxels_);
}

glm::dvec3 VoxelMedium::gradient(
    const glm::dvec3& position
) const
{
    const Cell cell =
        locateCell(position, origin_, size_, resolution_);

    const InterpolationWeights weights =
        computeWeights(cell, size_, resolution_);

    glm::dvec3 result(0.0);

    const int nz = zCornerCount(resolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const std::size_t i = a ? cell.i1 : cell.i0;
                const std::size_t j = b ? cell.j1 : cell.j0;
                const std::size_t k = c ? cell.k1 : cell.k0;

                result += weights.gradient[a][b][c] *
                    voxel(i, j, k);
            }
        }
    }

    return result;
}

glm::dmat3 VoxelMedium::hessian(
    const glm::dvec3& position
) const
{
    const Cell cell =
        locateCell(position, origin_, size_, resolution_);

    const InterpolationWeights weights =
        computeWeights(cell, size_, resolution_);

    glm::dmat3 H(0.0);

    const int nz = zCornerCount(resolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const std::size_t i = a ? cell.i1 : cell.i0;
                const std::size_t j = b ? cell.j1 : cell.j0;
                const std::size_t k = c ? cell.k1 : cell.k0;

                const double theta = voxel(i, j, k);

                const double dxy =
                    weights.hessian[a][b][c][3] * theta;

                const double dxz =
                    weights.hessian[a][b][c][4] * theta;

                const double dyz =
                    weights.hessian[a][b][c][5] * theta;

                // GLM usa [columna][fila].
                H[0][1] += dxy;
                H[1][0] += dxy;

                H[0][2] += dxz;
                H[2][0] += dxz;

                H[1][2] += dyz;
                H[2][1] += dyz;
            }
        }
    }

    return H;
}

// ------------------------------------------------------------
// Parameter derivatives
// ------------------------------------------------------------

std::vector<double> VoxelMedium::indexParameterGradient(
    const glm::dvec3& position
) const
{
    const Cell cell =
        locateCell(position, origin_, size_, resolution_);

    const InterpolationWeights weights =
        computeWeights(cell, size_, resolution_);

    std::vector<double> result(
        parameterCount(), 0.0
    );

    const int nz = zCornerCount(resolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const std::size_t i = a ? cell.i1 : cell.i0;
                const std::size_t j = b ? cell.j1 : cell.j0;
                const std::size_t k = c ? cell.k1 : cell.k0;

                result[index(i, j, k)] =
                    weights.value[a][b][c];
            }
        }
    }

    return result;
}

std::vector<glm::dvec3> VoxelMedium::gradientParameterGradient(
    const glm::dvec3& position
) const
{
    const Cell cell =
        locateCell(position, origin_, size_, resolution_);

    const InterpolationWeights weights =
        computeWeights(cell, size_, resolution_);

    std::vector<glm::dvec3> result(
        parameterCount(), glm::dvec3(0.0)
    );

    const int nz = zCornerCount(resolution_);

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < nz; ++c) {

                const std::size_t i = a ? cell.i1 : cell.i0;
                const std::size_t j = b ? cell.j1 : cell.j0;
                const std::size_t k = c ? cell.k1 : cell.k0;

                result[index(i, j, k)] =
                    weights.gradient[a][b][c];
            }
        }
    }

    return result;
}

std::size_t VoxelMedium::parameterCount() const
{
    return voxels_.size();
}

// ------------------------------------------------------------
// Parameters
// ------------------------------------------------------------

std::vector<double> VoxelMedium::getParameters() const
{
    return voxels_;
}

void VoxelMedium::setParameters(
    const std::vector<double>& parameters
)
{
    if (parameters.size() != parameterCount()) {
        throw std::invalid_argument(
            "VoxelMedium::setParameters: incorrect parameter count"
        );
    }

    voxels_ = parameters;
}

// ------------------------------------------------------------
// Grid refinement
// ------------------------------------------------------------

void VoxelMedium::refine(
    const glm::uvec3& newResolution
)
{
    if (newResolution.x < 2 ||
        newResolution.y < 2 ||
        newResolution.z < 1) {
        throw std::invalid_argument(
            "VoxelMedium::refine: invalid resolution"
        );
    }

    if (newResolution.x < resolution_.x ||
        newResolution.y < resolution_.y ||
        newResolution.z < resolution_.z) {
        throw std::invalid_argument(
            "VoxelMedium::refine: new resolution cannot be smaller"
        );
    }

    // No cambiar la dimensionalidad durante el refinamiento.
    if ((resolution_.z == 1) != (newResolution.z == 1)) {
        throw std::invalid_argument(
            "VoxelMedium::refine: cannot change dimensionality"
        );
    }

    if (newResolution == resolution_) {
        return;
    }

    const glm::uvec3 oldResolution = resolution_;
    const std::vector<double> oldVoxels = voxels_;

    const std::size_t newCount =
        static_cast<std::size_t>(newResolution.x) *
        newResolution.y *
        newResolution.z;

    std::vector<double> newVoxels(newCount, 0.0);

    // Muestrear el campo anterior en los nodos de la nueva grilla.
    for (std::size_t k = 0; k < newResolution.z; ++k) {
        for (std::size_t j = 0; j < newResolution.y; ++j) {
            for (std::size_t i = 0; i < newResolution.x; ++i) {

                const double x =
                    static_cast<double>(i) * size_.x /
                    static_cast<double>(newResolution.x - 1);

                const double y =
                    static_cast<double>(j) * size_.y /
                    static_cast<double>(newResolution.y - 1);

                double z = 0.0;

                if (newResolution.z >= 2) {
                    z =
                        static_cast<double>(k) * size_.z /
                        static_cast<double>(newResolution.z - 1);
                }

                const glm::dvec3 position =
                    origin_ + glm::dvec3(x, y, z);

                const Cell cell = locateCell(
                    position,
                    origin_,
                    size_,
                    oldResolution
                );

                const InterpolationWeights weights =
                    computeWeights(
                        cell,
                        size_,
                        oldResolution
                    );

                double value = 0.0;

                const int nz = zCornerCount(oldResolution);

                for (int a = 0; a < 2; ++a) {
                    for (int b = 0; b < 2; ++b) {
                        for (int c = 0; c < nz; ++c) {

                            const std::size_t oi =
                                a ? cell.i1 : cell.i0;

                            const std::size_t oj =
                                b ? cell.j1 : cell.j0;

                            const std::size_t ok =
                                c ? cell.k1 : cell.k0;

                            value +=
                                weights.value[a][b][c] *
                                oldVoxels[
                                    flatten(oi, oj, ok, oldResolution)
                                ];
                        }
                    }
                }

                newVoxels[
                    flatten(i, j, k, newResolution)
                ] = value;
            }
        }
    }

    resolution_ = newResolution;
    voxels_ = std::move(newVoxels);
}

// ------------------------------------------------------------
// Is Inside
// ------------------------------------------------------------

bool VoxelMedium::isInside(
    const glm::dvec3& position
) const
{
    const glm::dvec3 local = position - origin_;

    const bool insideXY =
        local.x >= 0.0 && local.x <= size_.x &&
        local.y >= 0.0 && local.y <= size_.y;

    if (!insideXY) {
        return false;
    }

    // En 2D, el medio no tiene extensión ni restricción en Z.
    if (resolution_.z == 1) {
        return true;
    }

    return local.z >= 0.0 && local.z <= size_.z;
}

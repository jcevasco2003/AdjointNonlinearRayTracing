#pragma once

#include "Camera.h"
#include "Light.h"
#include "Sensor.h"
#include "Simulation/Medium.h"

#include <memory>
#include <vector>

class Scene {
public:
    Camera camera;
    std::vector<Light> lights;
    std::vector<Sensor> sensors;

    std::vector<std::shared_ptr<Medium>> media;
};
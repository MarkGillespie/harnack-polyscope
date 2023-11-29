#pragma once

#include "fcpw/fcpw.h" // eigen
#include "geometry_utils.h"
#include "polyscope/polyscope.h" // glm


// compute generalized barycentric coordinates for point v0 inside of the
// polygon v using the method of Wachpress [1975] "A Rational Finite Element
// Basis", as formulated by Meyer et al [2001] in "Generalized Barycentric
// Coordinates on Irregular Polygons"
std::vector<double> wachpressCoords(glm::vec2 v0,
                                    const std::vector<glm::vec2>& v);
std::vector<std::array<glm::vec3, 3>>
wachpressInterpolate(const std::vector<glm::vec3>& pts);

// compute generalized barycentric coordinates for point v0 inside of the
// polygon v using the method of Floater [2003] "Mean value coordinates"
std::vector<double> meanValueCoords(glm::vec2 v0,
                                    const std::vector<glm::vec2>& v);
std::vector<std::array<glm::vec3, 3>>
meanValueInterpolate(const std::vector<glm::vec3>& pts);

std::vector<std::array<glm::vec3, 3>> barycentricInterpolate(
    const std::vector<glm::vec3>& pts,
    const std::function<std::vector<double>(glm::vec2,
                                            const std::vector<glm::vec2>&)>& f,
    uint subdivisions = 64);

std::vector<std::array<glm::vec3, 3>>
bilinearInterpolate(const std::vector<glm::vec3>& pts);

std::vector<std::array<glm::vec3, 3>>
astridInterpolate(const std::vector<glm::vec3>& pts);

std::vector<std::array<glm::vec3, 3>>
minimalSurface(const std::vector<glm::vec3>& pts);

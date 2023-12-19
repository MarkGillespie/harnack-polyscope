#pragma once

#include <vector>

#include "geometrycentral/utilities/vector3.h"

#include "fftw3.h"

using namespace geometrycentral;

// solve Poisson problem on an n x n x n grid
void solve_poisson_problem(std::vector<double>& source, size_t n);

std::vector<double> divergence(std::function<Vector3(Vector3)> V, size_t n);

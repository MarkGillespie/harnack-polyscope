#pragma once
#define ccl_device
#define ccl_private

class float3 {
  public:
    float x, y, z;
};
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }

class uint3 {
  public:
    uint x, y, z;
};
inline uint3 make_uint3(uint x, uint y, uint z) { return {x, y, z}; }

using packed_float3 = float3;
using packed_uint3  = uint3;

#include "math_elliptic_integral.h"
#include "spherical_harmonics.h"

// helpers
#include "vector_arithmetic.ipp"

#include "harnack.ipp"

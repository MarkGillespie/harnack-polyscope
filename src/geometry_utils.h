#pragma once
#include "fcpw/fcpw.h"                         // fcpw::Vector
#include "geometrycentral/utilities/vector3.h" // Vector3
#include "harnack.h"                           // acceleration_stats
#include "polyscope/polyscope.h"               // glm

inline float dist_to_segment(const glm::vec3& x, const glm::vec3& p1,
                             const glm::vec3& p2) {
    glm::vec3 m = p2 - p1;
    glm::vec3 v = x - p1;
    // dot = |a|*|b|cos(theta) * n, isolating |a|sin(theta)
    float t = fmin(fmax(dot(m, v) / dot(m, m), 0.), 1.);
    return length(v - t * m);
}

inline fcpw::Vector<3> to_fcpw(glm::vec3 v) {
    return fcpw::Vector<3>{v.x, v.y, v.z};
}

typedef struct sphere_trace_intersection_params {
    glm::vec3 ray_P;
    glm::vec3 ray_D;
    float ray_tmin;
    float ray_tmax;
    float epsilon;
    int max_iterations;
    bool capture_misses;
    bool use_overstepping;
    bool use_extrapolation;
    bool use_newton;
    float epsilon_loose;
    bool fixed_step_count;
} sphere_trace_intersection_params;

inline bool
sphere_trace_intersection(const sphere_trace_intersection_params& params,
                          fcpw::Scene<3>& scene, float* t_out,
                          acceleration_stats* stats = nullptr) {

    fcpw::Vector<3> ro = to_fcpw(params.ray_P);
    fcpw::Vector<3> rd = to_fcpw(params.ray_D);

    float t  = params.ray_tmin;
    int iter = 0;

    float ld = rd.norm();

    auto report_stats = [&]() {
        if (stats) {
            stats->total_iterations = iter;
        }
    };

    float t_overstep         = 0;
    static bool exceeded_max = false;
    while (t < params.ray_tmax) {
        fcpw::Vector<3> pos = ro + (t + t_overstep) * rd;

        // If we've exceeded the maximum number of iterations,
        // print a warning
        if (iter > params.max_iterations) {
            if (!exceeded_max) {
                exceeded_max = true;
                printf("Warning: exceeded maximum number of sphere tracing "
                       "iterations.\n");
            }

            if (t_out) *t_out = t + t_overstep;
            report_stats();
            return false;
        }

        // perform a closest point query
        fcpw::Interaction<3> interaction;
        scene.findClosestPoint(pos, interaction);

        // value of sdf at pos
        float val = interaction.signedDistance(pos);

        // safe step size
        float r = abs(val) / ld;

        if (stats) {
            stats->ts.push_back(t + t_overstep);
            stats->vals.push_back(val);
            stats->times.push_back(t + t_overstep);
            stats->omegas.push_back(val);
            stats->Rs.push_back(0);
            stats->rs.push_back(r);
        }

        if (r >= t_overstep) { // commit to step
            // If we're close enough to the level set, report a hit
            if (!params.fixed_step_count && abs(val) < params.epsilon) {
                if (t_out) *t_out = t + t_overstep;
                report_stats();
                return true;
            }
            t += t_overstep + r;
            if (params.use_overstepping) t_overstep = r * .75;
            if (params.use_overstepping && stats) stats->successful_oversteps++;
        } else { // step back and try again
            t_overstep = 0;
            if (stats) stats->failed_oversteps++;
        }
        iter++;
        /* if (inside_loose_shell && stats) stats->n_steps_after_eps++; */
    }

    if (t_out) *t_out = t + t_overstep;
    report_stats();
    return false;
}

template <typename T>
std::vector<std::array<T, 3>>
triangulate_polygon(const std::vector<T>& vertices, uint subdivisions) {
    std::vector<std::array<T, 3>> triangles;

    T center = T(0);
    for (const T& pt : vertices) center += pt;
    center /= (float)vertices.size();

    //===== Build some surface filling in the curve
    for (uint iE = 0; iE < vertices.size(); iE++) {
        // build subdivided triangle between a, b, c
        T a = vertices[iE];
        T b = vertices[(iE + 1) % vertices.size()];
        T c = center;

        T o  = a;
        T dx = b - a;
        T dy = c - a;

        for (uint iR = 0; iR < subdivisions + 1; iR++) { // rows
            float t0 = ((float)(iR + 0)) / ((float)(subdivisions + 1));
            float t1 = ((float)(iR + 1)) / ((float)(subdivisions + 1));
            for (uint iC = 0; iC + iR < subdivisions + 1; iC++) { // columns
                float s0 = ((float)(iC + 0)) / ((float)(subdivisions + 1));
                float s1 = ((float)(iC + 1)) / ((float)(subdivisions + 1));

                triangles.push_back(std::array<T, 3>{o + t0 * dx + s0 * dy,
                                                     o + t1 * dx + s0 * dy,
                                                     o + t0 * dx + s1 * dy});

                if (iC + iR < subdivisions) {
                    triangles.push_back(std::array<T, 3>{
                        o + t1 * dx + s0 * dy, o + t1 * dx + s1 * dy,
                        o + t0 * dx + s1 * dy});
                }
            }
        }
    }

    return triangles;
}

void compute_vertex_face_lists(
    const std::vector<std::array<glm::vec3, 3>>& tri_list,
    std::vector<geometrycentral::Vector3>& vertex_coordinates,
    std::vector<std::vector<size_t>>& faces, double epsilon = 1e-5);

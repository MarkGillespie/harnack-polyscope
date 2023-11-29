#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "fcpw/fcpw.h"

#include "args/args.hxx"
#include "imgui.h"

#include <fstream>

#include "generalized_barycentric_coordinates.h"
#include "geometry_utils.h"
#include "harnack.h"
#include "utils.h"

//====== Scene parameters

float s = 0.5;
std::vector<float3> pts{float3{1, s, 1}, float3{-1, -s, 1}, float3{-1, s, -1},
                        float3{1, -s, -1}};
// float s = 0.1;
// std::vector<float3> pts{float3{1, 0, 1}, float3{0, s, -1}, float3{-1, 0, 1},
//                         float3{0, -s, -.5}};
std::vector<uint3> loops{uint3{0, 4, 0}};

float tmin = 0, tmax = 10, epsilon = .001;
int max_iterations        = 2500;
int resolution_x          = 10;
int resolution_y          = 10;
bool use_grad_termination = true;
bool use_overstepping     = false;
bool use_extrapolation    = false;
bool use_newton           = false;
bool fixed_step_count     = false;

std::vector<std::array<glm::vec3, 3>> sphere_tracing_mesh;
std::map<std::string, std::vector<std::array<glm::vec3, 3>>> comparison_meshes;
fcpw::Scene<3> scene;

polyscope::PointCloud* psCloud;

//====== Helpers
typedef struct convergence_statistics {
    std::vector<float> ts, vals;
} convergence_statistics;

std::string tracing_mode;
std::vector<convergence_statistics> pixel_convergence_statistics;

float3 to_float3(glm::vec3 v) { return make_float3(v.x, v.y, v.z); }
glm::vec3 to_vec3(float3 v) { return glm::vec3{v.x, v.y, v.z}; }

bool intersect(glm::vec3 ro, glm::vec3 rd, float* t = nullptr,
               float* iter_frac = nullptr, float* omega = nullptr,
               acceleration_stats* stats = nullptr) {
    solid_angle_intersection_params sa_params;
    sa_params.ray_P                = to_float3(ro);
    sa_params.ray_D                = to_float3(rd);
    sa_params.ray_tmin             = tmin;
    sa_params.ray_tmax             = tmax;
    sa_params.loops                = loops.data();
    sa_params.pts                  = pts.data();
    sa_params.n_loops              = 1;
    sa_params.epsilon              = epsilon;
    sa_params.levelset             = 2. * M_PI;
    sa_params.frequency            = 0;
    sa_params.solid_angle_formula  = 0;
    sa_params.use_grad_termination = use_grad_termination;
    sa_params.max_iterations       = max_iterations;
    sa_params.clip_y               = false;
    sa_params.capture_misses       = false;
    sa_params.use_overstepping     = use_overstepping;
    sa_params.use_extrapolation    = use_extrapolation;
    sa_params.use_newton           = use_newton;
    sa_params.epsilon_loose        = sqrt(epsilon);
    sa_params.fixed_step_count     = fixed_step_count;

    float ignore_t, ignore_iter, ignore_omega;
    if (!t) t = &ignore_t;
    if (!iter_frac) iter_frac = &ignore_iter;
    if (!omega) omega = &ignore_omega;

    return ray_nonplanar_polygon_intersect_T<double>(sa_params, omega,
                                                     iter_frac, t, stats);
}

bool intersect_sphere_tracing(glm::vec3 ro, glm::vec3 rd, float* t = nullptr,
                              float* iter_frac          = nullptr,
                              float* omega              = nullptr,
                              acceleration_stats* stats = nullptr) {
    sphere_trace_intersection_params st_params;
    st_params.ray_P             = ro;
    st_params.ray_D             = rd;
    st_params.ray_tmin          = tmin;
    st_params.ray_tmax          = tmax;
    st_params.epsilon           = epsilon;
    st_params.max_iterations    = max_iterations;
    st_params.capture_misses    = false;
    st_params.use_overstepping  = use_overstepping;
    st_params.use_extrapolation = use_extrapolation;
    st_params.use_newton        = use_newton;
    st_params.epsilon_loose     = sqrt(epsilon);
    st_params.fixed_step_count  = fixed_step_count;

    float ignore_t, ignore_iter, ignore_omega;
    if (!t) t = &ignore_t;
    if (!iter_frac) iter_frac = &ignore_iter;
    if (!omega) omega = &ignore_omega;

    return sphere_trace_intersection(st_params, scene, t, stats);
}

glm::vec3 normal(glm::vec3 pos) {
    return to_vec3(ray_nonplanar_polygon_normal_T<double>(
        to_float3(pos), loops.data(), pts.data(), 1));
}

double radians(double degrees) { return degrees / 180. * M_PI; }

std::ostream& operator<<(std::ostream& out, const glm::vec3& vec) {
    out << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return out;
}

std::vector<std::array<glm::vec3, 3>> mesh_levelset(uint subdivisions = 64) {
    std::vector<glm::vec3> glm_pts;
    for (const float3& pt : pts) glm_pts.push_back(to_vec3(pt));
    std::vector<std::array<glm::vec3, 3>> triangles =
        triangulate_polygon(glm_pts, subdivisions);

    //===== Project onto the level set
    // check if p lies on any edge of polygon
    auto on_boundary = [&](const glm::vec3& p) -> bool {
        for (uint iE = 0; iE < loops[0].y; iE++) {
            glm::vec3 a = to_vec3(pts[loops[0].x + iE]);
            glm::vec3 b = to_vec3(pts[loops[0].x + ((iE + 1) % loops[0].y)]);
            if (dist_to_segment(p, a, b) < 1e-8) return true;
        }
        return false;
    };
    glm::vec3 proj_dir{0, -1, 0};
    // configure harnack params
    float old_epsilon = epsilon;
    epsilon           = fmin(epsilon, 0.0001);
    for (std::array<glm::vec3, 3>& face : triangles) {
        for (glm::vec3& vertex : face) {
            // leave points on boundary fixed
            if (on_boundary(vertex)) continue;

            glm::vec3 start = vertex - (float)2 * proj_dir;
            float t;
            bool projects_onto_levelset = intersect(start, proj_dir, &t);

            if (projects_onto_levelset) vertex = start + t * proj_dir;
        }
    }
    // restore old harnack params
    epsilon = old_epsilon;

    return triangles;
}

//====== Experiment code
enum class TracingMethod { Harnack, Sphere };
std::map<std::string, acceleration_stats> test_results;
polyscope::CameraParameters camParams;
void shootCameraRays(std::string name     = "default",
                     TracingMethod method = TracingMethod::Harnack) {
    camParams            = polyscope::view::getCameraParametersForCurrentView();
    glm::vec3 camPos     = camParams.getPosition();
    glm::vec3 lookDir    = camParams.getLookDir();
    glm::vec3 upDir      = camParams.getUpDir();
    glm::vec3 rightDir   = camParams.getRightDir();
    double fovY          = camParams.getFoVVerticalDegrees();
    glm::vec2 tanHalfFov = glm::vec2(tan(radians(fovY) * 0.5));

    std::vector<glm::vec3> intersections, normals, viewRayPts;
    std::vector<std::array<size_t, 2>> viewRayLines;
    std::vector<float> omegas, iterationCounts, overstep_success_rate,
        steps_after_epsilon_loose, newton_steps;
    std::vector<char> didHit;
    size_t successful_extrapolations = 0;
    float s                          = 0.05;
    intersections.reserve(resolution_x * resolution_y);
    double start       = std::clock();
    float aspect_ratio = (float)resolution_x / (float)resolution_y;
    pixel_convergence_statistics.clear();
    pixel_convergence_statistics.reserve(resolution_x * resolution_y);
    tracing_mode = (method == TracingMethod::Harnack) ? "harnack" : "sphere";
    for (int iY = 0; iY < resolution_y; iY++) {
        for (int iX = 0; iX < resolution_x; iX++) {

            glm::vec2 cCoord = (resolution_x * resolution_y <= 1)
                                   ? glm::vec2{0, 0}
                                   : glm::vec2{iX / (float)(resolution_x - 1),
                                               iY / (float)(resolution_y - 1)} *
                                             (float)2 -
                                         (float)1;
            cCoord.x *= aspect_ratio;

            // create view ray
            glm::vec3 rd = normalize(cCoord.x * rightDir + cCoord.y * upDir +
                                     (float)3 * lookDir);
            glm::vec3 ro = camPos;

            viewRayLines.push_back(
                {static_cast<size_t>(resolution_x * resolution_y),
                 viewRayPts.size()});
            viewRayPts.push_back(ro + tmax * rd);

            float t, omega, iter_frac;
            acceleration_stats stats;
            bool hit = method == TracingMethod::Harnack
                           ? intersect(ro, rd, &t, &iter_frac, &omega, &stats)
                           : intersect_sphere_tracing(ro, rd, &t, &iter_frac,
                                                      &omega, &stats);

            if (hit) {
                glm::vec3 intersection = ro + t * rd;
                intersections.push_back(intersection);

                omegas.push_back(omega);
                normals.push_back(normal(intersections.back()));
            }

            iterationCounts.push_back(stats.total_iterations);
            overstep_success_rate.push_back((double)stats.successful_oversteps /
                                            (double)stats.total_iterations);
            successful_extrapolations += stats.successful_extrapolations;
            didHit.push_back(hit);
            steps_after_epsilon_loose.push_back(stats.n_steps_after_eps);
            newton_steps.push_back(stats.n_newton_steps);
            pixel_convergence_statistics.push_back(convergence_statistics{});
            pixel_convergence_statistics.back().ts   = stats.ts;
            pixel_convergence_statistics.back().vals = stats.vals;

            if (stats.newton_ts.size() >= test_results[name].newton_ts.size())
                test_results[name] = stats;

            // for (size_t iI = 0; iI < stats.times.size(); iI++) {
            //     std::cout << std::setfill(' ') << std::setw(3) << iI
            //               << "| t = " << std::setw(8) << stats.times[iI]
            //               << "  ω = " << std::setw(8) << stats.omegas[iI]
            //               << "  R = " << std::setw(8) << stats.Rs[iI]
            //               << std::endl;
            // }
            // for (size_t iI = 0; iI < stats.newton_ts.size(); iI++) {
            //     std::cout << std::setfill(' ') << std::setw(3) << iI
            //               << "| t = " << std::setw(8) <<
            //               stats.newton_ts[iI]
            //               << "  f = " << std::setw(8) <<
            //               stats.newton_vals[iI]
            //               << " 4π = " << std::setw(8) << 4. * M_PI
            //               << " dt = " << std::setw(8) <<
            //               stats.newton_dts[iI]
            //               << " df = " << std::setw(8) <<
            //               stats.newton_dfs[iI]
            //               << std::endl;
            // }
        }
    }
    double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    double meanTime = duration / viewRayPts.size();

    psCloud = polyscope::registerPointCloud("intersections", intersections);
    psCloud->addScalarQuantity("omega", omegas);
    psCloud->addVectorQuantity("normal", normals);

    auto viewPts = polyscope::registerPointCloud("view ray points", viewRayPts);
    viewPts->addScalarQuantity("did hit", didHit);
    viewPts->addScalarQuantity("overstep success rate", overstep_success_rate);
    viewPts->addScalarQuantity("iteration counts", iterationCounts);
    viewPts->setPointRenderMode(polyscope::PointRenderMode::Quad);

    auto mean = [](const std::vector<float>& v) -> float {
        return std::accumulate(v.begin(), v.end(), 0.) / v.size();
    };
    float meanIterations = mean(iterationCounts);
    float maxIterations =
        *std::max_element(iterationCounts.begin(), iterationCounts.end());
    float meanOverstepSuccessRate        = mean(overstep_success_rate);
    float mean_steps_after_epsilon_loose = mean(steps_after_epsilon_loose);
    float mean_newton_steps              = mean(newton_steps);

    std::cout << "==== Stats (" << name << ")    " << vendl;
    std::cout << "           mean iterations: " << meanIterations << vendl;
    std::cout << "            max iterations: " << maxIterations << vendl;
    std::cout << "     overstep success rate: " << meanOverstepSuccessRate
              << vendl;
    std::cout << " successful extrapolations: " << successful_extrapolations
              << vendl;
    std::cout << "       steps after ε loose: "
              << mean_steps_after_epsilon_loose << vendl;
    std::cout << "              newton_steps: " << mean_newton_steps << vendl;
    std::cout << "                 mean time: " << meanTime << " s" << vendl;

    viewRayPts.push_back(camPos);
    polyscope::registerCurveNetwork("view rays", viewRayPts, viewRayLines)
        ->setEnabled(false);
}

void write_convergence_statistics(
    std::string filename = tracing_mode + "_convergence_statistics.csv") {
    std::ofstream out;
    out.open(filename);

    //=== column headers
    out << "grad_termination,";
    out << "overstepping,";
    out << "newton_steps,";
    size_t max_run_length = 0;
    for (size_t iP = 0; iP < pixel_convergence_statistics.size(); iP++) {
        out << std::to_string(iP) << "_ts,";
        out << std::to_string(iP) << "_vals";
        if (iP + 1 < pixel_convergence_statistics.size()) out << ",";
        max_run_length = std::max(max_run_length,
                                  pixel_convergence_statistics[iP].ts.size());
    }
    out << std::endl;
    //=== first row
    out << std::boolalpha;
    out << use_grad_termination << "," << use_overstepping << "," << use_newton
        << ",";

    for (size_t iP = 0; iP < pixel_convergence_statistics.size(); iP++) {
        const convergence_statistics& stats = pixel_convergence_statistics[iP];
        out << stats.ts[0] << "," << stats.vals[0];
        if (iP + 1 < pixel_convergence_statistics.size()) out << ",";
    }
    out << std::endl;

    //=== later rows
    for (size_t iR = 1; iR < max_run_length; iR++) {
        out << ",,,";
        for (size_t iP = 0; iP < pixel_convergence_statistics.size(); iP++) {
            const convergence_statistics& stats =
                pixel_convergence_statistics[iP];
            if (iR < stats.ts.size()) {
                out << stats.ts[iR] << ",";
            } else {
                out << ",";
            }
            if (iR < stats.vals.size()) {
                out << stats.vals[iR];
            }
            if (iP + 1 < pixel_convergence_statistics.size()) out << ",";
        }
        out << std::endl;
    }
    out.close();
}

void print_test_results() {
    std::vector<std::string> tests; // get list of keys
    size_t max_step_len = 0, max_newton_len = 0;
    for (auto it = test_results.begin(); it != test_results.end(); ++it) {
        tests.push_back(it->first);
        max_step_len   = std::max(max_step_len, it->second.times.size());
        max_newton_len = std::max(max_newton_len, it->second.newton_ts.size());
    }

    std::cout << "iter ";
    for (std::string test : tests) {
        std::cout << "| " << std::setfill(' ') << std::setw(54) << test;
    }
    std::cout << std::endl;

    if (false) {
        for (size_t iI = 0; iI < max_step_len; iI++) {
            std::cout << std::setfill(' ') << std::setw(4) << iI << " ";
            for (std::string t : tests) {
                const acceleration_stats& stats = test_results[t];
                float omega = -1, R = -1, r = -1;
                if (iI < stats.omegas.size()) {
                    omega = stats.omegas[iI];
                    R     = stats.Rs[iI];
                    r     = stats.rs[iI];
                }
                std::cout << std::fixed;
                std::cout.precision(5);
                // //=== overstepping
                // std::cout << "| t = " << std::setw(8) << stats.times[iI]
                //           << "  ω = " << std::setw(8) << omega
                //           << "  R = " << std::setw(8) << R
                //           << "  r = " << std::setw(8) << r;
                // //=== extrapolation
                // std::cout << "| t  = " << std::setw(8) << stats.times[iI]
                //           << "  v  = " << std::setw(8) << omega
                //           << "  te = " << std::setw(8)
                //           << stats.extrapolation_times[iI]
                //           << "  ve = " << std::setw(8)
                //           << stats.extrapolation_values[iI]
                //           << "  VT = " << std::setw(8) <<
                //           stats.true_values[iI]
                //           << "  a  = " << std::setw(8) << stats.as[iI]
                //           << "  b  = " << std::setw(8) << stats.bs[iI];
            }
            std::cout << std::endl;
        }
    }
    for (size_t i = 0; i < 60; i++) std::cout << "-";
    std::cout << vendl;
    for (size_t iI = 0; iI < max_newton_len; iI++) {
        std::cout << std::setfill(' ') << std::setw(4) << iI << " ";
        for (std::string t : tests) {
            const acceleration_stats& stats = test_results[t];
            std::cout << std::fixed;
            std::cout.precision(5);
            //=== newton
            std::cout << "| t  = " << std::setw(8) << stats.newton_ts[iI]
                      << "  f  = " << std::setw(8) << stats.newton_vals[iI]
                      << "  4π = " << std::setw(8) << 4. * M_PI
                      << "  dt = " << std::setw(8) << stats.newton_dts[iI]
                      << "  df = " << std::setw(8) << stats.newton_dfs[iI];
        }
        std::cout << std::endl;
    }
}

void print_camera_view() {
    glm::vec3 camPos   = camParams.getPosition();
    glm::vec3 lookDir  = camParams.getLookDir();
    glm::vec3 upDir    = camParams.getUpDir();
    glm::vec3 rightDir = camParams.getRightDir();
    double fovY        = camParams.getFoVVerticalDegrees();

    glm::mat3 default_cam_mat(rightDir, upDir, lookDir);
    glm::vec3 camSpaceCamPos = inverse(default_cam_mat) * camPos;

    std::cout << "//== Camera Parameters" << std::endl;
    std::cout << "const vec3 cam_pos   = vec3" << camPos << ";" << std::endl;
    std::cout << "const vec3 look_dir  = vec3" << lookDir << ";" << std::endl;
    std::cout << "const vec3 up_dir    = vec3" << upDir << ";" << std::endl;
    std::cout << "const vec3 right_dir = vec3" << rightDir << ";" << std::endl;
    std::cout
        << "const mat3 default_cam_mat = mat3(right_dir, up_dir, look_dir);"
        << std::endl;
    std::cout << "const vec3 cam_space_cam_pos   = vec3" << camSpaceCamPos
              << ";" << std::endl;
    std::cout << "const float fovY = " << fovY << ";" << std::endl;
}

void build_fcpw_scene() {
    uint n_triangles = sphere_tracing_mesh.size();
    uint n_vertices  = 3 * n_triangles;

    // initialize a 3d scene
    scene = fcpw::Scene<3>();

    // set the PrimitiveType for each object in the scene;
    // in this case, we have a single object consisting of triangles
    scene.setObjectTypes({{fcpw::PrimitiveType::Triangle}});

    // set the vertex and triangle count of the (0th) object
    scene.setObjectVertexCount(n_vertices, 0);
    scene.setObjectTriangleCount(n_triangles, 0);


    // specify the triangle indices
    for (int iT = 0; iT < n_triangles; iT++) {
        std::array<int, 3> indices{3 * iT + 0, 3 * iT + 1, 3 * iT + 2};
        scene.setObjectTriangle(indices.data(), iT, 0);

        // specify the vertex positions
        const std::array<glm::vec3, 3>& t = sphere_tracing_mesh[iT];
        for (int iV = 0; iV < 3; iV++) {
            scene.setObjectVertex(fcpw::Vector<3>{t[iV].x, t[iV].y, t[iV].z},
                                  3 * iT + iV, 0);
        }
    }

    // once geometry has been specified, build acceleration structure
    // the second boolean argument toggles vectorization
    scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false);

    // perform a closest point query
    fcpw::Interaction<3> interaction;
    fcpw::Vector<3> query_point;
    scene.findClosestPoint(query_point, interaction);
}

void build_comparison_mesh(
    std::string comparison_name,
    const std::function<std::vector<std::array<glm::vec3, 3>>(
        const std::vector<glm::vec3>&)>& build_mesh) {

    std::vector<glm::vec3> glm_pts;
    for (const float3& pt : pts) glm_pts.push_back(to_vec3(pt));
    auto mesh = build_mesh(glm_pts);
    std::vector<glm::vec3> positions;
    std::vector<std::vector<size_t>> faceIndices;
    for (size_t iF = 0; iF < mesh.size(); iF++) {
        faceIndices.push_back(std::vector<size_t>{
            positions.size(), positions.size() + 1, positions.size() + 2});
        for (size_t i = 0; i < 3; i++) positions.push_back(mesh[iF][i]);
    }
    polyscope::registerSurfaceMesh(comparison_name + " mesh", positions,
                                   faceIndices);
    comparison_meshes[comparison_name] = mesh;
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {
    if (ImGui::Button("Shoot Camera Rays")) {
        std::string name =
            std::string(use_grad_termination ? "grad-terminated " : "") +
            std::string(use_overstepping ? "overstepped " : "") +
            std::string(use_extrapolation ? "extrapolated " : "") +
            std::string(use_newton ? "newton-accelerated " : "");
        if (name.length() == 0) name = "default ";
        name += "Harnack tracing";
        shootCameraRays(name);
    }
    if (ImGui::Button("Restore Camera View")) {
        polyscope::view::setViewToCamera(camParams);
    }
    if (ImGui::Button("Print Camera View")) {
        print_camera_view();
    }
    if (ImGui::Button("Print log")) {
        print_test_results();
    }
    if (ImGui::Button("Test overstepping")) {
        bool old_overstepping = use_overstepping;
        use_overstepping      = true;
        shootCameraRays("overstep");
        use_overstepping = false;
        shootCameraRays("normalstep");
        print_test_results();
    }
    if (ImGui::Button("Build Sphere Tracing Mesh")) {
        sphere_tracing_mesh = mesh_levelset();
        std::vector<glm::vec3> positions;
        std::vector<std::vector<size_t>> faceIndices;
        for (size_t iF = 0; iF < sphere_tracing_mesh.size(); iF++) {
            faceIndices.push_back(std::vector<size_t>{
                positions.size(), positions.size() + 1, positions.size() + 2});
            for (size_t i = 0; i < 3; i++)
                positions.push_back(sphere_tracing_mesh[iF][i]);
        }
        polyscope::registerSurfaceMesh("mesh", positions, faceIndices);
        build_fcpw_scene();
    }
    if (ImGui::Button("Sphere Trace")) {
        std::string name =
            std::string(use_grad_termination ? "grad-terminated " : "") +
            std::string(use_overstepping ? "overstepped " : "") +
            std::string(use_extrapolation ? "extrapolated " : "") +
            std::string(use_newton ? "newton-accelerated " : "");
        if (name.length() == 0) name = "default ";
        name += "sphere tracing";
        shootCameraRays(name, TracingMethod::Sphere);
    }
    if (ImGui::Button("Save Convergence Statistics")) {
        write_convergence_statistics();
    }
    ImGui::Separator();
    if (ImGui::Button("Build Wachpress Mesh")) {
        build_comparison_mesh("wachpress", wachpressInterpolate);
    }
    if (ImGui::Button("Build Mean Value Mesh")) {
        build_comparison_mesh("mean_value", meanValueInterpolate);
    }
    if (ImGui::Button("Build Astrid Mesh")) {
        build_comparison_mesh("astrid", astridInterpolate);
    }
    if (ImGui::Button("Build Bilinear Mesh")) {
        if (pts.size() == 4) {
            build_comparison_mesh("bilinear", bilinearInterpolate);
        } else {
            polyscope::warning("Bilinear interpolation only makes sense if the "
                               "polygon has four sizes, but this polygon has " +
                               std::to_string(pts.size()) + " sides");
        }
    }
    if (ImGui::Button("Build Minimal Surface")) {
        build_comparison_mesh("minimal", minimalSurface);
    }
    if (ImGui::Button("Save Comparison Meshes")) {
        std::cout << "Writing meshes:" << vendl;
        for (auto const& comparison : comparison_meshes) {
            geometrycentral::surface::SimplePolygonMesh mesh;
            compute_vertex_face_lists(comparison.second, mesh.vertexCoordinates,
                                      mesh.polygons);
            std::string filename = comparison.first + "_mesh.obj";
            std::cout << "  Writing " << filename << "..." << vendl;
            mesh.writeMesh(filename, "obj");
        }
        std::cout << "Done writing meshes" << vendl;
    }

    ImGui::Separator();
    ImGui::SliderFloat("epsilon (log)", &epsilon, .00000001f, .0001f, "%.4f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::DragFloat("tmin", &tmin, .1f, 0.f, 20.f);
    ImGui::DragFloat("tmax", &tmax, .1f, 0.f, 20.f);
    ImGui::DragInt("max_iterations", &max_iterations, 10, 1, 50000);
    ImGui::DragInt("resolution x", &resolution_x, 1, 1, 2000);
    ImGui::DragInt("resolution y", &resolution_y, 1, 1, 2000);
    ImGui::Checkbox("use_grad_termination", &use_grad_termination);
    ImGui::Checkbox("use_overstepping", &use_overstepping);
    ImGui::Checkbox("use_extrapolation", &use_extrapolation);
    ImGui::Checkbox("use_newton", &use_newton);
    ImGui::Checkbox("fixed_step_count", &fixed_step_count);
}

int main(int argc, char** argv) {

    // Configure the argument parser
    args::ArgumentParser parser("Harnack debugger");
    polyscope::options::programName = "Harnack Debugger";

    // Parse args
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    // Initialize polyscope
    polyscope::init();

    // Set the callback function
    polyscope::state::userCallback = myCallback;

    polyscope::registerCurveNetworkLoop("polygon", pts);

    // shootCameraRays();

    // Give control to the polyscope gui
    polyscope::show();

    return EXIT_SUCCESS;
}

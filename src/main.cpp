#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"

#include "args/args.hxx"
#include "imgui.h"

#include "harnack.h"
#include "utils.h"

//====== Scene parameters
float s = 0.5;
std::vector<float3> pts{float3{1, s, 1}, float3{-1, -s, 1}, float3{-1, s, -1},
                        float3{1, -s, -1}};
std::vector<uint3> loops{uint3{0, 4, 0}};

float tmin = 0, tmax = 10, epsilon = .001;
int max_iterations        = 2500;
int resolution            = 10;
bool use_grad_termination = true;
bool use_overstepping     = false;
bool use_extrapolation    = false;
bool use_newton           = true;

polyscope::PointCloud* psCloud;

//====== Helpers
float3 to_float3(glm::vec3 v) { return make_float3(v.x, v.y, v.z); }
glm::vec3 to_vec3(float3 v) { return glm::vec3{v.x, v.y, v.z}; }

bool intersect(glm::vec3 ro, glm::vec3 rd, float* t, float* iter_frac,
               float* omega, acceleration_stats* stats) {
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

    return ray_nonplanar_polygon_intersect_T<double>(sa_params, omega,
                                                     iter_frac, t, stats);
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

//====== Experiment code
std::map<std::string, acceleration_stats> test_results;
polyscope::CameraParameters camParams;
void shootCameraRays(size_t N = 50, std::string name = "default") {
    camParams = polyscope::view::getCameraParametersForCurrentView();
    glm::vec3 lookDir, upDir, rightDir;
    polyscope::view::getCameraFrame(lookDir, upDir, rightDir);
    glm::vec3 camPos     = polyscope::view::getCameraWorldPosition();
    double fovY          = 50.0;
    glm::vec2 tanHalfFov = glm::vec2(tan(radians(fovY) * 0.5));

    std::vector<glm::vec3> intersections, normals, viewRayPts;
    std::vector<std::array<size_t, 2>> viewRayLines;
    std::vector<float> omegas, iterationCounts, overstep_success_rate,
        steps_after_epsilon_loose, newton_steps;
    std::vector<char> didHit;
    size_t successful_extrapolations = 0;
    float s                          = 0.05;
    intersections.reserve(N * N);
    double start = std::clock();
    for (size_t iX = 0; iX < N; iX++) {
        for (size_t iY = 0; iY < N; iY++) {

            glm::vec2 cCoord =
                (N <= 1)
                    ? glm::vec2{0, 0}
                    : glm::vec2{iX, iY} / (float)(N - 1) * (float)2 - (float)1;

            // create view ray
            glm::vec3 rd = normalize(cCoord.x * rightDir + cCoord.y * upDir +
                                     (float)3 * lookDir);
            glm::vec3 ro = camPos;

            viewRayLines.push_back({N * N, viewRayPts.size()});
            viewRayPts.push_back(ro + tmax * rd);

            float t, omega, iter_frac;
            acceleration_stats stats;
            bool hit = intersect(ro, rd, &t, &iter_frac, &omega, &stats);

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
            //               << "| t = " << std::setw(8) << stats.newton_ts[iI]
            //               << "  f = " << std::setw(8) <<
            //               stats.newton_vals[iI]
            //               << " 4π = " << std::setw(8) << 4. * M_PI
            //               << " dt = " << std::setw(8) << stats.newton_dts[iI]
            //               << " df = " << std::setw(8) << stats.newton_dfs[iI]
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
        name = name.substr(0, name.size() - 1); // trim trailing space
        shootCameraRays(resolution, name);
    }
    if (ImGui::Button("Restore Camera View")) {
        polyscope::view::setViewToCamera(camParams);
    }
    if (ImGui::Button("Print log")) {
        print_test_results();
    }
    if (ImGui::Button("Test overstepping")) {
        bool old_overstepping = use_overstepping;
        use_overstepping      = true;
        shootCameraRays(resolution, "overstep");
        use_overstepping = false;
        shootCameraRays(resolution, "normalstep");
        print_test_results();
    }

    ImGui::Separator();
    ImGui::SliderFloat("epsilon (log)", &epsilon, .000001f, .01f, "%.4f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::DragFloat("tmin", &tmin, .1f, 0.f, 20.f);
    ImGui::DragFloat("tmax", &tmax, .1f, 0.f, 20.f);
    ImGui::DragInt("max_iterations", &max_iterations, 10, 1, 50000);
    ImGui::DragInt("resolution", &resolution, 1, 1, 500);
    ImGui::Checkbox("use_grad_termination", &use_grad_termination);
    ImGui::Checkbox("use_overstepping", &use_overstepping);
    ImGui::Checkbox("use_extrapolation", &use_extrapolation);
    ImGui::Checkbox("use_newton", &use_newton);
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

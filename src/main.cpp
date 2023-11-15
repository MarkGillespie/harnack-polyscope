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
int resolution            = 200;
bool use_grad_termination = true;

polyscope::PointCloud* psCloud;

//====== Helpers
float3 to_float3(glm::vec3 v) { return make_float3(v.x, v.y, v.z); }
glm::vec3 to_vec3(float3 v) { return glm::vec3{v.x, v.y, v.z}; }

bool intersect(glm::vec3 ro, glm::vec3 rd, float* t, float* iter_frac,
               float* omega) {
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

    return ray_nonplanar_polygon_intersect_T<double>(sa_params, omega,
                                                     iter_frac, t);
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
polyscope::CameraParameters camParams;
void shootCameraRays(size_t N = 50) {
    camParams = polyscope::view::getCameraParametersForCurrentView();
    glm::vec3 lookDir, upDir, rightDir;
    polyscope::view::getCameraFrame(lookDir, upDir, rightDir);
    glm::vec3 camPos     = polyscope::view::getCameraWorldPosition();
    double fovY          = 50.0;
    glm::vec2 tanHalfFov = glm::vec2(tan(radians(fovY) * 0.5));

    std::vector<glm::vec3> intersections, normals, viewRayPts;
    std::vector<std::array<size_t, 2>> viewRayLines;
    std::vector<float> omegas, iterationCounts;
    std::vector<char> didHit;
    float s = 0.05;
    intersections.reserve(N * N);
    viewRayPts.push_back(camPos);
    iterationCounts.push_back(0);
    didHit.push_back(false);
    for (size_t iX = 0; iX < N; iX++) {
        for (size_t iY = 0; iY < N; iY++) {

            glm::vec2 cCoord =
                glm::vec2{iX, iY} / (float)(N - 1) * (float)2 - (float)1;

            // create view ray
            glm::vec3 rd = normalize(cCoord.x * rightDir + cCoord.y * upDir +
                                     (float)3 * lookDir);
            glm::vec3 ro = camPos;

            viewRayLines.push_back({0, viewRayPts.size()});
            viewRayPts.push_back(ro + tmax * rd);

            float t, omega, iter_frac;
            bool hit = intersect(ro, rd, &t, &iter_frac, &omega);

            if (hit) {
                glm::vec3 intersection = ro + t * rd;
                intersections.push_back(intersection);

                omegas.push_back(omega);
                normals.push_back(normal(intersections.back()));
            }

            iterationCounts.push_back(iter_frac * max_iterations);
            didHit.push_back(hit);
        }
    }
    psCloud = polyscope::registerPointCloud("intersections", intersections);
    psCloud->addScalarQuantity("omega", omegas);
    psCloud->addVectorQuantity("normal", normals);

    polyscope::registerCurveNetwork("view rays", viewRayPts, viewRayLines)
        ->setEnabled(false);
    auto viewPts = polyscope::registerPointCloud("view ray points", viewRayPts);
    viewPts->addScalarQuantity("did hit", didHit);
    viewPts->addScalarQuantity("iteration counts", iterationCounts)
        ->setEnabled(true);
    viewPts->setPointRenderMode(polyscope::PointRenderMode::Quad);

    float meanIterations =
        std::accumulate(iterationCounts.begin(), iterationCounts.end(), 0.) /
        (iterationCounts.size() - 1.);
    float maxIterations =
        *std::max_element(iterationCounts.begin(), iterationCounts.end());

    std::cout << "==== Stats    " << vendl;
    std::cout << "  mean iterations: " << meanIterations << vendl;
    std::cout << "   max iterations: " << maxIterations << vendl;
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {
    if (ImGui::Button("Shoot Camera Rays")) {
        shootCameraRays(resolution);
    }
    if (ImGui::Button("Restore Camera View")) {
        polyscope::view::setViewToCamera(camParams);
    }

    ImGui::Separator();
    ImGui::SliderFloat("epsilon (log)", &epsilon, .000001f, .01f, "%.4f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::DragFloat("tmin", &tmin, .1f, 0.f, 20.f);
    ImGui::DragFloat("tmax", &tmax, .1f, 0.f, 20.f);
    ImGui::DragInt("max_iterations", &max_iterations, 10, 1, 50000);
    ImGui::DragInt("resolution", &resolution, 1, 1, 500);
    ImGui::Checkbox("use_grad_termination", &use_grad_termination);
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

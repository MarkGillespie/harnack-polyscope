#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"

#include "args/args.hxx"
#include "imgui.h"

#include "harnack.h"
#include "utils.h"

polyscope::PointCloud* psCloud;

float s = 0.5;
std::vector<float3> pts{float3{1, s, 1}, float3{-1, -s, 1}, float3{-1, s, -1},
                        float3{1, -s, -1}};
std::vector<uint3> loops{uint3{0, 4, 0}};

float3 to_float3(glm::vec3 v) { return make_float3(v.x, v.y, v.z); }
glm::vec3 to_vec3(float3 v) { return glm::vec3{v.x, v.y, v.z}; }

bool intersect(glm::vec3 ro, glm::vec3 rd, float* t, float* iter_frac,
               float* omega) {
    solid_angle_intersection_params sa_params;
    sa_params.ray_P                = to_float3(ro);
    sa_params.ray_D                = to_float3(rd);
    sa_params.ray_tmin             = 0;
    sa_params.ray_tmax             = 5;
    sa_params.loops                = loops.data();
    sa_params.pts                  = pts.data();
    sa_params.n_loops              = 1;
    sa_params.epsilon              = .001;
    sa_params.levelset             = 2. * M_PI;
    sa_params.frequency            = 0;
    sa_params.solid_angle_formula  = 0;
    sa_params.use_grad_termination = true;
    sa_params.max_iterations       = 2500;
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

void shootCameraRays(size_t N = 50) {
    glm::mat4 camView    = polyscope::view::getCameraViewMatrix();
    glm::vec3 camPos     = polyscope::view::getCameraWorldPosition();
    double fovY          = 50.0;
    glm::vec2 tanHalfFov = glm::vec2(tan(radians(fovY) * 0.5));

    std::vector<glm::vec3> intersections, normals, viewRayPts;
    std::vector<float> omegas;
    float s = 0.05;
    intersections.reserve(N * N);
    for (size_t iX = 0; iX < N; iX++) {
        for (size_t iY = 0; iY < N; iY++) {

            glm::vec2 cCoord =
                glm::vec2{iX, iY} / (float)(N - 1) * (float)2 - (float)1;
            glm::vec3 vDir = normalize(glm::vec3(cCoord * tanHalfFov, -1.0));

            // create view ray
            glm::vec3 rd = glm::vec3(camView * glm::vec4(vDir, 0.0));
            glm::vec3 ro = camPos;

            viewRayPts.push_back(ro);
            viewRayPts.push_back(ro + (float)5 * rd);

            float t, omega, iter_frac;
            bool hit = intersect(ro, rd, &t, &iter_frac, &omega);

            if (hit) {
                glm::vec3 intersection = ro + t * rd;
                intersections.push_back(intersection);

                omegas.push_back(omega);
                normals.push_back(normal(intersections.back()));
            }
        }
    }
    psCloud = polyscope::registerPointCloud("intersections", intersections);
    psCloud->addScalarQuantity("omega", omegas);
    psCloud->addVectorQuantity("normal", normals);

    polyscope::registerCurveNetworkSegments("view rays", viewRayPts);
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {
    if (ImGui::Button("Shoot Camera Rays")) {
        shootCameraRays();
    }
}

int main(int argc, char** argv) {

    // Configure the argument parser
    args::ArgumentParser parser("Harnack debugger");

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

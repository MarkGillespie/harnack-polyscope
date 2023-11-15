#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"

#include "args/args.hxx"
#include "imgui.h"

#include "harnack.h"
#include "utils.h"

float s = 0.5;
std::vector<float3> pts{float3{1, s, 1}, float3{-1, -s, 1}, float3{-1, s, -1},
                        float3{1, -s, -1}};
std::vector<uint3> loops{uint3{0, 4, 0}};

float3 intersect(float3 ro, float3 rd, float* omega) {
    float u, v, t;
    ray_nonplanar_polygon_intersect_T<double>(
        ro, rd, 0, 5, loops.data(), pts.data(), 1, 0.001, 2. * M_PI, 0, false,
        1500, &u, &v, &t, omega);

    return float3{ro.x + t * rd.x, ro.y + t * rd.y, ro.z + t * rd.z};
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {}

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

    polyscope::init();

    polyscope::registerCurveNetworkLoop("polygon", pts);

    size_t N = 50;
    std::vector<float3> intersections, normals;
    std::vector<float> omegas;
    float s = 0.05;
    intersections.reserve(N * N);
    for (size_t iX = 0; iX < N; iX++) {
        for (size_t iY = 0; iY < N; iY++) {
            float x = ((float)iX) / ((float)N - 1.) * 2. * (1. - s) - (1. - s);
            float y = ((float)iY) / ((float)N - 1.) * 2. * (1. - s) - (1. - s);
            float omega;
            intersections.push_back(
                intersect(float3{x, 3, y}, float3{0, -1, 0}, &omega));
            omegas.push_back(omega);
            normals.push_back(ray_nonplanar_polygon_normal_T<double>(
                intersections.back(), loops.data(), pts.data(), 1));
        }
    }
    auto cloud = polyscope::registerPointCloud("intersections", intersections);
    cloud->addScalarQuantity("omega", omegas);
    cloud->addVectorQuantity("normal", normals)->setEnabled(true);

    // Give control to the polyscope gui
    polyscope::show();

    return EXIT_SUCCESS;
}

#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "fcpw/fcpw.h"

#include "args/args.hxx"
#include "imgui.h"

#include <fstream>

#define HAS_POLYSCOPE
#include "generalized_barycentric_coordinates.h"
#include "geometry_utils.h"
#include "harnack.h"
#include "utils.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

//====== Scene parameters

float tmin = 0, tmax = 10, epsilon = .001;
int max_iterations        = 2500;
int resolution_x          = 1;
int resolution_y          = 1;
bool use_grad_termination = true;
bool use_overstepping     = false;
bool use_extrapolation    = false;
bool use_newton           = false;
bool fixed_step_count     = false;
bool intersect_with_mesh  = false;
bool polygon_with_holes   = false;
int loop_id               = 0;
float target_levelset     = .5;

static std::vector<const char*> tracing_mode_names{
    "Harnack Tracing", "Sphere Tracing", "Newton's Method", "Bisection Search",
    "Interval Arithmetic"};
static int i_tracing_mode = 0;

static std::vector<const char*> solid_angle_mode_names{
    "Triangulated", "Prequantum", "Gauss-Bonnet"};
static int i_solid_angle_mode = 0;

float s = 0.5; // used in convergence tests
std::vector<packed_float3> pts{float3{1, s, 1}, float3{-1, -s, 1},
                               float3{-1, s, -1}, float3{1, -s, -1},
                               // center
                               float3{0, -1, 0}};
std::vector<packed_uint3> loops{uint3{0, 4, 0}};

typedef struct named_polygon {
    std::string name;
    std::vector<float3> pts;
    std::vector<uint3> loops;
    size_t cam_view;
} named_polygon;
const float r2                            = 1. / sqrt(2.);
std::vector<named_polygon> named_polygons = {
    {"default",
     {float3{1, s, 1}, float3{-1, -s, 1}, float3{-1, s, -1}, float3{1, -s, -1},
      // center
      float3{0, 0, 0}},
     {uint3{0, 4, 0}},
     0},
    {"nonconvex_planar_quad",
     {float3{1, 0, 1}, float3{0, 0, -1}, float3{-1, 0, 1}, float3{0, -0, -.5},
      // center
      float3{0, 0, 0}},
     {uint3{0, 4, 0}},
     1},
    {"nonconvex_nonplanar_quad",
     {float3{1, 0.1, 1}, float3{0, 0.2, -1}, float3{-1, 0.1, 1},
      float3{0, 0, -.5},
      // center
      float3{0, 0, 0}},
     {uint3{0, 4, 0}},
     2},
    {"nonconvex_planar_octagon",
     {float3{1, 0, -1}, float3{1, 0, 1}, float3{1 / 3., 0, 1},
      float3{1 / 3., 0, -1 / 3.}, float3{-1 / 3., 0, -1 / 3.},
      float3{-1 / 3., 0, 1}, float3{-1, 0, 1}, float3{-1, 0, -1},
      // center
      float3{0, 0, 0}},
     {uint3{0, 8, 0}},
     3},
    {"nonconvex_nonplanar_octagon",
     {float3{1, 0, -1}, float3{1, 0.1, 1}, float3{1 / 3., 0.2, 1},
      float3{1 / 3., 0., -1 / 3.}, float3{-1 / 3., 0., -1 / 3.},
      float3{-1 / 3., 0.2, 1}, float3{-1, 0.1, 1}, float3{-1, 0, -1},
      // center
      float3{0, 0, 0}},
     {uint3{0, 8, 0}},
     3},
    {"nice_nonplanar_octagon",
     {float3{1, 0, 0}, float3{r2, 1, r2}, float3{0, 0, 1}, float3{-r2, 1, r2},
      float3{-1, 0, 0}, float3{-r2, 1., -r2}, float3{0, 0, -1},
      float3{r2, 1, -r2},
      // center
      float3{0, 0, 0}},
     {uint3{0, 8, 0}},
     4},
};

void construct_approaching_circles() {
    std::vector<double> Ts{.2, .3, .455, .5, .7};
    size_t circle_resolution = 4;
    for (size_t iT = 0; iT < Ts.size(); iT++) {
        float d = Ts[iT];
        named_polygons.push_back(named_polygon{
            "approaching_circles_" + std::to_string(iT),
            {},
            {uint3{0, (uint)circle_resolution, 0},
             uint3{(uint)circle_resolution + 1, (uint)circle_resolution, 0}},
            5});
        for (size_t iR = 0; iR < circle_resolution; iR++) {
            float s = 2. * M_PI * (float)iR / ((float)circle_resolution);
            named_polygons.back().pts.push_back(float3{cos(s), sin(s) + 1, d});
        }
        named_polygons.back().pts.push_back(float3{0, 0, 0}); // center
        for (size_t iR = 0; iR < circle_resolution; iR++) {
            float s = -2. * M_PI * (float)iR / ((float)circle_resolution);
            named_polygons.back().pts.push_back(float3{cos(s), sin(s) + 1, -d});
        }
        named_polygons.back().pts.push_back(float3{0, 0, 0}); // center
    }
}

void load_loop_file(std::string filepath) {
    //==== extract basename from filename https://stackoverflow.com/a/8520815
    // If the path contains a slash, take the substring after it, otherwise use
    // the whole path
    size_t last_slash    = filepath.find_last_of('/');
    std::string filename = (last_slash != std::string::npos)
                               ? filepath.substr(last_slash + 1)
                               : filepath;

    // If the path contains a dot, take the substring before it, otherwise use
    // the whole filename
    size_t last_dot      = filename.find_last_of('.');
    std::string basename = (last_dot != std::string::npos)
                               ? filename.substr(0, last_dot)
                               : filename;
    std::string filetype = (last_dot != std::string::npos)
                               ? filename.substr(last_dot + 1)
                               : "loops";

    std::vector<float3> file_pts;
    std::vector<uint3> file_loops;
    std::vector<std::vector<size_t>> face_vert_adj_list;
    if (filetype == "loops") {
        std::ifstream inStream(filepath);
        if (!inStream)
            throw std::runtime_error("couldn't open file " + filepath);

        std::string line;
        while (getline(inStream, line)) {
            std::stringstream ss(line);
            float x, y, z;

            uint loop_start = file_pts.size(), loop_size = 0;
            face_vert_adj_list.push_back({});

            // Read three floats at a time from the line
            while (ss >> x >> y >> z) {
                face_vert_adj_list.back().push_back(file_pts.size());
                file_pts.push_back(float3{x, y, z});
                loop_size++;
            }

            // center
            file_pts.push_back(float3{0, 0, 0});
            file_loops.push_back(uint3{loop_start, loop_size, 0});
        }
    } else {
        SimplePolygonMesh mesh;
        mesh.readMeshFromFile(filepath);
        for (const std::vector<size_t>& face : mesh.polygons) {
            uint loop_start = file_pts.size(), loop_size = 0;
            face_vert_adj_list.push_back({});
            for (size_t iV : face) {
                face_vert_adj_list.back().push_back(file_pts.size());
                file_pts.push_back(to_float3(mesh.vertexCoordinates[iV]));
                loop_size++;
            }
            // center
            file_pts.push_back(float3{0, 0, 0});
            file_loops.push_back(uint3{loop_start, loop_size, 0});
        }
    }

    named_polygons.push_back(named_polygon{basename, file_pts, file_loops, 0});
    polyscope::registerSurfaceMesh(basename, file_pts, face_vert_adj_list);
}

std::vector<std::string> camera_positions = {
    // default
    "{\"farClipRatio\":20.0,\"fov\":45.0,\"nearClipRatio\":0.005,"
    "\"projectionMode\":\"Perspective\",\"viewMat\":[1.0,-0.0,0.0,-0.0,0.0,0."
    "997785151004791,-0.0665190145373344,-0.0,-0.0,0.0665190145373344,0."
    "997785151004791,-4.50998878479004,0.0,0.0,0.0,1.0],\"windowHeight\":1015,"
    "\"windowWidth\":1457}",
    // nonconvex planar quad
    "{\"farClipRatio\":20.0,\"fov\":45.0,\"nearClipRatio\":0.005,"
    "\"projectionMode\":\"Perspective\",\"viewMat\":[-0.998822569847107,5."
    "355104804039e-09,-0.0485050119459629,-0.216645479202271,-0."
    "0358295626938343,0.674063801765442,0.737804353237152,0.165525689721107,0."
    "0326956212520599,0.738674163818359,-0.673270523548126,-2.33209872245789,0."
    "0,0.0,0.0,1.0],\"windowHeight\":1015,\"windowWidth\":1457}",
    // nonconvex nonplanar quad
    "{\"farClipRatio\":20.0,\"fov\":45.0,\"nearClipRatio\":0.005,"
    "\"projectionMode\":\"Perspective\",\"viewMat\":[-0.171657741069794,-7."
    "56699591875076e-10,-0.985156536102295,7.5669963350844e-11,-0."
    "717151820659637,0.685624599456787,0.124959386885166,0.0312160551548004,0."
    "675447225570679,0.72795706987381,-0.117692723870277,-3.24677133560181,0.0,"
    "0.0,0.0,1.0],\"windowHeight\":1015,\"windowWidth\":1457}",
    // planar octagon
    "{\"farClipRatio\":20.0,\"fov\":45.0,\"nearClipRatio\":0.005,"
    "\"projectionMode\":\"Perspective\",\"viewMat\":[-0.999916672706604,-3."
    "63797880709171e-10,-0.0128343626856804,-0.0905372425913811,-0."
    "00934288371354342,0.685624301433563,0.727900147438049,0.164369627833366,0."
    "00879903137683868,0.727957427501678,-0.685569226741791,-3.2100293636322,0."
    "0,0.0,0.0,1.0],\"windowHeight\":1015,\"windowWidth\":1457}",
    // nice nonplanar octagon
    "{\"farClipRatio\":20.0,\"fov\":45.0,\"nearClipRatio\":0.005,"
    "\"projectionMode\":\"Perspective\",\"viewMat\":[0.987349390983582,-2."
    "3283064365387e-10,0.158559530973434,-0.0699826627969742,0."
    "0682378336787224,0.902657330036163,-0.424917191267014,-0.179787904024124,-"
    "0.143124654889107,0.430361896753311,0.891238331794739,-2.61236953735352,0."
    "0,0.0,0.0,1.0],\"windowHeight\":1015,\"windowWidth\":1457}",
    // approaching circles
    "{\"farClipRatio\":20.0,\"fov\":45.0,\"nearClipRatio\":0.005,"
    "\"projectionMode\":\"Perspective\",\"viewMat\":[0.000446119578555226,-3."
    "51883500115946e-09,0.999999165534973,-0.0651277303695679,0."
    "437462627887726,0.899236857891083,-0.000195159620488994,-0."
    "594492673873901,-0.899236619472504,0.437462538480759,0.000401298108045012,"
    "-4.23821687698364,0.0,0.0,0.0,1.0],\"windowHeight\":1015,\"windowWidth\":"
    "1457}"
    // done
};

// std::vector<float3> pts{float3{1, 0, 1}, float3{0, 0.1, -1}, float3{-1, 0,
// 1},
//                         float3{0, -0.1, -.5}};

SimplePolygonMesh sphere_tracing_mesh;
std::vector<std::pair<std::string, std::vector<Vector3>>>
    sphere_tracing_mesh_normals;
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

double evaluate_solid_angle(glm::vec3 x, size_t solid_angle_formula) {
    uint globalStart = loops[0].x;
    return polygon_solid_angle<double>(
        pts.data(), loops.data(), globalStart, {0}, {x[0], x[1], x[2]},
        solid_angle_formula, nullptr, 0, false, 1);
}

bool intersect(glm::vec3 ro, glm::vec3 rd, float* t = nullptr,
               float* iter_frac = nullptr, float* omega = nullptr,
               acceleration_stats* stats = nullptr, size_t i_loop = 0) {
    solid_angle_intersection_params sa_params;
    sa_params.ray_P                   = to_float3(ro);
    sa_params.ray_D                   = to_float3(rd);
    sa_params.ray_tmin                = tmin;
    sa_params.ray_tmax                = tmax;
    sa_params.loops                   = &loops[i_loop];
    sa_params.pts                     = pts.data();
    sa_params.n_loops                 = polygon_with_holes ? loops.size() : 1;
    sa_params.epsilon                 = epsilon;
    sa_params.levelset                = target_levelset * 4. * M_PI;
    sa_params.frequency               = 0;
    sa_params.solid_angle_formula     = i_solid_angle_mode;
    sa_params.use_grad_termination    = use_grad_termination;
    sa_params.max_iterations          = max_iterations;
    sa_params.clip_y                  = false;
    sa_params.capture_misses          = false;
    sa_params.use_overstepping        = use_overstepping;
    sa_params.use_extrapolation       = use_extrapolation;
    sa_params.use_newton              = use_newton;
    sa_params.use_quick_triangulation = false;
    sa_params.epsilon_loose           = sqrt(epsilon);
    sa_params.fixed_step_count        = fixed_step_count;

    float ignore_t, ignore_iter, ignore_omega;
    if (!t) t = &ignore_t;
    if (!iter_frac) iter_frac = &ignore_iter;
    if (!omega) omega = &ignore_omega;

    return ray_nonplanar_polygon_intersect_T<double>(sa_params, omega,
                                                     iter_frac, t, stats);
}

bool intersect_mesh(glm::vec3 ro, glm::vec3 rd, float* t = nullptr,
                    float* iter_frac = nullptr, float* omega = nullptr,
                    acceleration_stats* stats = nullptr) {

    float t_min         = tmax;
    float iter_frac_min = 0;
    float omega_min     = 0;
    acceleration_stats stats_min;
    bool did_hit = false;

    for (size_t i_loop = 0; i_loop < loops.size(); i_loop++) {
        float t_i, iter_frac_i, omega_i;
        acceleration_stats stats_i;
        bool hit_i =
            intersect(ro, rd, &t_i, &iter_frac_i, &omega_i, &stats_i, i_loop);
        did_hit = did_hit || hit_i;
        if (t_i < t_min) {
            t_min         = t_i;
            omega_min     = omega_i;
            iter_frac_min = iter_frac_i;
            stats_min     = stats_i;
        }
    }

    if (t) *t = t_min;
    if (iter_frac) *iter_frac = iter_frac_min;
    if (omega) *omega = omega_min;
    if (stats) *stats = stats_min;

    return did_hit;
}

bool intersect_newton(glm::vec3 ro, glm::vec3 rd, float* t = nullptr,
                      float* iter_frac = nullptr, float* omega = nullptr,
                      acceleration_stats* stats = nullptr, int verbosity = 0) {
    solid_angle_intersection_params sa_params;
    sa_params.ray_P                   = to_float3(ro);
    sa_params.ray_D                   = to_float3(rd);
    sa_params.ray_tmin                = tmin;
    sa_params.ray_tmax                = tmax;
    sa_params.loops                   = loops.data();
    sa_params.pts                     = pts.data();
    sa_params.n_loops                 = polygon_with_holes ? loops.size() : 1;
    sa_params.epsilon                 = epsilon;
    sa_params.levelset                = target_levelset * 4. * M_PI;
    sa_params.frequency               = 0;
    sa_params.solid_angle_formula     = i_solid_angle_mode;
    sa_params.use_grad_termination    = use_grad_termination;
    sa_params.max_iterations          = max_iterations;
    sa_params.clip_y                  = false;
    sa_params.capture_misses          = false;
    sa_params.use_overstepping        = use_overstepping;
    sa_params.use_extrapolation       = use_extrapolation;
    sa_params.use_newton              = use_newton;
    sa_params.use_quick_triangulation = false;
    sa_params.epsilon_loose           = sqrt(epsilon);
    sa_params.fixed_step_count        = fixed_step_count;

    float ignore_t, ignore_iter, ignore_omega;
    if (!t) t = &ignore_t;
    if (!iter_frac) iter_frac = &ignore_iter;
    if (!omega) omega = &ignore_omega;

    return newton_intersect_T<double>(sa_params, omega, iter_frac, t, nullptr,
                                      stats, verbosity);
}

bool intersect_bisection(glm::vec3 ro, glm::vec3 rd, float* t = nullptr,
                         float* iter_frac = nullptr, float* omega = nullptr,
                         acceleration_stats* stats = nullptr,
                         int verbosity             = 0) {
    solid_angle_intersection_params sa_params;
    sa_params.ray_P                   = to_float3(ro);
    sa_params.ray_D                   = to_float3(rd);
    sa_params.ray_tmin                = tmin;
    sa_params.ray_tmax                = tmax;
    sa_params.loops                   = loops.data();
    sa_params.pts                     = pts.data();
    sa_params.n_loops                 = polygon_with_holes ? loops.size() : 1;
    sa_params.epsilon                 = epsilon;
    sa_params.levelset                = target_levelset * 4. * M_PI;
    sa_params.frequency               = 0;
    sa_params.solid_angle_formula     = i_solid_angle_mode;
    sa_params.use_grad_termination    = use_grad_termination;
    sa_params.max_iterations          = max_iterations;
    sa_params.clip_y                  = false;
    sa_params.capture_misses          = false;
    sa_params.use_overstepping        = use_overstepping;
    sa_params.use_extrapolation       = use_extrapolation;
    sa_params.use_newton              = use_newton;
    sa_params.use_quick_triangulation = false;
    sa_params.epsilon_loose           = sqrt(epsilon);
    sa_params.fixed_step_count        = fixed_step_count;

    float ignore_t, ignore_iter, ignore_omega;
    if (!t) t = &ignore_t;
    if (!iter_frac) iter_frac = &ignore_iter;
    if (!omega) omega = &ignore_omega;

    return bisection_intersect_T<double>(sa_params, omega, iter_frac, t,
                                         nullptr, stats, verbosity);
}

bool intersect_interval(glm::vec3 ro, glm::vec3 rd, float* t = nullptr,
                        float* iter_frac = nullptr, float* omega = nullptr,
                        acceleration_stats* stats = nullptr,
                        int verbosity             = 0) {
    solid_angle_intersection_params sa_params;
    sa_params.ray_P                   = to_float3(ro);
    sa_params.ray_D                   = to_float3(rd);
    sa_params.ray_tmin                = tmin;
    sa_params.ray_tmax                = tmax;
    sa_params.loops                   = loops.data();
    sa_params.pts                     = pts.data();
    sa_params.n_loops                 = polygon_with_holes ? loops.size() : 1;
    sa_params.epsilon                 = epsilon;
    sa_params.levelset                = target_levelset * 4. * M_PI;
    sa_params.frequency               = 0;
    sa_params.solid_angle_formula     = i_solid_angle_mode;
    sa_params.use_grad_termination    = use_grad_termination;
    sa_params.max_iterations          = max_iterations;
    sa_params.clip_y                  = false;
    sa_params.capture_misses          = false;
    sa_params.use_overstepping        = use_overstepping;
    sa_params.use_extrapolation       = use_extrapolation;
    sa_params.use_newton              = use_newton;
    sa_params.use_quick_triangulation = false;
    sa_params.epsilon_loose           = sqrt(epsilon);
    sa_params.fixed_step_count        = fixed_step_count;

    float ignore_t, ignore_iter, ignore_omega;
    if (!t) t = &ignore_t;
    if (!iter_frac) iter_frac = &ignore_iter;
    if (!omega) omega = &ignore_omega;

    float t_star;
    bool should_hit = intersect(ro, rd, &t_star);

    return interval_intersect_T<double>(sa_params, omega, iter_frac, t,
                                        &t_star);
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

std::vector<glm::vec3> loop_glm_pts(const std::vector<float3>& pts,
                                    uint3 loop) {
    std::vector<glm::vec3> glm_pts;
    size_t s = loop.x; // start
    size_t N = loop.y; // loop size
    for (size_t i = s; i < s + N; i++) glm_pts.push_back(to_vec3(pts[i]));
    return glm_pts;
}

SimplePolygonMesh mesh_levelset(uint subdivisions = 64,
                                float contraction = 0.) {
    std::vector<glm::vec3> glm_pts = loop_glm_pts(pts, loops[0]);
    glm::vec3 center               = computeVirtualVertex(glm_pts);
    SimplePolygonMesh mesh = simple_mesh_polygon(glm_pts, center, subdivisions);

    if (contraction > 0) {
        Vector3 c = to_gc(center);
        for (Vector3& v : mesh.vertexCoordinates)
            v = c + (v - c) * (1. - contraction);
    }

    //===== Project onto the level set
    // check if p lies on any edge of polygon
    auto on_boundary = [&](const Vector3& p) -> bool {
        for (uint iE = 0; iE < loops[0].y; iE++) {
            Vector3 a = to_gc(pts[loops[0].x + iE]);
            Vector3 b = to_gc(pts[loops[0].x + ((iE + 1) % loops[0].y)]);
            if (dist_to_segment(p, a, b) < 1e-8) return true;
        }
        return false;
    };
    glm::vec3 proj_dir{0, -1, 0};
    // configure harnack params
    float old_epsilon = epsilon;
    epsilon           = fmin(epsilon, 0.0001);
    for (Vector3& v : mesh.vertexCoordinates) {
        // leave points on boundary fixed
        if (on_boundary(v)) continue;

        glm::vec3 start = to_vec3(v);
        float t_up, t_down;
        bool projects_up_onto_levelset   = intersect(start, proj_dir, &t_up);
        bool projects_down_onto_levelset = intersect(start, -proj_dir, &t_down);

        if (projects_up_onto_levelset && projects_down_onto_levelset) {
            // if levelset exists in both directions, take closer
            if (t_up < t_down) {
                projects_down_onto_levelset = false;
            } else {
                projects_up_onto_levelset = false;
            }
        }

        if (projects_up_onto_levelset) {
            v = to_gc(start + t_up * proj_dir);
        } else if (projects_down_onto_levelset) {
            v = to_gc(start - t_down * proj_dir);
        }
    }
    // restore old harnack params
    epsilon = old_epsilon;

    return mesh;
}

//====== Experiment code
enum class TracingMethod { Harnack, Sphere, Newton, Bisection, Interval };
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
    std::vector<float> ts, omegas, iterationCounts, overstep_success_rate,
        steps_after_epsilon_loose, newton_steps, values_triangulated,
        values_prequantum, values_gauss_bonnet;
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
            bool hit;
            int verbosity = didHit.size() == 317;
            switch (method) {
            case TracingMethod::Harnack:
                hit =
                    intersect_with_mesh
                        ? intersect_mesh(ro, rd, &t, &iter_frac, &omega, &stats)
                        : intersect(ro, rd, &t, &iter_frac, &omega, &stats,
                                    loop_id);
                break;
            case TracingMethod::Sphere:
                hit = intersect_sphere_tracing(ro, rd, &t, &iter_frac, &omega,
                                               &stats);
                break;
            case TracingMethod::Newton:
                hit = intersect_newton(ro, rd, &t, &iter_frac, &omega, &stats,
                                       verbosity);
                break;
            case TracingMethod::Bisection:
                hit = intersect_bisection(ro, rd, &t, &iter_frac, &omega,
                                          &stats, verbosity);
                break;
            case TracingMethod::Interval:
                hit = intersect_interval(ro, rd, &t, &iter_frac, &omega, &stats,
                                         verbosity);
                break;
            }

            if (hit) {
                glm::vec3 intersection = ro + t * rd;
                intersections.push_back(intersection);

                ts.push_back(t);
                omegas.push_back(omega);
                normals.push_back(normal(intersections.back()));

                if (false) {
                    values_triangulated.push_back(
                        evaluate_solid_angle(intersection, 0));
                    values_prequantum.push_back(
                        evaluate_solid_angle(intersection, 1));
                    values_gauss_bonnet.push_back(
                        evaluate_solid_angle(intersection, 2));
                }
            }

            bool debug_query = false;
            if (debug_query) {
                std::cout << "   final t value: " << stats.times.back()
                          << vendl;
                std::cout << "   final ω value: " << stats.vals.back() << vendl;
                std::vector<Vector2> t_w_pairs;
                for (size_t i = 0; i < stats.times.size(); i++) {
                    t_w_pairs.push_back({stats.times[i], stats.vals[i]});
                }
                polyscope::registerCurveNetworkLine2D("t-ω plot", t_w_pairs);
            }

            iterationCounts.push_back(stats.total_iterations);
            overstep_success_rate.push_back((double)stats.successful_oversteps /
                                            (double)stats.total_iterations);
            successful_extrapolations += stats.successful_extrapolations;

            if (verbosity > 0) {
                didHit.push_back(verbosity + 1);
            } else {
                didHit.push_back(hit);
            }

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
            bool print_newton_stats = false && didHit.size() == 54;
            if (print_newton_stats) {
                for (size_t iI = 0; iI < stats.newton_ts.size(); iI++) {
                    std::cout
                        << std::setfill(' ') << std::setw(3) << iI
                        << "| t = " << std::setw(8) << stats.newton_ts[iI]
                        << "  f = " << std::setw(8) << stats.newton_vals[iI]
                        << " 4π = " << std::setw(8) << 4. * M_PI
                        << " dt = " << std::setw(8) << stats.newton_dts[iI]
                        << " df = " << std::setw(8) << stats.newton_dfs[iI]
                        << std::endl;
                }
            }
        }
    }
    double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    double meanTime = duration / viewRayPts.size();

    psCloud = polyscope::registerPointCloud("intersections", intersections);
    psCloud->addScalarQuantity("t", ts);
    psCloud->addScalarQuantity("omega", omegas);
    psCloud->addVectorQuantity("normal", normals);
    if (false) {
        psCloud->addScalarQuantity("values (triangulated)",
                                   values_triangulated);
        psCloud->addScalarQuantity("values (prequantum)", values_prequantum);
        psCloud->addScalarQuantity("values (gauss-bonnet)",
                                   values_gauss_bonnet);
    }

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
    std::string filename = tracing_mode + "_convergence_statistics_full.csv") {
    std::ofstream out;
    out.open(filename);

    // Use full precision
    out.precision(std::numeric_limits<double>::max_digits10);

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

void print_camera_view(polyscope::CameraParameters params) {
    glm::vec3 camPos   = params.getPosition();
    glm::vec3 lookDir  = params.getLookDir();
    glm::vec3 upDir    = params.getUpDir();
    glm::vec3 rightDir = params.getRightDir();
    double fovY        = params.getFoVVerticalDegrees();

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
    std::cout << "const float fovY = " << fovY << ".;" << std::endl;
}

void build_fcpw_scene() {
    uint n_triangles = sphere_tracing_mesh.polygons.size();
    uint n_vertices  = 3 * n_triangles;

    // initialize a 3d scene
    scene = fcpw::Scene<3>();

    // set the PrimitiveType for each object in the scene;
    // in this case, we have a single object consisting of triangles
    scene.setObjectTypes({{fcpw::PrimitiveType::Triangle}});

    // set the vertex and triangle count of the (0th) object
    scene.setObjectVertexCount(n_vertices, 0);
    scene.setObjectTriangleCount(n_triangles, 0);

    // specify the vertex positions
    for (int iV = 0; iV < sphere_tracing_mesh.vertexCoordinates.size(); iV++) {
        const Vector3& v = sphere_tracing_mesh.vertexCoordinates[iV];
        scene.setObjectVertex(to_fcpw(v), iV, 0);
    }

    // specify the triangle indices
    for (int iT = 0; iT < n_triangles; iT++) {
        const std::vector<size_t>& face = sphere_tracing_mesh.polygons[iT];
        std::array<int, 3> indices{(int)face[0], (int)face[1], (int)face[2]};
        scene.setObjectTriangle(indices.data(), iT, 0);
    }

    // once geometry has been specified, build acceleration structure
    // the second boolean argument toggles vectorization
    scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false);

    // perform a closest point query
    fcpw::Interaction<3> interaction;
    fcpw::Vector<3> query_point;
    scene.findClosestPoint(query_point, interaction);
}

polyscope::SurfaceMesh*
build_comparison_mesh(std::string comparison_name,
                      const std::vector<std::array<glm::vec3, 3>>& mesh) {
    std::vector<glm::vec3> positions;
    std::vector<std::vector<size_t>> faceIndices;
    for (size_t iF = 0; iF < mesh.size(); iF++) {
        faceIndices.push_back(std::vector<size_t>{
            positions.size(), positions.size() + 1, positions.size() + 2});
        for (size_t i = 0; i < 3; i++) positions.push_back(mesh[iF][i]);
    }
    comparison_meshes[comparison_name] = mesh;
    return polyscope::registerSurfaceMesh(comparison_name + " mesh", positions,
                                          faceIndices);
}

polyscope::SurfaceMesh*
build_comparison_mesh(std::string comparison_name,
                      const std::function<std::vector<std::array<glm::vec3, 3>>(
                          const std::vector<glm::vec3>&)>& build_mesh) {

    std::vector<glm::vec3> glm_pts = loop_glm_pts(pts, loops[0]);
    auto mesh                      = build_mesh(glm_pts);
    return build_comparison_mesh(comparison_name, mesh);
}

void export_polygons_to_shadertoy() {
    polyscope::CameraParameters prev_params =
        polyscope::view::getCameraParametersForCurrentView();

    std::cout << "//====== named polygons" << std::endl;
    for (size_t iP = 0; iP < named_polygons.size(); iP++) {
        std::cout << "//  P" << iP << " : " << named_polygons[iP].name
                  << std::endl;
    }
    std::cout << "#define P0" << std::endl << std::endl;

    std::cout << "//====== polygon data" << std::endl;
    for (size_t iP = 0; iP < named_polygons.size(); iP++) {
        std::cout << "#ifdef P" << iP << std::endl;
        std::cout << "#define nV " << named_polygons[iP].pts.size()
                  << std::endl;
        std::cout << "vec3 points[nV] = vec3[](";
        for (size_t iPt = 0; iPt < named_polygons[iP].pts.size(); iPt++) {
            const float3& pt = named_polygons[iP].pts[iPt];
            std::cout << "vec3( " << pt.x << ", " << pt.y << ", " << pt.z
                      << " )";
            if (iPt + 1 < named_polygons[iP].pts.size()) std::cout << ", ";
        }
        std::cout << " );" << std::endl;

        std::cout << "#define nE " << named_polygons[iP].pts.size()
                  << std::endl;
        std::cout << "vec2 edges[nE] = vec2[](";
        for (size_t iL = 0; iL < named_polygons[iP].loops.size(); iL++) {
            const uint3& loop = named_polygons[iP].loops[iL];
            for (size_t iPt = 0; iPt < loop.y; iPt++) {
                std::cout << "vec2( " << loop.x + iPt << ", "
                          << (loop.x + ((iPt + 1) % loop.y)) << " )";
                if (iPt + 1 < loop.y) std::cout << ", ";
            }
            if (iL + 1 < named_polygons[iP].loops.size()) std::cout << ", ";
        }
        std::cout << " );" << std::endl;

        polyscope::view::setCameraFromJson(
            camera_positions[named_polygons[iP].cam_view], false);
        print_camera_view(polyscope::view::getCameraParametersForCurrentView());


        std::cout << "#endif // P" << iP << std::endl;
    }


    polyscope::view::setViewToCamera(prev_params);
}

polyscope::CurveNetwork* drawPolygon(const std::vector<float3>& pts,
                                     const std::vector<uint3>& loops) {
    std::vector<std::array<size_t, 2>> polygon_segments;
    for (size_t iL = 0; iL < loops.size(); iL++) {
        const uint3& loop = loops[iL];
        for (size_t iPt = 0; iPt < loop.y; iPt++) {
            polygon_segments.push_back(std::array<size_t, 2>{
                loop.x + iPt, loop.x + ((iPt + 1) % loop.y)});
        }
    }
    return polyscope::registerCurveNetwork("polygon", pts, polygon_segments);
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
static int selected_polygon = 0;
void myCallback() {
    ImGui::Combo("Intersection Mode", &i_tracing_mode,
                 tracing_mode_names.data(), tracing_mode_names.size());

    if (ImGui::Button("Shoot Camera Rays")) {
        std::string name =
            std::string(use_grad_termination ? "grad-terminated " : "") +
            std::string(use_overstepping ? "overstepped " : "") +
            std::string(use_extrapolation ? "extrapolated " : "") +
            std::string(use_newton ? "newton-accelerated " : "");
        if (name.length() == 0) name = "default ";

        switch (i_tracing_mode) {
        case 0:
            name += "Harnack tracing";
            shootCameraRays(name, TracingMethod::Harnack);
            break;
        case 1:
            name += "sphere tracing";
            if (sphere_tracing_mesh.vertexCoordinates.empty()) {
                sphere_tracing_mesh = mesh_levelset(3);
                polyscope::registerSurfaceMesh(
                    "mesh", sphere_tracing_mesh.vertexCoordinates,
                    sphere_tracing_mesh.polygons);
                build_fcpw_scene();
            }

            shootCameraRays(name, TracingMethod::Sphere);
            break;
        case 2:
            name += "Newton's method";
            shootCameraRays(name, TracingMethod::Newton);
            break;
        case 3:
            name += "bisection search";
            shootCameraRays(name, TracingMethod::Bisection);
            { // draw triangulation used to evaluate solid angle
                std::vector<std::vector<size_t>> vert_face_adj;
                for (uint3 loop : loops) {
                    size_t s = loop.x; // start
                    size_t N = loop.y;
                    for (size_t i = 0; i < N; i++) {
                        size_t j = (i + 1) % N;
                        vert_face_adj.push_back({s + i, s + j, s + N});
                    }
                }
                polyscope::registerSurfaceMesh("Solid Angle Triangulation", pts,
                                               vert_face_adj);
            }
            break;
        case 4:
            name += "interval arithmetic";
            shootCameraRays(name, TracingMethod::Interval);
            break;
        }
    }
    if (ImGui::Button("Restore Camera View")) {
        polyscope::view::setViewToCamera(camParams);
    }
    if (ImGui::Button("Print ShaderToy Camera Parameters")) {
        print_camera_view(camParams);
    }
    if (ImGui::Button("Print Camera JSON")) {
        std::cout << polyscope::view::getCameraJson() << std::endl;
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
    if (ImGui::Button("Save Convergence Statistics")) {
        write_convergence_statistics();
    }
    if (ImGui::TreeNode("Sphere Tracing")) {
        if (ImGui::Button("Build Small Sphere Tracing Mesh")) {
            sphere_tracing_mesh = mesh_levelset(3, 0.05);
            polyscope::registerSurfaceMesh(
                "mesh", sphere_tracing_mesh.vertexCoordinates,
                sphere_tracing_mesh.polygons);
            build_fcpw_scene();
        }
        if (ImGui::Button("Build Sphere Tracing Mesh")) {
            sphere_tracing_mesh = mesh_levelset();
            polyscope::registerSurfaceMesh(
                "mesh", sphere_tracing_mesh.vertexCoordinates,
                sphere_tracing_mesh.polygons);
            build_fcpw_scene();
        }
        if (ImGui::Button("Compute Sphere Tracing Normals")) {
            sphere_tracing_mesh_normals =
                std::vector<std::pair<std::string, std::vector<Vector3>>>{
                    std::make_pair("nicole", std::vector<Vector3>{}),
                    std::make_pair("finite_differences",
                                   std::vector<Vector3>{}),
                    std::make_pair("architecture", std::vector<Vector3>{})};

            for (size_t iV = 0;
                 iV < sphere_tracing_mesh.vertexCoordinates.size(); iV++) {
                Vector3 v = sphere_tracing_mesh.vertexCoordinates[iV];
                for (size_t i = 0; i < 3; i++) {
                    Vector3 n = -to_gc(ray_nonplanar_polygon_normal_T<double>(
                        to_float3(v), loops.data(), pts.data(), 1, i));
                    sphere_tracing_mesh_normals[i].second.push_back(n);
                }
            }

            for (size_t i = 0; i < 3; i++) {
                polyscope::getSurfaceMesh("mesh")->addVertexVectorQuantity(
                    "normals (" + sphere_tracing_mesh_normals[i].first + ")",
                    sphere_tracing_mesh_normals[i].second);
            }
        }
        if (ImGui::Button("Save sphere tracing normals")) {
            std::string polygon_name = named_polygons[selected_polygon].name;

            double min_y = 10000; // find min y coord to shift everything up to
                                  // start at CZ-plane
            std::vector<Vector3> gc_pts;
            std::vector<std::vector<size_t>> face_vertex_lists;
            for (uint3 loop : loops) {
                size_t s = loop.x;
                size_t N = loop.y;
                face_vertex_lists.push_back({});
                for (size_t i = s; i < s + N; i++) {
                    face_vertex_lists.back().push_back(gc_pts.size());
                    gc_pts.push_back(to_gc(pts[i]));
                    min_y = fmin(min_y, gc_pts.back().y);
                }
            }
            for (Vector3& p : gc_pts) p.y -= min_y;
            SimplePolygonMesh nonplanar_polygon(face_vertex_lists, gc_pts);
            nonplanar_polygon.writeMesh(polygon_name + "_polygon.obj", "obj");


            std::array<std::vector<Vector3>, 3> point_faces;
            for (size_t i = 0; i < 3; i++) {
                point_faces[i] =
                    generate_point_faces(sphere_tracing_mesh.vertexCoordinates,
                                         sphere_tracing_mesh_normals[i].second);
                for (Vector3& p : point_faces[i]) p.y -= min_y;
            }
            std::vector<std::vector<size_t>> face_vertex_indices;
            for (size_t iF = 0; iF < point_faces[0].size(); iF++)
                face_vertex_indices.push_back(
                    std::vector<size_t>{3 * iF, 3 * iF + 1, 3 * iF + 2});

            for (size_t i = 0; i < 3; i++) {
                SimplePolygonMesh cloud_tris(face_vertex_indices,
                                             point_faces[i]);
                cloud_tris.writeMesh(polygon_name + "_normals_" +
                                         sphere_tracing_mesh_normals[i].first +
                                         ".obj",
                                     "obj");
            }
        }
        if (ImGui::Button("Save sphere tracing mesh")) {
            std::string polygon_name = named_polygons[selected_polygon].name;

            double min_y = 10000; // find min y coord to shift everything up to
                                  // start at xz-plane
            std::vector<Vector3> shifted_pts;
            for (Vector3 pt : sphere_tracing_mesh.vertexCoordinates) {
                shifted_pts.push_back(pt);
                min_y = fmin(min_y, pt.y);
            }
            for (Vector3& p : shifted_pts) p.y -= min_y;

            SimplePolygonMesh shifted_mesh(sphere_tracing_mesh.polygons,
                                           shifted_pts);
            std::cout << "Writing sphere tracing mesh to " << polygon_name
                      << "_mesh.obj" << vendl;
            shifted_mesh.writeMesh(polygon_name + "_mesh.obj", "obj");
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Other Meshes")) {
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
                polyscope::warning(
                    "Bilinear interpolation only makes sense if the "
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
                compute_vertex_face_lists(
                    comparison.second, mesh.vertexCoordinates, mesh.polygons);
                std::string filename = comparison.first + "_mesh.obj";
                std::cout << "  Writing " << filename << "..." << vendl;
                mesh.writeMesh(filename, "obj");
            }
            std::cout << "Done writing meshes" << vendl;
        }
        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Parameters")) {
        ImGui::Combo("Solid Angle Mode", &i_solid_angle_mode,
                     solid_angle_mode_names.data(),
                     solid_angle_mode_names.size());
        ImGui::SliderFloat("epsilon (log)", &epsilon, .00000001f, .1f, "%.4f",
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
        ImGui::Checkbox("intersect_with_mesh", &intersect_with_mesh);
        ImGui::Checkbox("polygon_with_holes", &polygon_with_holes);
        ImGui::DragInt("loop_id", &loop_id, 1, 0, loops.size());
        ImGui::DragFloat("target_levelset", &target_levelset, .1f, 0.f, 1.f);
        ImGui::TreePop();
    }
    static std::vector<const char*> polygon_names;
    if (polygon_names.empty())
        for (auto& poly : named_polygons)
            polygon_names.push_back(poly.name.c_str());
    if (ImGui::TreeNode("Experiments")) {
        auto display_polygon = [&](size_t iP) {
            pts          = named_polygons[iP].pts;
            loops        = named_polygons[iP].loops;
            auto polygon = drawPolygon(pts, loops);
            polygon->setColor(glm::vec3(0));
            // set soft shadows on the ground
            polyscope::options::groundPlaneMode =
                polyscope::GroundPlaneMode::ShadowOnly;
            polyscope::options::groundPlaneHeightFactor = 0.00; // adjust the
            // plane height
            polyscope::options::shadowDarkness = 0.15; // lighter shadows
        };
        if (ImGui::Combo("Select Polygon", &selected_polygon,
                         polygon_names.data(), polygon_names.size())) {
            display_polygon(selected_polygon);
        }
        if (ImGui::Button("View From Polygon Camera")) {
            polyscope::view::setCameraFromJson(
                camera_positions[named_polygons[selected_polygon].cam_view],
                false);
        }
        if (ImGui::Button("Render All Polygons")) {
            for (size_t iP = 0; iP < named_polygons.size(); iP++) {
                display_polygon(iP);
                polyscope::view::setCameraFromJson(
                    camera_positions[named_polygons[iP].cam_view], false);
                std::string polygon_name = named_polygons[iP].name;

                size_t nL = named_polygons[iP].loops.size();
                if (nL == 2) {
                    // special case where we only do minimal surfaces
                    polyscope::screenshot(polygon_name + "--outline.png");

                    // first try optimizing boundary components separately
                    const std::vector<float3>& pts = named_polygons[iP].pts;
                    uint3 loop_a = named_polygons[iP].loops[0];
                    uint3 loop_b = named_polygons[iP].loops[1];
                    std::vector<glm::vec3> pts_a, pts_b;
                    for (size_t iA = 0; iA < loop_a.y; iA++)
                        pts_a.push_back(to_vec3(pts[loop_a.x + iA]));
                    for (size_t iB = 0; iB < loop_b.y; iB++)
                        pts_b.push_back(to_vec3(pts[loop_b.x + iB]));

                    // HACK : TKTKTKTKT : what's the deal with boundary loop
                    // orientations? why do solid angle and minimal surfaces
                    // disagree?
                    for (glm::vec3& p_b : pts_b) p_b.y = 2. - p_b.y;

                    double area_a, area_b;
                    std::vector<std::array<glm::vec3, 3>> side_a =
                        minimalSurfaceArea(pts_a, &area_a);
                    std::vector<std::array<glm::vec3, 3>> side_b =
                        minimalSurfaceArea(pts_b, &area_b);

                    double area_tube;
                    std::vector<std::array<glm::vec3, 3>> tube;
                    try {
                        tube = minimalSurfaceArea(pts_a, pts_b, &area_tube);
                    } catch (std::logic_error& e) {
                        // if mesh degenerates, set big area so we choose the
                        // other option
                        area_tube = 999;
                    }

                    polyscope::SurfaceMesh* psMesh;
                    if (area_tube < area_a + area_b) {
                        psMesh = build_comparison_mesh("minimal", tube);
                    } else {
                        side_a.insert(side_a.end(), side_b.begin(),
                                      side_b.end());
                        psMesh = build_comparison_mesh("minimal", side_a);
                    }
                    psMesh->setTransparency(0.75)->setEnabled(true);
                    polyscope::screenshot(polygon_name + "--minimal.png");
                    psMesh->setEnabled(false);
                    // polyscope::show();
                    continue;
                } else if (nL > 2) {
                    polyscope::warning("Minimal surfaces with >2 boundary "
                                       "components not supported");
                    continue;
                }

                polyscope::screenshot(polygon_name + "--outline.png");

                auto psMesh =
                    build_comparison_mesh("wachpress", wachpressInterpolate)
                        ->setTransparency(0.75);
                psMesh->setEnabled(true);
                polyscope::screenshot(polygon_name + "--wachpress.png");
                psMesh->setEnabled(false);

                psMesh =
                    build_comparison_mesh("mean_value", meanValueInterpolate)
                        ->setTransparency(0.75);
                psMesh->setEnabled(true);
                polyscope::screenshot(polygon_name + "--mean_value.png");
                psMesh->setEnabled(false);

                psMesh = build_comparison_mesh("astrid", astridInterpolate)
                             ->setTransparency(0.75);
                psMesh->setEnabled(true);
                polyscope::screenshot(polygon_name + "--astrid.png");
                psMesh->setEnabled(false);

                psMesh = build_comparison_mesh("minimal", minimalSurface)
                             ->setTransparency(0.75);
                psMesh->setEnabled(true);
                polyscope::screenshot(polygon_name + "--minimal.png");
                psMesh->setEnabled(false);

                if (pts.size() == 4) {
                    psMesh =
                        build_comparison_mesh("bilinear", bilinearInterpolate)
                            ->setTransparency(0.75);
                    psMesh->setEnabled(true);
                    polyscope::screenshot(polygon_name + "--bilinear.png");
                    psMesh->setEnabled(false);
                }
            }
        }
        if (ImGui::Button("Export Polygons to ShaderToy")) {
            export_polygons_to_shadertoy();
        }
        ImGui::TreePop();
    }
}

int main(int argc, char** argv) {
    polyscope::init(); // Initialize polyscope
    polyscope::options::programName = "Harnack Debugger";

    // Configure the argument parser
    args::ArgumentParser parser("Harnack debugger");

    args::Positional<std::string> inputFilename(parser, "loop_file",
                                                ".loops file to load.");

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

    construct_approaching_circles();
    if (inputFilename) {
        load_loop_file(args::get(inputFilename));
        pts   = named_polygons.back().pts;
        loops = named_polygons.back().loops;
    }

    // Set the callback function
    polyscope::state::userCallback = myCallback;

    drawPolygon(pts, loops);

    // store initial camera position
    camParams = polyscope::view::getCameraParametersForCurrentView();
    // shootCameraRays();

    // Give control to the polyscope gui
    polyscope::show();

    return EXIT_SUCCESS;
}

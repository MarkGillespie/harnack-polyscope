#include "geometrycentral/utilities/vector3.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "fcpw/fcpw.h"

#include "fftw3.h"

#include "args/args.hxx"
#include "imgui.h"

#include <fstream>

#include "generalized_barycentric_coordinates.h"
#include "geometry_utils.h"
#include "harnack.h"
#include "utils.h"

using namespace geometrycentral;

//====== Scene parameters

std::vector<Vector3> pts, normals;
double levelset = 0.5; // icosphere

typedef struct named_dipole {
    std::string name;
    std::vector<Vector3> pts, normals;
    size_t cam_view;
} named_dipole;

double tIco = (1. + sqrt(5.)) / 2.;
double aIco = (4. * PI) / 20.; // icosphere

std::vector<named_dipole> named_dipoles = {
    // clang-format off
    {"icosphere", {
        Vector3{ -1,  tIco,  0 },
        Vector3{  1,  tIco,  0 },
        Vector3{ -1, -tIco,  0 },
        Vector3{  1, -tIco,  0 },

        Vector3{  0, -1,  tIco },
        Vector3{  0,  1,  tIco },
        Vector3{  0, -1, -tIco },
        Vector3{  0,  1, -tIco },

        Vector3{  tIco,  0, -1 },
        Vector3{  tIco,  0,  1 },
        Vector3{ -tIco,  0, -1 },
        Vector3{ -tIco,  0,  1 }
    }, {
        Vector3{ -1,  tIco,  0 }.normalize() * aIco,
        Vector3{  1,  tIco,  0 }.normalize() * aIco,
        Vector3{ -1, -tIco,  0 }.normalize() * aIco,
        Vector3{  1, -tIco,  0 }.normalize() * aIco,

        Vector3{  0, -1,  tIco }.normalize() * aIco,
        Vector3{  0,  1,  tIco }.normalize() * aIco,
        Vector3{  0, -1, -tIco }.normalize() * aIco,
        Vector3{  0,  1, -tIco }.normalize() * aIco,

        Vector3{  tIco,  0, -1 }.normalize() * aIco,
        Vector3{  tIco,  0,  1 }.normalize() * aIco,
        Vector3{ -tIco,  0, -1 }.normalize() * aIco,
        Vector3{ -tIco,  0,  1 }.normalize() * aIco
    }, 0},
    {"spot", {
        Vector3{ 0.306682, -0.404653, 0.383932},	Vector3{ 0.358149, -0.171814, -0.0067112},	Vector3{ -0.19394, -0.449722, 0.557741},	Vector3{ -0.365797, -0.345487, 0.183516},	Vector3{ 0.297829, -0.475641, -0.117192},	Vector3{ 0.102813, -0.590646, 0.884883},	Vector3{ 0.340888, 0.019913, 0.0839484},	Vector3{ 0.294191, -0.561968, 0.676349},	Vector3{ 0.0924641, -0.26676, 0.942342},	Vector3{ 0.205547, -0.568616, 0.180048},	Vector3{ 0.250984, -0.169652, 0.897166},	Vector3{ 0.351356, -0.63048, 0.0736117},	Vector3{ -0.148554, -0.254757, -0.137669},	Vector3{ -0.00320457, 0.313508, 0.263692},	Vector3{ 0.234424, -0.546378, 0.925362},	Vector3{ -0.146923, -0.505335, 0.149239},	Vector3{ 0.109633, -0.465052, -0.0387675},	Vector3{ -0.131672, -0.149403, 0.958545},	Vector3{ -0.240129, -0.578025, -0.119354},	Vector3{ 0.361912, -0.0642679, 0.398502},	Vector3{ -0.28464, -0.408323, 0.900592},	Vector3{ 0.00294226, -0.447451, 0.139488},	Vector3{ -0.115035, -0.624938, 0.698428},	Vector3{ -0.0332526, -0.0243671, -0.227731},	Vector3{ 0.00878529, -0.492536, 0.58569},	Vector3{ 0.105753, 0.317392, 0.186009},	Vector3{ 0.0500551, 0.609591, 0.00245756},	Vector3{ -0.295486, 0.474562, -0.288572},	Vector3{ 0.324769, -0.389326, 0.501175},	Vector3{ -0.357239, -0.0836032, 0.51056},	Vector3{ 0.276488, 0.0759707, 0.661792},	Vector3{ 0.192402, -0.0642831, -0.155336},	Vector3{ -0.0412089, 0.10166, -0.421542},	Vector3{ -0.170017, 0.169369, 0.715236},	Vector3{ -0.327356, -0.589277, 0.804906},	Vector3{ 0.230041, -0.639426, -0.103619},	Vector3{ -0.110596, -0.535272, 0.902005},	Vector3{ -0.342371, 0.036362, 0.145877},	Vector3{ 0.24905, -0.293581, 0.906185},	Vector3{ -0.247021, 0.45102, -0.552863},	Vector3{ -0.32145, -0.426584, 0.673947},	Vector3{ 0.122936, -0.663889, 0.0508158},	Vector3{ -0.161405, 0.151953, -0.127951},	Vector3{ 0.051275, 0.058084, 0.943437},	Vector3{ -0.121045, 0.046851, 0.926983},	Vector3{ -0.206236, 0.759014, -0.323287},	Vector3{ 0.126489, 0.183638, 0.750458},	Vector3{ -0.0362124, 0.231166, 0.504113},	Vector3{ -0.354147, -0.0147716, 0.334118},	Vector3{ -0.296698, -0.246412, 0.859819},	Vector3{ 0.136865, -0.506635, 0.456064},	Vector3{ -0.209169, 0.157045, 0.527834},	Vector3{ 0.00367757, -0.0867345, 0.992075},	Vector3{ -0.383434, -0.324812, 0.0641477},	Vector3{ -0.365795, -0.612288, 0.0439601},	Vector3{ 0.0277583, 0.378706, 0.0416162},	Vector3{ 0.355872, -0.112496, 0.605408},	Vector3{ -0.360554, -0.324764, 0.40733},	Vector3{ 0.277368, 0.417234, -0.151495},	Vector3{ -0.186198, -0.469504, 0.346607},	Vector3{ 0.342467, -0.296243, 0.680109},	Vector3{ 0.374423, -0.219038, 0.390429},	Vector3{ -0.236825, -0.711007, 0.851152},	Vector3{ 0.384589, -0.194908, 0.281083},	Vector3{ 0.252692, 0.0750571, -0.0412684},	Vector3{ 0.373658, -0.305609, -0.024758},	Vector3{ 0.350413, -0.453177, 0.158818},	Vector3{ 0.224368, 0.208949, 0.339743},	Vector3{ 0.07022, -0.226575, -0.174293},	Vector3{ 0.184936, 0.541373, -0.529854},	Vector3{ -0.323632, -0.047496, 0.758027},	Vector3{ 0.306797, 0.57648, -0.208208},	Vector3{ 0.1775, -0.710941, 0.714975},	Vector3{ -0.161812, 0.252072, -0.650599},	Vector3{ -0.228639, 0.498725, -0.0642101},	Vector3{ -0.206675, 0.179157, -0.290203},	Vector3{ 0.212557, 0.254618, -0.642058},	Vector3{ -0.349235, -0.0977347, 0.00828974},	Vector3{ -0.343705, -0.211543, 0.691228},	Vector3{ -0.0639344, 0.413, -0.623514},	Vector3{ -0.0835288, -0.361608, 0.905793},	Vector3{ -0.215308, 0.214672, 0.0460902},	Vector3{ 0.0122909, -0.444126, 0.705749},	Vector3{ 0.168202, 0.163275, -0.26553},	Vector3{ -0.24705, 0.0290526, -0.080353},	Vector3{ 0.0406593, 0.790636, -0.421032},	Vector3{ -0.248916, 0.602213, -0.466259},	Vector3{ 0.330574, 0.279548, -0.470598},	Vector3{ -0.248857, 0.188787, 0.307984},	Vector3{ 0.242355, 0.209323, 0.100013},	Vector3{ -0.210851, 0.137126, -0.50908},	Vector3{ 0.207829, 0.162209, 0.503176},	Vector3{ -0.389944, 0.692829, -0.149178},	Vector3{ -0.165372, -0.712837, 0.0543033},	Vector3{ 0.307525, 0.433808, -0.344528},	Vector3{ 0.209844, 0.898253, -0.315469},	Vector3{ -0.324978, 0.296029, -0.379304},	Vector3{ -0.12576, 0.901885, -0.276456},	Vector3{ 0.427666, 0.708736, -0.158823},	Vector3{ -0.108179, -0.417472, -0.0443039}
    }, {
	Vector3{ 0.470802, -0.878987, -0.0756743},	Vector3{ 0.949295, 0.188295, -0.251763},	Vector3{ -0.116417, -0.986796, -0.112612},	Vector3{ -0.956961, -0.23568, 0.169354},	Vector3{ 0.089121, 0.0219348, -0.995779},	Vector3{ -0.835313, -0.186112, 0.517314},	Vector3{ 0.894238, 0.408996, -0.181824},	Vector3{ 0.560654, -0.17881, -0.808513},	Vector3{ 0.0634305, -0.313669, 0.947411},	Vector3{ -0.239355, -0.0988842, 0.965884},	Vector3{ 0.706132, 0.0534297, 0.706061},	Vector3{ 0.889134, -0.411228, 0.20083},	Vector3{ -0.17479, -0.36364, -0.914994},	Vector3{ 0.00646096, 0.987032, 0.160392},	Vector3{ 0.186361, 0.0236685, 0.982196},	Vector3{ 0.88744, -0.10354, 0.449143},	Vector3{ -0.885827, -0.0397447, -0.462311},	Vector3{ -0.269237, -0.0993826, 0.957933},	Vector3{ 0.223599, -0.0735065, -0.971906},	Vector3{ 0.967596, 0.225971, 0.112674},	Vector3{ -0.54388, 0.0770996, 0.835614},	Vector3{ 0.0446201, -0.987525, -0.151007},	Vector3{ 0.854165, -0.225179, -0.468718},	Vector3{ -0.127811, -0.113037, -0.985336},	Vector3{ 0.0413367, -0.916754, 0.397307},	Vector3{ 0.152579, 0.98724, 0.0455809},Vector3{ 0.193733, 0.496452, 0.84617},	Vector3{ -0.97239, 0.229544, 0.0420264},	Vector3{ 0.636796, -0.769403, 0.0501069},	Vector3{ -0.978724, 0.199967, 0.0459705},	Vector3{ 0.780339, 0.621219, 0.0718232},	Vector3{ 0.463881, 0.137221, -0.875205},	Vector3{ -0.00996557, -0.999506, 0.02979},	Vector3{ -0.564124, 0.802431, 0.194599},	Vector3{ -0.9715, -0.212012, 0.106015},	Vector3{ -0.231085, -0.332122, -0.914492},	Vector3{ 0.489272, -0.0489389, 0.870757},	Vector3{ -0.916242, 0.391323, -0.08583},	Vector3{ 0.372297, -0.00644172, 0.928091},	Vector3{ -0.434264, 0.541757, -0.719663},	Vector3{ -0.940724, -0.212361, -0.264463},	Vector3{ -0.918881, -0.358226, 0.165321},	Vector3{ -0.954172, -0.178993, -0.239827},	Vector3{ 0.167732, 0.531093, 0.830545},	Vector3{ -0.334805, 0.533731, 0.776554},	Vector3{ -0.670961, 0.736813, -0.0831823},	Vector3{ 0.365724, 0.901564, 0.231144},Vector3{ -0.0349137, 0.993728, 0.106238},	Vector3{ -0.928397, 0.351126, 0.121613},	Vector3{ -0.86271, 0.171405, 0.475764},	Vector3{ 0.31046, -0.94923, 0.0507637},	Vector3{ -0.674361, 0.728795, 0.118721},	Vector3{ 0.057859, -0.276969, 0.959135},	Vector3{ -0.998307, 0.0577521, 0.0069451},	Vector3{ -0.955731, -0.200488, 0.215367},	Vector3{ 0.0401766, -0.276091, 0.960291},	Vector3{ 0.997876, 0.0407043, 0.0508664},	Vector3{ -0.961326, -0.266501, 0.0694912},	Vector3{ 0.958434, 0.0680224, 0.277088},	Vector3{ -0.521292, -0.799682, -0.297932},	Vector3{ 0.997475, -0.066242, 0.0255876},	Vector3{ 0.993232, -0.0276986, 0.112795},	Vector3{ -0.262368, -0.770587, 0.580825},	Vector3{ 0.998247, -0.0269705, 0.0526758},	Vector3{ 0.658245, 0.457662, -0.597712},	Vector3{ 0.97793, 0.109962, -0.177656},	Vector3{ 0.959576, -0.16549, 0.227657},	Vector3{ 0.777325, 0.552755, 0.30038},Vector3{ 0.151759, -0.438399, -0.885875},	Vector3{ 0.192548, 0.398646, -0.896664},	Vector3{ -0.91846, 0.312854, 0.24198},	Vector3{ 0.305609, -0.933474, -0.187694},	Vector3{ -0.228292, -0.833116, -0.503786},	Vector3{ -0.168643, -0.400281, -0.900741},	Vector3{ -0.744789, 0.354368, 0.565432},	Vector3{ -0.61835, -0.728608, 0.294573},	Vector3{ 0.177088, -0.410259, -0.89461},	Vector3{ -0.937711, 0.151554, -0.312618},	Vector3{ -0.988376, -0.0749892, 0.13225},	Vector3{ -0.00371562, 0.567442, -0.823405},	Vector3{ 0.419944, -0.184713, 0.888554},	Vector3{ -0.74935, 0.512485, -0.419325},	Vector3{ -0.00636286, -0.92509, 0.379694},	Vector3{ 0.424936, -0.850699, 0.30942},	Vector3{ -0.615926, 0.39455, -0.681884},	Vector3{ 0.0538194, 0.697757, -0.71431},	Vector3{ -0.868991, 0.296983, -0.395797},	Vector3{ 0.992768, -0.108284, -0.0518313},	Vector3{ -0.849131, 0.480322, 0.2197},	Vector3{ 0.794698, 0.527072, -0.301082},	Vector3{ -0.299771, -0.936505, -0.181924},	Vector3{ 0.607995, 0.784541, 0.12181},	Vector3{ 0.0478653, 0.330065, 0.942744},	Vector3{ 0.614006, -0.778682, 0.129038},	Vector3{ 0.97239, 0.229544, 0.0420264},	Vector3{ 0.643057, 0.232252, -0.729751},	Vector3{ -0.93381, -0.296121, 0.200777},	Vector3{ 0.796891, 0.503806, -0.333382},	Vector3{ 0.15329, 0.562936, 0.812161},	Vector3{ 0.892825, -0.140929, -0.427788}
    }, 1}
    // clang-format on
};

void construct_fancy_dipoles() {}

std::vector<std::string> camera_positions = {
    // default
    "{\"farClipRatio\":20.0,\"fov\":45.0,\"nearClipRatio\":0.005,"
    "\"projectionMode\":\"Perspective\",\"viewMat\":[1.0,0.0,0.0,0.0,0.0,0."
    "945845246315002,-0.324617743492126,-2.98023223876953e-08,0.0,0."
    "324617743492126,0.945845246315002,-9.21889400482178,0.0,0.0,0.0,1.0],"
    "\"windowHeight\":1015,\"windowWidth\":1457}",
    // spot
    "{\"farClipRatio\":20.0,\"fov\":45.0,\"nearClipRatio\":0.005,"
    "\"projectionMode\":\"Perspective\",\"viewMat\":[-0.765372693538666,2."
    "09547579288483e-09,0.643587410449982,-0.198119655251503,0.263162732124329,"
    "0.912580132484436,0.312960684299469,-0.058085098862648,-0.587324678897858,"
    "0.408900290727615,-0.698463261127472,-2.85905742645264,0.0,0.0,0.0,1.0],"
    "\"windowHeight\":1015,\"windowWidth\":1457}",
    // done
};

float tmin = 0, tmax = 25, epsilon = .075;
int max_iterations        = 5000;
int resolution_x          = 80;
int resolution_y          = 80;
bool use_grad_termination = false;
bool use_overstepping     = false;
bool use_extrapolation    = false;
bool use_newton           = false;
bool fixed_step_count     = false;
float domain_shrink       = 0.3;

polyscope::PointCloud* psCloud;

//====== Helpers
std::string tracing_mode;

double radians(double degrees) { return degrees / 180. * M_PI; }

std::ostream& operator<<(std::ostream& out, const glm::vec3& vec) {
    out << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return out;
}

// uniformly random point inside of ball
// https://stats.stackexchange.com/a/30622
Vector3 randInBall(const Vector3& center, double R) {
    double phi       = randomReal(0, 2 * M_PI);
    double r         = pow(unitRand(), 1. / 3.);
    double cos_theta = randomReal(-1, 1);
    double sin_theta = sqrt(1 - pow(cos_theta, 2));
    double x         = r * sin_theta * cos(phi);
    double y         = r * sin_theta * sin(phi);
    double z         = r * cos_theta;
    return center + R * Vector3{x, y, z};
}

//====== Experiment code
double closestPoint(Vector3 pt, int& iP) {
    double minD2 = 1e10;
    iP           = -1;
    for (int i = 0; i < pts.size(); i++) {
        double d2 = (pt - pts[i]).norm2();
        if (d2 < minD2) iP = i;
        minD2 = fmin(minD2, d2);
    }
    return sqrt(minD2);
}

// find the potential for a dipole at position x with direction n evaluated at
// point p
double dipolePotential(Vector3 x, Vector3 n, Vector3 p) {
    Vector3 q = x - p;
    return dot(q, n) / pow(dot(q, q), 1.5);
}

Vector3 dipoleGradient(Vector3 x, Vector3 n, Vector3 p) {
    Vector3 q = p - x;
    double q2 = dot(q, q);
    return 3. * dot(q, n) * q / pow(q2, 2.5) - n / pow(q2, 1.5);
}

double totalPotential(Vector3 p) {
    double potential = 0.;
    for (int i = 0; i < pts.size(); i++) {
        potential += dipolePotential(pts[i], normals[i], p);
    }

    return potential;
}

Vector3 gradient(Vector3 p) {
    Vector3 grad = Vector3{0, 0, 0};
    for (int i = 0; i < pts.size(); i++) {
        grad += dipoleGradient(pts[i], normals[i], p);
    }

    return grad;
}


// takes a plane with origin x and normal n,
// and a sphere at point p with radius r,
// and returns the minimum of <p - x, n> over all points p on the sphere
double sphereDistToPlane(Vector3 x, Vector3 n, Vector3 p, double r) {
    return dot(x - p, n) - r * n.norm();
}

// takes a point x and a sphere centered at point p with radius r,
// and returns the mdistance from the sphere to x
double sphereMinDistToPoint(Vector3 x, Vector3 p, double r) {
    return (x - p).norm() - r;
}

// takes a point x and a sphere centered at point p with radius r,
// and returns the max distance from the sphere to x
double sphereMaxDistToPoint(Vector3 x, Vector3 p, double r) {
    return (x - p).norm() + r;
}

// lower bound the potential of a dipole at position x with direction n,
// over all points on the sphere centered at point p with radius r
double dipoleLowerBound(Vector3 x, Vector3 n, Vector3 p, double r) {
    double d_plane = sphereDistToPlane(x, n, p, r);
    double r_bound = d_plane < 0. ? sphereMinDistToPoint(x, p, r)
                                  : sphereMaxDistToPoint(x, p, r);
    return d_plane / pow(r_bound, 3.);
}

double totalBound(Vector3 p, double r) {
    double bound = 0.;
    for (int i = 0; i < pts.size(); i++) {
        bound += dipoleLowerBound(pts[i], normals[i], p, r);
    }

    return bound;
}

// For harnack's checks if we're close to our desired level set
bool closeToLevelset(double val, double levelset, double tol, double gradNorm) {
    double eps = use_grad_termination ? tol * gradNorm : tol;
    return abs(val - levelset) < eps;
}

double getMaxStep(double fx, double levelset, double R, double shift) {
    double a = (fx + shift) / (levelset + shift);
    return R * abs(a + 2. - sqrt(a * a + 8. * a));
}

bool intersect_harnack(const Vector3& ro, const Vector3& rd, double& t,
                       double& val, bool careful = false) {
    int iter = 0;
    t        = 0.;

    while (t < tmax) {
        Vector3 pos = ro + t * rd;
        if (iter >= max_iterations) {
            return false;
        }

        val = totalPotential(pos);

        int iClosest;
        double R = closestPoint(pos, iClosest);

        if (closeToLevelset(val, levelset, epsilon, 1.)) {
            return true;
        } else if (R < epsilon / 10. && iClosest >= 0) {
            // Vector3 n_closest = normals[iClosest];
            // n                 = -normalize(n_closest);
            return true;
        }

        double offset = R * domain_shrink;
        R -= offset;

        double bound = totalBound(pos, R);
        if (careful) {
            for (size_t iCheck = 0; iCheck < 100; iCheck++) {
                Vector3 p_check = randInBall(pos, R);
                double val_p    = totalPotential(p_check);

                if (val_p < bound) {
                    std::cout
                        << "Error: function at point " << p_check
                        << " takes value " << val_p
                        << " which is lower than the purported lower bound of "
                        << bound << vendl;

                    for (size_t iP = 0; iP < pts.size(); iP++) {
                        double dipole_i_p =
                            dipolePotential(pts[iP], normals[iP], p_check);
                        double bound_i_p =
                            dipoleLowerBound(pts[iP], normals[iP], pos, R);

                        if (dipole_i_p < bound_i_p) {
                            std::cout << "      in particular, dipole " << iP
                                      << " takes value " << dipole_i_p
                                      << " when its bound is " << bound_i_p
                                      << vendl;

                            Vector3 x = pts[iP];
                            Vector3 n = normals[iP];
                            Vector3 p = pos;

                            double d_plane_bound =
                                sphereDistToPlane(x, n, p, R);
                            double r_bound =
                                d_plane_bound < 0.
                                    ? sphereMinDistToPoint(x, p, R)
                                    : sphereMaxDistToPoint(x, p, R);

                            double d_plane_true = dot(x - p_check, n);
                            double r_true       = (x - p_check).norm();
                            double dipole_true = d_plane_true / pow(r_true, 3.);

                            std::cout
                                << "     d_plane_bound = " << d_plane_bound
                                << ", d_plane_true = " << d_plane_true
                                << "    | r_bound = " << r_bound
                                << ", r_true = " << r_true
                                << "    | dipole_true = " << dipole_true
                                << vendl;

                            // Vector3 q = x - p;
                            // return dot(q, n) / pow(dot(q, q), 1.5);
                        }
                    }
                }
            }
        }

        double shift = -bound;
        double r     = getMaxStep(val, levelset, R, shift);
        t += r;
        iter++;
    }

    return false;
}

double clamp(double x, double x_min, double x_max) {
    return fmin(fmax(x, x_min), x_max);
}

bool intersect_newton(const Vector3& ro, const Vector3& rd, double& t,
                      double& val, bool careful = false) {
    t = 0.;

    Vector3 pos  = ro + t * rd;
    Vector3 grad = gradient(pos);

    for (size_t iter = 0; iter < 10 && grad.norm2() < 1e-6; iter++) {
        t += .5;
        Vector3 pos  = ro + t * rd;
        Vector3 grad = gradient(pos);

        if (careful) {
            std::cout << "   grad: " << grad << std::endl;
            auto pr = std::setprecision(4);
            std::cout << std::setfill(' ') << std::setw(3) << iter
                      << "| t = " << std::setw(8) << std::fixed << pr << t;
            std::cout << " grad_f = " << grad << std::endl;
        }
    }


    if (careful) {

        std::ofstream o("dipole_potential_along_ray.csv");
        if (!o) throw std::runtime_error("couldn't open output file ");
        o << "t,val" << std::endl;
        for (double s = 0; s < tmax; s += 0.05) {
            Vector3 pos = ro + s * rd;
            double val  = totalPotential(pos);
            o << s << "," << val << std::endl;
        }
    }

    double trust_radius       = 1.;
    double accuracy_threshold = 0.25;

    for (int iN = 0; iN < 16; iN++) {
        pos = ro + t * rd;

        val  = totalPotential(pos);
        grad = gradient(pos);

        double df    = dot(rd, grad);
        double f_err = val - levelset;

        double dt = -f_err / df;
        dt        = clamp(dt, -trust_radius, trust_radius);

        double accuracy =
            (totalPotential(ro + (t + dt) * rd) - val) / (dt * df);

        if (accuracy > .75 && abs(abs(dt) - trust_radius) < 0.01) {
            trust_radius = fmin(2. * trust_radius, 4.);
        } else if (accuracy < accuracy_threshold) {
            trust_radius /= 4.;
        }

        if (careful) {
            // std::cout << "   grad: " << grad << std::endl;
            auto pr = std::setprecision(4);
            std::cout << std::setfill(' ') << std::setw(3) << iN
                      << "| t = " << std::setw(8) << std::fixed << pr << t
                      << "  f = " << std::setw(8) << std::fixed << pr << val
                      << " dt = " << std::setw(8) << std::fixed << pr << dt
                      << " ferr = " << std::setw(8) << std::fixed << pr << f_err
                      << " df = " << std::setw(8) << std::fixed << pr << df
                      << " pos = " << pos << " grad_f = " << grad << std::endl;
        }

        if (abs(f_err) < epsilon) {
            return true;
        }

        if (accuracy > accuracy_threshold) {
            t += dt;
        }
    }
    return false;
}

enum class TracingMethod { Harnack, Raymarch, Newton };
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

    std::vector<Vector3> intersections, normals, viewRayPts, coordinates;
    std::vector<std::array<size_t, 2>> viewRayLines;
    std::vector<float> omegas;
    std::vector<int> didHit;
    size_t successful_extrapolations = 0;
    float s                          = 0.05;
    intersections.reserve(resolution_x * resolution_y);
    double start       = std::clock();
    float aspect_ratio = (float)resolution_x / (float)resolution_y;
    tracing_mode = (method == TracingMethod::Harnack) ? "harnack" : "sphere";
    // int xBad = 52, yBad = 76;
    int xBad = 32, yBad = 52; // debug newton's method on icosahedron
    for (int iY = 0; iY < resolution_y; iY++) {
        for (int iX = 0; iX < resolution_x; iX++) {
            coordinates.push_back(
                {static_cast<double>(iX), static_cast<double>(iY), 0});

            glm::vec2 cCoord = (resolution_x * resolution_y <= 1)
                                   ? glm::vec2{0, 0}
                                   : glm::vec2{iX / (float)(resolution_x - 1),
                                               iY / (float)(resolution_y - 1)} *
                                             (float)2 -
                                         (float)1;
            cCoord.x *= aspect_ratio;

            // create view ray
            Vector3 rd = to_gc(normalize(
                cCoord.x * rightDir + cCoord.y * upDir + (float)3 * lookDir));
            Vector3 ro = to_gc(camPos);

            viewRayLines.push_back(
                {static_cast<size_t>(resolution_x * resolution_y),
                 viewRayPts.size()});
            viewRayPts.push_back(ro + tmax * rd);

            double t, omega, iter_frac;
            bool careful = iX == xBad && iY == yBad;

            bool hit;
            switch (method) {
            case TracingMethod::Harnack:
                hit = intersect_harnack(ro, rd, t, omega, careful);
                break;
            case TracingMethod::Raymarch:
                throw std::runtime_error("raymarching not implemented yet");
                // hit = intersect_raymarch(ro, rd, t, omega, careful);
                break;
            case TracingMethod::Newton:
                hit = intersect_newton(ro, rd, t, omega, careful);
                break;
            }

            if (hit) {
                Vector3 intersection = ro + t * rd;
                intersections.push_back(intersection);

                omegas.push_back(omega);
                normals.push_back(
                    gradient(intersections.back())); // todo: normalize
            }
            if (careful) {
                didHit.push_back(hit * 3 + 3);
            } else {
                didHit.push_back(hit);
            }
        }
    }
    double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    double meanTime = duration / viewRayPts.size();

    psCloud = polyscope::registerPointCloud("intersections", intersections);
    psCloud->addScalarQuantity("omega", omegas);
    psCloud->addVectorQuantity("normal", normals);
    psCloud->setEnabled(false);

    auto viewPts = polyscope::registerPointCloud("view ray points", viewRayPts);
    viewPts->addScalarQuantity("did hit", didHit)->setEnabled(true);
    viewPts->setPointRenderMode(polyscope::PointRenderMode::Quad);
    viewPts->setPointRadius(.003);
    viewPts->addVectorQuantity("coordinates", coordinates);
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
    std::cout << "const Vector3 cam_pos   = Vector3" << camPos << ";"
              << std::endl;
    std::cout << "const Vector3 look_dir  = Vector3" << lookDir << ";"
              << std::endl;
    std::cout << "const Vector3 up_dir    = Vector3" << upDir << ";"
              << std::endl;
    std::cout << "const Vector3 right_dir = Vector3" << rightDir << ";"
              << std::endl;
    std::cout
        << "const mat3 default_cam_mat = mat3(right_dir, up_dir, look_dir);"
        << std::endl;
    std::cout << "const Vector3 cam_space_cam_pos   = Vector3" << camSpaceCamPos
              << ";" << std::endl;
    std::cout << "const float fovY = " << fovY << ".;" << std::endl;
}

void export_dipoles_to_shadertoy() {
    polyscope::CameraParameters prev_params =
        polyscope::view::getCameraParametersForCurrentView();

    std::cout << "//====== named dipoles" << std::endl;
    for (size_t iP = 0; iP < named_dipoles.size(); iP++) {
        std::cout << "//  P" << iP << " : " << named_dipoles[iP].name
                  << std::endl;
    }
    std::cout << "#define P0" << std::endl << std::endl;

    std::cout << "//====== dipole data" << std::endl;

    polyscope::view::setViewToCamera(prev_params);
}

polyscope::PointCloud* drawDipoles(const std::vector<Vector3>& pts,
                                   const std::vector<Vector3>& normals) {
    auto psCloud = polyscope::registerPointCloud("dipoles", pts);
    psCloud->addVectorQuantity("N", normals);
    return psCloud;
}

void display_dipole(size_t iP) {
    polyscope::view::setCameraFromJson(
        camera_positions[named_dipoles[iP].cam_view], false);
    pts         = named_dipoles[iP].pts;
    normals     = named_dipoles[iP].normals;
    auto dipole = drawDipoles(pts, normals);
    // dipole->setColor(glm::vec3(0));
    // set soft shadows on the ground
    // polyscope::options::groundPlaneMode =
    //     polyscope::GroundPlaneMode::ShadowOnly;
    // polyscope::options::groundPlaneHeightFactor = 0.00; // adjust the
    // // plane height
    // polyscope::options::shadowDarkness = 0.15; // lighter shadows
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {
    static std::vector<const char*> tracing_mode_names{
        "Harnack Tracing", "Ray marching", "Newton's Method"};
    static int tracing_mode = 2;
    ImGui::Combo("Intersection Mode", &tracing_mode, tracing_mode_names.data(),
                 tracing_mode_names.size());
    if (ImGui::Button("Shoot Camera Rays")) {
        std::string name =
            std::string(use_grad_termination ? "grad-terminated " : "") +
            std::string(use_overstepping ? "overstepped " : "") +
            std::string(use_extrapolation ? "extrapolated " : "") +
            std::string(use_newton ? "newton-accelerated " : "");
        if (name.length() == 0) name = "default ";

        switch (tracing_mode) {
        case 0:
            name += "Harnack tracing";
            shootCameraRays(name, TracingMethod::Harnack);
            break;
        case 1:
            name += "ray marching";
            shootCameraRays(name, TracingMethod::Raymarch);
            break;
        case 2:
            name += "Newton's method";
            shootCameraRays(name, TracingMethod::Newton);
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

    ImGui::Separator();
    ImGui::SliderFloat("epsilon (log)", &epsilon, .001f, .1f, "%.4f",
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
    ImGui::Separator();
    static std::vector<const char*> dipole_names;
    if (dipole_names.empty())
        for (auto& poly : named_dipoles)
            dipole_names.push_back(poly.name.c_str());
    static int selected_dipole = 0;
    if (ImGui::Combo("Select Dipole", &selected_dipole, dipole_names.data(),
                     dipole_names.size())) {
        display_dipole(selected_dipole);
    }

    if (ImGui::Button("Export Dipoles to ShaderToy")) {
        export_dipoles_to_shadertoy();
    }
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

    construct_fancy_dipoles();

    // Initialize polyscope
    polyscope::init();

    // Set the callback function
    polyscope::state::userCallback = myCallback;

    polyscope::registerCurveNetworkLoop("polygon", pts);

    // store initial camera position
    camParams = polyscope::view::getCameraParametersForCurrentView();
    // shootCameraRays();

    display_dipole(0);

    // Give control to the polyscope gui
    polyscope::show();

    return EXIT_SUCCESS;
}

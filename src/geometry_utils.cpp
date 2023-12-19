#include "geometry_utils.h"

void compute_vertex_face_lists(
    const std::vector<std::array<glm::vec3, 3>>& tri_list,
    std::vector<geometrycentral::Vector3>& vertex_coordinates,
    std::vector<std::vector<size_t>>& faces, double epsilon) {

    using namespace geometrycentral;

    auto to_gc = [](const glm::vec3& v) -> Vector3 {
        return Vector3{v.x, v.y, v.z};
    };

    for (const std::array<glm::vec3, 3>& tri : tri_list) {
        size_t iA  = vertex_coordinates.size();
        Vector3 pA = to_gc(tri[0]);
        for (size_t iV = 0; iV < vertex_coordinates.size(); iV++) {
            if ((pA - vertex_coordinates[iV]).norm2() < 1e-6) {
                iA = iV;
                break;
            }
        }
        if (iA >= vertex_coordinates.size()) vertex_coordinates.push_back(pA);

        size_t iB  = vertex_coordinates.size();
        Vector3 pB = to_gc(tri[1]);
        for (size_t iV = 0; iV < vertex_coordinates.size(); iV++) {
            if ((pB - vertex_coordinates[iV]).norm2() < 1e-6) {
                iB = iV;
                break;
            }
        }
        if (iB >= vertex_coordinates.size()) vertex_coordinates.push_back(pB);

        size_t iC  = vertex_coordinates.size();
        Vector3 pC = to_gc(tri[2]);
        for (size_t iV = 0; iV < vertex_coordinates.size(); iV++) {
            if ((pC - vertex_coordinates[iV]).norm2() < 1e-6) {
                iC = iV;
                break;
            }
        }
        if (iC >= vertex_coordinates.size()) vertex_coordinates.push_back(pC);

        faces.push_back(std::vector<size_t>{iA, iB, iC});
    }
}

std::vector<geometrycentral::Vector3>
generate_point_faces(const std::vector<geometrycentral::Vector3>& points,
                     const std::vector<geometrycentral::Vector3>& normals,
                     double epsilon) {
    using geometrycentral::Vector2;
    using geometrycentral::Vector3;
    Vector2 v1 = epsilon * Vector2{0, 1};
    Vector2 v2 = epsilon * Vector2{-sin(2. * M_PI / 3), cos(2. * M_PI / 3)};
    Vector2 v3 = epsilon * Vector2{-sin(4. * M_PI / 3), cos(4. * M_PI / 3)};

    std::vector<Vector3> faceVerts;
    for (size_t iP = 0; iP < points.size(); iP++) {
        Vector3 x, y;
        branchlessONB(normals[iP], x, y);
        faceVerts.push_back(points[iP] + v1.x * x + v1.y * y);
        faceVerts.push_back(points[iP] + v2.x * x + v2.y * y);
        faceVerts.push_back(points[iP] + v3.x * x + v3.y * y);
    }

    return faceVerts;
}

void branchlessONB(const geometrycentral::Vector3& n,
                   geometrycentral::Vector3& b1, geometrycentral::Vector3& b2) {
    using geometrycentral::Vector3;
    double sign    = copysign(1.0f, n.z);
    const double a = -1.0f / (sign + n.z);
    const double b = n.x * n.y * a;
    b1 = Vector3{1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x};
    b2 = Vector3{b, sign + n.y * n.y * a, -n.y};
}

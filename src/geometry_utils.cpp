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

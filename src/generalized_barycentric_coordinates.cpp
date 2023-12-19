#include "generalized_barycentric_coordinates.h"

#include "utils.h" // verbose_assert

#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/subdivide.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/surface_mesh.h"

double cross(const glm::vec2& a, const glm::vec2& b) {
    return a.x * b.y - a.y * b.x;
}

double angle(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
    glm::vec2 ba = a - b;
    glm::vec2 bc = c - b;
    return atan2(cross(bc, ba), glm::dot(bc, ba));
}

double cotangent(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
    glm::vec2 ba = a - b;
    glm::vec2 bc = c - b;
    return glm::dot(bc, ba) / cross(bc, ba);
}

// compute generalized barycentric coordinates for point v0 inside of the
// polygon v using the method of Wachpress [1975] "A Rational Finite Element
// Basis", as formulated by Meyer et al [2001] in "Generalized Barycentric
// Coordinates on Irregular Polygons"
std::vector<double> wachpressCoords(glm::vec2 v0,
                                    const std::vector<glm::vec2>& v) {
    std::vector<double> weights(v.size());

    double weight_sum = 0;
    for (size_t j = 0; j < v.size(); j++) {
        const glm::vec2& q_prev = v[(j + v.size() - 1) % v.size()];
        const glm::vec2& q_next = v[(j + 1) % v.size()];
        const glm::vec2& q_j    = v[j];

        // check if v0 lies on line q_prev -> q_j
        // Note: computation largely redundant with cotangent call below
        if (abs(cross(q_j - q_prev, v0 - q_prev)) <
            0.001 * glm::length(q_j - q_prev)) {
            // if on line, just find linear interpolation along line

            glm::vec2 m = q_j - q_prev;
            glm::vec2 x = v0 - q_prev;
            // dot = |a|*|b|cos(theta) * n, isolating |a|sin(theta)
            double t = fmin(fmax(glm::dot(m, x) / glm::dot(m, m), 0.), 1.);
            weights  = std::vector<double>(v.size(), 0);
            weights[(j + v.size() - 1) % v.size()] = 1 - t;
            weights[j]                             = t;
            return weights;
        }

        weights[j] = (cotangent(q_prev, q_j, v0) + cotangent(v0, q_j, q_next)) /
                     glm::length2(v0 - q_j);
        weight_sum += weights[j];
    }

    // normalize weights to sum to 1
    for (size_t j = 0; j < weights.size(); j++) weights[j] /= weight_sum;

    return weights;
}

std::vector<std::array<glm::vec3, 3>>
wachpressInterpolate(const std::vector<glm::vec3>& pts) {
    return barycentricInterpolate(pts, wachpressCoords);
}

// compute generalized barycentric coordinates for point v0 inside of the
// polygon v using the method of Floater [2003] "Mean value coordinates"
std::vector<double> meanValueCoords(glm::vec2 v0,
                                    const std::vector<glm::vec2>& v) {
    std::vector<double> weights(v.size());

    double weight_sum = 0;
    for (size_t j = 0; j < v.size(); j++) {
        const glm::vec2& q_prev = v[(j + v.size() - 1) % v.size()];
        const glm::vec2& q_next = v[(j + 1) % v.size()];
        const glm::vec2& q_j    = v[j];

        // check if v0 lies on line q_prev -> q_j
        // Note: computation largely redundant with angle call below
        if (abs(cross(q_j - q_prev, v0 - q_prev)) <
            0.001 * glm::length(q_j - q_prev)) {
            // if on line, just find linear interpolation along line

            glm::vec2 m = q_j - q_prev;
            glm::vec2 x = v0 - q_prev;
            // dot = |a|*|b|cos(theta) * n, isolating |a|sin(theta)
            double t = fmin(fmax(glm::dot(m, x) / glm::dot(m, m), 0.), 1.);
            weights  = std::vector<double>(v.size(), 0);
            weights[(j + v.size() - 1) % v.size()] = 1 - t;
            weights[j]                             = t;
            return weights;
        }

        double alpha_prev = angle(q_prev, v0, q_j);
        double alpha_next = angle(q_j, v0, q_next);

        // you could probably replace the trig with judicious use of dot and
        // cross products
        weights[j] = (tan(alpha_prev / 2.) + tan(alpha_next / 2.)) /
                     glm::length(v0 - q_j);
        weight_sum += weights[j];
    }

    // normalize weights to sum to 1
    for (size_t j = 0; j < weights.size(); j++) weights[j] /= weight_sum;

    return weights;
}

std::vector<std::array<glm::vec3, 3>>
meanValueInterpolate(const std::vector<glm::vec3>& pts) {
    return barycentricInterpolate(pts, meanValueCoords);
}

std::vector<std::array<glm::vec3, 3>> barycentricInterpolate(
    const std::vector<glm::vec3>& pts,
    const std::function<std::vector<double>(glm::vec2,
                                            const std::vector<glm::vec2>&)>& f,
    uint subdivisions) {

    size_t N = pts.size();

    std::vector<glm::vec2> planar_pts;
    for (size_t iP = 0; iP < N; iP++) {
        double theta =
            2. * M_PI * static_cast<double>(iP) / static_cast<double>(N);
        planar_pts.push_back(glm::vec2(cos(theta), sin(theta)));
    }

    std::vector<std::array<glm::vec2, 3>> planar_triangles =
        triangulate_polygon(planar_pts, subdivisions);

    auto map = [&](const glm::vec2& p) -> glm::vec3 {
        std::vector<double> weights = f(p, planar_pts);
        glm::vec3 result{0, 0, 0};
        for (size_t iV = 0; iV < pts.size(); iV++)
            result += static_cast<float>(weights[iV]) * pts[iV];
        return result;
    };

    std::vector<std::array<glm::vec3, 3>> spatial_triangles;
    for (const std::array<glm::vec2, 3>& tri : planar_triangles) {
        spatial_triangles.push_back(
            std::array<glm::vec3, 3>{map(tri[0]), map(tri[1]), map(tri[2])});
    }

    return spatial_triangles;
}

std::vector<std::array<glm::vec3, 3>>
bilinearInterpolate(const std::vector<glm::vec3>& pts) {
    verbose_assert(pts.size() == 4,
                   "bilinear interpolation requires four input points, but " +
                       std::to_string(pts.size()) + " points were provided");

    uint subdivisions = 64;

    const glm::vec3& a = pts[0];
    const glm::vec3& b = pts[1];
    const glm::vec3& c = pts[2];
    const glm::vec3& d = pts[3];

    auto interp = [&](float s, float t) -> glm::vec3 {
        return (1 - t) * ((1 - s) * pts[0] + s * pts[1]) +
               t * ((1 - s) * pts[3] + s * pts[2]);
    };

    std::vector<std::array<glm::vec3, 3>> triangles;
    for (uint iR = 0; iR < subdivisions + 1; iR++) { // rows
        float t0 = ((float)(iR + 0)) / ((float)(subdivisions + 1));
        float t1 = ((float)(iR + 1)) / ((float)(subdivisions + 1));
        for (uint iC = 0; iC < subdivisions + 1; iC++) { // columns
            float s0 = ((float)(iC + 0)) / ((float)(subdivisions + 1));
            float s1 = ((float)(iC + 1)) / ((float)(subdivisions + 1));

            triangles.push_back(std::array<glm::vec3, 3>{
                interp(s0, t0), interp(s1, t0), interp(s1, t1)});
            triangles.push_back(std::array<glm::vec3, 3>{
                interp(s0, t0), interp(s1, t1), interp(s0, t1)});
        }
    }
    return triangles;
}

std::vector<std::array<glm::vec3, 3>>
astridInterpolate(const std::vector<glm::vec3>& pts) {
    glm::vec3 center = computeVirtualVertex(pts);
    std::vector<std::array<glm::vec3, 3>> triangles;
    for (size_t iE = 0; iE < pts.size(); iE++) {
        triangles.push_back(std::array<glm::vec3, 3>{
            pts[iE], pts[(iE + 1) % pts.size()], center});
    }
    return triangles;
}

void flowToMinimalSurface(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geom, double* area) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    VertexData<bool> is_interior(mesh, true);
    for (BoundaryLoop b : mesh.boundaryLoops()) {
        for (Vertex v : b.adjacentVertices()) is_interior[v] = false;
    }

    geom.requireCotanLaplacian();
    geom.requireVertexIndices();

    auto read_positions = [&](Vector<double>& x, Vector<double>& y,
                              Vector<double>& z) {
        const VertexData<size_t>& iV = geom.vertexIndices;
        for (Vertex v : mesh.vertices()) {
            x(iV[v]) = geom.vertexPositions[v].x;
            y(iV[v]) = geom.vertexPositions[v].y;
            z(iV[v]) = geom.vertexPositions[v].z;
        }
    };

    auto set_positions = [&](const Vector<double>& x, const Vector<double>& y,
                             const Vector<double>& z) {
        const VertexData<size_t>& iV = geom.vertexIndices;
        for (Vertex v : mesh.vertices()) {
            geom.vertexPositions[v].x = x(iV[v]);
            geom.vertexPositions[v].y = y(iV[v]);
            geom.vertexPositions[v].z = z(iV[v]);
        }
    };

    size_t N = mesh.nVertices();
    Vector<double> x(N), y(N), z(N), x_int, x_bdy, y_int, y_bdy, z_int, z_bdy;

    double change = 1;
    size_t iter   = 0;
    while (change > 1e-3 && iter < 25) {
        SparseMatrix<double> L = geom.cotanLaplacian;
        BlockDecompositionResult<double> decomp =
            blockDecomposeSquare(L, is_interior.raw());

        read_positions(x, y, z);
        decomposeVector(decomp, x, x_int, x_bdy);
        decomposeVector(decomp, y, y_int, y_bdy);
        decomposeVector(decomp, z, z_int, z_bdy);

        PositiveDefiniteSolver<double> solver(decomp.AA);
        x_int = solver.solve(-decomp.AB * x_bdy);
        y_int = solver.solve(-decomp.AB * y_bdy);
        z_int = solver.solve(-decomp.AB * z_bdy);

        Vector<double> new_x, new_y, new_z;
        new_x = reassembleVector(decomp, x_int, x_bdy);
        new_y = reassembleVector(decomp, y_int, y_bdy);
        new_z = reassembleVector(decomp, z_int, z_bdy);

        change = (new_x - x).squaredNorm() + (new_y - y).squaredNorm() +
                 (new_z - z).squaredNorm();

        set_positions(new_x, new_y, new_z);

        x = new_x;
        y = new_y;
        z = new_z;
        geom.refreshQuantities();

        iter++;
    }

    geom.unrequireVertexIndices();
    geom.unrequireCotanLaplacian();

    if (area) {
        *area = 0;
        geom.requireFaceAreas();
        for (Face f : mesh.faces()) *area += geom.faceAreas[f];
        geom.unrequireFaceAreas();
    }
}

std::vector<std::array<glm::vec3, 3>>
minimalSurface(const std::vector<glm::vec3>& pts) {
    return minimalSurfaceArea(pts, nullptr);
}

std::vector<std::array<glm::vec3, 3>>
minimalSurfaceArea(const std::vector<glm::vec3>& pts, double* area) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    uint subdivisions = 64;

    glm::vec3 center = computeVirtualVertex(pts);
    center.x += 2.7;
    center.y += 5;
    std::vector<std::array<glm::vec3, 3>> init_positions =
        triangulate_polygon(pts, center, subdivisions);

    SimplePolygonMesh smp;
    compute_vertex_face_lists(init_positions, smp.vertexCoordinates,
                              smp.polygons);

    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geom;
    std::tie(mesh, geom) =
        makeManifoldSurfaceMeshAndGeometry(smp.polygons, smp.vertexCoordinates);

    flowToMinimalSurface(*mesh, *geom, area);

    std::vector<std::array<glm::vec3, 3>> triangles;
    const VertexData<Vector3>& p = geom->vertexPositions;
    for (Face f : mesh->faces()) {
        Vector3 pi = p[f.halfedge().tailVertex()];
        Vector3 pj = p[f.halfedge().tipVertex()];
        Vector3 pk = p[f.halfedge().next().tipVertex()];
        triangles.push_back(std::array<glm::vec3, 3>{
            glm::vec3{pi.x, pi.y, pi.z}, glm::vec3{pj.x, pj.y, pj.z},
            glm::vec3{pk.x, pk.y, pk.z}});
    }
    return triangles;
}

std::vector<std::array<glm::vec3, 3>>
minimalSurfaceArea(const std::vector<glm::vec3>& pts_a,
                   const std::vector<glm::vec3>& pts_b, double* area) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;
    std::vector<Vector3> positions;
    std::vector<std::vector<size_t>> polygons;

    verbose_assert(pts_a.size() == pts_b.size(),
                   "boundary loops in minimal surface must have same number of "
                   "components");

    auto to_gc_vec3 = [](glm::vec3 p) -> Vector3 { return {p.x, p.y, p.z}; };
    size_t N        = pts_a.size();
    for (size_t iP = 0; iP < N; iP++)
        positions.push_back(to_gc_vec3(pts_a[iP]));
    for (size_t iP = 0; iP < N; iP++)
        positions.push_back(to_gc_vec3(pts_b[iP]));

    for (size_t iP = 0; iP < N; iP++) {
        size_t jP = (iP + 1) % N;
        polygons.push_back(std::vector<size_t>{iP, jP, N + jP, N + iP});
    }

    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geom;
    std::tie(mesh, geom) =
        makeManifoldSurfaceMeshAndGeometry(polygons, positions);

    for (size_t iS = 0; iS < 6; iS++) linearSubdivide(*mesh, *geom);
    for (Face f : mesh->faces()) mesh->triangulate(f);

    flowToMinimalSurface(*mesh, *geom, area);

    std::vector<std::array<glm::vec3, 3>> triangles;
    const VertexData<Vector3>& p = geom->vertexPositions;
    for (Face f : mesh->faces()) {
        Vector3 pi = p[f.halfedge().tailVertex()];
        Vector3 pj = p[f.halfedge().tipVertex()];
        Vector3 pk = p[f.halfedge().next().tipVertex()];
        triangles.push_back(std::array<glm::vec3, 3>{
            glm::vec3{pi.x, pi.y, pi.z}, glm::vec3{pj.x, pj.y, pj.z},
            glm::vec3{pk.x, pk.y, pk.z}});
    }
    return triangles;
}

// from
// https://github.com/mbotsch/polygon-laplacian/blob/master/src/PolyDiffGeo.cpp#L496
void compute_virtual_vertex(const Eigen::MatrixXd& poly,
                            Eigen::VectorXd& weights) {
    int val = poly.rows();
    Eigen::MatrixXd J(val, val);
    Eigen::VectorXd b(val);
    weights.resize(val);

    for (int i = 0; i < val; i++) {
        Eigen::Vector3d pk = poly.row(i);

        double Bk1_d2 = 0.0;
        double Bk1_d1 = 0.0;

        double Bk2_d0 = 0.0;
        double Bk2_d2 = 0.0;

        double Bk3_d0 = 0.0;
        double Bk3_d1 = 0.0;

        double CBk        = 0.0;
        Eigen::Vector3d d = Eigen::MatrixXd::Zero(3, 1);

        for (int j = 0; j < val; j++) {
            Eigen::Vector3d pi = poly.row(j);
            Eigen::Vector3d pj = poly.row((j + 1) % val);
            d                  = pi - pj;

            double Bik1 = d(1) * pk(2) - d(2) * pk(1);
            double Bik2 = d(2) * pk(0) - d(0) * pk(2);
            double Bik3 = d(0) * pk(1) - d(1) * pk(0);

            double Ci1 = d(1) * pi(2) - d(2) * pi(1);
            double Ci2 = d(2) * pi(0) - d(0) * pi(2);
            double Ci3 = d(0) * pi(1) - d(1) * pi(0);

            Bk1_d1 += d(1) * Bik1;
            Bk1_d2 += d(2) * Bik1;

            Bk2_d0 += d(0) * Bik2;
            Bk2_d2 += d(2) * Bik2;

            Bk3_d0 += d(0) * Bik3;
            Bk3_d1 += d(1) * Bik3;

            CBk += Ci1 * Bik1 + Ci2 * Bik2 + Ci3 * Bik3;
        }
        for (int k = 0; k < val; k++) {
            Eigen::Vector3d xj = poly.row(k);
            J(i, k) = 0.5 * (xj(2) * Bk1_d1 - xj(1) * Bk1_d2 + xj(0) * Bk2_d2 -
                             xj(2) * Bk2_d0 + xj(1) * Bk3_d0 - xj(0) * Bk3_d1);
        }
        b(i) = 0.5 * CBk;
    }

    Eigen::MatrixXd M(val + 1, val);
    M.block(0, 0, val, val) = 4 * J;
    M.block(val, 0, 1, val).setOnes();

    Eigen::VectorXd b_(val + 1);
    b_.block(0, 0, val, 1) = 4 * b;

    b_(val) = 1.;

    weights = M.completeOrthogonalDecomposition().solve(b_).topRows(val);
}

glm::vec3 computeVirtualVertex(const std::vector<glm::vec3>& pts) {
    Eigen::MatrixXd pt_matrix(pts.size(), 3);
    for (size_t iP = 0; iP < pts.size(); iP++)
        pt_matrix.row(iP) << pts[iP].x, pts[iP].y, pts[iP].z;
    Eigen::VectorXd weights;
    compute_virtual_vertex(pt_matrix, weights);

    Eigen::VectorXd eigen_center = weights.transpose() * pt_matrix;
    return glm::vec3{eigen_center(0), eigen_center(1), eigen_center(2)};
}

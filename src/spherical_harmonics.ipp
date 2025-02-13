// evaluate the homogeneous harmonic polynomial representing
// the (l, m) spherical harmonic at point pos
template <typename T>
T evaluateSphericalHarmonic(int l, int m, const std::array<T, 3>& pos) {
    return static_cast<T>(evaluateSphericalHarmonicDouble(l, m, pos));
}

template <typename T>
double evaluateSphericalHarmonicDouble(int l, int m,
                                       const std::array<T, 3>& pos) {
    double x = static_cast<double>(pos[0]);
    double y = static_cast<double>(pos[1]);
    double z = static_cast<double>(pos[2]);
    if (l == 1 && m == -1) {
        return -0.5 * (std::sqrt(3 / M_PI) * y);
    } else if (l == 1 && m == 0) {
        return std::sqrt(3 / (2. * M_PI)) * z;
    } else if (l == 1 && m == 1) {
        return -0.5 * (std::sqrt(3 / M_PI) * x);
    } else if (l == 2 && m == -2) {
        return (std::sqrt(15 / M_PI) * x * y) / 2.;
    } else if (l == 2 && m == -1) {
        return -0.5 * (std::sqrt(15 / M_PI) * y * z);
    } else if (l == 2 && m == 0) {
        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;
        return (-0.5 * x2 - y2 / 2. + z2) * std::sqrt(5 / (2. * M_PI));
    } else if (l == 2 && m == 1) {
        return -0.5 * (std::sqrt(15 / M_PI) * x * z);
    } else if (l == 2 && m == 2) {
        double x2 = x * x;
        double y2 = y * y;
        return ((x2 - y2) * std::sqrt(15 / M_PI)) / 4.;
    } else if (l == 3 && m == -3) {
        double x2 = x * x;
        double y3 = y * y * y;
        return (std::sqrt(35 / (2. * M_PI)) * (y3 - 3 * x2 * y)) / 4.;
    } else if (l == 3 && m == -2) {
        return (std::sqrt(105 / M_PI) * x * y * z) / 2.;
    } else if (l == 3 && m == -1) {
        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;
        return (std::sqrt(21 / (2. * M_PI)) * (x2 * y + (y2 - 4 * z2) * y)) /
               4.;
    } else if (l == 3 && m == 0) {
        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;
        return std::sqrt(7 / (2. * M_PI)) *
               ((-3 * x2 * z) / 2. + ((-3 * y2) / 2. + z2) * z);
    } else if (l == 3 && m == 1) {
        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;
        return ((x2 + y2 - 4 * z2) * std::sqrt(21 / (2. * M_PI)) * x) / 4.;
    } else if (l == 3 && m == 2) {
        double x2 = x * x;
        double y2 = y * y;
        return (std::sqrt(105 / M_PI) * (x2 * z - y2 * z)) / 4.;
    } else if (l == 3 && m == 3) {
        double x2 = x * x;
        double y2 = y * y;
        return -0.25 * ((x2 - 3 * y2) * std::sqrt(35 / (2. * M_PI)) * x);
    } else if (l == 4 && m == -4) {
        double x2 = x * x;
        double y3 = y * y * y;
        return (3 * std::sqrt(35 / M_PI) * x * (-y3 + x2 * y)) / 4.;
    } else if (l == 4 && m == -3) {
        double x2 = x * x;
        double y3 = y * y * y;
        return (-3 * std::sqrt(35 / (2. * M_PI)) *
                (-(y3 * z) + 3 * x2 * y * z)) /
               4.;
    } else if (l == 4 && m == -2) {
        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;
        return (-3 * std::sqrt(5 / M_PI) * x * (x2 * y + (y2 - 6 * z2) * y)) /
               4.;
    } else if (l == 4 && m == -1) {
        double x2 = x * x;
        double y2 = y * y;
        double z3 = z * z * z;
        return (3 * std::sqrt(5 / (2. * M_PI)) *
                (3 * x2 * y * z + y * (-4 * z3 + 3 * y2 * z))) /
               4.;
    } else if (l == 4 && m == 0) {
        double x2 = x * x;
        double y2 = y * y;
        double z4 = z * z * z * z;
        double z2 = z * z;
        return (3 * (y2 * ((3 * y2) / 8. - 3 * z2) +
                     x2 * ((3 * x2) / 8. + (3 * y2) / 4. - 3 * z2) + z4)) /
               std::sqrt(2 * M_PI);
    } else if (l == 4 && m == 1) {
        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;
        return (3 * std::sqrt(5 / (2. * M_PI)) * x *
                (3 * x2 * z + (3 * y2 - 4 * z2) * z)) /
               4.;
    } else if (l == 4 && m == 2) {
        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;
        return (-3 * (x2 * (x2 - 6 * z2) + y2 * (-y2 + 6 * z2)) *
                std::sqrt(5 / M_PI)) /
               8.;
    } else if (l == 4 && m == 3) {
        double x2 = x * x;
        double y2 = y * y;
        return (-3 * std::sqrt(35 / (2. * M_PI)) * x * (x2 * z - 3 * y2 * z)) /
               4.;
    } else if (l == 4 && m == 4) {
        double x2 = x * x;
        double y4 = y * y * y * y;
        double y2 = y * y;
        return (3 * (x2 * (x2 - 6 * y2) + y4) * std::sqrt(35 / M_PI)) / 16.;
    }
    return 0;
}

// return a bound on the absolute value of the (l, m) spherical harmonic on the
// unit sphere
double sphericalHarmonicBound(int l, int m) {
    if (l == 1 && m == -1) {
        return 0.488601;
    } else if (l == 1 && m == 0) {
        return 0.690989;
    } else if (l == 1 && m == 1) {
        return 0.488601;
    } else if (l == 2 && m == -2) {
        return 0.546274;
    } else if (l == 2 && m == -1) {
        return 0.546274;
    } else if (l == 2 && m == 0) {
        return 0.446031;
    } else if (l == 2 && m == 1) {
        return 0.546274;
    } else if (l == 2 && m == 2) {
        return 0.546274;
    } else if (l == 3 && m == -3) {
        return 0.590044;
    } else if (l == 3 && m == -2) {
        return 0.556298;
    } else if (l == 3 && m == -1) {
        return 0.62938;
    } else if (l == 3 && m == 0) {
        return 1.0555;
    } else if (l == 3 && m == 1) {
        return 0.62938;
    } else if (l == 3 && m == 2) {
        return 0.556298;
    } else if (l == 3 && m == 3) {
        return 0.590044;
    } else if (l == 4 && m == -4) {
        return 0.625836;
    } else if (l == 4 && m == -3) {
        return 0.574867;
    } else if (l == 4 && m == -2) {
        return 0.608255;
    } else if (l == 4 && m == -1) {
        return 0.706531;
    } else if (l == 4 && m == 0) {
        return 0.512926;
    } else if (l == 4 && m == 1) {
        return 0.706531;
    } else if (l == 4 && m == 2) {
        return 0.473087;
    } else if (l == 4 && m == 3) {
        return 0.574867;
    } else if (l == 4 && m == 4) {
        return 0.625836;
    } else if (l == 5 && m == -5) {
        return 0.656382;
    } else if (l == 5 && m == -4) {
        return 0.594089;
    }
    return 0;
}

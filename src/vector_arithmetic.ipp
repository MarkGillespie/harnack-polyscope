namespace {
using f3 = std::array<float, 3>;
using f4 = std::array<float, 4>;

float dot(const f3& a, const f3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
f3 cross(const f3& a, const f3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
float len_squared(const f3& a) { return dot(a, a); }
float len(const f3& a) { return sqrt(len_squared(a)); }
void normalize(f3& vec) {
    float s = len(vec);
    for (uint i = 0; i < 3; i++) vec[i] /= s;
}
f3 normalized(const f3& vec) {
    float s = len(vec);
    return {vec[0] / s, vec[1] / s, vec[2] / s};
}
f3 diff(const f3& a, const f3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
f3 operator-(const f3& a, const f3& b) { return diff(a, b); }
f3 operator+(const f3& a, const f3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
f3 diff_f(const float3& a, const f3& b) {
    return {(float)a.x - b[0], (float)a.y - b[1], (float)a.z - b[2]};
}
// a + s * b
f3 fma(const f3& a, float s, const f3& b) {
    return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]};
}

f3 over(const f3& a, float s) { return {a[0] / s, a[1] / s, a[2] / s}; }

f3 times(const f3& a, float s) { return {a[0] * s, a[1] * s, a[2] * s}; }

//== Quaternions
f3 orthogonal(const f3& v) // find a vector orthogonal to v
{
    if (std::abs(v[0]) <= std::abs(v[1]) && std::abs(v[0]) <= std::abs(v[2])) {
        return f3{0., -v[2], v[1]};
    } else if (std::abs(v[1]) <= std::abs(v[0]) &&
               std::abs(v[1]) <= std::abs(v[2])) {
        return f3{v[2], 0., -v[0]};
    } else {
        return f3{-v[1], v[0], 0.};
    }
}
f3 mul_s(float s, const f3& v) { return {s * v[0], s * v[1], s * v[2]}; }
float q_re(const f4& q) { return q[0]; }
f3 q_im(const f4& q) { return {q[1], q[2], q[3]}; }
f4 build_T4(float x, const f3& yzw) { return {x, yzw[0], yzw[1], yzw[2]}; }
float q_dot(const f4& a, const f4& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
f4 q_mul(const f4& a, const f4& b) {
    f3 u = mul_s(q_re(a), q_im(b));
    f3 v = mul_s(q_re(b), q_im(a));
    f3 w = cross(q_im(a), q_im(b));
    return {q_re(a) * q_re(b) - dot(q_im(a), q_im(b)), u[0] + v[0] + w[0],
            u[1] + v[1] + w[1], u[2] + v[2] + w[2]};
}
f4 q_conj(const f4& q) { return {q[0], -q[1], -q[2], -q[3]}; }
f4 q_div_s(const f4& q, float s) {
    return {q[0] / s, q[1] / s, q[2] / s, q[3] / s};
}
f4 q_inv(const f4& q) { return q_div_s(q_conj(q), q_dot(q, q)); }
f4 q_div(const f4& a, const f4& b) { return q_mul(a, q_inv(b)); }

// dihedral of two points on the unit sphere, as defined by Chern & Ishida
// https://arxiv.org/abs/2303.14555
// https://stackoverflow.com/a/11741520
f4 dihedral(const f3& p1, const f3& p2) {
    float lengthProduct = len(p1) * len(p2);

    // antiparallel vectors
    if (std::abs(dot(p1, p2) / lengthProduct + (float)1.) < (float)0.0001)
        return build_T4(0., orthogonal(p1));

    // can skip normalization since we don't care about magnitude
    return build_T4(dot(p1, p2) + lengthProduct, cross(p1, p2));
}
// arg(\bar{q2} q1) as defined by Chern & Ishida
// https://arxiv.org/abs/2303.14555
float fiberArg(const f4& q1, const f4& q2) {
    f4 s = q_mul(q_conj(q2), q1);
    return atan2(s[1], s[0]);
}

using d3 = std::array<double, 3>;
using d4 = std::array<double, 4>;

double dot(const d3& a, const d3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
d3 cross(const d3& a, const d3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
double len_squared(const d3& a) { return dot(a, a); }
double len(const d3& a) { return sqrt(len_squared(a)); }
void normalize(d3& vec) {
    double s = len(vec);
    for (uint i = 0; i < 3; i++) vec[i] /= s;
}
d3 normalized(const d3& vec) {
    double s = len(vec);
    return {vec[0] / s, vec[1] / s, vec[2] / s};
}
d3 diff(const d3& a, const d3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
d3 operator-(const d3& a, const d3& b) { return diff(a, b); }
d3 operator+(const d3& a, const d3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
d3 diff_f(const float3& a, const d3& b) {
    return {(double)a.x - b[0], (double)a.y - b[1], (double)a.z - b[2]};
}
d3 diff_f(const d3& a, const float3& b) {
    return {a[0] - (double)b.x, a[1] - (double)b.y, a[2] - (double)b.z};
}
// a + s * b
d3 fma(const d3& a, double s, const d3& b) {
    return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]};
}

d3 over(const d3& a, float s) { return {a[0] / s, a[1] / s, a[2] / s}; }

d3 times(const d3& a, float s) { return {a[0] * s, a[1] * s, a[2] * s}; }

//== Quaternions
d3 orthogonal(const d3& v) // dind a vector orthogonal to v
{
    if (std::abs(v[0]) <= std::abs(v[1]) && std::abs(v[0]) <= std::abs(v[2])) {
        return d3{0., -v[2], v[1]};
    } else if (std::abs(v[1]) <= std::abs(v[0]) &&
               std::abs(v[1]) <= std::abs(v[2])) {
        return d3{v[2], 0., -v[0]};
    } else {
        return d3{-v[1], v[0], 0.};
    }
}
d3 mul_s(double s, const d3& v) { return {s * v[0], s * v[1], s * v[2]}; }
double q_re(const d4& q) { return q[0]; }
d3 q_im(const d4& q) { return {q[1], q[2], q[3]}; }
d4 build_T4(double x, const d3& yzw) { return {x, yzw[0], yzw[1], yzw[2]}; }
double q_dot(const d4& a, const d4& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
d4 q_mul(const d4& a, const d4& b) {
    d3 u = mul_s(q_re(a), q_im(b));
    d3 v = mul_s(q_re(b), q_im(a));
    d3 w = cross(q_im(a), q_im(b));
    return {q_re(a) * q_re(b) - dot(q_im(a), q_im(b)), u[0] + v[0] + w[0],
            u[1] + v[1] + w[1], u[2] + v[2] + w[2]};
}
d4 q_conj(const d4& q) { return {q[0], -q[1], -q[2], -q[3]}; }
d4 q_div_s(const d4& q, double s) {
    return {q[0] / s, q[1] / s, q[2] / s, q[3] / s};
}
d4 q_inv(const d4& q) { return q_div_s(q_conj(q), q_dot(q, q)); }
d4 q_div(const d4& a, const d4& b) { return q_mul(a, q_inv(b)); }

// dihedral of two points on the unit sphere, as defined by Chern & Ishida
// https://arxiv.org/abs/2303.14555
// https://stackoverflow.com/a/11741520
d4 dihedral(const d3& p1, const d3& p2) {
    double lengthProduct = len(p1) * len(p2);

    // antiparallel vectors
    if (std::abs(dot(p1, p2) / lengthProduct + 1.) < 0.0001)
        return build_T4(0., orthogonal(p1));

    // can skip normalization since we don't care about magnitude
    return build_T4(dot(p1, p2) + lengthProduct, cross(p1, p2));
}
// arg(\bar{q2} q1) as defined by Chern & Ishida
// https://arxiv.org/abs/2303.14555
double fiberArg(const d4& q1, const d4& q2) {
    d4 s = q_mul(q_conj(q2), q1);
    return atan2(s[1], s[0]);
}

template <typename T>
std::array<T, 3> from_float3(const float3& p) {
    return {(T)p.x, (T)p.y, (T)p.z};
}

template <typename T>
float3 to_float3(const std::array<T, 3>& p) {
    return make_float3(p[0], p[1], p[2]);
}

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::array<T, 3>& v) {
    o << "(" << std::setw(8) << std::fixed << std::setprecision(4) << v[0]
      << ", " << std::setw(8) << std::fixed << std::setprecision(4) << v[1]
      << ", " << std::setw(8) << std::fixed << std::setprecision(4) << v[2]
      << ")";
    return o;
}

// interval arithmetic stolen from https://www.shadertoy.com/view/7tKfz1 by fad
template <typename T>
struct interval {
    T l, u;
};

template <typename T>
interval<T> iMinMax(T x, T y, T z, T w) {
    return {std::min(x, std::min(y, std::min(z, w))),
            std::max(x, std::max(y, std::max(z, w)))};
}

template <typename T>
interval<T> iNeg(interval<T> x) { // -x
    return {-x.u, -x.l};
}

template <typename T>
interval<T> iInv(interval<T> x) { // 1 / x
    if (x.l > 0.0 || x.u < 0.0) {
        return {1.0 / x.u, 1.0 / x.l};
    } else if (x.l < 0.0 && x.u > 0.0) {
        return {-std::numeric_limits<T>::infinity(),
                std::numeric_limits<T>::infinity()};
    } else if (x.u == 0.0) {
        return {-std::numeric_limits<T>::infinity(), 1.0 / x.l};
    } else {
        return {1.0 / x.u, std::numeric_limits<T>::infinity()};
    }
}

template <typename T>
interval<T> iAdd(interval<T> x, interval<T> y) {
    return {x.l + y.l, x.u + y.u};
}
template <typename T>
interval<T> iSub(interval<T> x, interval<T> y) {
    return {x.l - y.u, x.u - y.l};
}
template <typename T>
interval<T> iMul(interval<T> x, interval<T> y) {
    return iMinMax(x.l * y.l, x.l * y.u, x.u * y.l, x.u * y.u);
}
template <typename T>
interval<T> iDiv(interval<T> x, interval<T> y) {
    return iMul(x, iInv(y));
}

template <typename T>
interval<T> iFloor(interval<T> x) {
    return {floor(x.l), floor(x.u)};
}

template <typename T>
interval<T> iMod(interval<T> x, interval<T> y) {
    return iSub(x, iMul(y, iFloor(iDiv(x, y))));
}

template <typename T>
interval<T> iSqrt(interval<T> x) {
    if (x.l > 0.0) return {sqrt(x.l), sqrt(x.u)};
    if (x.u > 0.0) return {0.0, sqrt(x.u)};
    if (x.u == 0.0) return {0, 0};
    return {sqrt(x.l), sqrt(x.u)}; // NaNs?
    // return EMPTY_SET;
}


template <typename T>
interval<T> iAtan(interval<T> x) {
    return {atan(x.l), atan(x.u)};
}

template <typename T>
interval<T> iAtan2(interval<T> y, interval<T> x) {
    if (x.u < 0.) {
        if (y.u < 0.) return {atan2(y.u, x.l), atan2(y.l, x.u)};
        if (y.l < 0.) return {-M_PI, M_PI};
        return {atan2(y.u, x.u), atan2(y.l, x.l)};
    }

    if (x.u == 0.) {
        if (x.l < 0.) {
            if (y.u < 0.) return {atan2(y.u, x.l), -M_PI / 2.};
            if (y.l < 0.) return {-M_PI, M_PI};
            if (y.l == 0.) return {0., M_PI};
            return {M_PI / 2., atan2(y.l, x.l)};
        }

        if (y.u < 0.) return {-M_PI / 2., -M_PI / 2.};

        if (y.u == 0.) {
            if (y.l < 0.) return {-M_PI / 2., 0.};
            return {0., 0.};
        }

        if (y.l < 0.) return {-M_PI / 2., M_PI / 2.};
        if (y.l == 0.) return {0., M_PI / 2.};
        return {M_PI / 2., M_PI / 2.};
    }

    if (x.l < 0.) {
        if (y.u < 0.) return {atan2(y.u, x.l), atan2(y.u, x.u)};
        if (y.u == 0.) {
            if (y.l == 0.) return {0., M_PI};
            return {-M_PI, M_PI};
        }
        if (y.l < 0.) return {-M_PI, M_PI};
        if (y.l == 0.) return {0., M_PI};
        return {atan2(y.l, x.u), atan2(y.l, x.l)};
    }

    if (x.l == 0.) {
        if (y.u <= 0.) return {atan2(y.u, x.u), -M_PI / 2.};
        if (y.l < 0.) return {-M_PI / 2., M_PI / 2.};
        if (y.l == 0.) return {0., M_PI / 2.};
        return {atan2(y.l, x.u), M_PI / 2.};
    }

    return iAtan(iDiv(y, x));
}

template <typename T>
interval<T> operator+(const interval<T>& a, const interval<T>& b) {
    return iAdd(a, b);
}

template <typename T>
interval<T> operator-(const interval<T>& a, const interval<T>& b) {
    return iSub(a, b);
}

template <typename T>
interval<T> operator*(const interval<T>& a, const interval<T>& b) {
    return iMul(a, b);
}
template <typename T>
interval<T> operator/(const interval<T>& a, const interval<T>& b) {
    return iDiv(a, b);
}

#define MAKE_OVERLOADS(fn)                                                     \
    template <typename T>                                                      \
    interval<T> fn(T x, const interval<T>& y) {                                \
        return fn(interval<T>{x, x}, y);                                       \
    }                                                                          \
    template <typename T>                                                      \
    interval<T> fn(const interval<T>& x, T y) {                                \
        return fn(x, interval<T>{y, y});                                       \
    }

// 4.56199
MAKE_OVERLOADS(iAdd)
MAKE_OVERLOADS(iSub)
MAKE_OVERLOADS(iMul)
MAKE_OVERLOADS(iDiv)
MAKE_OVERLOADS(iFloor)
MAKE_OVERLOADS(iMod)
MAKE_OVERLOADS(operator+)
MAKE_OVERLOADS(operator-)
MAKE_OVERLOADS(operator*)
MAKE_OVERLOADS(operator/)

template <typename T>
struct i_vec3 {
    interval<T> x, y, z;
};

template <typename T>
i_vec3<T> iAdd(const i_vec3<T>& a, const i_vec3<T>& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
template <typename T>
i_vec3<T> iSub(const i_vec3<T>& a, const i_vec3<T>& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
template <typename T>
i_vec3<T> iMul(const i_vec3<T>& a, const interval<T>& b) {
    return {a.x * b, a.y * b, a.z * b};
}
template <typename T>
i_vec3<T> iDiv(const i_vec3<T>& a, const interval<T>& b) {
    return {a.x / b, a.y / b, a.z / b};
}
template <typename T>
interval<T> iDot(const i_vec3<T>& a, const i_vec3<T>& b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}
template <typename T>
i_vec3<T> iCross(const i_vec3<T>& a, const i_vec3<T>& b) {
    return {a.y * b.z - a.z * b.y, //
            a.z * b.x - a.x * b.z, //
            a.x * b.y - a.y * b.x};
}
template <typename T>
interval<T> iLen(const i_vec3<T>& a) {
    return iSqrt(iDot(a, a));
}

template <typename T>
i_vec3<T> operator+(const i_vec3<T>& a, const i_vec3<T>& b) {
    return iAdd(a, b);
}

template <typename T>
i_vec3<T> operator-(const i_vec3<T>& a, const i_vec3<T>& b) {
    return iSub(a, b);
}

template <typename T>
i_vec3<T> operator*(const i_vec3<T>& a, const interval<T>& b) {
    return iMul(a, b);
}
template <typename T>
i_vec3<T> operator*(const interval<T>& a, const i_vec3<T>& b) {
    return iMul(b, a);
}
template <typename T>
i_vec3<T> operator/(const i_vec3<T>& a, const interval<T>& b) {
    return iDiv(a, b);
}

template <typename T>
struct i_cplx {
    interval<T> re, im;
    i_cplx& operator*=(const i_cplx& other) {
        interval<T> a = re;
        interval<T> b = im;
        re            = (a * other.re) - (b * other.im);
        im            = (a * other.im) + (b * other.re);
        return *this;
    }
};

template <typename T>
interval<T> iArg(const i_cplx<T>& z) {
    return iAtan2(z.im, z.re);
}
} // namespace

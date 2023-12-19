#include "poisson_problem.h"

#include <cmath>

void solve_poisson_problem(std::vector<double>& source, size_t n) {
    double h = 1. / (double)n;

    // double domain to implement Neumann boundary conditions
    size_t n_fftw = 2 * n;
    // output size of last dimension is smaller due to symmetry
    size_t n_fftw_i = (n_fftw / 2) + 1;
    size_t Nin      = n_fftw * n_fftw * n_fftw;
    size_t Nout     = n_fftw * n_fftw * n_fftw_i;

    // allocate array
    double* in        = (double*)fftw_malloc(sizeof(double) * Nin);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nout);

    // initialize fftw
    fftw_plan forwards =
        fftw_plan_dft_r2c_3d(n_fftw, n_fftw, n_fftw, in, out, FFTW_ESTIMATE);
    fftw_plan backwards =
        fftw_plan_dft_c2r_3d(n_fftw, n_fftw, n_fftw, out, in, FFTW_ESTIMATE);

    // fill arrays
    for (size_t k = 0; k < n; k++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t i = 0; i < n; i++) {
                double source_val = source[i + j * n + k * n * n];
                // iterate over all reflections of the (i, j, k) cell
                for (size_t ri : {i, n_fftw - i}) {
                    for (size_t rj : {j, n_fftw - j}) {
                        for (size_t rk : {k, n_fftw - k}) {
                            size_t fftw_ind =
                                ri + rj * n_fftw + rk * n_fftw * n_fftw;
                            in[fftw_ind] = source_val;
                        }
                    }
                }
            }
        }
    }

    // do fft
    fftw_execute(forwards); /* repeat as needed */
    auto mu = [&](size_t p) -> double {
        return 4 * pow(sin((double)(p - 1) * M_PI / n_fftw), 2) / pow(h, 2);
    };
    for (size_t k = 0; k < n_fftw; k++) {
        double mu_k = mu(k);
        for (size_t j = 0; j < n_fftw; j++) {
            double mu_j = mu(j);
            for (size_t i = 0; i < n_fftw_i; i++) {
                double mu_i     = mu(i);
                size_t fftw_ind = i + j * n_fftw_i + k * n_fftw_i * n_fftw;
                out[fftw_ind][0] /= -(mu_i + mu_j + mu_k);
                out[fftw_ind][1] /= -(mu_i + mu_j + mu_k);
            }
        }
    }
    // request mean-zero solution
    out[0][0] = 0;
    out[0][1] = 0;

    fftw_execute(backwards);

    // read answer back into source array
    for (size_t k = 0; k < n; k++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t i = 0; i < n; i++) {
                size_t source_ind  = i + j * n + k * n * n;
                size_t fftw_ind    = i + j * n_fftw + k * n_fftw * n_fftw;
                source[source_ind] = in[fftw_ind];
            }
        }
    }

    // clean up
    fftw_destroy_plan(forwards);
    fftw_destroy_plan(backwards);
    fftw_free(in);
    fftw_free(out);
}

std::vector<double> divergence(std::function<Vector3(Vector3)> V, size_t n) {
    double h = 1. / (double)n;

    std::vector<double> out;
    out.reserve(n * n * n);

    for (size_t k = 0; k < n; k++) {
        double z = h * (double)k;
        for (size_t j = 0; j < n; j++) {
            double y = h * (double)j;
            for (size_t i = 0; i < n; i++) {
                double x = h * (double)i;

                double dxx = (V({x + h, y, z}).x - V({x, y, z}).x) / h;
                double dyy = (V({x, y + h, z}).y - V({x, y, z}).y) / h;
                double dzz = (V({x, y, z + h}).z - V({x, y, z}).z) / h;

                out.push_back(dxx + dyy + dzz);
            }
        }
    }

    return out;
}

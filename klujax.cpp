// version: 0.5.3
// Imports

#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "klu.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

// Split numeric allocations only; fused solves own and free their temporaries.
static std::atomic<int64_t> live_numeric_allocations{0};
static std::atomic<uint64_t> total_numeric_allocations{0};

ffi::Error validate_args(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions ds_Ax,
    const ffi::AnyBuffer::Dimensions ds_x) {
    int d_x = ds_x.size();
    if (d_x != 3) {
        return ffi::Error::InvalidArgument("x is not 3D.");
    }
    int n_lhs = (int)ds_x[0];
    int n_col = (int)ds_x[1];

    int d_Ax = ds_Ax.size();
    if (d_Ax != 2) {
        return ffi::Error::InvalidArgument("Ax is not 2D.");
    }
    int n_lhs_bis = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];

    if (n_lhs != n_lhs_bis) {
        return ffi::Error::InvalidArgument(
            "n_lhs mismatch: Ax.shape[0] != x.shape[0]: Got " + std::to_string(n_lhs_bis) + " != " + std::to_string(n_lhs));
    }

    auto ds_Ai = Ai.dimensions();
    int d_Ai = ds_Ai.size();
    if (d_Ai != 1) {
        return ffi::Error::InvalidArgument("Ai is not 1D.");
    }
    int n_nz_bis = (int)ds_Ai[0];
    if (n_nz != n_nz_bis) {
        return ffi::Error::InvalidArgument(
            "n_nz mismatch: Ai.shape[0] != Ax.shape[1]: Got " + std::to_string(n_nz_bis) + " != " + std::to_string(n_nz));
    }

    auto ds_Aj = Aj.dimensions();
    int d_Aj = ds_Aj.size();
    if (d_Aj != 1) {
        return ffi::Error::InvalidArgument("Aj is not 1D.");
    }
    n_nz_bis = (int)ds_Aj[0];
    if (n_nz != n_nz_bis) {
        return ffi::Error::InvalidArgument(
            "n_nz mismatch: Aj.shape[0] != Ax.shape[1]: Got " + std::to_string(n_nz_bis) + " != " + std::to_string(n_nz));
    }

    int i;
    int j;
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    for (int n = 0; n < n_nz; n++) {
        i = _Ai[n];
        if (i < 0) {
            return ffi::Error::InvalidArgument("Ai contains negative index");
        }
        if (i >= n_col) {
            return ffi::Error::InvalidArgument("Ai.max() >= n_col");
        }
        j = _Aj[n];
        if (j < 0) {
            return ffi::Error::InvalidArgument("Aj contains negative index");
        }
        if (j >= n_col) {
            return ffi::Error::InvalidArgument("Aj.max() >= n_col");
        }
    }
    return ffi::Error::Success();
}

void coo_to_csc_analyze(
    const int n_col,
    const int n_nz,
    const int* Ai,
    const int* Aj,
    int* Bi,
    int* Bp,
    int* Bk) {
    // compute number of non-zero entries per row of A
    for (int n = 0; n < n_nz; n++) {
        Bp[Aj[n]] += 1;
    }

    // cumsum the n_nz per row to get Bp
    int cumsum = 0;
    int temp = 0;
    for (int j = 0; j <= n_col; j++) {
        temp = Bp[j];
        Bp[j] = cumsum;
        cumsum += temp;
    }

    // write Ai, Aj into Bi, Bk
    int col = 0;
    int dest = 0;
    for (int n = 0; n < n_nz; n++) {
        col = Aj[n];
        dest = Bp[col];
        Bi[dest] = Ai[n];
        Bk[dest] = n;
        Bp[col] += 1;
    }

    int last = 0;
    for (int i = 0; i <= n_col; i++) {
        temp = Bp[i];
        Bp[i] = last;
        last = temp;
    }
}

using Complex = std::complex<double>;

template <typename T>
struct KluTraits;

template <>
struct KluTraits<double> {
    static klu_numeric* factor(int* Ap, int* Ai, double* Ax, klu_symbolic* Symbolic, klu_common* Common) {
        return klu_factor(Ap, Ai, Ax, Symbolic, Common);
    }
    static int refactor(int* Ap, int* Ai, double* Ax, klu_symbolic* Symbolic, klu_numeric* Numeric, klu_common* Common) {
        return klu_refactor(Ap, Ai, Ax, Symbolic, Numeric, Common);
    }
    static int solve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, double* B, klu_common* Common) {
        return klu_solve(Symbolic, Numeric, d, nrhs, B, Common);
    }
    static int tsolve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, double* B, klu_common* Common) {
        return klu_tsolve(Symbolic, Numeric, d, nrhs, B, Common);
    }
};

template <>
struct KluTraits<Complex> {
    static klu_numeric* factor(int* Ap, int* Ai, Complex* Ax, klu_symbolic* Symbolic, klu_common* Common) {
        return klu_z_factor(Ap, Ai, reinterpret_cast<double*>(Ax), Symbolic, Common);
    }
    static int refactor(int* Ap, int* Ai, Complex* Ax, klu_symbolic* Symbolic, klu_numeric* Numeric, klu_common* Common) {
        return klu_z_refactor(Ap, Ai, reinterpret_cast<double*>(Ax), Symbolic, Numeric, Common);
    }
    static int solve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, Complex* B, klu_common* Common) {
        return klu_z_solve(Symbolic, Numeric, d, nrhs, reinterpret_cast<double*>(B), Common);
    }
    // conj_solve=0 gives plain transpose A^T; use 1 for conjugate transpose A^H
    static int tsolve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, Complex* B, klu_common* Common) {
        return klu_z_tsolve(Symbolic, Numeric, d, nrhs, reinterpret_cast<double*>(B), 0, Common);
    }
};

template <typename T>
ffi::Error dot_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_x,
    const T* _Ax,
    const T* _x,
    T* _b) {
    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_x);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_x[0];
    int n_col = (int)ds_x[1];
    int n_rhs = (int)ds_x[2];
    int n_nz = (int)ds_Ax[1];

    // initialize empty result
    for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
        _b[i] = 0.0;
    }

    // fill result (all multi-dim arrays are row-major)
    // x_mik = A_mij × x_mjk (einsum)
    // sizes: m<n_lhs; i<n_col<--Ai; j<n_col<--Aj; k<n_rhs
    // Loop order: m (batch) outer for better cache locality on Ax
    int i;
    int j;
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_nz; n++) {
            i = _Ai[n];
            j = _Aj[n];
            for (int k = 0; k < n_rhs; k++) {
                _b[m * n_col * n_rhs + i * n_rhs + k] += _Ax[m * n_nz + n] * _x[m * n_col * n_rhs + j * n_rhs + k];
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error dot_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> x,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> b) {
    return dot_impl<double>(Ai, Aj, Ax.dimensions(), x.dimensions(),
                            Ax.typed_data(), x.typed_data(), b->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    dot_f64_handler, dot_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // b
);

ffi::Error dot_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> x,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> b) {
    return dot_impl<Complex>(Ai, Aj, Ax.dimensions(), x.dimensions(),
                             reinterpret_cast<const Complex*>(Ax.typed_data()),
                             reinterpret_cast<const Complex*>(x.typed_data()),
                             reinterpret_cast<Complex*>(b->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    dot_c128_handler, dot_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // b
);

template <typename T>
ffi::Error solve_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const T* _Ax,
    const T* _b,
    T* _x) {
    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_b);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // get COO -> CSC transformation information (using RAII for automatic cleanup)
    auto _Bk = std::make_unique<int[]>(n_nz);  // Ax -> Bx transformation indices
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // copy _b into _x_temp and transpose the last two dimensions since KLU expects col-major layout
    // _b itself won't be used anymore. KLU works on _x_temp in-place.
    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    // initialize KLU for given sparsity pattern
    klu_symbolic* Symbolic;
    klu_numeric* Numeric;
    klu_common Common;
    klu_defaults(&Common);
    Symbolic = klu_analyze(n_col, _Bp.get(), _Bi.get(), &Common);

    // solve for all elements in batch:
    // NOTE: same sparsity pattern for each element in batch assumed
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        // convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        // solve using KLU
        Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_factor/z_factor failed (singular matrix?)");
        }
        KluTraits<T>::solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_solve/z_solve failed");
        }
        klu_free_numeric(&Numeric, &Common);
    }

    // copy _x_temp into _x and transpose the last two dimensions since JAX expects row-major layout
    // NOTE: it feels a bit weird to have to do all this copying and transposing here. This might actually be
    // pretty inefficient. Ideally I'd like to get rid of this transpose. Maybe just represent b/x in python
    // as n_lhs x n_rhs x n_col in stead of n_lhs x n_col x n_rhs?
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    klu_free_symbolic(&Symbolic, &Common);
    return ffi::Error::Success();
}

ffi::Error solve_f64(
    ffi::Buffer<ffi::DataType::S32> Ai,
    ffi::Buffer<ffi::DataType::S32> Aj,
    ffi::Buffer<ffi::DataType::F64> Ax,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return solve_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(),
                              Ax.typed_data(), b.typed_data(), x->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    solve_f64_handler, solve_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

ffi::Error solve_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return solve_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(),
                               reinterpret_cast<const Complex*>(Ax.typed_data()),
                               reinterpret_cast<const Complex*>(b.typed_data()),
                               reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    solve_c128_handler, solve_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

template <typename T>
ffi::Error solve_with_symbol_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const T* _Ax,
    const T* _b,
    T* _x) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_b);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // get COO -> CSC transformation information (using RAII for automatic cleanup)
    auto _Bk = std::make_unique<int[]>(n_nz);  // Ax -> Bx transformation indices
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // copy _b into _x_temp and transpose the last two dimensions since KLU expects col-major layout
    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    klu_numeric* Numeric;
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        // convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        // solve using KLU with provided Symbolic handle
        Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            return ffi::Error::InvalidArgument("klu_factor/z_factor failed (singular matrix?)");
        }
        KluTraits<T>::solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            return ffi::Error::InvalidArgument("klu_solve/z_solve failed");
        }
        klu_free_numeric(&Numeric, &Common);
    }

    // copy _x_temp into _x and transpose
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    return ffi::Error::Success();
}

ffi::Error solve_with_symbol_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return solve_with_symbol_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                          Ax.typed_data(), b.typed_data(), x->typed_data());
}

ffi::Error solve_with_symbol_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return solve_with_symbol_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                           reinterpret_cast<const Complex*>(Ax.typed_data()),
                                           reinterpret_cast<const Complex*>(b.typed_data()),
                                           reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_symbol_f64_handler, solve_with_symbol_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_symbol_c128_handler, solve_with_symbol_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

// tsolve_with_symbol: identical to solve_with_symbol_impl but uses KluTraits<T>::tsolve
// Solves A^T x = b (transpose solve) reusing a pre-computed symbolic factorization.
template <typename T>
ffi::Error tsolve_with_symbol_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const T* _Ax,
    const T* _b,
    T* _x) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_b);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    klu_numeric* Numeric;
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            return ffi::Error::InvalidArgument("klu_factor/z_factor failed (singular matrix?)");
        }
        // NOTE: tsolve instead of solve
        KluTraits<T>::tsolve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            return ffi::Error::InvalidArgument("klu_tsolve/z_tsolve failed");
        }
        klu_free_numeric(&Numeric, &Common);
    }

    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    return ffi::Error::Success();
}

ffi::Error tsolve_with_symbol_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return tsolve_with_symbol_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                           Ax.typed_data(), b.typed_data(), x->typed_data());
}

ffi::Error tsolve_with_symbol_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return tsolve_with_symbol_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                            reinterpret_cast<const Complex*>(Ax.typed_data()),
                                            reinterpret_cast<const Complex*>(b.typed_data()),
                                            reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_symbol_f64_handler, tsolve_with_symbol_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_symbol_c128_handler, tsolve_with_symbol_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

template <typename T>
ffi::Error factor_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const T* _Ax,
    uint64_t* _numeric) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    int n_lhs = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];
    int n_col = Symbolic->n;

    if (Ai.dimensions().size() != 1 || Aj.dimensions().size() != 1) return ffi::Error::InvalidArgument("Ai/Aj must be 1D");
    if (Ai.dimensions()[0] != n_nz || Aj.dimensions()[0] != n_nz) return ffi::Error::InvalidArgument("Ai/Aj size mismatch with Ax");

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // get COO -> CSC transformation information
    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        // convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        klu_numeric* Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            // Cleanup already allocated numerics in this batch?
            // For simplicity, we return error and let user handle cleanup (or leak, but this is exception path).
            // Ideally we should cleanup.
            for (int j = 0; j < i; j++) {
                klu_numeric* num = reinterpret_cast<klu_numeric*>(_numeric[j]);
                klu_free_numeric(&num, &Common);
                --live_numeric_allocations;
            }
            return ffi::Error::InvalidArgument("klu_factor/z_factor failed (singular matrix?)");
        }
        _numeric[i] = reinterpret_cast<uint64_t>(Numeric);
        ++live_numeric_allocations;
        ++total_numeric_allocations;
    }
    return ffi::Error::Success();
}

ffi::Error factor_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> numeric) {
    return factor_impl<double>(Ai, Aj, Ax.dimensions(), symbolic, Ax.typed_data(), numeric->typed_data());
}

ffi::Error factor_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> numeric) {
    return factor_impl<Complex>(Ai, Aj, Ax.dimensions(), symbolic,
                                reinterpret_cast<const Complex*>(Ax.typed_data()),
                                numeric->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factor_f64_handler, factor_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Ret<ffi::Buffer<ffi::DataType::U64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factor_c128_handler, factor_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::C128>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Ret<ffi::Buffer<ffi::DataType::U64>>());

template <typename T>
ffi::Error refactor_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _Ax,
    uint64_t* _out_numeric) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    int n_lhs = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];
    int n_col = Symbolic->n;

    if (Ai.dimensions().size() != 1 || Aj.dimensions().size() != 1) return ffi::Error::InvalidArgument("Ai/Aj must be 1D");
    if (Ai.dimensions()[0] != n_nz || Aj.dimensions()[0] != n_nz) return ffi::Error::InvalidArgument("Ai/Aj size mismatch with Ax");
    if ((int)numeric.element_count() != n_lhs) return ffi::Error::InvalidArgument("numeric array size must match n_lhs");

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const uint64_t* _numeric = numeric.typed_data();

    // get COO -> CSC transformation information
    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        uint64_t num_addr = _numeric[i];
        if (num_addr == 0) return ffi::Error::InvalidArgument("numeric pointer is null");
        klu_numeric* Numeric = reinterpret_cast<klu_numeric*>(num_addr);

        int m = i * n_nz;
        // convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        int status = KluTraits<T>::refactor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, Numeric, &Common);
        if (!status || Common.status < KLU_OK) {
            return ffi::Error::InvalidArgument("klu_refactor/z_refactor failed (singular matrix?)");
        }
        // Pass through the same pointer so XLA sees a data dependency: refactor → solve
        _out_numeric[i] = num_addr;
    }
    return ffi::Error::Success();
}

ffi::Error refactor_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric) {
    return refactor_impl<double>(Ai, Aj, Ax.dimensions(), symbolic, numeric, Ax.typed_data(), out_numeric->typed_data());
}

ffi::Error refactor_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric) {
    return refactor_impl<Complex>(Ai, Aj, Ax.dimensions(), symbolic, numeric,
                                  reinterpret_cast<const Complex*>(Ax.typed_data()),
                                  out_numeric->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_f64_handler, refactor_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // out_numeric (same handles, for XLA dep edge)
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_c128_handler, refactor_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()   // out_numeric (same handles, for XLA dep edge)
);

// refactor_and_solve: fused in-place refactorization followed by triangular solve.
// Avoids allocating the CSC buffer twice and saves a JAX dispatch round-trip.
// Returns two outputs: x (the solution) and out_numeric (same pointers as input numeric,
// threaded through so XLA sees the dependency edge: refactor → solve → next-op).
template <typename T>
ffi::Error refactor_and_solve_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _Ax,
    const T* _b,
    T* _x,
    uint64_t* _out_numeric) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    int n_lhs_ax = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];
    int n_col = Symbolic->n;

    if (Ai.dimensions().size() != 1 || Aj.dimensions().size() != 1) return ffi::Error::InvalidArgument("Ai/Aj must be 1D");
    if (Ai.dimensions()[0] != n_nz || Aj.dimensions()[0] != n_nz) return ffi::Error::InvalidArgument("Ai/Aj size mismatch with Ax");
    if ((int)numeric.element_count() != n_lhs_ax) return ffi::Error::InvalidArgument("numeric array size must match n_lhs");

    // Validate b dimensions: must be 3D (n_lhs, n_col, n_rhs)
    if (ds_b.size() != 3) return ffi::Error::InvalidArgument("b must be 3D (n_lhs, n_col, n_rhs)");
    int n_lhs_b = (int)ds_b[0];
    int n_rhs = (int)ds_b[2];

    if (n_lhs_ax != n_lhs_b) return ffi::Error::InvalidArgument("n_lhs mismatch between Ax and b");

    int n_lhs = n_lhs_ax;

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const uint64_t* _numeric = numeric.typed_data();

    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // Transpose b into col-major temp buffer for KLU
    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        uint64_t num_addr = _numeric[i];
        if (num_addr == 0) return ffi::Error::InvalidArgument("numeric pointer is null");
        klu_numeric* Numeric = reinterpret_cast<klu_numeric*>(num_addr);

        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        // Convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        // Refactor in-place (reuses pivots from original factor call)
        int status = KluTraits<T>::refactor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, Numeric, &Common);
        if (!status || Common.status < KLU_OK) {
            return ffi::Error::InvalidArgument("klu_refactor/z_refactor failed (singular matrix?)");
        }

        // Solve immediately using the freshly refactored numeric
        KluTraits<T>::solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            return ffi::Error::InvalidArgument("klu_solve/z_solve failed");
        }

        // Pass through the same numeric pointer for the XLA dependency edge
        _out_numeric[i] = num_addr;
    }

    // Transpose result back to row-major for JAX
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    return ffi::Error::Success();
}

ffi::Error refactor_and_solve_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric) {
    return refactor_and_solve_impl<double>(
        Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic, numeric,
        Ax.typed_data(), b.typed_data(), x->typed_data(), out_numeric->typed_data());
}

ffi::Error refactor_and_solve_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric) {
    return refactor_and_solve_impl<Complex>(
        Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic, numeric,
        reinterpret_cast<const Complex*>(Ax.typed_data()),
        reinterpret_cast<const Complex*>(b.typed_data()),
        reinterpret_cast<Complex*>(x->typed_data()),
        out_numeric->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_and_solve_f64_handler, refactor_and_solve_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x (solution)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // out_numeric (same handles, for XLA dep edge)
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_and_solve_c128_handler, refactor_and_solve_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x (solution)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()   // out_numeric (same handles, for XLA dep edge)
);

template <typename T>
ffi::Error solve_with_numeric_impl(
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::AnyBuffer::Dimensions& ds_x,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _b,
    T* _x) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    int n_numeric = numeric.element_count();
    const uint64_t* _numeric = numeric.typed_data();

    // Parse b dimensions
    int d_b = ds_b.size();
    int n_lhs_b, n_col, n_rhs;

    if (d_b == 1) {
        // b is (n_col,) -> treat as (1, n_col, 1)
        n_lhs_b = 1;
        n_col = ds_b[0];
        n_rhs = 1;
    } else if (d_b == 2) {
        // b is (n_lhs, n_col) or (n_col, n_rhs)
        // Decide based on numeric batch size
        if (n_numeric > 1 && (int)ds_b[0] == n_numeric) {
            // b is (n_lhs, n_col) -> (n_lhs, n_col, 1)
            n_lhs_b = ds_b[0];
            n_col = ds_b[1];
            n_rhs = 1;
        } else {
            // b is (n_col, n_rhs) -> (1, n_col, n_rhs)
            n_lhs_b = 1;
            n_col = ds_b[0];
            n_rhs = ds_b[1];
        }
    } else if (d_b == 3) {
        // b is already (n_lhs, n_col, n_rhs)
        n_lhs_b = ds_b[0];
        n_col = ds_b[1];
        n_rhs = ds_b[2];
    } else {
        return ffi::Error::InvalidArgument("b must be 1D, 2D, or 3D");
    }

    // Determine output batch size
    bool broadcast_numeric = (n_numeric == 1);
    bool broadcast_b = (n_lhs_b == 1);

    int n_lhs;
    if (!broadcast_numeric && !broadcast_b) {
        if (n_numeric != n_lhs_b) {
            return ffi::Error::InvalidArgument("numeric and b batch size mismatch");
        }
        n_lhs = n_numeric;
    } else if (!broadcast_numeric) {
        n_lhs = n_numeric;
    } else if (!broadcast_b) {
        n_lhs = n_lhs_b;
    } else {
        n_lhs = 1;
    }

    // Validate output dimensions match expected
    if (ds_x.size() != d_b) {
        return ffi::Error::InvalidArgument("output dimensions don't match input");
    }

    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);

    // Copy b to x_temp (transpose) with broadcasting
    for (int m = 0; m < n_lhs; m++) {
        int m_b = broadcast_b ? 0 : m;
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m_b * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        int n = i * n_rhs * n_col;
        uint64_t num_addr = broadcast_numeric ? _numeric[0] : _numeric[i];
        if (num_addr == 0) return ffi::Error::InvalidArgument("numeric pointer is null");

        klu_numeric* Numeric = reinterpret_cast<klu_numeric*>(num_addr);
        KluTraits<T>::solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);

        if (Common.status < KLU_OK) {
            return ffi::Error::InvalidArgument("klu_solve/z_solve failed");
        }
    }

    // Copy x_temp to x (transpose)
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error solve_with_numeric_f64(
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return solve_with_numeric_impl<double>(b.dimensions(), x->dimensions(), symbolic, numeric, b.typed_data(), x->typed_data());
}

ffi::Error solve_with_numeric_c128(
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return solve_with_numeric_impl<Complex>(b.dimensions(), x->dimensions(), symbolic, numeric,
                                            reinterpret_cast<const Complex*>(b.typed_data()),
                                            reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_numeric_f64_handler, solve_with_numeric_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_numeric_c128_handler, solve_with_numeric_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::C128>>()
        .Ret<ffi::Buffer<ffi::DataType::C128>>());

// tsolve_with_numeric: identical to solve_with_numeric_impl but uses KluTraits<T>::tsolve.
// Solves A^T x = b using a pre-computed numeric factorization.
template <typename T>
ffi::Error tsolve_with_numeric_impl(
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::AnyBuffer::Dimensions& ds_x,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _b,
    T* _x) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    int n_numeric = numeric.element_count();
    const uint64_t* _numeric = numeric.typed_data();

    int d_b = ds_b.size();
    int n_lhs_b, n_col, n_rhs;

    if (d_b == 1) {
        n_lhs_b = 1;
        n_col = ds_b[0];
        n_rhs = 1;
    } else if (d_b == 2) {
        if (n_numeric > 1 && (int)ds_b[0] == n_numeric) {
            n_lhs_b = ds_b[0];
            n_col = ds_b[1];
            n_rhs = 1;
        } else {
            n_lhs_b = 1;
            n_col = ds_b[0];
            n_rhs = ds_b[1];
        }
    } else if (d_b == 3) {
        n_lhs_b = ds_b[0];
        n_col = ds_b[1];
        n_rhs = ds_b[2];
    } else {
        return ffi::Error::InvalidArgument("b must be 1D, 2D, or 3D");
    }

    bool broadcast_numeric = (n_numeric == 1);
    bool broadcast_b = (n_lhs_b == 1);

    int n_lhs;
    if (!broadcast_numeric && !broadcast_b) {
        if (n_numeric != n_lhs_b) return ffi::Error::InvalidArgument("numeric and b batch size mismatch");
        n_lhs = n_numeric;
    } else if (!broadcast_numeric) {
        n_lhs = n_numeric;
    } else if (!broadcast_b) {
        n_lhs = n_lhs_b;
    } else {
        n_lhs = 1;
    }

    if (ds_x.size() != d_b) return ffi::Error::InvalidArgument("output dimensions don't match input");

    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);

    for (int m = 0; m < n_lhs; m++) {
        int m_b = broadcast_b ? 0 : m;
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m_b * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        int n = i * n_rhs * n_col;
        uint64_t num_addr = broadcast_numeric ? _numeric[0] : _numeric[i];
        if (num_addr == 0) return ffi::Error::InvalidArgument("numeric pointer is null");

        klu_numeric* Numeric = reinterpret_cast<klu_numeric*>(num_addr);
        // NOTE: tsolve instead of solve
        KluTraits<T>::tsolve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);

        if (Common.status < KLU_OK) {
            return ffi::Error::InvalidArgument("klu_tsolve/z_tsolve failed");
        }
    }

    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error tsolve_with_numeric_f64(
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return tsolve_with_numeric_impl<double>(b.dimensions(), x->dimensions(), symbolic, numeric, b.typed_data(), x->typed_data());
}

ffi::Error tsolve_with_numeric_c128(
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return tsolve_with_numeric_impl<Complex>(b.dimensions(), x->dimensions(), symbolic, numeric,
                                             reinterpret_cast<const Complex*>(b.typed_data()),
                                             reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_numeric_f64_handler, tsolve_with_numeric_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_numeric_c128_handler, tsolve_with_numeric_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

ffi::Error free_numeric(
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::RemainingArgs dependencies,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> status) {
    int n = numeric.element_count();
    const uint64_t* _numeric = numeric.typed_data();

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n; i++) {
        uint64_t addr = _numeric[i];
        if (addr != 0) {
            klu_numeric* Numeric = reinterpret_cast<klu_numeric*>(addr);
            klu_free_numeric(&Numeric, &Common);
            --live_numeric_allocations;
        }
    }

    // returning value so function can be traced
    *status->typed_data() = 1;
    return ffi::Error::Success();
}

ffi::Error free_symbolic(
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> status) {
    if (symbolic.element_count() != 1) {
        return ffi::Error::InvalidArgument("symbolic must be a scalar.");
    }
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) {
        *status->typed_data() = 0;  // 0 = nothing to free
        return ffi::Error::Success();
    }

    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);
    klu_common Common;
    klu_defaults(&Common);
    klu_free_symbolic(&Symbolic, &Common);

    // returning value so function can be traced
    *status->typed_data() = 1;
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    free_numeric_handler, free_numeric,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric
        .RemainingArgs()                         // consumer results establish execution dependencies
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // status (instead of no return)
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    free_symbolic_handler, free_symbolic,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // status
);

struct KLUSymbolic {
    klu_symbolic* ptr = nullptr;
    klu_common common{};
    bool closed = false;

    KLUSymbolic(uint64_t raw) : ptr(reinterpret_cast<klu_symbolic*>(raw)) {
        klu_defaults(&common);
    }
    KLUSymbolic(const KLUSymbolic&) = delete;
    KLUSymbolic& operator=(const KLUSymbolic&) = delete;

    uint64_t raw() const { return reinterpret_cast<uint64_t>(ptr); }

    void close() {
        if (!closed && ptr) {
            klu_free_symbolic(&ptr, &common);
            closed = true;
        }
    }

    ~KLUSymbolic() { close(); }
};

struct KLUNumeric {
    std::vector<uint64_t> ptrs;
    klu_common common{};
    bool closed = false;

    KLUNumeric(std::vector<uint64_t> data) : ptrs(std::move(data)) {
        klu_defaults(&common);
    }
    KLUNumeric(const KLUNumeric&) = delete;
    KLUNumeric& operator=(const KLUNumeric&) = delete;

    size_t size() const { return ptrs.size(); }

    std::vector<uint64_t> as_list() const { return ptrs; }

    void close() {
        if (!closed) {
            for (auto addr : ptrs) {
                if (addr != 0) {
                    auto* num = reinterpret_cast<klu_numeric*>(addr);
                    klu_free_numeric(&num, &common);
                    --live_numeric_allocations;
                }
            }
            ptrs.clear();
            closed = true;
        }
    }

    ~KLUNumeric() { close(); }
};

ffi::Error analyze(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::S32> n_col_buf,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> symbolic) {
    if (n_col_buf.element_count() != 1) {
        return ffi::Error::InvalidArgument("n_col must be a scalar.");
    }
    int n_col = *n_col_buf.typed_data();

    auto ds_Ai = Ai.dimensions();
    if (ds_Ai.size() != 1) return ffi::Error::InvalidArgument("Ai must be 1D.");
    int n_nz = ds_Ai[0];

    auto ds_Aj = Aj.dimensions();
    if (ds_Aj.size() != 1) return ffi::Error::InvalidArgument("Aj must be 1D.");
    if (ds_Aj[0] != n_nz) return ffi::Error::InvalidArgument("Aj size mismatch.");

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // Validate indices
    for (int n = 0; n < n_nz; n++) {
        if (_Ai[n] < 0 || _Ai[n] >= n_col) return ffi::Error::InvalidArgument("Ai index out of bounds.");
        if (_Aj[n] < 0 || _Aj[n] >= n_col) return ffi::Error::InvalidArgument("Aj index out of bounds.");
    }

    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bk = std::make_unique<int[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    klu_common Common;
    klu_defaults(&Common);
    klu_symbolic* Symbolic = klu_analyze(n_col, _Bp.get(), _Bi.get(), &Common);

    if (!Symbolic) {
        return ffi::Error::Internal("klu_analyze failed.");
    }

    *symbolic->typed_data() = reinterpret_cast<uint64_t>(Symbolic);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analyze_handler, analyze,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // n_col
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // symbolic ptr
);

// Python wrappers
PYBIND11_MODULE(klujax_cpp, m) {
    m.def("dot_f64",
          []() { return py::capsule((void*)&dot_f64_handler); });
    m.def("dot_c128",
          []() { return py::capsule((void*)&dot_c128_handler); });
    m.def("solve_f64",
          []() { return py::capsule((void*)&solve_f64_handler); });
    m.def("solve_c128",
          []() { return py::capsule((void*)&solve_c128_handler); });
    m.def("solve_with_symbol_f64",
          []() { return py::capsule((void*)&solve_with_symbol_f64_handler); });
    m.def("solve_with_symbol_c128",
          []() { return py::capsule((void*)&solve_with_symbol_c128_handler); });
    m.def("tsolve_with_symbol_f64",
          []() { return py::capsule((void*)&tsolve_with_symbol_f64_handler); });
    m.def("tsolve_with_symbol_c128",
          []() { return py::capsule((void*)&tsolve_with_symbol_c128_handler); });
    m.def("factor_f64",
          []() { return py::capsule((void*)&factor_f64_handler); });
    m.def("factor_c128",
          []() { return py::capsule((void*)&factor_c128_handler); });
    m.def("solve_with_numeric_f64",
          []() { return py::capsule((void*)&solve_with_numeric_f64_handler); });
    m.def("solve_with_numeric_c128",
          []() { return py::capsule((void*)&solve_with_numeric_c128_handler); });
    m.def("tsolve_with_numeric_f64",
          []() { return py::capsule((void*)&tsolve_with_numeric_f64_handler); });
    m.def("tsolve_with_numeric_c128",
          []() { return py::capsule((void*)&tsolve_with_numeric_c128_handler); });
    m.def("free_numeric",
          []() { return py::capsule((void*)&free_numeric_handler); });
    m.def("analyze",
          []() { return py::capsule((void*)&analyze_handler); });
    m.def("free_symbolic",
          []() { return py::capsule((void*)&free_symbolic_handler); });
    m.def("refactor_f64",
          []() { return py::capsule((void*)&refactor_f64_handler); });
    m.def("refactor_c128",
          []() { return py::capsule((void*)&refactor_c128_handler); });
    m.def("refactor_and_solve_f64",
          []() { return py::capsule((void*)&refactor_and_solve_f64_handler); });
    m.def("refactor_and_solve_c128",
          []() { return py::capsule((void*)&refactor_and_solve_c128_handler); });

    m.def("_numeric_allocation_counts", []() {
        return py::make_tuple(live_numeric_allocations.load(), total_numeric_allocations.load());
    });

    py::class_<KLUSymbolic>(m, "KLUSymbolic")
        .def(py::init<uint64_t>())
        .def_property_readonly("raw", &KLUSymbolic::raw)
        .def_property_readonly("handle", [](KLUSymbolic& s) -> KLUSymbolic& { return s; }, py::return_value_policy::reference)
        .def("close", &KLUSymbolic::close)
        .def("__enter__", [](KLUSymbolic& s) -> KLUSymbolic& { return s; }, py::return_value_policy::reference)
        .def("__exit__", [](KLUSymbolic& s, py::args) { s.close(); });

    py::class_<KLUNumeric>(m, "KLUNumeric")
        .def(py::init<std::vector<uint64_t>>())
        .def_property_readonly("size", &KLUNumeric::size)
        .def("as_list", &KLUNumeric::as_list)
        .def("close", &KLUNumeric::close)
        .def("__enter__", [](KLUNumeric& s) -> KLUNumeric& { return s; }, py::return_value_policy::reference)
        .def("__exit__", [](KLUNumeric& s, py::args) { s.close(); });
}
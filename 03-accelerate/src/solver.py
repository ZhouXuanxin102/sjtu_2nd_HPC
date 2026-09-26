from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any

import numpy as np

_LIB = None
_LIB_LOCK = threading.Lock()

_C_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

/* pi = PI_HI + PI_LO. PI_HI has trailing zeros so n*PI_HI stays exact for moderate n. */
static const double INV_PI = 0x1.45f306dc9c883p-2;
static const double PI_HI = 0x1.921fb54400000p+1;
static const double PI_LO = 0x1.0b4611a626331p-33;

static const double LOG2E = 1.44269504088896338700;
static const double LN2_HI = 6.93147180369123816490e-01;
static const double LN2_LO = 1.90821492927058770002e-10;

static inline double exp2_int(int n) {
    union {
        uint64_t u;
        double d;
    } v;
    v.u = (uint64_t)(n + 1023) << 52;
    return v.d;
}

__attribute__((always_inline)) static inline double sin_poly(double y) {
    double z = y * y;
    double u = -0x1.761b41316381ap-75;
    u = fma(u, z, 0x1.71b8ef6dcf572p-66);
    u = fma(u, z, -0x1.2f49b46814157p-57);
    u = fma(u, z, 0x1.952c77030ad4ap-49);
    u = fma(u, z, -0x1.ae7f3e733b81fp-41);
    u = fma(u, z, 0x1.6124613a86d09p-33);
    u = fma(u, z, -0x1.ae64567f544e4p-26);
    u = fma(u, z, 0x1.71de3a556c734p-19);
    u = fma(u, z, -0x1.a01a01a01a01ap-13);
    u = fma(u, z, 0x1.1111111111111p-7);
    u = fma(u, z, -0x1.5555555555555p-3);
    u = fma(u, z, 1.0);
    return y * u;
}

__attribute__((always_inline)) static inline double fast_sin(double x) {
    if (!isfinite(x) || fabs(x) > 1.0e6) {
        return sin(x);
    }
    double n_d = nearbyint(x * INV_PI);
    double r = (x - n_d * PI_HI) - n_d * PI_LO;
    if (fabs(r) > 1.6) {
        return sin(x);
    }
    double s = sin_poly(r);
    return ((int)n_d & 1) ? -s : s;
}

static inline double fast_exp(double x) {
    if (x < -150.0) {
        return 0.0;
    }
    if (!isfinite(x) || x > 709.0) {
        return exp(x);
    }
    double n_d = nearbyint(x * LOG2E);
    int n = (int)n_d;
    double r = (x - n_d * LN2_HI) - n_d * LN2_LO;
    double p = 1.0 / 87178291200.0;
    p = fma(p, r, 1.0 / 6227020800.0);
    p = fma(p, r, 1.0 / 479001600.0);
    p = fma(p, r, 1.0 / 39916800.0);
    p = fma(p, r, 1.0 / 3628800.0);
    p = fma(p, r, 1.0 / 362880.0);
    p = fma(p, r, 1.0 / 40320.0);
    p = fma(p, r, 1.0 / 5040.0);
    p = fma(p, r, 1.0 / 720.0);
    p = fma(p, r, 1.0 / 120.0);
    p = fma(p, r, 1.0 / 24.0);
    p = fma(p, r, 1.0 / 6.0);
    p = fma(p, r, 1.0 / 2.0);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    return p * exp2_int(n);
}

__attribute__((always_inline)) static inline void dots_pair(const double *x, const float *mu, const float *vv, int dimension,
                             double *dot_mu, double *dot_v) {
    double dm = 0.0;
    double dv = 0.0;
    int d = 0;
#if HAS_NEON
    float64x2_t sm0 = vdupq_n_f64(0.0);
    float64x2_t sm1 = vdupq_n_f64(0.0);
    float64x2_t sv0 = vdupq_n_f64(0.0);
    float64x2_t sv1 = vdupq_n_f64(0.0);
    for (; d + 3 < dimension; d += 4) {
        float64x2_t x0 = vld1q_f64(x + d);
        float64x2_t x1 = vld1q_f64(x + d + 2);
        float32x4_t m = vld1q_f32(mu + d);
        float32x4_t v = vld1q_f32(vv + d);
        float64x2_t diff0 = vsubq_f64(x0, vcvt_f64_f32(vget_low_f32(m)));
        float64x2_t diff1 = vsubq_f64(x1, vcvt_f64_f32(vget_high_f32(m)));
        sm0 = vfmaq_f64(sm0, diff0, diff0);
        sm1 = vfmaq_f64(sm1, diff1, diff1);
        sv0 = vfmaq_f64(sv0, x0, vcvt_f64_f32(vget_low_f32(v)));
        sv1 = vfmaq_f64(sv1, x1, vcvt_f64_f32(vget_high_f32(v)));
    }
    sm0 = vaddq_f64(sm0, sm1);
    sv0 = vaddq_f64(sv0, sv1);
    dm = vgetq_lane_f64(sm0, 0) + vgetq_lane_f64(sm0, 1);
    dv = vgetq_lane_f64(sv0, 0) + vgetq_lane_f64(sv0, 1);
#endif
    for (; d < dimension; ++d) {
        double xd = x[d];
        double diff = xd - (double)mu[d];
        dm += diff * diff;
        dv += xd * (double)vv[d];
    }
    *dot_mu = dm;
    *dot_v = dv;
}

static inline double term_from_dots(double r2, double dot_v, float weight, float scale, float bias,
                                    float trig_scale, double exp_arg_limit) {
    if (r2 < 0.0) {
        r2 = 0.0;
    }
    double arg = -(double)scale * r2;
    double gaussian = arg < exp_arg_limit ? 0.0 : (double)weight * fast_exp(arg);
    double sine = (double)bias * fast_sin((double)trig_scale * dot_v);
    return gaussian + sine;
}

void compute_field_omp(const float *points, const float *centers, const float *weights,
                       const float *scales, const float *bias, const float *trig_scale,
                       const float *trig_vec, float *out, int q_count, int c_count,
                       int dimension) {
    if (q_count <= 0) {
        return;
    }
    if (c_count <= 0 || dimension < 0) {
        memset(out, 0, (size_t)q_count * sizeof(float));
        return;
    }

    double *exp_arg_limit = (double *)malloc((size_t)c_count * sizeof(double));
    if (exp_arg_limit == NULL) {
        memset(out, 0, (size_t)q_count * sizeof(float));
        return;
    }
    for (int c = 0; c < c_count; ++c) {
        double aw = fabs((double)weights[c]);
        exp_arg_limit[c] = aw < 1e-300 ? 0.0 : -42.0 - log(aw);
    }

#pragma omp parallel for schedule(static)
    for (int q = 0; q < q_count; ++q) {
        const float *point = points + (size_t)q * (size_t)dimension;
        double x_stack[64];
        double *x = x_stack;
        double *heap_x = NULL;
        if (dimension > 64) {
            heap_x = (double *)malloc((size_t)dimension * sizeof(double));
            x = heap_x;
        }
        for (int d = 0; d < dimension; ++d) {
            x[d] = (double)point[d];
        }
        double acc = 0.0;
        int c = 0;
        for (; c + 3 < c_count; c += 4) {
            double r0, v0, r1, v1, r2, v2, r3, v3;
            const size_t stride = (size_t)dimension;
            dots_pair(x, centers + (size_t)c * stride, trig_vec + (size_t)c * stride, dimension, &r0, &v0);
            dots_pair(x, centers + (size_t)(c + 1) * stride, trig_vec + (size_t)(c + 1) * stride, dimension, &r1, &v1);
            dots_pair(x, centers + (size_t)(c + 2) * stride, trig_vec + (size_t)(c + 2) * stride, dimension, &r2, &v2);
            dots_pair(x, centers + (size_t)(c + 3) * stride, trig_vec + (size_t)(c + 3) * stride, dimension, &r3, &v3);
            double s0 = fast_sin((double)trig_scale[c] * v0);
            double s1 = fast_sin((double)trig_scale[c + 1] * v1);
            double s2 = fast_sin((double)trig_scale[c + 2] * v2);
            double s3 = fast_sin((double)trig_scale[c + 3] * v3);
            double a0 = -(double)scales[c] * (r0 < 0.0 ? 0.0 : r0);
            double a1 = -(double)scales[c + 1] * (r1 < 0.0 ? 0.0 : r1);
            double a2 = -(double)scales[c + 2] * (r2 < 0.0 ? 0.0 : r2);
            double a3 = -(double)scales[c + 3] * (r3 < 0.0 ? 0.0 : r3);
            double g0 = a0 < exp_arg_limit[c] ? 0.0 : (double)weights[c] * fast_exp(a0);
            double g1 = a1 < exp_arg_limit[c + 1] ? 0.0 : (double)weights[c + 1] * fast_exp(a1);
            double g2 = a2 < exp_arg_limit[c + 2] ? 0.0 : (double)weights[c + 2] * fast_exp(a2);
            double g3 = a3 < exp_arg_limit[c + 3] ? 0.0 : (double)weights[c + 3] * fast_exp(a3);
            acc += g0 + (double)bias[c] * s0;
            acc += g1 + (double)bias[c + 1] * s1;
            acc += g2 + (double)bias[c + 2] * s2;
            acc += g3 + (double)bias[c + 3] * s3;
        }
        for (; c + 1 < c_count; c += 2) {
            double r0, v0, r1, v1;
            dots_pair(x, centers + (size_t)c * (size_t)dimension,
                      trig_vec + (size_t)c * (size_t)dimension, dimension, &r0, &v0);
            dots_pair(x, centers + (size_t)(c + 1) * (size_t)dimension,
                      trig_vec + (size_t)(c + 1) * (size_t)dimension, dimension, &r1, &v1);
            acc += term_from_dots(r0, v0, weights[c], scales[c], bias[c], trig_scale[c],
                                  exp_arg_limit[c]);
            acc += term_from_dots(r1, v1, weights[c + 1], scales[c + 1], bias[c + 1],
                                  trig_scale[c + 1], exp_arg_limit[c + 1]);
        }
        if (c < c_count) {
            double r0, v0;
            dots_pair(x, centers + (size_t)c * (size_t)dimension,
                      trig_vec + (size_t)c * (size_t)dimension, dimension, &r0, &v0);
            acc += term_from_dots(r0, v0, weights[c], scales[c], bias[c], trig_scale[c],
                                  exp_arg_limit[c]);
        }
        out[q] = (float)acc;
        free(heap_x);
    }
    free(exp_arg_limit);
}


"""


def _library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
        directory = Path(tmp) / f"kernel-field-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / "field_kernel.c"
        binary = directory / "field_kernel.so"
        source.write_text(_C_SOURCE, encoding="utf-8")
        command = [
            "gcc",
            "-O3",
            "-march=native",
            "-fopenmp",
            "-fPIC",
            "-shared",
            "-std=c11",
            "-ffp-contract=off",
            "-o",
            str(binary),
            str(source),
            "-lm",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "gcc failed").strip()
            raise RuntimeError(detail)
        lib = ctypes.CDLL(str(binary))
        lib.compute_field_omp.restype = None
        lib.compute_field_omp.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _LIB = lib
        return lib


def compute_field(
    points: Any,
    centers: Any,
    weights: Any,
    scales: Any,
    bias: Any,
    trig_scale: Any,
    trig_vec: Any,
) -> np.ndarray:
    """Field response. The numeric kernel is compiled from the C source below on first call."""
    q_count, dimension = _shape_2d(points, "points")
    c_count, center_dimension = _shape_2d(centers, "centers")
    trig_rows, trig_dimension = _shape_2d(trig_vec, "trig_vec")
    _shape_1d(weights, "weights", expected=c_count)
    _shape_1d(scales, "scales", expected=c_count)
    _shape_1d(bias, "bias", expected=c_count)
    _shape_1d(trig_scale, "trig_scale", expected=c_count)
    if center_dimension != dimension:
        raise ValueError(f"centers has dim {center_dimension}, expected {dimension}")
    if trig_rows != c_count or trig_dimension != dimension:
        raise ValueError(
            f"trig_vec must have shape ({c_count}, {dimension}), "
            f"got ({trig_rows}, {trig_dimension})"
        )

    points_a = np.ascontiguousarray(points, dtype=np.float32)
    centers_a = np.ascontiguousarray(centers, dtype=np.float32)
    weights_a = np.ascontiguousarray(weights, dtype=np.float32)
    scales_a = np.ascontiguousarray(scales, dtype=np.float32)
    bias_a = np.ascontiguousarray(bias, dtype=np.float32)
    trig_scale_a = np.ascontiguousarray(trig_scale, dtype=np.float32)
    trig_vec_a = np.ascontiguousarray(trig_vec, dtype=np.float32)
    output = np.empty(q_count, dtype=np.float32)
    if q_count == 0:
        return output
    pointer = ctypes.POINTER(ctypes.c_float)
    _library().compute_field_omp(
        points_a.ctypes.data_as(pointer),
        centers_a.ctypes.data_as(pointer),
        weights_a.ctypes.data_as(pointer),
        scales_a.ctypes.data_as(pointer),
        bias_a.ctypes.data_as(pointer),
        trig_scale_a.ctypes.data_as(pointer),
        trig_vec_a.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer),
        int(q_count),
        int(c_count),
        int(dimension),
    )
    return output


def _shape_2d(value: Any, name: str) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"{name} must be 2D, got shape {shape}")
        return int(shape[0]), int(shape[1])
    rows = len(value)
    columns = len(value[0]) if rows else 0
    if any(len(row) != columns for row in value):
        raise ValueError(f"{name} must be rectangular")
    return rows, columns


def _shape_1d(value: Any, name: str, expected: int | None = None) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 1:
            raise ValueError(f"{name} must be 1D, got shape {shape}")
        size = int(shape[0])
    else:
        size = len(value)
    if expected is not None and size != expected:
        raise ValueError(f"{name} must have length {expected}, got {size}")
    return size

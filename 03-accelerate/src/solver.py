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
#else
static int omp_get_thread_num(void) { return 0; }
static int omp_get_num_threads(void) { return 1; }
#endif

/* Query panel kept in L2 while a thread streams center panels. */
enum { QUERY_BLOCK = 256, MR = 4, NR = 8 };

static const double INV_PI = 0x1.45f306dc9c883p-2;
static const double PI_HI = 0x1.921fb54400000p+1;
static const double PI_LO = 0x1.0b4611a626331p-33;
static const double LOG2E = 1.44269504088896338700;
static const double LN2_HI = 6.93147180369123816490e-01;
static const double LN2_LO = 1.90821492927058770002e-10;

static void *xalloc(size_t bytes) {
    if (bytes == 0) {
        bytes = 64;
    }
    bytes = (bytes + 63u) & ~(size_t)63u;
    return aligned_alloc(64, bytes);
}

static inline double exp_limit(float weight) {
    double aw = fabs((double)weight);
    if (aw < 1e-300) {
        return 0.0;
    }
    return -42.0 - log(aw);
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

static inline double exp2_int(int n) {
    union {
        uint64_t u;
        double d;
    } v;
    v.u = (uint64_t)(n + 1023) << 52;
    return v.d;
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

static inline double pair_term(const float *x, const float *mu, const float *vv, int dimension,
                               float weight, float scale, float bias, float trig_scale) {
    double r = 0.0;
    double p = 0.0;
    for (int k = 0; k < dimension; ++k) {
        double xd = (double)x[k];
        double diff = xd - (double)mu[k];
        r = fma(diff, diff, r);
        p = fma(xd, (double)vv[k], p);
    }
    double arg = -(double)scale * r;
    double limit = exp_limit(weight);
    double gaussian = arg < limit ? 0.0 : (double)weight * fast_exp(arg);
    return gaussian + (double)bias * fast_sin((double)trig_scale * p);
}

static void compute_scalar_all(const float *points, const float *centers, const float *weights,
                               const float *scales, const float *bias, const float *trig_scale,
                               const float *trig_vec, float *out, int q_count, int c_count,
                               int dimension) {
#pragma omp parallel for schedule(static)
    for (int q = 0; q < q_count; ++q) {
        const float *x = points + (size_t)q * (size_t)dimension;
        double sum = 0.0;
        for (int c = 0; c < c_count; ++c) {
            sum += pair_term(x, centers + (size_t)c * (size_t)dimension,
                             trig_vec + (size_t)c * (size_t)dimension, dimension, weights[c],
                             scales[c], bias[c], trig_scale[c]);
        }
        out[q] = (float)sum;
    }
}

#if HAS_NEON

static inline float64x2_t make2(double a, double b) {
    float64x2_t v = vdupq_n_f64(a);
    return vsetq_lane_f64(b, v, 1);
}

static inline float64x2_t exp2_vec(float64x2_t n_d) {
    /* Fast path only. n is inside the normal exponent range there. */
    int64x2_t n = vcvtq_s64_f64(n_d);
    int64x2_t bits = vshlq_n_s64(vaddq_s64(n, vdupq_n_s64(1023)), 52);
    return vreinterpretq_f64_s64(bits);
}

static inline void fast_exp_pair(float64x2_t x0, float64x2_t x1, float64x2_t *o0, float64x2_t *o1) {
    if (!(vmaxvq_f64(vmaxq_f64(x0, x1)) <= 709.0)) {
        *o0 = make2(fast_exp(vgetq_lane_f64(x0, 0)), fast_exp(vgetq_lane_f64(x0, 1)));
        *o1 = make2(fast_exp(vgetq_lane_f64(x1, 0)), fast_exp(vgetq_lane_f64(x1, 1)));
        return;
    }
    float64x2_t log2e = vdupq_n_f64(LOG2E);
    float64x2_t n0 = vrndnq_f64(vmulq_f64(x0, log2e));
    float64x2_t n1 = vrndnq_f64(vmulq_f64(x1, log2e));
    float64x2_t hi = vdupq_n_f64(LN2_HI);
    float64x2_t lo = vdupq_n_f64(LN2_LO);
    float64x2_t r0 = vsubq_f64(vsubq_f64(x0, vmulq_f64(n0, hi)), vmulq_f64(n0, lo));
    float64x2_t r1 = vsubq_f64(vsubq_f64(x1, vmulq_f64(n1, hi)), vmulq_f64(n1, lo));
    float64x2_t p0 = vdupq_n_f64(1.0 / 87178291200.0);
    float64x2_t p1 = p0;
#define EP(coef)                                      \
    p0 = vfmaq_f64(vdupq_n_f64(coef), p0, r0);        \
    p1 = vfmaq_f64(vdupq_n_f64(coef), p1, r1);
    EP(1.0 / 6227020800.0)
    EP(1.0 / 479001600.0)
    EP(1.0 / 39916800.0)
    EP(1.0 / 3628800.0)
    EP(1.0 / 362880.0)
    EP(1.0 / 40320.0)
    EP(1.0 / 5040.0)
    EP(1.0 / 720.0)
    EP(1.0 / 120.0)
    EP(1.0 / 24.0)
    EP(1.0 / 6.0)
    EP(1.0 / 2.0)
    EP(1.0)
    EP(1.0)
#undef EP
    p0 = vmulq_f64(p0, exp2_vec(n0));
    p1 = vmulq_f64(p1, exp2_vec(n1));
    uint64x2_t z0 = vcltq_f64(x0, vdupq_n_f64(-150.0));
    uint64x2_t z1 = vcltq_f64(x1, vdupq_n_f64(-150.0));
    float64x2_t zero = vdupq_n_f64(0.0);
    *o0 = vbslq_f64(z0, zero, p0);
    *o1 = vbslq_f64(z1, zero, p1);
}

static inline void fast_sin_pair(float64x2_t x0, float64x2_t x1, float64x2_t *o0, float64x2_t *o1) {
    float64x2_t amax = vmaxq_f64(vabsq_f64(x0), vabsq_f64(x1));
    if (!(vmaxvq_f64(amax) <= 1.0e6)) {
        *o0 = make2(fast_sin(vgetq_lane_f64(x0, 0)), fast_sin(vgetq_lane_f64(x0, 1)));
        *o1 = make2(fast_sin(vgetq_lane_f64(x1, 0)), fast_sin(vgetq_lane_f64(x1, 1)));
        return;
    }
    float64x2_t inv_pi = vdupq_n_f64(INV_PI);
    float64x2_t n0 = vrndnq_f64(vmulq_f64(x0, inv_pi));
    float64x2_t n1 = vrndnq_f64(vmulq_f64(x1, inv_pi));
    float64x2_t hi = vdupq_n_f64(PI_HI);
    float64x2_t lo = vdupq_n_f64(PI_LO);
    float64x2_t r0 = vsubq_f64(vsubq_f64(x0, vmulq_f64(n0, hi)), vmulq_f64(n0, lo));
    float64x2_t r1 = vsubq_f64(vsubq_f64(x1, vmulq_f64(n1, hi)), vmulq_f64(n1, lo));
    if (vmaxvq_f64(vmaxq_f64(vabsq_f64(r0), vabsq_f64(r1))) > 1.6) {
        *o0 = make2(fast_sin(vgetq_lane_f64(x0, 0)), fast_sin(vgetq_lane_f64(x0, 1)));
        *o1 = make2(fast_sin(vgetq_lane_f64(x1, 0)), fast_sin(vgetq_lane_f64(x1, 1)));
        return;
    }
    float64x2_t z0 = vmulq_f64(r0, r0);
    float64x2_t z1 = vmulq_f64(r1, r1);
    float64x2_t u0 = vdupq_n_f64(-0x1.761b41316381ap-75);
    float64x2_t u1 = u0;
#define SP(coef)                                      \
    u0 = vfmaq_f64(vdupq_n_f64(coef), u0, z0);        \
    u1 = vfmaq_f64(vdupq_n_f64(coef), u1, z1);
    SP(0x1.71b8ef6dcf572p-66)
    SP(-0x1.2f49b46814157p-57)
    SP(0x1.952c77030ad4ap-49)
    SP(-0x1.ae7f3e733b81fp-41)
    SP(0x1.6124613a86d09p-33)
    SP(-0x1.ae64567f544e4p-26)
    SP(0x1.71de3a556c734p-19)
    SP(-0x1.a01a01a01a01ap-13)
    SP(0x1.1111111111111p-7)
    SP(-0x1.5555555555555p-3)
    SP(1.0)
#undef SP
    u0 = vmulq_f64(r0, u0);
    u1 = vmulq_f64(r1, u1);
    int64x2_t i0 = vcvtq_s64_f64(n0);
    int64x2_t i1 = vcvtq_s64_f64(n1);
    uint64x2_t odd0 = vandq_u64(vreinterpretq_u64_s64(i0), vdupq_n_u64(1));
    uint64x2_t odd1 = vandq_u64(vreinterpretq_u64_s64(i1), vdupq_n_u64(1));
    uint64x2_t sign0 = vshlq_n_u64(odd0, 63);
    uint64x2_t sign1 = vshlq_n_u64(odd1, 63);
    *o0 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(u0), sign0));
    *o1 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(u1), sign1));
}

static inline double row_sum(double nx, float64x2_t dmu0, float64x2_t dmu1, float64x2_t dv0,
                             float64x2_t dv1, const double *nmu, const float *weight, const float *scale,
                             const float *bias, const float *trig_scale, const double *elim) {
    float64x2_t nxv = vdupq_n_f64(nx);
    float64x2_t nm0 = vld1q_f64(nmu);
    float64x2_t nm1 = vld1q_f64(nmu + 2);
    float64x2_t two = vdupq_n_f64(2.0);
    float64x2_t r0 = vfmsq_f64(vaddq_f64(nxv, nm0), dmu0, two);
    float64x2_t r1 = vfmsq_f64(vaddq_f64(nxv, nm1), dmu1, two);
    float64x2_t zero = vdupq_n_f64(0.0);
    r0 = vmaxq_f64(r0, zero);
    r1 = vmaxq_f64(r1, zero);

    float32x4_t sc32 = vld1q_f32(scale);
    float64x2_t sc0 = vcvt_f64_f32(vget_low_f32(sc32));
    float64x2_t sc1 = vcvt_f64_f32(vget_high_f32(sc32));
    float64x2_t arg0 = vnegq_f64(vmulq_f64(sc0, r0));
    float64x2_t arg1 = vnegq_f64(vmulq_f64(sc1, r1));
    float64x2_t e0, e1;
    fast_exp_pair(arg0, arg1, &e0, &e1);
    uint64x2_t m0 = vcltq_f64(arg0, vld1q_f64(elim));
    uint64x2_t m1 = vcltq_f64(arg1, vld1q_f64(elim + 2));
    float32x4_t w32 = vld1q_f32(weight);
    float64x2_t w0 = vcvt_f64_f32(vget_low_f32(w32));
    float64x2_t w1 = vcvt_f64_f32(vget_high_f32(w32));
    float64x2_t g0 = vbslq_f64(m0, zero, vmulq_f64(w0, e0));
    float64x2_t g1 = vbslq_f64(m1, zero, vmulq_f64(w1, e1));

    float32x4_t ts32 = vld1q_f32(trig_scale);
    float64x2_t ts0 = vcvt_f64_f32(vget_low_f32(ts32));
    float64x2_t ts1 = vcvt_f64_f32(vget_high_f32(ts32));
    float64x2_t ang0 = vmulq_f64(ts0, dv0);
    float64x2_t ang1 = vmulq_f64(ts1, dv1);
    float64x2_t s0, s1;
    fast_sin_pair(ang0, ang1, &s0, &s1);
    float32x4_t b32 = vld1q_f32(bias);
    float64x2_t b0 = vcvt_f64_f32(vget_low_f32(b32));
    float64x2_t b1 = vcvt_f64_f32(vget_high_f32(b32));
    float64x2_t h0 = vfmaq_f64(g0, b0, s0);
    float64x2_t h1 = vfmaq_f64(g1, b1, s1);
    float64x2_t h = vaddq_f64(h0, h1);
    return vgetq_lane_f64(h, 0) + vgetq_lane_f64(h, 1);
}

static inline void fast_exp4(float64x2_t x0, float64x2_t x1, float64x2_t x2, float64x2_t x3,
                             float64x2_t *o0, float64x2_t *o1, float64x2_t *o2, float64x2_t *o3) {
    float64x2_t top = vmaxq_f64(vmaxq_f64(x0, x1), vmaxq_f64(x2, x3));
    if (!(vmaxvq_f64(top) <= 709.0)) {
        fast_exp_pair(x0, x1, o0, o1);
        fast_exp_pair(x2, x3, o2, o3);
        return;
    }
    float64x2_t log2e = vdupq_n_f64(LOG2E);
    float64x2_t hi = vdupq_n_f64(LN2_HI);
    float64x2_t lo = vdupq_n_f64(LN2_LO);
    float64x2_t n0 = vrndnq_f64(vmulq_f64(x0, log2e));
    float64x2_t n1 = vrndnq_f64(vmulq_f64(x1, log2e));
    float64x2_t n2 = vrndnq_f64(vmulq_f64(x2, log2e));
    float64x2_t n3 = vrndnq_f64(vmulq_f64(x3, log2e));
    float64x2_t r0 = vsubq_f64(vsubq_f64(x0, vmulq_f64(n0, hi)), vmulq_f64(n0, lo));
    float64x2_t r1 = vsubq_f64(vsubq_f64(x1, vmulq_f64(n1, hi)), vmulq_f64(n1, lo));
    float64x2_t r2 = vsubq_f64(vsubq_f64(x2, vmulq_f64(n2, hi)), vmulq_f64(n2, lo));
    float64x2_t r3 = vsubq_f64(vsubq_f64(x3, vmulq_f64(n3, hi)), vmulq_f64(n3, lo));
    float64x2_t p0 = vdupq_n_f64(1.0 / 87178291200.0);
    float64x2_t p1 = p0;
    float64x2_t p2 = p0;
    float64x2_t p3 = p0;
#define EP4(coef)                               \
    p0 = vfmaq_f64(vdupq_n_f64(coef), p0, r0);  \
    p1 = vfmaq_f64(vdupq_n_f64(coef), p1, r1);  \
    p2 = vfmaq_f64(vdupq_n_f64(coef), p2, r2);  \
    p3 = vfmaq_f64(vdupq_n_f64(coef), p3, r3);
    EP4(1.0 / 6227020800.0)
    EP4(1.0 / 479001600.0)
    EP4(1.0 / 39916800.0)
    EP4(1.0 / 3628800.0)
    EP4(1.0 / 362880.0)
    EP4(1.0 / 40320.0)
    EP4(1.0 / 5040.0)
    EP4(1.0 / 720.0)
    EP4(1.0 / 120.0)
    EP4(1.0 / 24.0)
    EP4(1.0 / 6.0)
    EP4(1.0 / 2.0)
    EP4(1.0)
    EP4(1.0)
#undef EP4
    p0 = vmulq_f64(p0, exp2_vec(n0));
    p1 = vmulq_f64(p1, exp2_vec(n1));
    p2 = vmulq_f64(p2, exp2_vec(n2));
    p3 = vmulq_f64(p3, exp2_vec(n3));
    float64x2_t zero = vdupq_n_f64(0.0);
    float64x2_t bound = vdupq_n_f64(-150.0);
    *o0 = vbslq_f64(vcltq_f64(x0, bound), zero, p0);
    *o1 = vbslq_f64(vcltq_f64(x1, bound), zero, p1);
    *o2 = vbslq_f64(vcltq_f64(x2, bound), zero, p2);
    *o3 = vbslq_f64(vcltq_f64(x3, bound), zero, p3);
}

static inline void fast_sin4(float64x2_t x0, float64x2_t x1, float64x2_t x2, float64x2_t x3,
                             float64x2_t *o0, float64x2_t *o1, float64x2_t *o2, float64x2_t *o3) {
    float64x2_t amax = vmaxq_f64(vmaxq_f64(vabsq_f64(x0), vabsq_f64(x1)),
                                 vmaxq_f64(vabsq_f64(x2), vabsq_f64(x3)));
    if (!(vmaxvq_f64(amax) <= 1.0e6)) {
        fast_sin_pair(x0, x1, o0, o1);
        fast_sin_pair(x2, x3, o2, o3);
        return;
    }
    float64x2_t inv_pi = vdupq_n_f64(INV_PI);
    float64x2_t hi = vdupq_n_f64(PI_HI);
    float64x2_t lo = vdupq_n_f64(PI_LO);
    float64x2_t n0 = vrndnq_f64(vmulq_f64(x0, inv_pi));
    float64x2_t n1 = vrndnq_f64(vmulq_f64(x1, inv_pi));
    float64x2_t n2 = vrndnq_f64(vmulq_f64(x2, inv_pi));
    float64x2_t n3 = vrndnq_f64(vmulq_f64(x3, inv_pi));
    float64x2_t r0 = vsubq_f64(vsubq_f64(x0, vmulq_f64(n0, hi)), vmulq_f64(n0, lo));
    float64x2_t r1 = vsubq_f64(vsubq_f64(x1, vmulq_f64(n1, hi)), vmulq_f64(n1, lo));
    float64x2_t r2 = vsubq_f64(vsubq_f64(x2, vmulq_f64(n2, hi)), vmulq_f64(n2, lo));
    float64x2_t r3 = vsubq_f64(vsubq_f64(x3, vmulq_f64(n3, hi)), vmulq_f64(n3, lo));
    float64x2_t rmax = vmaxq_f64(vmaxq_f64(vabsq_f64(r0), vabsq_f64(r1)),
                                 vmaxq_f64(vabsq_f64(r2), vabsq_f64(r3)));
    if (vmaxvq_f64(rmax) > 1.6) {
        fast_sin_pair(x0, x1, o0, o1);
        fast_sin_pair(x2, x3, o2, o3);
        return;
    }
    float64x2_t z0 = vmulq_f64(r0, r0);
    float64x2_t z1 = vmulq_f64(r1, r1);
    float64x2_t z2 = vmulq_f64(r2, r2);
    float64x2_t z3 = vmulq_f64(r3, r3);
    float64x2_t u0 = vdupq_n_f64(-0x1.761b41316381ap-75);
    float64x2_t u1 = u0;
    float64x2_t u2 = u0;
    float64x2_t u3 = u0;
#define SP4(coef)                               \
    u0 = vfmaq_f64(vdupq_n_f64(coef), u0, z0);  \
    u1 = vfmaq_f64(vdupq_n_f64(coef), u1, z1);  \
    u2 = vfmaq_f64(vdupq_n_f64(coef), u2, z2);  \
    u3 = vfmaq_f64(vdupq_n_f64(coef), u3, z3);
    SP4(0x1.71b8ef6dcf572p-66)
    SP4(-0x1.2f49b46814157p-57)
    SP4(0x1.952c77030ad4ap-49)
    SP4(-0x1.ae7f3e733b81fp-41)
    SP4(0x1.6124613a86d09p-33)
    SP4(-0x1.ae64567f544e4p-26)
    SP4(0x1.71de3a556c734p-19)
    SP4(-0x1.a01a01a01a01ap-13)
    SP4(0x1.1111111111111p-7)
    SP4(-0x1.5555555555555p-3)
    SP4(1.0)
#undef SP4
    u0 = vmulq_f64(r0, u0);
    u1 = vmulq_f64(r1, u1);
    u2 = vmulq_f64(r2, u2);
    u3 = vmulq_f64(r3, u3);
    uint64x2_t one = vdupq_n_u64(1);
    uint64x2_t s0 = vshlq_n_u64(vandq_u64(vreinterpretq_u64_s64(vcvtq_s64_f64(n0)), one), 63);
    uint64x2_t s1 = vshlq_n_u64(vandq_u64(vreinterpretq_u64_s64(vcvtq_s64_f64(n1)), one), 63);
    uint64x2_t s2 = vshlq_n_u64(vandq_u64(vreinterpretq_u64_s64(vcvtq_s64_f64(n2)), one), 63);
    uint64x2_t s3 = vshlq_n_u64(vandq_u64(vreinterpretq_u64_s64(vcvtq_s64_f64(n3)), one), 63);
    *o0 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(u0), s0));
    *o1 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(u1), s1));
    *o2 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(u2), s2));
    *o3 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(u3), s3));
}

static inline double horiz4(float64x2_t a, float64x2_t b, float64x2_t c, float64x2_t d) {
    float64x2_t s = vaddq_f64(vaddq_f64(a, b), vaddq_f64(c, d));
    return vgetq_lane_f64(s, 0) + vgetq_lane_f64(s, 1);
}

static inline double row_sum8(double nx, float32x4_t mu_lo, float32x4_t mu_hi, const double *pdot,
                              const double *nmu, const float *weight, const float *scale,
                              const float *bias, const float *trig_scale, const double *elim) {
    float64x2_t nxv = vdupq_n_f64(nx);
    float64x2_t two = vdupq_n_f64(2.0);
    float64x2_t zero = vdupq_n_f64(0.0);
    float64x2_t d0 = vcvt_f64_f32(vget_low_f32(mu_lo));
    float64x2_t d1 = vcvt_f64_f32(vget_high_f32(mu_lo));
    float64x2_t d2 = vcvt_f64_f32(vget_low_f32(mu_hi));
    float64x2_t d3 = vcvt_f64_f32(vget_high_f32(mu_hi));
    float64x2_t r0 = vfmsq_f64(vaddq_f64(nxv, vld1q_f64(nmu)), d0, two);
    float64x2_t r1 = vfmsq_f64(vaddq_f64(nxv, vld1q_f64(nmu + 2)), d1, two);
    float64x2_t r2 = vfmsq_f64(vaddq_f64(nxv, vld1q_f64(nmu + 4)), d2, two);
    float64x2_t r3 = vfmsq_f64(vaddq_f64(nxv, vld1q_f64(nmu + 6)), d3, two);
    r0 = vmaxq_f64(r0, zero);
    r1 = vmaxq_f64(r1, zero);
    r2 = vmaxq_f64(r2, zero);
    r3 = vmaxq_f64(r3, zero);
    float32x4_t sc_lo = vld1q_f32(scale);
    float32x4_t sc_hi = vld1q_f32(scale + 4);
    float64x2_t arg0 = vnegq_f64(vmulq_f64(vcvt_f64_f32(vget_low_f32(sc_lo)), r0));
    float64x2_t arg1 = vnegq_f64(vmulq_f64(vcvt_f64_f32(vget_high_f32(sc_lo)), r1));
    float64x2_t arg2 = vnegq_f64(vmulq_f64(vcvt_f64_f32(vget_low_f32(sc_hi)), r2));
    float64x2_t arg3 = vnegq_f64(vmulq_f64(vcvt_f64_f32(vget_high_f32(sc_hi)), r3));
    float64x2_t e0, e1, e2, e3;
    fast_exp4(arg0, arg1, arg2, arg3, &e0, &e1, &e2, &e3);
    e0 = vbslq_f64(vcltq_f64(arg0, vld1q_f64(elim)), zero, e0);
    e1 = vbslq_f64(vcltq_f64(arg1, vld1q_f64(elim + 2)), zero, e1);
    e2 = vbslq_f64(vcltq_f64(arg2, vld1q_f64(elim + 4)), zero, e2);
    e3 = vbslq_f64(vcltq_f64(arg3, vld1q_f64(elim + 6)), zero, e3);
    float32x4_t w_lo = vld1q_f32(weight);
    float32x4_t w_hi = vld1q_f32(weight + 4);
    float64x2_t g0 = vmulq_f64(vcvt_f64_f32(vget_low_f32(w_lo)), e0);
    float64x2_t g1 = vmulq_f64(vcvt_f64_f32(vget_high_f32(w_lo)), e1);
    float64x2_t g2 = vmulq_f64(vcvt_f64_f32(vget_low_f32(w_hi)), e2);
    float64x2_t g3 = vmulq_f64(vcvt_f64_f32(vget_high_f32(w_hi)), e3);
    float64x2_t p0 = vld1q_f64(pdot);
    float64x2_t p1 = vld1q_f64(pdot + 2);
    float64x2_t p2 = vld1q_f64(pdot + 4);
    float64x2_t p3 = vld1q_f64(pdot + 6);
    float32x4_t ts_lo = vld1q_f32(trig_scale);
    float32x4_t ts_hi = vld1q_f32(trig_scale + 4);
    float64x2_t ang0 = vmulq_f64(vcvt_f64_f32(vget_low_f32(ts_lo)), p0);
    float64x2_t ang1 = vmulq_f64(vcvt_f64_f32(vget_high_f32(ts_lo)), p1);
    float64x2_t ang2 = vmulq_f64(vcvt_f64_f32(vget_low_f32(ts_hi)), p2);
    float64x2_t ang3 = vmulq_f64(vcvt_f64_f32(vget_high_f32(ts_hi)), p3);
    float64x2_t s0, s1, s2, s3;
    fast_sin4(ang0, ang1, ang2, ang3, &s0, &s1, &s2, &s3);
    float32x4_t b_lo = vld1q_f32(bias);
    float32x4_t b_hi = vld1q_f32(bias + 4);
    g0 = vfmaq_f64(g0, vcvt_f64_f32(vget_low_f32(b_lo)), s0);
    g1 = vfmaq_f64(g1, vcvt_f64_f32(vget_high_f32(b_lo)), s1);
    g2 = vfmaq_f64(g2, vcvt_f64_f32(vget_low_f32(b_hi)), s2);
    g3 = vfmaq_f64(g3, vcvt_f64_f32(vget_high_f32(b_hi)), s3);
    return horiz4(g0, g1, g2, g3);
}

#define DECLARE_MU()                        \
    float32x4_t dm0_0 = vdupq_n_f32(0.0f);  \
    float32x4_t dm0_1 = vdupq_n_f32(0.0f);  \
    float32x4_t dm1_0 = vdupq_n_f32(0.0f);  \
    float32x4_t dm1_1 = vdupq_n_f32(0.0f);  \
    float32x4_t dm2_0 = vdupq_n_f32(0.0f);  \
    float32x4_t dm2_1 = vdupq_n_f32(0.0f);  \
    float32x4_t dm3_0 = vdupq_n_f32(0.0f);  \
    float32x4_t dm3_1 = vdupq_n_f32(0.0f);

#define MSTEP(kk)                                                                                  \
    do {                                                                                           \
        const float *ak = a + (size_t)(kk) * (size_t)QUERY_BLOCK;                                  \
        const float *mk = mu + (size_t)(kk) * (size_t)b_stride;                                    \
        float32x4_t avec = vld1q_f32(ak);                                                          \
        float32x4_t m0 = vld1q_f32(mk);                                                            \
        float32x4_t m1 = vld1q_f32(mk + 4);                                                        \
        float32x4_t t;                                                                             \
        t = vdupq_laneq_f32(avec, 0);                                                              \
        dm0_0 = vfmaq_f32(dm0_0, t, m0);                                                           \
        dm0_1 = vfmaq_f32(dm0_1, t, m1);                                                           \
        t = vdupq_laneq_f32(avec, 1);                                                              \
        dm1_0 = vfmaq_f32(dm1_0, t, m0);                                                           \
        dm1_1 = vfmaq_f32(dm1_1, t, m1);                                                           \
        t = vdupq_laneq_f32(avec, 2);                                                              \
        dm2_0 = vfmaq_f32(dm2_0, t, m0);                                                           \
        dm2_1 = vfmaq_f32(dm2_1, t, m1);                                                           \
        t = vdupq_laneq_f32(avec, 3);                                                              \
        dm3_0 = vfmaq_f32(dm3_0, t, m0);                                                           \
        dm3_1 = vfmaq_f32(dm3_1, t, m1);                                                           \
    } while (0)

#define M4(base) MSTEP(base); MSTEP((base) + 1); MSTEP((base) + 2); MSTEP((base) + 3)
#define M16(base) M4(base); M4((base) + 4); M4((base) + 8); M4((base) + 12)

#define STORE_MU()                              \
    vst1q_f32(dmu + 0, dm0_0);                  \
    vst1q_f32(dmu + 4, dm0_1);                  \
    vst1q_f32(dmu + 8, dm1_0);                  \
    vst1q_f32(dmu + 12, dm1_1);                 \
    vst1q_f32(dmu + 16, dm2_0);                 \
    vst1q_f32(dmu + 20, dm2_1);                 \
    vst1q_f32(dmu + 24, dm3_0);                 \
    vst1q_f32(dmu + 28, dm3_1)

__attribute__((noinline)) static void dots_mu_d32(const float *__restrict__ a, const float *__restrict__ mu,
                                                  int b_stride, float *__restrict__ dmu) {
    DECLARE_MU()
    M16(0);
    M16(16);
    STORE_MU();
}

__attribute__((noinline)) static void dots_mu_d16(const float *__restrict__ a, const float *__restrict__ mu,
                                                  int b_stride, float *__restrict__ dmu) {
    DECLARE_MU()
    M16(0);
    STORE_MU();
}

__attribute__((noinline)) static void dots_mu_var(const float *__restrict__ a, const float *__restrict__ mu,
                                                  int b_stride, int dimension, float *__restrict__ dmu) {
    DECLARE_MU()
    for (int k = 0; k < dimension; ++k) {
        MSTEP(k);
    }
    STORE_MU();
}

static inline void dots_mu_4x8(const float *a, const float *mu, int b_stride, int dimension, float *dmu) {
    if (dimension == 32) {
        dots_mu_d32(a, mu, b_stride, dmu);
    } else if (dimension == 16) {
        dots_mu_d16(a, mu, b_stride, dmu);
    } else {
        dots_mu_var(a, mu, b_stride, dimension, dmu);
    }
}

#define DECLARE_V()                         \
    float64x2_t p0_0 = vdupq_n_f64(0.0);    \
    float64x2_t p0_1 = vdupq_n_f64(0.0);    \
    float64x2_t p0_2 = vdupq_n_f64(0.0);    \
    float64x2_t p0_3 = vdupq_n_f64(0.0);    \
    float64x2_t p1_0 = vdupq_n_f64(0.0);    \
    float64x2_t p1_1 = vdupq_n_f64(0.0);    \
    float64x2_t p1_2 = vdupq_n_f64(0.0);    \
    float64x2_t p1_3 = vdupq_n_f64(0.0);    \
    float64x2_t p2_0 = vdupq_n_f64(0.0);    \
    float64x2_t p2_1 = vdupq_n_f64(0.0);    \
    float64x2_t p2_2 = vdupq_n_f64(0.0);    \
    float64x2_t p2_3 = vdupq_n_f64(0.0);    \
    float64x2_t p3_0 = vdupq_n_f64(0.0);    \
    float64x2_t p3_1 = vdupq_n_f64(0.0);    \
    float64x2_t p3_2 = vdupq_n_f64(0.0);    \
    float64x2_t p3_3 = vdupq_n_f64(0.0);

#define VSTEP(kk)                                                                                  \
    do {                                                                                           \
        float32x4_t avec = vld1q_f32(a + (size_t)(kk) * (size_t)QUERY_BLOCK);                      \
        const float *vk = vv + (size_t)(kk) * (size_t)b_stride;                                    \
        float32x4_t vc0 = vld1q_f32(vk);                                                           \
        float32x4_t vc1 = vld1q_f32(vk + 4);                                                       \
        float64x2_t ax_lo = vcvt_f64_f32(vget_low_f32(avec));                                      \
        float64x2_t ax_hi = vcvt_f64_f32(vget_high_f32(avec));                                     \
        float64x2_t v0 = vcvt_f64_f32(vget_low_f32(vc0));                                          \
        float64x2_t v1 = vcvt_f64_f32(vget_high_f32(vc0));                                         \
        float64x2_t v2 = vcvt_f64_f32(vget_low_f32(vc1));                                          \
        float64x2_t v3 = vcvt_f64_f32(vget_high_f32(vc1));                                         \
        float64x2_t x;                                                                             \
        x = vdupq_laneq_f64(ax_lo, 0);                                                             \
        p0_0 = vfmaq_f64(p0_0, x, v0);                                                             \
        p0_1 = vfmaq_f64(p0_1, x, v1);                                                             \
        p0_2 = vfmaq_f64(p0_2, x, v2);                                                             \
        p0_3 = vfmaq_f64(p0_3, x, v3);                                                             \
        x = vdupq_laneq_f64(ax_lo, 1);                                                             \
        p1_0 = vfmaq_f64(p1_0, x, v0);                                                             \
        p1_1 = vfmaq_f64(p1_1, x, v1);                                                             \
        p1_2 = vfmaq_f64(p1_2, x, v2);                                                             \
        p1_3 = vfmaq_f64(p1_3, x, v3);                                                             \
        x = vdupq_laneq_f64(ax_hi, 0);                                                             \
        p2_0 = vfmaq_f64(p2_0, x, v0);                                                             \
        p2_1 = vfmaq_f64(p2_1, x, v1);                                                             \
        p2_2 = vfmaq_f64(p2_2, x, v2);                                                             \
        p2_3 = vfmaq_f64(p2_3, x, v3);                                                             \
        x = vdupq_laneq_f64(ax_hi, 1);                                                             \
        p3_0 = vfmaq_f64(p3_0, x, v0);                                                             \
        p3_1 = vfmaq_f64(p3_1, x, v1);                                                             \
        p3_2 = vfmaq_f64(p3_2, x, v2);                                                             \
        p3_3 = vfmaq_f64(p3_3, x, v3);                                                             \
    } while (0)

#define V4(base) VSTEP(base); VSTEP((base) + 1); VSTEP((base) + 2); VSTEP((base) + 3)
#define V16(base) V4(base); V4((base) + 4); V4((base) + 8); V4((base) + 12)

#define STORE_V()                              \
    vst1q_f64(dst + 0, p0_0);                  \
    vst1q_f64(dst + 2, p0_1);                  \
    vst1q_f64(dst + 4, p0_2);                  \
    vst1q_f64(dst + 6, p0_3);                  \
    vst1q_f64(dst + 8, p1_0);                  \
    vst1q_f64(dst + 10, p1_1);                 \
    vst1q_f64(dst + 12, p1_2);                 \
    vst1q_f64(dst + 14, p1_3);                 \
    vst1q_f64(dst + 16, p2_0);                 \
    vst1q_f64(dst + 18, p2_1);                 \
    vst1q_f64(dst + 20, p2_2);                 \
    vst1q_f64(dst + 22, p2_3);                 \
    vst1q_f64(dst + 24, p3_0);                 \
    vst1q_f64(dst + 26, p3_1);                 \
    vst1q_f64(dst + 28, p3_2);                 \
    vst1q_f64(dst + 30, p3_3)

__attribute__((noinline)) static void dots_v_d32(const float *__restrict__ a, const float *__restrict__ vv,
                                                 int b_stride, double *__restrict__ dst) {
    DECLARE_V()
    V16(0);
    V16(16);
    STORE_V();
}

__attribute__((noinline)) static void dots_v_d16(const float *__restrict__ a, const float *__restrict__ vv,
                                                 int b_stride, double *__restrict__ dst) {
    DECLARE_V()
    V16(0);
    STORE_V();
}

__attribute__((noinline)) static void dots_v_var(const float *__restrict__ a, const float *__restrict__ vv,
                                                 int b_stride, int dimension, double *__restrict__ dst) {
    DECLARE_V()
    for (int k = 0; k < dimension; ++k) {
        VSTEP(k);
    }
    STORE_V();
}

static inline void dots_v8(const float *a, const float *vv, int b_stride, int dimension, double *dst) {
    if (dimension == 32) {
        dots_v_d32(a, vv, b_stride, dst);
    } else if (dimension == 16) {
        dots_v_d16(a, vv, b_stride, dst);
    } else {
        dots_v_var(a, vv, b_stride, dimension, dst);
    }
}

static int compute_fast(const float *points, const float *centers, const float *weights,
                        const float *scales, const float *bias, const float *trig_scale,
                        const float *trig_vec, float *out, int q_count, int c_count, int dimension) {
    int c_pad = (c_count + (NR - 1)) & ~(NR - 1);
    float *pack_mu = (float *)xalloc(sizeof(float) * (size_t)dimension * (size_t)c_pad);
    float *pack_v = (float *)xalloc(sizeof(float) * (size_t)dimension * (size_t)c_pad);
    double *nmu = (double *)xalloc(sizeof(double) * (size_t)c_pad);
    double *elim = (double *)xalloc(sizeof(double) * (size_t)c_pad);
    double *nx = (double *)xalloc(sizeof(double) * (size_t)q_count);
    double *acc = (double *)xalloc(sizeof(double) * (size_t)q_count);
    if (!pack_mu || !pack_v || !nmu || !elim || !nx || !acc) {
        free(pack_mu);
        free(pack_v);
        free(nmu);
        free(elim);
        free(nx);
        free(acc);
        return 0;
    }

#pragma omp parallel proc_bind(close)
    {
#pragma omp for schedule(static)
        for (int k = 0; k < dimension; ++k) {
            float *mu_row = pack_mu + (size_t)k * (size_t)c_pad;
            float *v_row = pack_v + (size_t)k * (size_t)c_pad;
            for (int c = 0; c < c_count; ++c) {
                mu_row[c] = centers[(size_t)c * (size_t)dimension + (size_t)k];
                v_row[c] = trig_vec[(size_t)c * (size_t)dimension + (size_t)k];
            }
        }
#pragma omp for schedule(static)
        for (int c = 0; c < c_count; ++c) {
            double nm = 0.0;
            for (int k = 0; k < dimension; ++k) {
                double m = (double)pack_mu[(size_t)k * (size_t)c_pad + (size_t)c];
                nm = fma(m, m, nm);
            }
            nmu[c] = nm;
            elim[c] = exp_limit(weights[c]);
        }
#pragma omp for schedule(static)
        for (int q = 0; q < q_count; ++q) {
            const float *src = points + (size_t)q * (size_t)dimension;
            double norm = 0.0;
            for (int k = 0; k < dimension; ++k) {
                double v = (double)src[k];
                norm = fma(v, v, norm);
            }
            nx[q] = norm;
        }

        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int q_begin = (int)(((long long)q_count * tid) / nth);
        int q_end = (int)(((long long)q_count * (tid + 1)) / nth);
        for (int q = q_begin; q < q_end; ++q) {
            acc[q] = 0.0;
        }
        float *pack_a = NULL;
        if (q_end > q_begin) {
            pack_a = (float *)xalloc(sizeof(float) * (size_t)dimension * (size_t)QUERY_BLOCK);
        }
        if (q_end > q_begin && pack_a == NULL) {
            for (int q = q_begin; q < q_end; ++q) {
                double sum = 0.0;
                const float *x = points + (size_t)q * (size_t)dimension;
                for (int c = 0; c < c_count; ++c) {
                    sum += pair_term(x, centers + (size_t)c * (size_t)dimension,
                                     trig_vec + (size_t)c * (size_t)dimension, dimension, weights[c],
                                     scales[c], bias[c], trig_scale[c]);
                }
                out[q] = (float)sum;
            }
        } else if (q_end > q_begin) {
            int c_main = c_count & ~(NR - 1);
            for (int q0 = q_begin; q0 < q_end; q0 += QUERY_BLOCK) {
                int n = q_end - q0;
                if (n > QUERY_BLOCK) {
                    n = QUERY_BLOCK;
                }
                int n4 = n & ~(MR - 1);
                for (int i = 0; i < n4; ++i) {
                    const float *src = points + (size_t)(q0 + i) * (size_t)dimension;
                    for (int k = 0; k < dimension; ++k) {
                        pack_a[(size_t)k * (size_t)QUERY_BLOCK + (size_t)i] = src[k];
                    }
                }
                for (int c = 0; c < c_main; c += NR) {
                    for (int qi = 0; qi < n4; qi += MR) {
                        float dmu[32] __attribute__((aligned(16)));
                        double pdot[32] __attribute__((aligned(16)));
                        dots_mu_4x8(pack_a + qi, pack_mu + c, c_pad, dimension, dmu);
                        dots_v8(pack_a + qi, pack_v + c, c_pad, dimension, pdot);
                        for (int i = 0; i < MR; ++i) {
                            float32x4_t mu_lo = vld1q_f32(dmu + 8 * i);
                            float32x4_t mu_hi = vld1q_f32(dmu + 8 * i + 4);
                            acc[q0 + qi + i] += row_sum8(
                                nx[q0 + qi + i], mu_lo, mu_hi, pdot + 8 * i, nmu + c, weights + c, scales + c,
                                bias + c, trig_scale + c, elim + c);
                        }
                    }
                }
                for (int i = 0; i < n4; ++i) {
                    int q = q0 + i;
                    const float *x = points + (size_t)q * (size_t)dimension;
                    for (int c = c_main; c < c_count; ++c) {
                        acc[q] += pair_term(x, centers + (size_t)c * (size_t)dimension,
                                            trig_vec + (size_t)c * (size_t)dimension, dimension,
                                            weights[c], scales[c], bias[c], trig_scale[c]);
                    }
                }
                for (int i = n4; i < n; ++i) {
                    int q = q0 + i;
                    const float *x = points + (size_t)q * (size_t)dimension;
                    double sum = 0.0;
                    for (int c = 0; c < c_count; ++c) {
                        sum += pair_term(x, centers + (size_t)c * (size_t)dimension,
                                         trig_vec + (size_t)c * (size_t)dimension, dimension, weights[c],
                                         scales[c], bias[c], trig_scale[c]);
                    }
                    acc[q] = sum;
                }
            }
            for (int q = q_begin; q < q_end; ++q) {
                out[q] = (float)acc[q];
            }
        }
        free(pack_a);
    }

    free(pack_mu);
    free(pack_v);
    free(nmu);
    free(elim);
    free(nx);
    free(acc);
    return 1;
}

#endif

void compute_field_omp(const float *points, const float *centers, const float *weights,
                       const float *scales, const float *bias, const float *trig_scale,
                       const float *trig_vec, float *out, int q_count, int c_count, int dimension) {
    if (q_count <= 0) {
        return;
    }
    if (c_count <= 0 || dimension < 0) {
        memset(out, 0, (size_t)q_count * sizeof(float));
        return;
    }
#if HAS_NEON
    if (dimension > 0 && compute_fast(points, centers, weights, scales, bias, trig_scale, trig_vec, out,
                                      q_count, c_count, dimension)) {
        return;
    }
#else
    (void)dimension;
#endif
    compute_scalar_all(points, centers, weights, scales, bias, trig_scale, trig_vec, out, q_count,
                       c_count, dimension);
}
"""


def _library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        if not os.environ.get("OMP_NUM_THREADS"):
            os.environ["OMP_NUM_THREADS"] = str(max(1, len(os.sched_getaffinity(0))))
        os.environ.setdefault("OMP_PROC_BIND", "close")
        os.environ.setdefault("OMP_PLACES", "cores")
        os.environ.setdefault("OMP_DYNAMIC", "FALSE")
        os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
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
            "-fno-math-errno",
            "-fno-trapping-math",
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
    """Field response. The numeric kernel is compiled from the embedded C source on import."""
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


_library()

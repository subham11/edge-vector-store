// ──────────────────────────────────────────────────────────────
//  SIMDKernels — batch distance computation with NEON/SSE2
//
//  Key optimisation: the query vector is loaded into registers
//  once per block of 4 floats, and reused across all candidates.
//  This eliminates redundant loads when comparing one query against
//  many stored vectors.
// ──────────────────────────────────────────────────────────────
#include "SIMDKernels.h"
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define EVS_HAS_NEON 1
#endif

namespace evs {

// ── Cosine ───────────────────────────────────────────────────

void batchCosineDistance(const float* query,
                         const float* const* vectors,
                         float* distances,
                         size_t count,
                         int32_t dims) {
#if EVS_HAS_NEON
    // Accumulators for each candidate: dot, normA, normB
    float32x4_t dot[4], nA[4], nB[4];
    for (size_t c = 0; c < count; ++c) {
        dot[c] = vdupq_n_f32(0.0f);
        nA[c]  = vdupq_n_f32(0.0f);
        nB[c]  = vdupq_n_f32(0.0f);
    }

    int32_t i = 0;
    for (; i + 4 <= dims; i += 4) {
        // Load query block once — reuse across all candidates
        float32x4_t vq = vld1q_f32(query + i);
        float32x4_t vqSq = vmulq_f32(vq, vq);

        for (size_t c = 0; c < count; ++c) {
            float32x4_t vv = vld1q_f32(vectors[c] + i);
            dot[c] = vfmaq_f32(dot[c], vq, vv);
            nA[c]  = vaddq_f32(nA[c], vqSq);
            nB[c]  = vfmaq_f32(nB[c], vv, vv);
        }
    }

    // Horizontal reduce and compute final distances
    for (size_t c = 0; c < count; ++c) {
        float d = vaddvq_f32(dot[c]);
        float a = vaddvq_f32(nA[c]);
        float b = vaddvq_f32(nB[c]);

        // Handle scalar tail
        for (int32_t j = i; j < dims; ++j) {
            d += query[j] * vectors[c][j];
            a += query[j] * query[j];
            b += vectors[c][j] * vectors[c][j];
        }

        float denom = std::sqrt(a) * std::sqrt(b);
        distances[c] = (denom > 0.0f) ? (1.0f - d / denom) : 1.0f;
    }
#else
    // Scalar fallback
    for (size_t c = 0; c < count; ++c) {
        float d = 0.0f, a = 0.0f, b = 0.0f;
        for (int32_t j = 0; j < dims; ++j) {
            d += query[j] * vectors[c][j];
            a += query[j] * query[j];
            b += vectors[c][j] * vectors[c][j];
        }
        float denom = std::sqrt(a) * std::sqrt(b);
        distances[c] = (denom > 0.0f) ? (1.0f - d / denom) : 1.0f;
    }
#endif
}

// ── Euclidean ────────────────────────────────────────────────

void batchEuclideanDistance(const float* query,
                            const float* const* vectors,
                            float* distances,
                            size_t count,
                            int32_t dims) {
#if EVS_HAS_NEON
    float32x4_t sum[4];
    for (size_t c = 0; c < count; ++c) {
        sum[c] = vdupq_n_f32(0.0f);
    }

    int32_t i = 0;
    for (; i + 4 <= dims; i += 4) {
        float32x4_t vq = vld1q_f32(query + i);
        for (size_t c = 0; c < count; ++c) {
            float32x4_t vd = vsubq_f32(vq, vld1q_f32(vectors[c] + i));
            sum[c] = vfmaq_f32(sum[c], vd, vd);
        }
    }

    for (size_t c = 0; c < count; ++c) {
        float s = vaddvq_f32(sum[c]);
        for (int32_t j = i; j < dims; ++j) {
            float d = query[j] - vectors[c][j];
            s += d * d;
        }
        distances[c] = s;
    }
#else
    for (size_t c = 0; c < count; ++c) {
        float s = 0.0f;
        for (int32_t j = 0; j < dims; ++j) {
            float d = query[j] - vectors[c][j];
            s += d * d;
        }
        distances[c] = s;
    }
#endif
}

// ── Inner Product ────────────────────────────────────────────

void batchInnerProductDistance(const float* query,
                               const float* const* vectors,
                               float* distances,
                               size_t count,
                               int32_t dims) {
#if EVS_HAS_NEON
    float32x4_t sum[4];
    for (size_t c = 0; c < count; ++c) {
        sum[c] = vdupq_n_f32(0.0f);
    }

    int32_t i = 0;
    for (; i + 4 <= dims; i += 4) {
        float32x4_t vq = vld1q_f32(query + i);
        for (size_t c = 0; c < count; ++c) {
            sum[c] = vfmaq_f32(sum[c], vq, vld1q_f32(vectors[c] + i));
        }
    }

    for (size_t c = 0; c < count; ++c) {
        float s = vaddvq_f32(sum[c]);
        for (int32_t j = i; j < dims; ++j) {
            s += query[j] * vectors[c][j];
        }
        distances[c] = 1.0f - s;
    }
#else
    for (size_t c = 0; c < count; ++c) {
        float s = 0.0f;
        for (int32_t j = 0; j < dims; ++j) {
            s += query[j] * vectors[c][j];
        }
        distances[c] = 1.0f - s;
    }
#endif
}

} // namespace evs

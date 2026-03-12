#pragma once
// ──────────────────────────────────────────────────────────────
//  SIMD distance kernels for float32 reranking
//
//  Provides batch distance computation: compute distances for
//  up to 4 candidates simultaneously while keeping the query
//  vector in registers. Includes dimension-aligned fast paths
//  for common embedding sizes (128, 256, 384, 512, 768).
// ──────────────────────────────────────────────────────────────
#include "../core/Types.h"
#include <cstddef>
#include <cstdint>

namespace evs {

/// Compute cosine distance for a batch of candidates.
/// @param query     Query vector (dims floats)
/// @param vectors   Array of candidate vector pointers (up to 4)
/// @param distances Output distances (one per candidate)
/// @param count     Number of candidates (1..4)
/// @param dims      Vector dimensionality
void batchCosineDistance(const float* query,
                         const float* const* vectors,
                         float* distances,
                         size_t count,
                         int32_t dims);

/// Compute euclidean (L2²) distance for a batch of candidates.
void batchEuclideanDistance(const float* query,
                            const float* const* vectors,
                            float* distances,
                            size_t count,
                            int32_t dims);

/// Compute inner product distance for a batch of candidates.
void batchInnerProductDistance(const float* query,
                               const float* const* vectors,
                               float* distances,
                               size_t count,
                               int32_t dims);

/// Dispatch to the correct batch kernel based on metric.
inline void batchDistance(Metric metric,
                          const float* query,
                          const float* const* vectors,
                          float* distances,
                          size_t count,
                          int32_t dims) {
    switch (metric) {
        case Metric::Cosine:
            batchCosineDistance(query, vectors, distances, count, dims);
            break;
        case Metric::Euclidean:
            batchEuclideanDistance(query, vectors, distances, count, dims);
            break;
        case Metric::InnerProduct:
            batchInnerProductDistance(query, vectors, distances, count, dims);
            break;
    }
}

/// Single-pair distance (delegates to the batch kernels with count=1).
inline float singleDistance(Metric metric,
                            const float* a,
                            const float* b,
                            int32_t dims) {
    float dist = 0.0f;
    const float* ptrs[1] = { b };
    batchDistance(metric, a, ptrs, &dist, 1, dims);
    return dist;
}

} // namespace evs

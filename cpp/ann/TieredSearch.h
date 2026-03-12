#pragma once
// ──────────────────────────────────────────────────────────────
//  TieredSearch — merges hot-cache + cold-index results
//                 with optional float32 reranking
// ──────────────────────────────────────────────────────────────
#include "ANNEngine.h"
#include "HotCache.h"
#include "../core/Config.h"
#include "../storage/VectorStore.h"
#include <string>
#include <vector>

namespace evs {

class TieredSearch {
public:
    TieredSearch(HotCache& hot, ANNEngine& cold, const StoreConfig& cfg,
                 VectorStore* vectors = nullptr)
        : hot_(hot), cold_(cold), cfg_(cfg), vectors_(vectors) {}

    /**
     * Execute a tiered search according to the given profile.
     * Returns up to `topK` ANNResults with normalised distances.
     */
    std::vector<ANNResult> search(const float* query, size_t topK,
                                   SearchProfile profile) const;

private:
    HotCache&         hot_;
    ANNEngine&        cold_;
    const StoreConfig& cfg_;
    VectorStore*      vectors_;

    /// Normalise a raw distance into [0, 1].
    float normalise(float rawDist) const;

    /// Compute cosine / L2 / IP distance between two float32 vectors.
    float computeDistance(const float* a, const float* b) const;

    /// Rerank candidates using float32 vectors from VectorStore.
    std::vector<ANNResult> rerank(const float* query,
                                  const std::vector<ANNResult>& candidates,
                                  size_t topK) const;

    /// Merge two sorted result lists, deduplicate by key, keep top-K.
    static std::vector<ANNResult> merge(
        const std::vector<ANNResult>& a,
        const std::vector<ANNResult>& b,
        size_t topK);
};

} // namespace evs

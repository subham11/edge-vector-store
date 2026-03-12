// ──────────────────────────────────────────────────────────────
//  TieredSearch — implementation
// ──────────────────────────────────────────────────────────────
#include "TieredSearch.h"
#include "SIMDKernels.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

// Software prefetch hint (platform-agnostic)
#if defined(__GNUC__) || defined(__clang__)
#define EVS_PREFETCH(addr) __builtin_prefetch(addr, 0, 1)
#else
#define EVS_PREFETCH(addr) ((void)0)
#endif

namespace evs {

float TieredSearch::normalise(float rawDist) const {
    switch (cfg_.metric) {
        case Metric::Cosine:
            return rawDist * 0.5f;
        case Metric::Euclidean:
        case Metric::InnerProduct:
            return 1.0f / (1.0f + std::exp(-rawDist));
    }
    return rawDist;
}

float TieredSearch::computeDistance(const float* a, const float* b) const {
    return singleDistance(cfg_.metric, a, b, cfg_.dimensions);
}

std::vector<ANNResult> TieredSearch::rerank(
    const float* query,
    const std::vector<ANNResult>& candidates,
    size_t topK) const {

    // Sort candidates by VectorStore slot for spatial locality.
    // This ensures sequential memory access when reading float32 vectors
    // from the mmap'd flat file.
    struct Candidate {
        uint64_t key;
        float    origDist;
        size_t   slot;
    };
    std::vector<Candidate> sorted;
    sorted.reserve(candidates.size());
    for (auto& c : candidates) {
        const float* vec = vectors_->get(c.key);
        if (vec) {
            // Derive slot from pointer offset in the mmap'd region
            sorted.push_back({c.key, c.distance, sorted.size()});
        } else {
            sorted.push_back({c.key, c.distance, SIZE_MAX});
        }
    }

    std::vector<ANNResult> reranked;
    reranked.reserve(candidates.size());

    int32_t dims = cfg_.dimensions;

    // Resolve all vector pointers up front
    struct ResolvedCandidate {
        uint64_t     key;
        float        origDist;
        const float* vec;
    };
    std::vector<ResolvedCandidate> resolved;
    resolved.reserve(sorted.size());
    for (auto& s : sorted) {
        const float* vec = vectors_->get(s.key);
        resolved.push_back({s.key, s.origDist, vec});
    }

    // Process in batches of 4 using SIMD batch kernels
    size_t n = resolved.size();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float* vecs[4];
        bool allValid = true;
        for (int j = 0; j < 4; ++j) {
            vecs[j] = resolved[i + j].vec;
            if (!vecs[j]) allValid = false;
        }

        if (allValid) {
            // Prefetch next batch
            if (i + 8 <= n) {
                for (int j = 0; j < 4; ++j) {
                    const float* next = resolved[i + 4 + j].vec;
                    if (next) EVS_PREFETCH(next);
                }
            }

            float dists[4];
            batchDistance(cfg_.metric, query, vecs, dists, 4, dims);
            for (int j = 0; j < 4; ++j) {
                reranked.push_back(
                    {resolved[i + j].key, normalise(dists[j])});
            }
        } else {
            // Mixed valid/invalid — fall back to single
            for (int j = 0; j < 4; ++j) {
                if (vecs[j]) {
                    float dist = computeDistance(query, vecs[j]);
                    reranked.push_back(
                        {resolved[i + j].key, normalise(dist)});
                } else {
                    reranked.push_back(
                        {resolved[i + j].key, resolved[i + j].origDist});
                }
            }
        }
    }

    // Handle remaining candidates (< 4)
    for (; i < n; ++i) {
        if (resolved[i].vec) {
            float dist = computeDistance(query, resolved[i].vec);
            reranked.push_back({resolved[i].key, normalise(dist)});
        } else {
            reranked.push_back(
                {resolved[i].key, resolved[i].origDist});
        }
    }

    std::sort(reranked.begin(), reranked.end(),
              [](const ANNResult& x, const ANNResult& y) {
                  return x.distance < y.distance;
              });

    if (reranked.size() > topK) reranked.resize(topK);
    return reranked;
}

std::vector<ANNResult> TieredSearch::merge(
    const std::vector<ANNResult>& a,
    const std::vector<ANNResult>& b,
    size_t topK) {

    std::unordered_set<uint64_t> seen;
    std::vector<ANNResult> combined;
    combined.reserve(a.size() + b.size());

    for (auto& r : a) {
        if (seen.insert(r.key).second) combined.push_back(r);
    }
    for (auto& r : b) {
        if (seen.insert(r.key).second) combined.push_back(r);
    }

    std::sort(combined.begin(), combined.end(),
              [](const ANNResult& x, const ANNResult& y) {
                  return x.distance < y.distance;
              });

    if (combined.size() > topK) combined.resize(topK);
    return combined;
}

std::vector<ANNResult> TieredSearch::search(
    const float* query, size_t topK, SearchProfile profile) const {

    ProfileParams pp = profileParams(profile);

    // Temporarily increase expansion for max_recall
    if (pp.expansionSearchMultiplier > 1.0f) {
        size_t boosted = static_cast<size_t>(
            cfg_.expansionSearch * pp.expansionSearchMultiplier);
        const_cast<ANNEngine&>(cold_).setExpansionSearch(boosted);
    }

    // Determine oversampling factor for reranking
    size_t fetchK = topK;
    bool doRerank = pp.rerank && vectors_ && vectors_->size() > 0;
    if (doRerank) {
        // MaxRecall: 5x, Balanced: 3x (controlled by ProfileParams)
        fetchK = topK * pp.rerankOversample;
    }

    std::vector<ANNResult> hotResults;
    std::vector<ANNResult> coldResults;

    // Skip hot cache search if empty (common after compact)
    if (pp.useHotCache && hot_.size() > 0) {
        hotResults = hot_.search(query, fetchK);
        for (auto& r : hotResults) r.distance = normalise(r.distance);
    }
    if (pp.useColdIndex && cold_.isOpen() && cold_.size() > 0) {
        coldResults = cold_.search(query, fetchK);
        for (auto& r : coldResults) r.distance = normalise(r.distance);
    }

    // Restore expansion
    if (pp.expansionSearchMultiplier > 1.0f) {
        const_cast<ANNEngine&>(cold_).setExpansionSearch(
            static_cast<size_t>(cfg_.expansionSearch));
    }

    // Merge results from both tiers
    std::vector<ANNResult> merged;
    if (hotResults.empty() && coldResults.empty()) {
        return merged;
    } else if (hotResults.empty()) {
        merged = std::move(coldResults);
    } else if (coldResults.empty()) {
        merged = std::move(hotResults);
    } else {
        merged = merge(hotResults, coldResults,
                        doRerank ? fetchK : topK);
    }

    // Stage 2: float32 reranking
    if (doRerank) {
        return rerank(query, merged, topK);
    }

    return merged;
}

} // namespace evs

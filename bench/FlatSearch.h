#pragma once
// ──────────────────────────────────────────────────────────────
//  FlatSearch — brute-force baseline for ground-truth recall
// ──────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace evs {

struct FlatResult {
    uint64_t key;
    float    distance;  // cosine distance = 1 - similarity
};

class FlatSearch {
public:
    void reserve(size_t n, int dims) {
        dims_ = dims;
        keys_.reserve(n);
        data_.reserve(n * dims);
    }

    void add(uint64_t key, const float* vec) {
        keys_.push_back(key);
        data_.insert(data_.end(), vec, vec + dims_);
    }

    size_t size() const { return keys_.size(); }

    std::vector<FlatResult> search(const float* query, size_t topK) const {
        size_t n = keys_.size();
        std::vector<std::pair<float, uint64_t>> scores(n);

        for (size_t i = 0; i < n; ++i) {
            const float* vec = &data_[i * dims_];
            float dot = 0.0f;
            for (int d = 0; d < dims_; ++d)
                dot += query[d] * vec[d];
            // cosine distance = 1 - cosine_similarity
            // (assumes normalized vectors)
            scores[i] = {1.0f - dot, keys_[i]};
        }

        size_t k = std::min(topK, n);
        std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });

        std::vector<FlatResult> results(k);
        for (size_t i = 0; i < k; ++i)
            results[i] = {scores[i].second, scores[i].first};
        return results;
    }

private:
    int                  dims_ = 0;
    std::vector<uint64_t> keys_;
    std::vector<float>    data_;
};

} // namespace evs

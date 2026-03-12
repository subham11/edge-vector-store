#pragma once
// ──────────────────────────────────────────────────────────────
//  HotCache — small in-RAM USearch index with LRU eviction
// ──────────────────────────────────────────────────────────────
#include "ANNEngine.h"
#include "../core/Config.h"
#include <list>
#include <unordered_map>
#include <vector>

namespace evs {

/**
 * A fixed-capacity in-RAM HNSW index for recently-upserted or
 * frequently-accessed vectors.  When the cache is full the
 * least-recently-used keys are evicted and (optionally) moved
 * to the cold index.
 */
class HotCache {
public:
    HotCache() = default;

    bool init(const StoreConfig& cfg);

    /// Insert or touch a vector. Returns evicted keys (if any).
    std::vector<uint64_t> put(uint64_t key, const float* data, int dims);

    /// Search the hot index.
    std::vector<ANNResult> search(const float* query, size_t topK) const;

    /// Remove a key from the cache.
    bool remove(uint64_t key);

    size_t size() const { return lruMap_.size(); }
    size_t capacity() const { return capacity_; }

    /// Drain all keys (used during compact). Returns all keys.
    std::vector<uint64_t> drain();

private:
    ANNEngine engine_;
    size_t    capacity_ = 10000;

    // LRU tracking: front = most recent, back = least recent
    std::list<uint64_t> lruList_;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lruMap_;

    void touch(uint64_t key);
    std::vector<uint64_t> evictIfNeeded();
};

} // namespace evs

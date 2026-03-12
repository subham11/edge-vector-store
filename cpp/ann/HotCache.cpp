// ──────────────────────────────────────────────────────────────
//  HotCache — implementation
// ──────────────────────────────────────────────────────────────
#include "HotCache.h"

namespace evs {

bool HotCache::init(const StoreConfig& cfg) {
    capacity_ = static_cast<size_t>(cfg.hotCacheCapacity);
    return engine_.init(cfg);
}

void HotCache::touch(uint64_t key) {
    auto it = lruMap_.find(key);
    if (it != lruMap_.end()) {
        lruList_.erase(it->second);
    }
    lruList_.push_front(key);
    lruMap_[key] = lruList_.begin();
}

std::vector<uint64_t> HotCache::evictIfNeeded() {
    std::vector<uint64_t> evicted;
    while (lruMap_.size() > capacity_) {
        uint64_t victim = lruList_.back();
        lruList_.pop_back();
        lruMap_.erase(victim);
        engine_.remove(victim);
        evicted.push_back(victim);
    }
    return evicted;
}

std::vector<uint64_t> HotCache::put(uint64_t key, const float* data,
                                     int dims) {
    // If already present, just touch + re-add (update)
    auto it = lruMap_.find(key);
    if (it != lruMap_.end()) {
        engine_.remove(key);
    }

    engine_.add(key, data);
    touch(key);
    return evictIfNeeded();
}

std::vector<ANNResult> HotCache::search(const float* query,
                                         size_t topK) const {
    return engine_.search(query, topK);
}

bool HotCache::remove(uint64_t key) {
    auto it = lruMap_.find(key);
    if (it == lruMap_.end()) return false;
    lruList_.erase(it->second);
    lruMap_.erase(it);
    engine_.remove(key);
    return true;
}

std::vector<uint64_t> HotCache::drain() {
    std::vector<uint64_t> keys;
    keys.reserve(lruMap_.size());
    for (auto& [k, _] : lruMap_) keys.push_back(k);
    lruList_.clear();
    lruMap_.clear();
    // We don't clear the engine here — caller does a full rebuild.
    return keys;
}

} // namespace evs

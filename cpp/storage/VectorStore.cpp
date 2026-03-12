// ──────────────────────────────────────────────────────────────
//  VectorStore — implementation
// ──────────────────────────────────────────────────────────────
#include "VectorStore.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace evs {

bool VectorStore::open(const std::string& dir, int32_t dims) {
    close();
    dir_  = dir;
    dims_ = dims;
    loadFromDisk();
    return true;
}

void VectorStore::close() {
    mmap_.close();
    keyToSlot_.clear();
    pendingKeys_.clear();
    pendingData_.clear();
    removedKeys_.clear();
    mmapVectors_ = nullptr;
    mmapCount_   = 0;
    dims_        = 0;
}

void VectorStore::put(uint64_t key, const float* data) {
    // Check if already in mmap region — mark for overwrite
    auto it = keyToSlot_.find(key);
    if (it != keyToSlot_.end()) {
        // Will be replaced during save; mark old slot as removed
        removedKeys_.push_back(key);
    }
    // Also check if already in pending (update in-place)
    for (size_t i = 0; i < pendingKeys_.size(); ++i) {
        if (pendingKeys_[i] == key) {
            std::memcpy(pendingData_.data() + i * dims_, data,
                        dims_ * sizeof(float));
            // Update keyToSlot to point to pending
            keyToSlot_[key] = mmapCount_ + i;
            return;
        }
    }

    size_t pendingIdx = pendingKeys_.size();
    pendingKeys_.push_back(key);
    pendingData_.insert(pendingData_.end(), data, data + dims_);
    keyToSlot_[key] = mmapCount_ + pendingIdx;
}

void VectorStore::remove(uint64_t key) {
    auto it = keyToSlot_.find(key);
    if (it != keyToSlot_.end()) {
        removedKeys_.push_back(key);
        keyToSlot_.erase(it);
    }
}

const float* VectorStore::get(uint64_t key) const {
    auto it = keyToSlot_.find(key);
    if (it == keyToSlot_.end()) return nullptr;

    size_t slot = it->second;
    if (slot < mmapCount_) {
        // In mmap region
        return mmapVectors_ + slot * dims_;
    }
    // In pending region
    size_t pendingIdx = slot - mmapCount_;
    if (pendingIdx < pendingKeys_.size()) {
        return pendingData_.data() + pendingIdx * dims_;
    }
    return nullptr;
}

bool VectorStore::save() {
    if (!isOpen()) return false;
    if (pendingKeys_.empty() && removedKeys_.empty()) return true;
    return writeToDisk();
}

// ── persistence ──────────────────────────────────────────────

bool VectorStore::loadFromDisk() {
    std::string path = filePath();
    if (!mmap_.open(path)) return false;

    if (mmap_.size() < kHeaderBytes) {
        mmap_.close();
        return false;
    }

    auto magic   = *mmap_.at<uint32_t>(0);
    auto version = *mmap_.at<uint32_t>(4);
    auto count   = *mmap_.at<uint32_t>(8);
    auto dims    = *mmap_.at<uint32_t>(12);

    if (magic != kMagic || version != kVersion ||
        static_cast<int32_t>(dims) != dims_) {
        mmap_.close();
        return false;
    }

    size_t slotTableBytes  = count * sizeof(uint64_t);
    size_t vectorDataBytes = count * dims_ * sizeof(float);
    size_t expectedSize    = kHeaderBytes + slotTableBytes + vectorDataBytes;
    if (mmap_.size() < expectedSize) {
        mmap_.close();
        return false;
    }

    // Build key→slot map from slot table
    const uint64_t* slots = mmap_.at<uint64_t>(kHeaderBytes);
    keyToSlot_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        keyToSlot_[slots[i]] = i;
    }

    mmapVectors_ = mmap_.at<float>(kHeaderBytes + slotTableBytes);
    mmapCount_   = count;
    return true;
}

bool VectorStore::writeToDisk() {
    // Collect all live vectors (existing + pending, minus removed)
    std::vector<uint64_t> allKeys;
    std::vector<float>    allData;

    // Existing mmap'd vectors (skip removed)
    for (size_t i = 0; i < mmapCount_; ++i) {
        const uint64_t* slots =
            mmap_.at<uint64_t>(kHeaderBytes);
        uint64_t key = slots[i];

        // Skip if removed or will be overwritten by pending
        bool skip = false;
        for (auto rk : removedKeys_) {
            if (rk == key) { skip = true; break; }
        }
        if (skip) continue;

        allKeys.push_back(key);
        const float* vec = mmapVectors_ + i * dims_;
        allData.insert(allData.end(), vec, vec + dims_);
    }

    // Add pending vectors
    for (size_t i = 0; i < pendingKeys_.size(); ++i) {
        allKeys.push_back(pendingKeys_[i]);
        const float* vec = pendingData_.data() + i * dims_;
        allData.insert(allData.end(), vec, vec + dims_);
    }

    // Close mmap before writing
    mmap_.close();
    mmapVectors_ = nullptr;
    mmapCount_   = 0;
    keyToSlot_.clear();

    // Write to disk
    std::string path = filePath();
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    uint32_t count = static_cast<uint32_t>(allKeys.size());
    uint32_t dims  = static_cast<uint32_t>(dims_);

    // Header
    std::fwrite(&kMagic, 4, 1, f);
    std::fwrite(&kVersion, 4, 1, f);
    std::fwrite(&count, 4, 1, f);
    std::fwrite(&dims, 4, 1, f);

    // Slot table (keys)
    std::fwrite(allKeys.data(), sizeof(uint64_t), count, f);

    // Vector data
    std::fwrite(allData.data(), sizeof(float),
                static_cast<size_t>(count) * dims_, f);

    std::fclose(f);

    // Clear pending state
    pendingKeys_.clear();
    pendingData_.clear();
    removedKeys_.clear();

    // Re-mmap the written file
    loadFromDisk();

    return true;
}

} // namespace evs

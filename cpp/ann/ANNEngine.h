#pragma once
// ──────────────────────────────────────────────────────────────
//  ANNEngine — wrapper around USearch C API
// ──────────────────────────────────────────────────────────────
#include "../core/Types.h"
#include "../core/Config.h"
#include <string>
#include <vector>
#include <cstddef>

// Forward-declare the opaque USearch type so we don't leak the header.
typedef void* usearch_index_t;

namespace evs {

struct ANNResult {
    uint64_t key;
    float    distance;
};

class ANNEngine {
public:
    ANNEngine() = default;
    ~ANNEngine();

    ANNEngine(const ANNEngine&) = delete;
    ANNEngine& operator=(const ANNEngine&) = delete;

    /// Create a new empty index.
    bool init(const StoreConfig& cfg);

    /// Load a previously saved index from disk.
    bool load(const std::string& path);

    /// Memory-map an index for read-only search (no RAM copy).
    bool view(const std::string& path);

    /// Save the current index to disk.
    bool save(const std::string& path);

    /// Add a vector. `data` points to `dimensions` float32 values.
    bool add(uint64_t key, const float* data);

    /// Remove a vector by key.
    bool remove(uint64_t key);

    /// Search. Returns up to `topK` results sorted by distance.
    std::vector<ANNResult> search(const float* query, size_t topK) const;

    /// Number of vectors in the index.
    size_t size() const;

    /// Approximate memory usage in bytes.
    size_t memoryUsage() const;

    /// Change the expansion parameter for search on the fly.
    void setExpansionSearch(size_t expansion);

    /// Destroy the index, releasing all resources.
    void close();

    bool isOpen() const { return index_ != nullptr; }
    bool isMmap() const { return isMmap_; }

private:
    usearch_index_t index_   = nullptr;
    int32_t         dims_    = 0;
    bool            isMmap_  = false;
};

} // namespace evs

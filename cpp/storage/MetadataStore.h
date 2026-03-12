#pragma once
// ──────────────────────────────────────────────────────────────
//  MetadataStore — in-memory hash maps + binary file persistence
//  Replaces the SQLite-based version for zero-overhead lookups.
// ──────────────────────────────────────────────────────────────
#include "../core/Types.h"
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace evs {

class MetadataStore {
public:
    MetadataStore() = default;
    ~MetadataStore();

    MetadataStore(const MetadataStore&) = delete;
    MetadataStore& operator=(const MetadataStore&) = delete;

    /// Open or create the store at `dirPath/`.
    bool open(const std::string& dirPath);

    /// Persist and close.
    void close();

    // ── document CRUD ────────────────────────────────────────

    bool upsertDocument(const std::string& id,
                        const std::string& payloadJson);

    bool deleteDocument(const std::string& id);

    /// Returns payload JSON, empty string if not found.
    std::string getDocument(const std::string& id) const;

    int64_t documentCount() const;

    // ── vector bookkeeping ───────────────────────────────────

    bool registerVector(const std::string& docId, uint64_t numericKey);

    bool unregisterVector(const std::string& docId);

    /// Look up the numeric key for a doc id. Returns 0 on miss.
    uint64_t numericKeyFor(const std::string& docId) const;

    /// Look up the doc id for a numeric key. Returns "" on miss.
    std::string docIdFor(uint64_t numericKey) const;

    /// Batch look up doc ids for multiple numeric keys.
    std::unordered_map<uint64_t, std::string> docIdsFor(
        const std::vector<uint64_t>& numericKeys) const;

    int64_t vectorCount() const;

    // ── stats / misc ─────────────────────────────────────────

    bool setStat(const std::string& key, const std::string& value);
    std::string getStat(const std::string& key) const;

    /// Persist all data to disk (checkpoint).
    bool vacuum();

    /// Iterate all registered vector ids.
    void forEachVector(
        std::function<void(const std::string& docId, uint64_t key)> cb) const;

private:
    std::string dir_;
    bool dirty_ = false;

    // In-memory maps — O(1) lookups, zero SQLite overhead
    std::unordered_map<std::string, std::string> docs_;      // id → payload
    std::unordered_map<std::string, uint64_t>    strToNum_;  // docId → key
    std::unordered_map<uint64_t, std::string>    numToStr_;  // key → docId
    std::unordered_map<std::string, std::string> stats_;     // key → value

    bool loadFromDisk();
    bool saveToDisk() const;
};

} // namespace evs

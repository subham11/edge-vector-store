#pragma once
// ──────────────────────────────────────────────────────────────
//  EdgeStore — top-level orchestrator
//  Owns MetadataStore, ANNEngine (cold), HotCache, Journal.
//  Exposes the full public API consumed by the JSI bridge.
// ──────────────────────────────────────────────────────────────
#include "../core/Config.h"
#include "../core/Journal.h"
#include "../core/Types.h"
#include "../storage/IDMapper.h"
#include "../storage/MetadataStore.h"
#include "../storage/VectorStore.h"
#include "../ann/ANNEngine.h"
#include "../ann/HotCache.h"
#include "../ann/TieredSearch.h"
#include "../pack/PackFormat.h"
#include "../pack/PackIO.h"

#include <mutex>
#include <string>
#include <vector>

namespace evs {

class EdgeStore {
public:
    EdgeStore() = default;
    ~EdgeStore();

    EdgeStore(const EdgeStore&) = delete;
    EdgeStore& operator=(const EdgeStore&) = delete;

    /// Initialise from a JSON config string (same shape as TS StoreConfig).
    bool init(const std::string& configJson);

    /// Shut everything down gracefully.
    void close();

    // ── document API ─────────────────────────────────────────

    bool upsertDocuments(const std::string& docsJson);

    bool upsertVectors(const std::string& entriesJson);

    std::string search(const std::string& optionsJson) const;

    bool remove(const std::string& idsJson);

    // ── direct API (no JSON parsing — called by JSI bridge) ──

    /// Direct search: zero-copy query pointer, structured results.
    std::vector<SearchResult> searchDirect(const float* query, int topK,
                                           SearchProfile profile,
                                           bool includePayload = true) const;

    /// Direct batch vector upsert: ids + flat packed float buffer.
    bool upsertVectorsDirect(const std::string* ids,
                              const float* vectors,
                              size_t count, size_t dims);

    // ── maintenance ──────────────────────────────────────────

    /// Flush hot cache to cold index, truncate journal, VACUUM.
    bool compact();

    std::string getStats() const;

    // ── pack import / export ─────────────────────────────────

    bool importPack(const std::string& path);
    bool exportPack(const std::string& path);

private:
    mutable std::mutex mu_;
    StoreConfig        cfg_;

    MetadataStore meta_;
    VectorStore   vectors_;
    Journal       journal_;
    ANNEngine     cold_;
    HotCache      hot_;

    std::string coldIndexPath() const;
    std::string lastSeqKey() const { return "__last_compact_seq"; }

    /// Replay journal entries after crash to re-populate hot cache.
    void recoverFromJournal();
};

} // namespace evs

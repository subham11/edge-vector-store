#pragma once
// ──────────────────────────────────────────────────────────────
//  IDMapper — deterministic string → uint64 with collision handling
// ──────────────────────────────────────────────────────────────
#include "../core/Types.h"
#include "MetadataStore.h"
#include <string>

namespace evs {

class IDMapper {
public:
    explicit IDMapper(MetadataStore& store) : store_(store) {}

    /// Map a string doc id to a uint64 key.
    /// Uses FNV-1a, then checks / registers with MetadataStore to
    /// detect and resolve collisions by linear probing.
    uint64_t toNumericKey(const std::string& docId) {
        // Already registered?
        uint64_t existing = store_.numericKeyFor(docId);
        if (existing != 0) return existing;

        uint64_t candidate = fnv1a(docId);
        if (candidate == 0) candidate = 1; // reserve 0 as "not found"

        // Linear probe if the candidate is taken by another docId
        while (true) {
            std::string owner = store_.docIdFor(candidate);
            if (owner.empty() || owner == docId) break;
            ++candidate;
            if (candidate == 0) candidate = 1;
        }

        store_.registerVector(docId, candidate);
        return candidate;
    }

    /// Reverse lookup: numeric key → string doc id.
    std::string toDocId(uint64_t key) const {
        return store_.docIdFor(key);
    }

private:
    MetadataStore& store_;
};

} // namespace evs

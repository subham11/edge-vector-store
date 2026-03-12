#pragma once
// ──────────────────────────────────────────────────────────────
//  EdgeVectorStore — shared type definitions
// ──────────────────────────────────────────────────────────────
#include <cstdint>
#include <string>
#include <vector>

namespace evs {

// ── Quantization ────────────────────────────────────────────
enum class Quantization : uint8_t {
    F32 = 0,
    F16 = 1,
    I8  = 2,
    B1  = 3,
};

// ── Distance metric ─────────────────────────────────────────
enum class Metric : uint8_t {
    Cosine       = 0,
    Euclidean    = 1,
    InnerProduct = 2,
};

// ── Search profile ──────────────────────────────────────────
enum class SearchProfile : uint8_t {
    Balanced    = 0,
    MemorySaver = 1,
    MaxRecall   = 2,
    MaxSpeed    = 3,
};

// ── Journal operation types ─────────────────────────────────
enum class JournalOp : uint8_t {
    Upsert = 1,
    Delete = 2,
};

// ── Search result ───────────────────────────────────────────
struct SearchResult {
    std::string id;       // document string id
    float       distance; // lower = closer
    std::string payload;  // JSON payload (optional)
};

// ── Store statistics ────────────────────────────────────────
struct StoreStats {
    int64_t documentCount     = 0;
    int64_t vectorCount       = 0;
    int64_t hotCacheCount     = 0;
    int64_t memoryUsageBytes  = 0;
    int64_t coldIndexSizeBytes = 0;
    Quantization quantization = Quantization::I8;
    int32_t dimensions        = 0;
};

// ── ID mapping ──────────────────────────────────────────────
/// Deterministic string → uint64 via FNV-1a.
/// Collisions are resolved in the SQLite id_map table.
inline uint64_t fnv1a(const std::string& s) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : s) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace evs

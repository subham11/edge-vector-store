#pragma once
// ──────────────────────────────────────────────────────────────
//  PackFormat — .evs pack file layout & manifest
//
//  An .evs file is a standard ZIP archive containing:
//    manifest.json   — metadata (version, dims, quantization, counts)
//    cold.usearch    — pre-built USearch index
//    metadata.db     — SQLite database (documents + vector map)
// ──────────────────────────────────────────────────────────────
#include "../core/Types.h"
#include <cstdint>
#include <string>

namespace evs {

struct PackManifest {
    std::string   version       = "1";
    int32_t       dimensions    = 768;
    Quantization  quantization  = Quantization::I8;
    Metric        metric        = Metric::Cosine;
    int64_t       vectorCount   = 0;
    int64_t       documentCount = 0;
    std::string   createdAt;    // ISO-8601

    /// Serialise to JSON.
    std::string toJson() const;

    /// Parse from JSON. Returns false on error.
    static bool fromJson(const std::string& json, PackManifest& out);
};

} // namespace evs

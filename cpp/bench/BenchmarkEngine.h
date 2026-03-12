#pragma once
// ──────────────────────────────────────────────────────────────
//  BenchmarkEngine — raw HNSW benchmark (build once, sweep ef)
//
//  Extracted from EdgeStore to keep core code focused on CRUD.
// ──────────────────────────────────────────────────────────────
#include <string>

namespace evs {

/// Builds an in-memory HNSW index once, then sweeps multiple ef
/// values — measuring recall, latency, and QPS for each.
/// Returns a JSON array of result objects.
std::string benchmarkRawANN(const std::string& configJson);

} // namespace evs

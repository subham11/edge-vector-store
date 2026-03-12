// ──────────────────────────────────────────────────────────────
//  EVS Benchmark — compare Flat, raw USearch, and EdgeVectorStore
// ──────────────────────────────────────────────────────────────
//  Usage:
//      cmake ../.. -DEVS_BENCHMARK=ON && make -j && ./evs_bench
//
//  Prerequisites:
//      python tools/generate_dataset.py
//      python tools/convert_vectors.py
//
//  Reads from data/:
//      farmers_100k_vectors.bin   (N * 384 * 4 bytes, float32)
//      farmers_100k_queries.bin   (1000 * 384 * 4 bytes, float32)
//      farmers_100k_groundtruth.bin (1000 * 10 * 8 bytes, uint64)
// ──────────────────────────────────────────────────────────────

#include "FlatSearch.h"
#include "ann/ANNEngine.h"
#include "core/Config.h"
#include "core/EdgeStore.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace fs = std::filesystem;
using Clock  = std::chrono::high_resolution_clock;

// ── Globals ─────────────────────────────────────────────────

static constexpr int    DIMS        = 384;
static constexpr int    TOP_K       = 10;
static constexpr int    NUM_QUERIES = 1000;
static constexpr size_t NUM_VECTORS = 100000;

struct BenchResult {
    std::string engine;
    double insertTimeSec    = 0;
    double insertThroughput = 0;
    double indexBuildSec    = 0;
    // latencies in ms
    double searchMean = 0, searchP50 = 0, searchP95 = 0, searchP99 = 0;
    double recallAt10       = 0;
    size_t diskUsageBytes   = 0;
    double memoryDeltaMB    = 0;
    double coldStartMs      = 0;
};

// ── Helpers ─────────────────────────────────────────────────

static double rss_mb() {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

static std::vector<float> load_bin_f32(const std::string& path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<float> data(expected);
    f.read(reinterpret_cast<char*>(data.data()), expected * sizeof(float));
    return data;
}

static std::vector<uint64_t> load_bin_u64(const std::string& path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<uint64_t> data(expected);
    f.read(reinterpret_cast<char*>(data.data()), expected * sizeof(uint64_t));
    return data;
}

static void compute_latency_stats(std::vector<double>& latencies,
                                   BenchResult& r) {
    std::sort(latencies.begin(), latencies.end());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    r.searchMean = sum / latencies.size();
    r.searchP50  = latencies[latencies.size() / 2];
    r.searchP95  = latencies[(size_t)(latencies.size() * 0.95)];
    r.searchP99  = latencies[(size_t)(latencies.size() * 0.99)];
}

static double recall_at_k(const std::vector<uint64_t>& retrieved,
                           const uint64_t* groundTruth, int k) {
    int hits = 0;
    for (int i = 0; i < k && i < (int)retrieved.size(); ++i) {
        for (int j = 0; j < k; ++j) {
            if (retrieved[i] == groundTruth[j]) { ++hits; break; }
        }
    }
    return (double)hits / k;
}

static std::string result_to_json(const BenchResult& r) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        R"({"engine":"%s","insertTimeSec":%.3f,"insertThroughput":%.0f,)"
        R"("indexBuildTimeSec":%.3f,)"
        R"("searchLatencyMs":{"mean":%.3f,"p50":%.3f,"p95":%.3f,"p99":%.3f},)"
        R"("recallAt10":%.4f,"diskUsageBytes":%zu,"diskUsageMB":%.2f,)"
        R"("memoryDeltaMB":%.2f,"coldStartMs":%.2f})",
        r.engine.c_str(), r.insertTimeSec, r.insertThroughput,
        r.indexBuildSec,
        r.searchMean, r.searchP50, r.searchP95, r.searchP99,
        r.recallAt10, r.diskUsageBytes, r.diskUsageBytes / 1e6,
        r.memoryDeltaMB, r.coldStartMs
    );
    return buf;
}

// ── Benchmark: Flat (brute-force) ───────────────────────────

static BenchResult bench_flat(const std::vector<float>& vectors,
                               const std::vector<float>& queries,
                               const std::vector<uint64_t>& groundTruth) {
    BenchResult r;
    r.engine = "Flat (brute-force)";
    std::printf("\n══ %s ══\n", r.engine.c_str());

    evs::FlatSearch flat;
    flat.reserve(NUM_VECTORS, DIMS);

    double rss0 = rss_mb();
    auto t0 = Clock::now();
    for (size_t i = 0; i < NUM_VECTORS; ++i)
        flat.add(i, &vectors[i * DIMS]);
    auto t1 = Clock::now();
    r.insertTimeSec    = std::chrono::duration<double>(t1 - t0).count();
    r.insertThroughput = NUM_VECTORS / r.insertTimeSec;
    r.memoryDeltaMB    = rss_mb() - rss0;

    std::printf("  Insert: %.2fs (%'.0f vec/s)\n", r.insertTimeSec, r.insertThroughput);

    // Search
    std::vector<double> latencies(NUM_QUERIES);
    double recall_sum = 0;
    for (int q = 0; q < NUM_QUERIES; ++q) {
        auto ts = Clock::now();
        auto results = flat.search(&queries[q * DIMS], TOP_K);
        auto te = Clock::now();
        latencies[q] = std::chrono::duration<double, std::milli>(te - ts).count();

        std::vector<uint64_t> ids;
        for (auto& rr : results) ids.push_back(rr.key);
        recall_sum += recall_at_k(ids, &groundTruth[q * TOP_K], TOP_K);
    }
    compute_latency_stats(latencies, r);
    r.recallAt10 = recall_sum / NUM_QUERIES;

    std::printf("  Search: mean=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms\n",
                r.searchMean, r.searchP50, r.searchP95, r.searchP99);
    std::printf("  Recall@10: %.4f\n", r.recallAt10);

    return r;
}

// ── Benchmark: Raw USearch ──────────────────────────────────

static BenchResult bench_usearch(const std::vector<float>& vectors,
                                  const std::vector<float>& queries,
                                  const std::vector<uint64_t>& groundTruth,
                                  const std::string& tmpDir) {
    BenchResult r;
    r.engine = "USearch (raw HNSW)";
    std::printf("\n══ %s ══\n", r.engine.c_str());

    evs::StoreConfig cfg;
    cfg.dimensions = DIMS;
    cfg.quantization = evs::Quantization::I8;
    cfg.metric = evs::Metric::Cosine;
    cfg.connectivity = 16;
    cfg.expansionAdd = 128;
    cfg.expansionSearch = 64;

    evs::ANNEngine engine;
    engine.init(cfg);

    double rss0 = rss_mb();
    auto t0 = Clock::now();
    for (size_t i = 0; i < NUM_VECTORS; ++i)
        engine.add(i, &vectors[i * DIMS]);
    auto t1 = Clock::now();
    r.insertTimeSec    = std::chrono::duration<double>(t1 - t0).count();
    r.insertThroughput = NUM_VECTORS / r.insertTimeSec;
    r.memoryDeltaMB    = rss_mb() - rss0;

    std::printf("  Insert: %.2fs (%'.0f vec/s)\n", r.insertTimeSec, r.insertThroughput);

    // Save and measure disk
    std::string indexPath = tmpDir + "/usearch_raw.idx";
    engine.save(indexPath);
    r.diskUsageBytes = fs::file_size(indexPath);

    // Search
    std::vector<double> latencies(NUM_QUERIES);
    double recall_sum = 0;
    for (int q = 0; q < NUM_QUERIES; ++q) {
        auto ts = Clock::now();
        auto results = engine.search(&queries[q * DIMS], TOP_K);
        auto te = Clock::now();
        latencies[q] = std::chrono::duration<double, std::milli>(te - ts).count();

        std::vector<uint64_t> ids;
        for (auto& rr : results) ids.push_back(rr.key);
        recall_sum += recall_at_k(ids, &groundTruth[q * TOP_K], TOP_K);
    }
    compute_latency_stats(latencies, r);
    r.recallAt10 = recall_sum / NUM_QUERIES;

    std::printf("  Search: mean=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms\n",
                r.searchMean, r.searchP50, r.searchP95, r.searchP99);
    std::printf("  Recall@10: %.4f\n", r.recallAt10);
    std::printf("  Disk: %.2f MB\n", r.diskUsageBytes / 1e6);

    // Cold start
    engine.close();
    evs::ANNEngine cold;
    cold.init(cfg);
    auto cs0 = Clock::now();
    cold.load(indexPath);
    auto cs1 = Clock::now();
    r.coldStartMs = std::chrono::duration<double, std::milli>(cs1 - cs0).count();
    std::printf("  Cold start: %.2fms\n", r.coldStartMs);

    return r;
}

// ── Benchmark: EdgeVectorStore (full stack) ─────────────────

static BenchResult bench_evs(const std::vector<float>& vectors,
                              const std::vector<float>& queries,
                              const std::vector<uint64_t>& groundTruth,
                              const std::string& tmpDir) {
    BenchResult r;
    r.engine = "EdgeVectorStore";
    std::printf("\n══ %s ══\n", r.engine.c_str());

    std::string storePath = tmpDir + "/evs_store";
    fs::create_directories(storePath);

    evs::EdgeStore store;
    std::ostringstream cfgJson;
    cfgJson << R"({"storagePath":")" << storePath
            << R"(","dimensions":)" << DIMS
            << R"(,"quantization":"i8","metric":"cosine"})";
    store.init(cfgJson.str());

    double rss0 = rss_mb();

    // Insert in batches of 1000 for realism
    auto t0 = Clock::now();
    const int BATCH = 1000;
    for (size_t batch_start = 0; batch_start < NUM_VECTORS; batch_start += BATCH) {
        size_t batch_end = std::min(batch_start + (size_t)BATCH, NUM_VECTORS);

        std::ostringstream docs;
        docs << "[";
        for (size_t i = batch_start; i < batch_end; ++i) {
            if (i > batch_start) docs << ",";
            docs << R"({"id":"f)" << i << R"(","payload":{"idx":)" << i << "}}";
        }
        docs << "]";
        store.upsertDocuments(docs.str());

        std::ostringstream vecs;
        vecs << "[";
        for (size_t i = batch_start; i < batch_end; ++i) {
            if (i > batch_start) vecs << ",";
            vecs << R"({"id":"f)" << i << R"(","vector":[)";
            for (int d = 0; d < DIMS; ++d) {
                if (d > 0) vecs << ",";
                vecs << vectors[i * DIMS + d];
            }
            vecs << "]}";
        }
        vecs << "]";
        store.upsertVectors(vecs.str());

        if ((batch_start / BATCH + 1) % 20 == 0)
            std::printf("  Inserted %zu/%zu...\n", batch_end, NUM_VECTORS);
    }
    auto t1 = Clock::now();
    r.insertTimeSec    = std::chrono::duration<double>(t1 - t0).count();
    r.insertThroughput = NUM_VECTORS / r.insertTimeSec;
    r.memoryDeltaMB    = rss_mb() - rss0;

    std::printf("  Insert: %.2fs (%'.0f vec/s)\n", r.insertTimeSec, r.insertThroughput);

    // Compact (flush hot → cold)
    std::printf("  Compacting...\n");
    auto tc0 = Clock::now();
    store.compact();
    auto tc1 = Clock::now();
    r.indexBuildSec = std::chrono::duration<double>(tc1 - tc0).count();
    std::printf("  Compact: %.2fs\n", r.indexBuildSec);

    // Measure disk
    size_t total = 0;
    for (auto& entry : fs::recursive_directory_iterator(storePath)) {
        if (entry.is_regular_file())
            total += entry.file_size();
    }
    r.diskUsageBytes = total;

    // Search
    std::vector<double> latencies(NUM_QUERIES);
    double recall_sum = 0;
    for (int q = 0; q < NUM_QUERIES; ++q) {
        std::ostringstream sq;
        sq << R"({"storagePath":")" << storePath << R"(","vector":[)";
        for (int d = 0; d < DIMS; ++d) {
            if (d > 0) sq << ",";
            sq << queries[q * DIMS + d];
        }
        sq << R"(],"topK":)" << TOP_K << "}";

        auto ts = Clock::now();
        std::string res = store.search(sq.str());
        auto te = Clock::now();
        latencies[q] = std::chrono::duration<double, std::milli>(te - ts).count();

        // Parse result IDs for recall — extract "id":"fNNN" patterns
        std::vector<uint64_t> ids;
        size_t pos = 0;
        while ((pos = res.find("\"id\":\"f", pos)) != std::string::npos) {
            pos += 7; // skip "id":"f
            size_t end = res.find('"', pos);
            if (end != std::string::npos) {
                uint64_t id = std::strtoull(res.c_str() + pos, nullptr, 10);
                ids.push_back(id);
            }
            pos = end;
        }
        recall_sum += recall_at_k(ids, &groundTruth[q * TOP_K], TOP_K);
    }
    compute_latency_stats(latencies, r);
    r.recallAt10 = recall_sum / NUM_QUERIES;

    std::printf("  Search: mean=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms\n",
                r.searchMean, r.searchP50, r.searchP95, r.searchP99);
    std::printf("  Recall@10: %.4f\n", r.recallAt10);
    std::printf("  Disk: %.2f MB\n", r.diskUsageBytes / 1e6);

    // Cold start
    store.close();
    evs::EdgeStore cold;
    auto cs0 = Clock::now();
    cold.init(cfgJson.str());
    // Run one search to fully warm
    {
        std::ostringstream sq;
        sq << R"({"storagePath":")" << storePath << R"(","vector":[)";
        for (int d = 0; d < DIMS; ++d) {
            if (d > 0) sq << ",";
            sq << queries[0];
        }
        sq << R"(],"topK":1})";
        cold.search(sq.str());
    }
    auto cs1 = Clock::now();
    r.coldStartMs = std::chrono::duration<double, std::milli>(cs1 - cs0).count();
    std::printf("  Cold start: %.2fms\n", r.coldStartMs);
    cold.close();

    return r;
}

// ── Main ────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // Find data directory (relative to executable or project root)
    std::string dataDir;
    // Try relative to CWD first
    for (auto candidate : {"data", "../../data", "../data"}) {
        if (fs::exists(std::string(candidate) + "/farmers_100k_vectors.bin")) {
            dataDir = candidate;
            break;
        }
    }
    if (dataDir.empty()) {
        std::fprintf(stderr, "Cannot find data/ directory with vector files.\n"
                             "Run generate_dataset.py and convert_vectors.py first.\n");
        return 1;
    }

    std::printf("=== EdgeVectorStore Benchmark ===\n");
    std::printf("Data directory: %s\n", fs::canonical(dataDir).c_str());
    std::printf("Vectors: %zu x %d, Queries: %d, Top-K: %d\n\n",
                NUM_VECTORS, DIMS, NUM_QUERIES, TOP_K);

    // Load data
    std::printf("Loading vectors...\n");
    auto vectors = load_bin_f32(dataDir + "/farmers_100k_vectors.bin",
                                NUM_VECTORS * DIMS);

    std::printf("Loading queries...\n");
    auto queries = load_bin_f32(dataDir + "/farmers_100k_queries.bin",
                                NUM_QUERIES * DIMS);

    std::printf("Loading ground-truth...\n");
    auto gt = load_bin_u64(dataDir + "/farmers_100k_groundtruth.bin",
                           NUM_QUERIES * TOP_K);

    // Temp directory for indexes
    std::string tmpDir = (fs::temp_directory_path() / "evs_bench").string();
    fs::create_directories(tmpDir);

    // Run benchmarks
    auto flatResult   = bench_flat(vectors, queries, gt);
    auto usearchResult = bench_usearch(vectors, queries, gt, tmpDir);
    auto evsResult    = bench_evs(vectors, queries, gt, tmpDir);

    // Output JSON
    std::string jsonOut = "{\"engines\":[";
    jsonOut += result_to_json(flatResult) + ",";
    jsonOut += result_to_json(usearchResult) + ",";
    jsonOut += result_to_json(evsResult);
    jsonOut += "]}";

    std::string outPath = dataDir + "/bench_cpp.json";
    {
        std::ofstream f(outPath);
        f << jsonOut;
    }

    std::printf("\n\n═══════════════════════════════════════════\n");
    std::printf("  Results saved to %s\n", outPath.c_str());
    std::printf("═══════════════════════════════════════════\n");

    // Summary table
    std::printf("\n%-25s %12s %10s %10s %10s %10s\n",
                "Engine", "Insert/s", "p50(ms)", "p95(ms)", "Recall@10", "Disk(MB)");
    std::printf("%-25s %12s %10s %10s %10s %10s\n",
                "─────────────────────────", "────────────", "──────────",
                "──────────", "──────────", "──────────");
    for (auto& r : {flatResult, usearchResult, evsResult}) {
        std::printf("%-25s %12.0f %10.3f %10.3f %10.4f %10.2f\n",
                    r.engine.c_str(), r.insertThroughput,
                    r.searchP50, r.searchP95, r.recallAt10,
                    r.diskUsageBytes / 1e6);
    }

    // Cleanup
    fs::remove_all(tmpDir);

    return 0;
}

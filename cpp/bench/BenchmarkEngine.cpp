// ──────────────────────────────────────────────────────────────
//  BenchmarkEngine — raw HNSW benchmark (build once, sweep ef)
// ──────────────────────────────────────────────────────────────
#include "BenchmarkEngine.h"
#include "../util/JsonParser.h"
#include "../core/Config.h"
#include "../core/Types.h"
#include "../ann/ANNEngine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <set>
#include <sstream>
#include <vector>

namespace evs {

using json::getString;
using json::getInt;
using json::parseFlat;

std::string benchmarkRawANN(const std::string& configJson) {
    auto kvs = parseFlat(configJson);
    int dims         = getInt("dims", kvs, 384);
    int numVectors   = getInt("numVectors", kvs, 10000);
    int numQueries   = getInt("numQueries", kvs, 100);
    int topK         = getInt("topK", kvs, 10);
    int warmup       = getInt("warmup", kvs, 10);
    bool doRerank    = getString("rerank", kvs) == "true";
    int oversample   = getInt("oversample", kvs, 3);
    int connectivity = getInt("connectivity", kvs, 16);
    int expansionAdd = getInt("expansionAdd", kvs, 128);

    std::string qstr = getString("quantization", kvs);
    if (qstr.empty()) qstr = "f32";

    Quantization quant = Quantization::F32;
    if (qstr == "f16") quant = Quantization::F16;
    else if (qstr == "i8") quant = Quantization::I8;
    else if (qstr == "b1") quant = Quantization::B1;

    // Parse sweepEf: comma-separated list of ef values
    std::string sweepStr = getString("sweepEf", kvs);
    std::vector<int> efValues;
    if (!sweepStr.empty()) {
        std::istringstream iss(sweepStr);
        std::string token;
        while (std::getline(iss, token, ',')) {
            int v = std::stoi(token);
            if (v > 0) efValues.push_back(v);
        }
    }
    if (efValues.empty()) {
        efValues.push_back(getInt("ef", kvs, 64));
    }

    std::string buildLabel = "M" + std::to_string(connectivity)
                           + "_A" + std::to_string(expansionAdd);

    StoreConfig cfg;
    cfg.dimensions      = dims;
    cfg.metric          = Metric::Cosine;
    cfg.quantization    = quant;
    cfg.connectivity    = static_cast<int32_t>(connectivity);
    cfg.expansionAdd    = static_cast<int32_t>(expansionAdd);
    cfg.expansionSearch = static_cast<int32_t>(efValues[0]);

    ANNEngine engine;
    if (!engine.init(cfg)) {
        return R"([{"error":"Failed to init ANNEngine"}])";
    }

    // Generate random vectors (seeded for reproducibility)
    std::srand(42);
    std::vector<std::vector<float>> vecs(numVectors, std::vector<float>(dims));
    for (int i = 0; i < numVectors; i++) {
        float norm = 0;
        for (int d = 0; d < dims; d++) {
            vecs[i][d] = ((float)std::rand() / RAND_MAX) * 2.0f - 1.0f;
            norm += vecs[i][d] * vecs[i][d];
        }
        norm = std::sqrt(norm);
        for (int d = 0; d < dims; d++) vecs[i][d] /= norm;
    }

    // Build
    auto tBuildStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numVectors; i++) {
        engine.add(static_cast<uint64_t>(i), vecs[i].data());
    }
    auto tBuildEnd = std::chrono::high_resolution_clock::now();
    double buildSec = std::chrono::duration<double>(tBuildEnd - tBuildStart).count();

    // Generate queries
    std::vector<std::vector<float>> queries(numQueries, std::vector<float>(dims));
    for (int q = 0; q < numQueries; q++) {
        float norm = 0;
        for (int d = 0; d < dims; d++) {
            queries[q][d] = ((float)std::rand() / RAND_MAX) * 2.0f - 1.0f;
            norm += queries[q][d] * queries[q][d];
        }
        norm = std::sqrt(norm);
        for (int d = 0; d < dims; d++) queries[q][d] /= norm;
    }

    // Brute-force ground truth (computed once)
    std::vector<std::set<int>> gtSets(numQueries);
    for (int q = 0; q < numQueries; q++) {
        std::vector<std::pair<float, int>> scores(numVectors);
        for (int i = 0; i < numVectors; i++) {
            float dot = 0;
            for (int d = 0; d < dims; d++) dot += queries[q][d] * vecs[i][d];
            scores[i] = {dot, i};
        }
        std::sort(scores.begin(), scores.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });
        for (int k = 0; k < topK && k < numVectors; k++) {
            gtSets[q].insert(scores[k].second);
        }
    }

    double memMB = engine.memoryUsage() / 1e6;

    // Sweep ef values
    std::ostringstream oss;
    oss << "[";

    for (size_t ei = 0; ei < efValues.size(); ei++) {
        int ef = efValues[ei];
        engine.setExpansionSearch(static_cast<size_t>(ef));

        // Warmup
        for (int w = 0; w < warmup && w < numQueries; w++) {
            engine.search(queries[w].data(), static_cast<size_t>(topK));
        }

        size_t fetchK = doRerank
            ? static_cast<size_t>(topK * oversample)
            : static_cast<size_t>(topK);

        std::vector<double> latencies(numQueries);
        int hits = 0;
        int total = 0;

        for (int q = 0; q < numQueries; q++) {
            auto ts = std::chrono::high_resolution_clock::now();

            auto coarse = engine.search(queries[q].data(), fetchK);

            std::vector<int> annKeys;
            if (doRerank && coarse.size() > static_cast<size_t>(topK)) {
                struct Scored { int idx; float dist; };
                std::vector<Scored> scored;
                scored.reserve(coarse.size());
                for (auto& c : coarse) {
                    int idx = static_cast<int>(c.key);
                    float dot = 0;
                    for (int d = 0; d < dims; d++)
                        dot += queries[q][d] * vecs[idx][d];
                    scored.push_back({idx, 1.0f - dot});
                }
                std::sort(scored.begin(), scored.end(),
                          [](auto& a, auto& b) { return a.dist < b.dist; });
                int keep = std::min(topK, static_cast<int>(scored.size()));
                for (int k = 0; k < keep; k++)
                    annKeys.push_back(scored[k].idx);
            } else {
                int keep = std::min(topK, static_cast<int>(coarse.size()));
                for (int k = 0; k < keep; k++)
                    annKeys.push_back(static_cast<int>(coarse[k].key));
            }

            auto te = std::chrono::high_resolution_clock::now();
            latencies[q] = std::chrono::duration<double, std::milli>(te - ts).count();

            for (int idx : annKeys) {
                if (gtSets[q].count(idx)) hits++;
            }
            total += topK;
        }

        std::sort(latencies.begin(), latencies.end());
        double mean = 0;
        for (auto v : latencies) mean += v;
        mean /= latencies.size();
        double p50 = latencies[latencies.size() / 2];
        double p95 = latencies[(size_t)(latencies.size() * 0.95)];
        double p99 = latencies[(size_t)(latencies.size() * 0.99)];
        double recall = (total > 0) ? (double)hits / total : 0.0;
        double qps = (mean > 0) ? 1000.0 / mean : 0.0;

        if (ei > 0) oss << ",";
        oss << "{";
        oss << "\"ef\":" << ef << ",";
        oss << "\"recallAtK\":" << recall << ",";
        oss << "\"searchMean\":" << mean << ",";
        oss << "\"searchP50\":" << p50 << ",";
        oss << "\"searchP95\":" << p95 << ",";
        oss << "\"searchP99\":" << p99 << ",";
        oss << "\"qps\":" << qps << ",";
        oss << "\"buildTimeSec\":" << buildSec << ",";
        oss << "\"insertThroughput\":" << (numVectors / buildSec) << ",";
        oss << "\"memoryMB\":" << memMB << ",";
        oss << "\"numVectors\":" << numVectors << ",";
        oss << "\"dims\":" << dims << ",";
        oss << "\"connectivity\":" << connectivity << ",";
        oss << "\"expansionAdd\":" << expansionAdd << ",";
        oss << "\"rerank\":" << (doRerank ? "true" : "false") << ",";
        oss << "\"oversample\":" << oversample << ",";
        oss << "\"quantization\":\"" << qstr << "\",";
        oss << "\"buildConfig\":\"" << buildLabel << "\"";
        oss << "}";
    }

    oss << "]";
    engine.close();
    return oss.str();
}

} // namespace evs

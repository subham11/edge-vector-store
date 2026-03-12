// ──────────────────────────────────────────────────────────────
//  EdgeStore — implementation
// ──────────────────────────────────────────────────────────────
#include "EdgeStore.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "../util/JsonParser.h"

// Bring shared JSON helpers into scope
using evs::json::getString;
using evs::json::getInt;
using evs::json::parseFlat;
using evs::json::parseFloatArray;
using evs::json::parseStringArray;
using evs::json::parseObjectArray;
using evs::json::toJsonString;

namespace evs {

// ── helpers ──────────────────────────────────────────────────

std::string EdgeStore::coldIndexPath() const {
    return cfg_.storagePath + "/cold.usearch";
}

// ── lifecycle ────────────────────────────────────────────────

EdgeStore::~EdgeStore() {
    close();
}

bool EdgeStore::init(const std::string& configJson) {
    std::lock_guard<std::mutex> lock(mu_);

    auto kv = parseFlat(configJson);
    cfg_.storagePath     = getString("storagePath", kv);
    cfg_.dimensions      = getInt("dimensions", kv, 768);
    cfg_.hotCacheCapacity = getInt("hotCacheCapacity", kv, 10000);
    cfg_.connectivity    = getInt("connectivity", kv, 16);
    cfg_.expansionAdd    = getInt("expansionAdd", kv, 128);
    cfg_.expansionSearch = getInt("expansionSearch", kv, 64);

    std::string q = getString("quantization", kv);
    if (q == "f32")      cfg_.quantization = Quantization::F32;
    else if (q == "f16") cfg_.quantization = Quantization::F16;
    else if (q == "b1")  cfg_.quantization = Quantization::B1;
    else                 cfg_.quantization = Quantization::I8;

    std::string m = getString("metric", kv);
    if (m == "euclidean")      cfg_.metric = Metric::Euclidean;
    else if (m == "ip")        cfg_.metric = Metric::InnerProduct;
    else                       cfg_.metric = Metric::Cosine;

    std::string p = getString("profile", kv);
    if (p == "memory_saver")      cfg_.profile = SearchProfile::MemorySaver;
    else if (p == "max_recall")   cfg_.profile = SearchProfile::MaxRecall;
    else if (p == "max_speed")    cfg_.profile = SearchProfile::MaxSpeed;
    else                          cfg_.profile = SearchProfile::Balanced;

    // Open metadata
    if (!meta_.open(cfg_.storagePath)) return false;

    // Open float32 vector store (for reranking)
    vectors_.open(cfg_.storagePath, cfg_.dimensions);

    // Open binary journal
    if (!journal_.open(cfg_.storagePath)) return false;

    // Initialise cold index
    if (!cold_.init(cfg_)) return false;

    // Try mmap'ing existing cold index (zero-copy, read-only).
    // Falls back to heap load if mmap isn't available.
    std::string coldPath = coldIndexPath();
    std::ifstream test(coldPath);
    if (test.good()) {
        test.close();
        if (!cold_.view(coldPath))
            cold_.load(coldPath);
    }

    // Initialise hot cache
    if (!hot_.init(cfg_)) return false;

    // Replay journal to recover any un-compacted writes
    recoverFromJournal();

    return true;
}

void EdgeStore::close() {
    std::lock_guard<std::mutex> lock(mu_);
    // Save cold index (only if it was heap-loaded, not mmap'd)
    if (cold_.isOpen() && !cold_.isMmap() && cold_.size() > 0) {
        cold_.save(coldIndexPath());
    }
    cold_.close();
    vectors_.save();
    vectors_.close();
    journal_.close();
    meta_.close();
}

// ── recovery ─────────────────────────────────────────────────

void EdgeStore::recoverFromJournal() {
    std::string seqStr = meta_.getStat(lastSeqKey());
    int64_t lastSeq = seqStr.empty() ? 0 : std::atoll(seqStr.c_str());

    IDMapper mapper(meta_);

    journal_.replay(lastSeq, [&](const JournalEntry& e) {
        if (e.op == JournalOp::Upsert && !e.vector.empty()) {
            uint64_t key = mapper.toNumericKey(e.targetId);
            hot_.put(key, e.vector.data(), cfg_.dimensions);
        } else if (e.op == JournalOp::Delete) {
            uint64_t key = meta_.numericKeyFor(e.targetId);
            if (key != 0) {
                hot_.remove(key);
                cold_.remove(key);
            }
        }
    });
}

// ── document API ─────────────────────────────────────────────

bool EdgeStore::upsertDocuments(const std::string& docsJson) {
    std::lock_guard<std::mutex> lock(mu_);
    auto docs = parseObjectArray(docsJson);
    for (auto& d : docs) {
        std::string id = getString("id", d);
        if (id.empty()) continue;
        std::string payload = "{}";
        auto pit = d.find("payload");
        if (pit != d.end()) payload = pit->second;
        meta_.upsertDocument(id, payload);
    }
    return true;
}

bool EdgeStore::upsertVectors(const std::string& entriesJson) {
    std::lock_guard<std::mutex> lock(mu_);

    auto entries = parseObjectArray(entriesJson);
    IDMapper mapper(meta_);

    for (auto& e : entries) {
        std::string id = getString("id", e);
        if (id.empty()) continue;

        auto vecStr = getString("vector", e);
        if (vecStr.empty()) {
            auto vit = e.find("vector");
            if (vit != e.end()) vecStr = vit->second;
        }
        std::vector<float> vec = parseFloatArray(vecStr);
        if (vec.empty()) continue;

        uint64_t key = mapper.toNumericKey(id);

        // Hot cache (evicted keys will be flushed on compact)
        hot_.put(key, vec.data(), static_cast<int>(vec.size()));

        // Float32 store for reranking
        vectors_.put(key, vec.data());

        // Journal (binary append — no locking conflicts)
        journal_.appendUpsert(id, vec.data(),
                              static_cast<int>(vec.size()));
    }
    journal_.flush();
    return true;
}

// ── direct API (no JSON) ─────────────────────────────────────

std::vector<SearchResult> EdgeStore::searchDirect(
    const float* query, int topK, SearchProfile profile,
    bool includePayload) const {

    std::lock_guard<std::mutex> lock(mu_);

    TieredSearch tiered(
        const_cast<HotCache&>(hot_),
        const_cast<ANNEngine&>(cold_),
        cfg_,
        const_cast<VectorStore*>(&vectors_));
    auto raw = tiered.search(query, static_cast<size_t>(topK), profile);

    std::vector<SearchResult> results;
    results.reserve(raw.size());
    for (auto& r : raw) {
        SearchResult sr;
        sr.id = meta_.docIdFor(r.key);
        sr.distance = r.distance;
        if (includePayload && !sr.id.empty()) {
            sr.payload = meta_.getDocument(sr.id);
        }
        results.push_back(std::move(sr));
    }
    return results;
}

bool EdgeStore::upsertVectorsDirect(const std::string* ids,
                                     const float* vectors,
                                     size_t count, size_t dims) {
    std::lock_guard<std::mutex> lock(mu_);

    IDMapper mapper(meta_);
    for (size_t i = 0; i < count; ++i) {
        const std::string& id = ids[i];
        const float* vec = vectors + i * dims;
        uint64_t key = mapper.toNumericKey(id);

        hot_.put(key, vec, static_cast<int>(dims));
        vectors_.put(key, vec);
        journal_.appendUpsert(id, vec, static_cast<int>(dims));
    }
    journal_.flush();
    return true;
}

std::string EdgeStore::search(const std::string& optionsJson) const {
    std::lock_guard<std::mutex> lock(mu_);

    auto kv = parseFlat(optionsJson);
    auto vecStr = getString("vector", kv);
    if (vecStr.empty()) {
        vecStr = getString("queryVector", kv);
    }
    if (vecStr.empty()) {
        auto vit = kv.find("vector");
        if (vit != kv.end()) vecStr = vit->second;
    }
    if (vecStr.empty()) {
        auto vit = kv.find("queryVector");
        if (vit != kv.end()) vecStr = vit->second;
    }
    std::vector<float> query = parseFloatArray(vecStr);
    if (query.empty()) return "[]";

    int topK = getInt("topK", kv, 10);
    bool includePayload = getString("includePayload", kv) != "false";

    SearchProfile profile = cfg_.profile;
    std::string pStr = getString("profile", kv);
    if (pStr.empty()) pStr = getString("mode", kv);
    if (pStr == "memory_saver")      profile = SearchProfile::MemorySaver;
    else if (pStr == "max_recall")   profile = SearchProfile::MaxRecall;
    else if (pStr == "max_speed")    profile = SearchProfile::MaxSpeed;
    else if (pStr == "balanced")     profile = SearchProfile::Balanced;

    TieredSearch tiered(
        const_cast<HotCache&>(hot_),
        const_cast<ANNEngine&>(cold_),
        cfg_,
        const_cast<VectorStore*>(&vectors_));
    auto results = tiered.search(query.data(),
                                  static_cast<size_t>(topK), profile);

    // Build JSON result directly into pre-allocated string
    std::string out;
    out.reserve(results.size() * 80); // ~80 bytes per result entry
    out += '[';

    char distBuf[32];
    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) out += ',';

        // Direct reverse-lookup per key (O(1) hash map)
        std::string docIdStr = meta_.docIdFor(results[i].key);

        out += "{\"id\":";
        out += toJsonString(docIdStr);
        out += ",\"distance\":";
        int n = std::snprintf(distBuf, sizeof(distBuf), "%.6g",
                              results[i].distance);
        out.append(distBuf, static_cast<size_t>(n));

        if (includePayload && !docIdStr.empty()) {
            std::string payload = meta_.getDocument(docIdStr);
            if (!payload.empty()) {
                out += ",\"payload\":";
                out += payload;
            }
        }
        out += '}';
    }
    out += ']';
    return out;
}

bool EdgeStore::remove(const std::string& idsJson) {
    std::lock_guard<std::mutex> lock(mu_);

    auto ids = parseStringArray(idsJson);
    for (auto& id : ids) {
        uint64_t key = meta_.numericKeyFor(id);
        if (key != 0) {
            journal_.appendDelete(id);
            hot_.remove(key);
            cold_.remove(key);
            vectors_.remove(key);
            meta_.unregisterVector(id);
        }
        meta_.deleteDocument(id);
    }
    journal_.flush();
    return true;
}

// ── maintenance ──────────────────────────────────────────────

bool EdgeStore::compact() {
    std::lock_guard<std::mutex> lock(mu_);

    // 1. Transition cold index from mmap (read-only) to heap (writable)
    //    so we can add journal vectors.
    cold_.close();
    cold_.init(cfg_);
    std::string coldPath = coldIndexPath();
    {
        std::ifstream test(coldPath);
        if (test.good()) {
            test.close();
            cold_.load(coldPath);
        }
    }

    // 2. Drain hot cache → cold index
    auto evictedKeys = hot_.drain();
    std::string seqStr = meta_.getStat(lastSeqKey());
    int64_t lastSeq = seqStr.empty() ? 0 : std::atoll(seqStr.c_str());

    IDMapper mapper(meta_);

    journal_.replay(lastSeq, [&](const JournalEntry& e) {
        if (e.op == JournalOp::Upsert && !e.vector.empty()) {
            uint64_t key = mapper.toNumericKey(e.targetId);
            cold_.add(key, e.vector.data());
        } else if (e.op == JournalOp::Delete) {
            uint64_t key = meta_.numericKeyFor(e.targetId);
            if (key != 0) cold_.remove(key);
        }
    });

    // 3. Save cold index, then switch back to mmap for fast reads
    cold_.save(coldPath);
    cold_.close();
    cold_.init(cfg_);
    if (!cold_.view(coldPath))
        cold_.load(coldPath);

    // 4. Flush vector store to disk
    vectors_.save();

    // 5. Truncate journal
    int64_t hw = journal_.highWatermark();
    journal_.truncate(hw);
    meta_.setStat(lastSeqKey(), std::to_string(hw));

    // 6. VACUUM
    meta_.vacuum();

    return true;
}

std::string EdgeStore::getStats() const {
    std::lock_guard<std::mutex> lock(mu_);

    StoreStats s;
    s.documentCount     = meta_.documentCount();
    s.vectorCount       = meta_.vectorCount();
    s.hotCacheCount     = static_cast<int64_t>(hot_.size());
    s.memoryUsageBytes  = static_cast<int64_t>(cold_.memoryUsage());
    s.dimensions        = cfg_.dimensions;
    s.quantization      = cfg_.quantization;

    // Approximate cold index file size
    std::string coldPath = coldIndexPath();
    std::ifstream f(coldPath, std::ios::binary | std::ios::ate);
    if (f.good()) {
        s.coldIndexSizeBytes = static_cast<int64_t>(f.tellg());
    }

    auto qStr = [](Quantization q) -> const char* {
        switch (q) {
            case Quantization::F32: return "f32";
            case Quantization::F16: return "f16";
            case Quantization::I8:  return "i8";
            case Quantization::B1:  return "b1";
        }
        return "i8";
    };

    std::ostringstream out;
    out << "{\"documentCount\":" << s.documentCount
        << ",\"vectorCount\":" << s.vectorCount
        << ",\"hotCacheCount\":" << s.hotCacheCount
        << ",\"memoryUsageBytes\":" << s.memoryUsageBytes
        << ",\"coldIndexSizeBytes\":" << s.coldIndexSizeBytes
        << ",\"dimensions\":" << s.dimensions
        << ",\"quantization\":\"" << qStr(s.quantization) << "\"}";
    return out.str();
}

// ── pack import / export ─────────────────────────────────────

bool EdgeStore::importPack(const std::string& path) {
    std::lock_guard<std::mutex> lock(mu_);

    PackManifest manifest;
    if (!PackReader::read(path, cfg_.storagePath, manifest))
        return false;

    // Reload cold index from the extracted file (mmap for fast reads)
    cold_.close();
    cold_.init(cfg_);
    if (!cold_.view(coldIndexPath()))
        cold_.load(coldIndexPath());

    // Reopen MetadataStore to load extracted binary files.
    meta_.close();
    meta_.open(cfg_.storagePath);

    // Reopen VectorStore
    vectors_.close();
    vectors_.open(cfg_.storagePath, cfg_.dimensions);

    // Reopen journal
    journal_.close();
    journal_.open(cfg_.storagePath);

    return true;
}

bool EdgeStore::exportPack(const std::string& path) {
    std::lock_guard<std::mutex> lock(mu_);

    // Ensure cold index is flushed to disk
    if (cold_.isOpen() && cold_.size() > 0)
        cold_.save(coldIndexPath());

    // Ensure vector store is flushed
    vectors_.save();

    // Ensure metadata binary files are on disk
    meta_.vacuum();

    PackManifest manifest;
    manifest.dimensions    = cfg_.dimensions;
    manifest.quantization  = cfg_.quantization;
    manifest.metric        = cfg_.metric;
    manifest.vectorCount   = meta_.vectorCount();
    manifest.documentCount = meta_.documentCount();

    return PackWriter::write(cfg_.storagePath, manifest, path);
}

} // namespace evs

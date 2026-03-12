// ──────────────────────────────────────────────────────────────
//  Smoke test — exercises EdgeStore without JSI
// ──────────────────────────────────────────────────────────────
#include "core/EdgeStore.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

static bool ok(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return false;
    }
    std::printf("  OK: %s\n", msg);
    return true;
}

static std::string makeConfig(const std::string& dir) {
    std::ostringstream o;
    o << R"({"storagePath":")" << dir
      << R"(","dimensions":384,"quantization":"i8","metric":"cosine"})";
    return o.str();
}

static std::string makeVectorEntry(const std::string& id,
                                    const std::vector<float>& vec) {
    std::ostringstream o;
    o << R"({"id":")" << id << R"(","vector":[)";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) o << ",";
        o << vec[i];
    }
    o << "]}";
    return o.str();
}

int main() {
    std::printf("=== EdgeVectorStore smoke test ===\n\n");

    // Create a temp directory
    std::string tmpDir = (fs::temp_directory_path() / "evs_smoke_test").string();
    fs::create_directories(tmpDir);

    // 1. Init
    std::printf("[init]\n");
    evs::EdgeStore store;
    if (!ok(store.init(makeConfig(tmpDir)), "store.init()")) return 1;

    // 2. Upsert documents
    std::printf("\n[upsert documents]\n");
    ok(store.upsertDocuments(
           R"([{"id":"doc1","payload":{"title":"Hello World"}},)"
           R"({"id":"doc2","payload":{"title":"Goodbye Moon"}}])"),
       "upsertDocuments (2 docs)");

    // 3. Upsert vectors  (384-dim random)
    std::printf("\n[upsert vectors]\n");
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    auto randVec = [&](int dims) {
        std::vector<float> v(dims);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    auto v1 = randVec(384), v2 = randVec(384);
    std::string entries = "[" + makeVectorEntry("doc1", v1) + "," +
                          makeVectorEntry("doc2", v2) + "]";
    ok(store.upsertVectors(entries), "upsertVectors (2 vectors)");

    // 4. Search
    std::printf("\n[search]\n");
    auto query = randVec(384);
    std::ostringstream sq;
    sq << R"({"storagePath":")" << tmpDir << R"(","vector":[)";
    for (size_t i = 0; i < query.size(); ++i) {
        if (i > 0) sq << ",";
        sq << query[i];
    }
    sq << R"(],"topK":2})";

    std::string results = store.search(sq.str());
    ok(!results.empty() && results != "[]", "search returned results");
    std::printf("  results: %s\n", results.c_str());

    // 5. Stats
    std::printf("\n[stats]\n");
    std::string stats = store.getStats();
    ok(!stats.empty(), "getStats returned JSON");
    std::printf("  stats: %s\n", stats.c_str());

    // 6. Compact
    std::printf("\n[compact]\n");
    ok(store.compact(), "compact()");

    // 7. Search after compact (cold index now)
    std::printf("\n[search after compact]\n");
    results = store.search(sq.str());
    ok(!results.empty() && results != "[]",
       "search after compact returned results");
    std::printf("  results: %s\n", results.c_str());

    // 8. Export pack
    std::printf("\n[export pack]\n");
    std::string packPath = tmpDir + "/test.evs";
    ok(store.exportPack(packPath), "exportPack()");
    ok(fs::exists(packPath), "pack file exists");
    std::printf("  pack size: %lld bytes\n",
                static_cast<long long>(fs::file_size(packPath)));

    // 9. Remove
    std::printf("\n[remove]\n");
    ok(store.remove(R"(["doc1"])"), "remove doc1");

    // ── Phase 1: Direct API tests (no JSON) ─────────────────

    // Re-insert vectors via direct API
    std::printf("\n[upsertVectorsDirect]\n");
    {
        auto v3 = randVec(384), v4 = randVec(384);
        std::string ids[] = {"doc1", "doc2"};

        // Pack vectors into a flat buffer (mimics Float32Array layout)
        std::vector<float> packed(384 * 2);
        std::memcpy(packed.data(), v3.data(), 384 * sizeof(float));
        std::memcpy(packed.data() + 384, v4.data(), 384 * sizeof(float));

        ok(store.upsertVectorsDirect(ids, packed.data(), 2, 384),
           "upsertVectorsDirect (2 vectors)");
    }

    // Search via direct API
    std::printf("\n[searchDirect]\n");
    {
        auto q = randVec(384);
        auto directResults = store.searchDirect(
            q.data(), 2, evs::SearchProfile::Balanced);
        ok(!directResults.empty(), "searchDirect returned results");
        for (auto& r : directResults) {
            std::printf("  id=%s distance=%.6f\n",
                        r.id.c_str(), r.distance);
        }
    }

    // 10. Cleanup
    store.close();
    fs::remove_all(tmpDir);

    std::printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}

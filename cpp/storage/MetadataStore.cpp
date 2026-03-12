// ──────────────────────────────────────────────────────────────
//  MetadataStore — in-memory hash maps + binary file persistence
//
//  Binary file formats (all little-endian, native on ARM/x86):
//
//  registry.bin — vector ID ↔ numeric key mappings
//    [uint32_t count]
//    per entry: [uint32_t id_len][char id[]][uint64_t key]
//
//  documents.bin — document payloads
//    [uint32_t count]
//    per entry: [uint32_t id_len][char id[]][uint32_t payload_len][char payload[]]
//
//  stats.bin — key-value stats
//    [uint32_t count]
//    per entry: [uint32_t key_len][char key[]][uint32_t val_len][char val[]]
// ──────────────────────────────────────────────────────────────
#include "MetadataStore.h"
#include "MmapFile.h"

#include <cstdio>
#include <cstring>

namespace evs {

// ── lifecycle ────────────────────────────────────────────────

bool MetadataStore::open(const std::string& dirPath) {
    dir_ = dirPath;
    dirty_ = false;
    docs_.clear();
    strToNum_.clear();
    numToStr_.clear();
    stats_.clear();
    loadFromDisk(); // OK if files don't exist yet
    return true;
}

void MetadataStore::close() {
    if (dirty_) saveToDisk();
    docs_.clear();
    strToNum_.clear();
    numToStr_.clear();
    stats_.clear();
    dirty_ = false;
}

MetadataStore::~MetadataStore() {
    close();
}

// ── document CRUD ────────────────────────────────────────────

bool MetadataStore::upsertDocument(const std::string& id,
                                    const std::string& payloadJson) {
    docs_[id] = payloadJson;
    dirty_ = true;
    return true;
}

bool MetadataStore::deleteDocument(const std::string& id) {
    unregisterVector(id);
    docs_.erase(id);
    dirty_ = true;
    return true;
}

std::string MetadataStore::getDocument(const std::string& id) const {
    auto it = docs_.find(id);
    return (it != docs_.end()) ? it->second : std::string();
}

int64_t MetadataStore::documentCount() const {
    return static_cast<int64_t>(docs_.size());
}

// ── vector bookkeeping ───────────────────────────────────────

bool MetadataStore::registerVector(const std::string& docId,
                                    uint64_t numericKey) {
    // Remove old mapping if this docId was previously registered
    auto oldIt = strToNum_.find(docId);
    if (oldIt != strToNum_.end()) {
        numToStr_.erase(oldIt->second);
    }
    strToNum_[docId] = numericKey;
    numToStr_[numericKey] = docId;
    dirty_ = true;
    return true;
}

bool MetadataStore::unregisterVector(const std::string& docId) {
    auto it = strToNum_.find(docId);
    if (it == strToNum_.end()) return true; // nothing to do
    numToStr_.erase(it->second);
    strToNum_.erase(it);
    dirty_ = true;
    return true;
}

uint64_t MetadataStore::numericKeyFor(const std::string& docId) const {
    auto it = strToNum_.find(docId);
    return (it != strToNum_.end()) ? it->second : 0;
}

std::string MetadataStore::docIdFor(uint64_t numericKey) const {
    auto it = numToStr_.find(numericKey);
    return (it != numToStr_.end()) ? it->second : std::string();
}

std::unordered_map<uint64_t, std::string> MetadataStore::docIdsFor(
    const std::vector<uint64_t>& numericKeys) const {
    std::unordered_map<uint64_t, std::string> result;
    result.reserve(numericKeys.size());
    for (auto k : numericKeys) {
        auto it = numToStr_.find(k);
        if (it != numToStr_.end()) result[k] = it->second;
    }
    return result;
}

int64_t MetadataStore::vectorCount() const {
    return static_cast<int64_t>(strToNum_.size());
}

// ── stats / misc ─────────────────────────────────────────────

bool MetadataStore::setStat(const std::string& key,
                             const std::string& value) {
    stats_[key] = value;
    dirty_ = true;
    return true;
}

std::string MetadataStore::getStat(const std::string& key) const {
    auto it = stats_.find(key);
    return (it != stats_.end()) ? it->second : std::string();
}

bool MetadataStore::vacuum() {
    return saveToDisk();
}

void MetadataStore::forEachVector(
    std::function<void(const std::string& docId, uint64_t key)> cb) const {
    for (auto& [docId, key] : strToNum_) {
        cb(docId, key);
    }
}

// ── binary persistence helpers (write: fwrite, read: mmap) ───

static inline void writeU32(FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
static inline void writeU64(FILE* f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
static inline void writeStr(FILE* f, const std::string& s) {
    uint32_t len = static_cast<uint32_t>(s.size());
    writeU32(f, len);
    if (len > 0) std::fwrite(s.data(), 1, len, f);
}

// ── mmap-based cursor for zero-copy reading ──────────────────

struct ReadCursor {
    const uint8_t* p;
    const uint8_t* end;

    bool hasBytes(size_t n) const { return p + n <= end; }

    bool readU32(uint32_t& v) {
        if (!hasBytes(4)) return false;
        std::memcpy(&v, p, 4);
        p += 4;
        return true;
    }

    bool readU64(uint64_t& v) {
        if (!hasBytes(8)) return false;
        std::memcpy(&v, p, 8);
        p += 8;
        return true;
    }

    bool readStr(std::string& s) {
        uint32_t len = 0;
        if (!readU32(len)) return false;
        if (!hasBytes(len)) return false;
        s.assign(reinterpret_cast<const char*>(p), len);
        p += len;
        return true;
    }
};

bool MetadataStore::saveToDisk() const {
    if (dir_.empty()) return false;

    // ── registry.bin ──
    {
        std::string path = dir_ + "/registry.bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        writeU32(f, static_cast<uint32_t>(strToNum_.size()));
        for (auto& [id, key] : strToNum_) {
            writeStr(f, id);
            writeU64(f, key);
        }
        std::fclose(f);
    }

    // ── documents.bin ──
    {
        std::string path = dir_ + "/documents.bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        writeU32(f, static_cast<uint32_t>(docs_.size()));
        for (auto& [id, payload] : docs_) {
            writeStr(f, id);
            writeStr(f, payload);
        }
        std::fclose(f);
    }

    // ── stats.bin ──
    {
        std::string path = dir_ + "/stats.bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        writeU32(f, static_cast<uint32_t>(stats_.size()));
        for (auto& [key, val] : stats_) {
            writeStr(f, key);
            writeStr(f, val);
        }
        std::fclose(f);
    }

    return true;
}

bool MetadataStore::loadFromDisk() {
    // Uses mmap for zero-copy reading — single mmap per file instead
    // of many fread calls.  The MmapFile is scoped; after parsing into
    // the hash maps the mapping is released.

    // ── registry.bin ──
    {
        MmapFile mf;
        if (mf.open(dir_ + "/registry.bin")) {
            ReadCursor c{mf.data(), mf.data() + mf.size()};
            uint32_t count = 0;
            if (c.readU32(count)) {
                strToNum_.reserve(count);
                numToStr_.reserve(count);
                for (uint32_t i = 0; i < count; ++i) {
                    std::string id;
                    uint64_t key = 0;
                    if (!c.readStr(id) || !c.readU64(key)) break;
                    strToNum_[id] = key;
                    numToStr_[key] = id;
                }
            }
        }
    }

    // ── documents.bin ──
    {
        MmapFile mf;
        if (mf.open(dir_ + "/documents.bin")) {
            ReadCursor c{mf.data(), mf.data() + mf.size()};
            uint32_t count = 0;
            if (c.readU32(count)) {
                docs_.reserve(count);
                for (uint32_t i = 0; i < count; ++i) {
                    std::string id, payload;
                    if (!c.readStr(id) || !c.readStr(payload)) break;
                    docs_[id] = std::move(payload);
                }
            }
        }
    }

    // ── stats.bin ──
    {
        MmapFile mf;
        if (mf.open(dir_ + "/stats.bin")) {
            ReadCursor c{mf.data(), mf.data() + mf.size()};
            uint32_t count = 0;
            if (c.readU32(count)) {
                stats_.reserve(count);
                for (uint32_t i = 0; i < count; ++i) {
                    std::string key, val;
                    if (!c.readStr(key) || !c.readStr(val)) break;
                    stats_[key] = std::move(val);
                }
            }
        }
    }

    return true;
}

} // namespace evs

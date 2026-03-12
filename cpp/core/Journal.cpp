// ──────────────────────────────────────────────────────────────
//  Journal — binary append-only WAL
//
//  Each entry layout (all native endian):
//    int64_t  seq
//    uint8_t  op          (1 = Upsert, 2 = Delete)
//    uint32_t id_len
//    char     id[id_len]
//    uint32_t vec_floats  (0 for Delete)
//    float    vec[vec_floats]
// ──────────────────────────────────────────────────────────────
#include "Journal.h"
#include <cstring>

namespace evs {

Journal::~Journal() {
    close();
}

void Journal::close() {
    if (file_) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

bool Journal::open(const std::string& dirPath) {
    path_ = dirPath + "/journal.bin";
    nextSeq_ = 1;

    // Open for reading first to scan high watermark
    scanHighWatermark();

    // Open for appending
    file_ = std::fopen(path_.c_str(), "ab");
    return file_ != nullptr;
}

void Journal::scanHighWatermark() {
    FILE* f = std::fopen(path_.c_str(), "rb");
    if (!f) return;

    while (true) {
        int64_t seq = 0;
        if (std::fread(&seq, 8, 1, f) != 1) break;

        uint8_t op = 0;
        if (std::fread(&op, 1, 1, f) != 1) break;

        uint32_t idLen = 0;
        if (std::fread(&idLen, 4, 1, f) != 1) break;
        if (std::fseek(f, static_cast<long>(idLen), SEEK_CUR) != 0) break;

        uint32_t vecFloats = 0;
        if (std::fread(&vecFloats, 4, 1, f) != 1) break;
        if (vecFloats > 0) {
            if (std::fseek(f, static_cast<long>(vecFloats * sizeof(float)),
                           SEEK_CUR) != 0)
                break;
        }

        if (seq >= nextSeq_) nextSeq_ = seq + 1;
    }
    std::fclose(f);
}

int64_t Journal::appendUpsert(const std::string& id,
                               const float* data, int dims) {
    if (!file_) return -1;

    int64_t seq = nextSeq_++;
    uint8_t op = static_cast<uint8_t>(JournalOp::Upsert);
    uint32_t idLen = static_cast<uint32_t>(id.size());
    uint32_t vecFloats = static_cast<uint32_t>(dims);

    std::fwrite(&seq, 8, 1, file_);
    std::fwrite(&op, 1, 1, file_);
    std::fwrite(&idLen, 4, 1, file_);
    std::fwrite(id.data(), 1, idLen, file_);
    std::fwrite(&vecFloats, 4, 1, file_);
    std::fwrite(data, sizeof(float), vecFloats, file_);

    return seq;
}

int64_t Journal::appendDelete(const std::string& id) {
    if (!file_) return -1;

    int64_t seq = nextSeq_++;
    uint8_t op = static_cast<uint8_t>(JournalOp::Delete);
    uint32_t idLen = static_cast<uint32_t>(id.size());
    uint32_t vecFloats = 0;

    std::fwrite(&seq, 8, 1, file_);
    std::fwrite(&op, 1, 1, file_);
    std::fwrite(&idLen, 4, 1, file_);
    std::fwrite(id.data(), 1, idLen, file_);
    std::fwrite(&vecFloats, 4, 1, file_);

    return seq;
}

void Journal::flush() {
    if (file_) std::fflush(file_);
}

void Journal::replay(int64_t since,
                     std::function<void(const JournalEntry&)> cb) const {
    FILE* f = std::fopen(path_.c_str(), "rb");
    if (!f) return;

    while (true) {
        JournalEntry e;
        e.timestamp = 0;

        if (std::fread(&e.seq, 8, 1, f) != 1) break;

        uint8_t op = 0;
        if (std::fread(&op, 1, 1, f) != 1) break;
        e.op = static_cast<JournalOp>(op);

        uint32_t idLen = 0;
        if (std::fread(&idLen, 4, 1, f) != 1) break;
        e.targetId.resize(idLen);
        if (idLen > 0 && std::fread(&e.targetId[0], 1, idLen, f) != idLen) break;

        uint32_t vecFloats = 0;
        if (std::fread(&vecFloats, 4, 1, f) != 1) break;
        if (vecFloats > 0) {
            e.vector.resize(vecFloats);
            if (std::fread(e.vector.data(), sizeof(float), vecFloats, f) != vecFloats)
                break;
        }

        if (e.seq > since) cb(e);
    }
    std::fclose(f);
}

bool Journal::truncate(int64_t /*upToSeq*/) {
    // After compaction, all entries are consumed.
    // Just delete the file and reopen empty.
    close();
    std::remove(path_.c_str());
    nextSeq_ = 1;
    file_ = std::fopen(path_.c_str(), "ab");
    return file_ != nullptr;
}

} // namespace evs

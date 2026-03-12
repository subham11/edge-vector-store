#pragma once
// ──────────────────────────────────────────────────────────────
//  Journal — binary append-only WAL for crash recovery
//
//  Each entry in journal.bin:
//    [int64_t seq][uint8_t op][uint32_t id_len][char id[]]
//    [uint32_t vec_floats][float vec[]]       (vec_floats=0 for Delete)
// ──────────────────────────────────────────────────────────────
#include "../core/Types.h"
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace evs {

struct JournalEntry {
    int64_t     seq;
    JournalOp   op;
    std::string targetId;
    int64_t     timestamp;
    std::vector<float> vector;
};

class Journal {
public:
    Journal() = default;
    ~Journal();

    Journal(const Journal&) = delete;
    Journal& operator=(const Journal&) = delete;

    /// Open journal in the given directory (creates journal.bin if needed).
    bool open(const std::string& dirPath);

    /// Close the journal file.
    void close();

    /// Append an upsert entry. Returns sequence number.
    int64_t appendUpsert(const std::string& id,
                          const float* data, int dims);

    /// Append a delete entry. Returns sequence number.
    int64_t appendDelete(const std::string& id);

    /// Flush buffered writes to disk.
    void flush();

    /// Replay un-applied entries (seq > since). Calls `cb` per entry.
    void replay(int64_t since,
                std::function<void(const JournalEntry&)> cb) const;

    /// Truncate entries up to (inclusive) the given seq.
    bool truncate(int64_t upToSeq);

    /// Highest sequence number written.
    int64_t highWatermark() const { return nextSeq_ - 1; }

private:
    std::string path_;
    FILE* file_ = nullptr;
    int64_t nextSeq_ = 1;

    /// Scan existing file to set nextSeq_.
    void scanHighWatermark();
};

} // namespace evs

#pragma once
// ──────────────────────────────────────────────────────────────
//  VectorStore — flat float32 vector storage for reranking
//
//  Stores raw float32 vectors in an append-only flat file
//  (vectors.f32) that can be mmap'd for zero-copy access.
//  Each vector is identified by its 64-bit numeric key.
//  Used by the 2-stage search pipeline to rerank candidates
//  with full precision after coarse int8 retrieval.
// ──────────────────────────────────────────────────────────────
#include "MmapFile.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace evs {

class VectorStore {
public:
    VectorStore() = default;
    ~VectorStore() = default;

    VectorStore(const VectorStore&) = delete;
    VectorStore& operator=(const VectorStore&) = delete;

    /// Open or create the store at dir/vectors.f32.
    /// dims = number of float32 elements per vector.
    bool open(const std::string& dir, int32_t dims);

    /// Close the store, releasing the mmap.
    void close();

    /// Store a vector (copied into in-memory buffer and flushed on save).
    void put(uint64_t key, const float* data);

    /// Remove a vector by key.
    void remove(uint64_t key);

    /// Get a pointer to the float32 vector for `key`.
    /// Returns nullptr if not found.
    const float* get(uint64_t key) const;

    /// Flush in-memory additions to disk and re-mmap.
    bool save();

    /// Number of vectors stored.
    size_t size() const { return keyToSlot_.size(); }

    /// Is the store open?
    bool isOpen() const { return dims_ > 0; }

private:
    std::string dir_;
    int32_t     dims_ = 0;

    // On-disk layout:
    //   Header (16 bytes): [magic:u32][version:u32][count:u32][dims:u32]
    //   Slot table (count * 8 bytes): [key:u64] per slot
    //   Vector data (count * dims * 4 bytes): float32 vectors, same order
    //
    // For read: mmap the file, build keyToSlot_ from slot table.
    // For write: append to pending_, flush on save().

    static constexpr uint32_t kMagic   = 0x45565646; // "EVVF"
    static constexpr uint32_t kVersion = 1;
    static constexpr size_t   kHeaderBytes = 16;

    MmapFile mmap_;

    // Maps key → slot index (position in the file's slot/vector arrays)
    std::unordered_map<uint64_t, size_t> keyToSlot_;

    // Pointer to the vector data region (inside mmap or pendingData_)
    const float* mmapVectors_ = nullptr;
    size_t       mmapCount_   = 0;

    // Pending writes not yet on disk
    std::vector<uint64_t>  pendingKeys_;
    std::vector<float>     pendingData_;
    std::vector<uint64_t>  removedKeys_;

    std::string filePath() const { return dir_ + "/vectors.f32"; }
    bool loadFromDisk();
    bool writeToDisk();
};

} // namespace evs

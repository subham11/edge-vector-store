#pragma once
// ──────────────────────────────────────────────────────────────
//  MmapFile — read-only memory-mapped file wrapper
//
//  Uses POSIX mmap (iOS / Android / macOS / Linux).
//  Provides a const uint8_t* view of the file contents without
//  any heap allocation or fread overhead.
// ──────────────────────────────────────────────────────────────
#include <cstddef>
#include <cstdint>
#include <string>

namespace evs {

class MmapFile {
public:
    MmapFile() = default;
    ~MmapFile();

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    /// Memory-map a file in read-only mode.
    bool open(const std::string& path);

    /// Unmap and close.
    void close();

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool isOpen() const { return data_ != nullptr; }

    /// Typed read at a byte offset (caller ensures bounds).
    template <typename T>
    const T* at(size_t byteOffset) const {
        return reinterpret_cast<const T*>(data_ + byteOffset);
    }

private:
    uint8_t* data_ = nullptr;
    size_t   size_ = 0;
    int      fd_   = -1;
};

} // namespace evs

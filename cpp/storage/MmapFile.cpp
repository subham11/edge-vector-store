// ──────────────────────────────────────────────────────────────
//  MmapFile — POSIX mmap implementation
// ──────────────────────────────────────────────────────────────
#include "MmapFile.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace evs {

MmapFile::~MmapFile() { close(); }

bool MmapFile::open(const std::string& path) {
    close();

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;

    struct stat st;
    if (::fstat(fd_, &st) != 0 || st.st_size == 0) {
        close();
        return false;
    }
    size_ = static_cast<size_t>(st.st_size);

    void* ptr = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) {
        close();
        return false;
    }

    data_ = static_cast<uint8_t*>(ptr);
    return true;
}

void MmapFile::close() {
    if (data_) {
        ::munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
}

} // namespace evs

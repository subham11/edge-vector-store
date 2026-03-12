// ──────────────────────────────────────────────────────────────
//  PackIO — ZIP-based .evs pack import/export
//
//  Uses miniz for ZIP I/O. For mobile builds miniz adds ~40KB.
//  Download miniz.h from: https://github.com/richgel999/miniz
//  and place in third_party/miniz/miniz.h
// ──────────────────────────────────────────────────────────────
#include "PackIO.h"

#include "miniz.h"

#include <fstream>
#include <vector>
#include <cstring>

namespace {

bool readFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) return false;
    auto sz = f.tellg();
    f.seekg(0);
    out.resize(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(sz));
    return f.good();
}

bool writeFile(const std::string& path, const void* data, size_t len) {
    std::ofstream f(path, std::ios::binary);
    if (!f.good()) return false;
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(len));
    return f.good();
}

} // anon namespace

namespace evs {

bool PackWriter::write(const std::string& srcDir,
                       const PackManifest& manifest,
                       const std::string& destPath) {
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));

    if (!mz_zip_writer_init_file(&zip, destPath.c_str(), 0))
        return false;

    // 1. manifest.json
    std::string mJson = manifest.toJson();
    if (!mz_zip_writer_add_mem(&zip, "manifest.json",
                                mJson.data(), mJson.size(),
                                MZ_BEST_SPEED)) {
        mz_zip_writer_end(&zip);
        return false;
    }

    // 2. cold.usearch
    std::vector<uint8_t> indexData;
    if (readFile(srcDir + "/cold.usearch", indexData)) {
        if (!mz_zip_writer_add_mem(&zip, "cold.usearch",
                                    indexData.data(), indexData.size(),
                                    MZ_NO_COMPRESSION)) {
            mz_zip_writer_end(&zip);
            return false;
        }
    }

    // 3. registry.bin
    std::vector<uint8_t> regData;
    if (readFile(srcDir + "/registry.bin", regData)) {
        if (!mz_zip_writer_add_mem(&zip, "registry.bin",
                                    regData.data(), regData.size(),
                                    MZ_NO_COMPRESSION)) {
            mz_zip_writer_end(&zip);
            return false;
        }
    }

    // 4. documents.bin
    std::vector<uint8_t> docData;
    if (readFile(srcDir + "/documents.bin", docData)) {
        if (!mz_zip_writer_add_mem(&zip, "documents.bin",
                                    docData.data(), docData.size(),
                                    MZ_NO_COMPRESSION)) {
            mz_zip_writer_end(&zip);
            return false;
        }
    }

    // 5. stats.bin
    std::vector<uint8_t> statsData;
    if (readFile(srcDir + "/stats.bin", statsData)) {
        if (!mz_zip_writer_add_mem(&zip, "stats.bin",
                                    statsData.data(), statsData.size(),
                                    MZ_NO_COMPRESSION)) {
            mz_zip_writer_end(&zip);
            return false;
        }
    }

    // 6. vectors.f32
    std::vector<uint8_t> vecData;
    if (readFile(srcDir + "/vectors.f32", vecData)) {
        if (!mz_zip_writer_add_mem(&zip, "vectors.f32",
                                    vecData.data(), vecData.size(),
                                    MZ_NO_COMPRESSION)) {
            mz_zip_writer_end(&zip);
            return false;
        }
    }

    bool ok = mz_zip_writer_finalize_archive(&zip);
    mz_zip_writer_end(&zip);
    return ok;
}

bool PackReader::read(const std::string& packPath,
                      const std::string& destDir,
                      PackManifest& manifestOut) {
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));

    if (!mz_zip_reader_init_file(&zip, packPath.c_str(), 0))
        return false;

    int nFiles = static_cast<int>(mz_zip_reader_get_num_files(&zip));
    bool gotManifest = false;

    for (int i = 0; i < nFiles; ++i) {
        char fname[256];
        mz_zip_reader_get_filename(&zip, static_cast<mz_uint>(i),
                                    fname, sizeof(fname));

        size_t sz = 0;
        void* data = mz_zip_reader_extract_to_heap(
            &zip, static_cast<mz_uint>(i), &sz, 0);
        if (!data) continue;

        std::string name(fname);
        if (name == "manifest.json") {
            std::string json(reinterpret_cast<char*>(data), sz);
            gotManifest = PackManifest::fromJson(json, manifestOut);
        } else {
            writeFile(destDir + "/" + name, data, sz);
        }
        mz_free(data);
    }

    mz_zip_reader_end(&zip);
    return gotManifest;
}

} // namespace evs

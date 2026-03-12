#pragma once
// ──────────────────────────────────────────────────────────────
//  PackReader / PackWriter — import/export .evs ZIP packs
//
//  Uses miniz (single-header) for ZIP I/O. The .evs file is a
//  standard ZIP containing:
//    manifest.json
//    cold.usearch
//    metadata.db
// ──────────────────────────────────────────────────────────────
#include "PackFormat.h"
#include <string>

namespace evs {

class PackWriter {
public:
    /// Create an .evs pack at `dest` from the store directory `srcDir`.
    /// `srcDir` must contain cold.usearch and metadata.db.
    static bool write(const std::string& srcDir,
                      const PackManifest& manifest,
                      const std::string& destPath);
};

class PackReader {
public:
    /// Extract an .evs pack into `destDir`.
    /// Returns the parsed manifest, or nullopt on failure.
    static bool read(const std::string& packPath,
                     const std::string& destDir,
                     PackManifest& manifestOut);
};

} // namespace evs

#pragma once
// ──────────────────────────────────────────────────────────────
//  EdgeVectorStore — initialisation configuration
// ──────────────────────────────────────────────────────────────
#include "Types.h"
#include <string>

namespace evs {

struct StoreConfig {
    std::string   storagePath;
    int32_t       dimensions       = 384;
    SearchProfile profile          = SearchProfile::Balanced;
    Quantization  quantization     = Quantization::I8;
    Metric        metric           = Metric::Cosine;
    int32_t       hotCacheCapacity = 10000;
    int32_t       connectivity     = 16;
    int32_t       expansionAdd     = 128;
    int32_t       expansionSearch  = 64;
};

/// Parameters that the tiered-search layer derives from a profile.
struct ProfileParams {
    bool     useHotCache               = true;
    bool     useColdIndex              = true;
    float    expansionSearchMultiplier = 1.0f;
    bool     rerank                    = false;
    uint32_t rerankOversample          = 3;  // fetch N*topK for reranking
};

inline ProfileParams profileParams(SearchProfile p) {
    switch (p) {
        case SearchProfile::Balanced:
            return {true, true, 1.0f, true, 3};
        case SearchProfile::MemorySaver:
            return {false, true, 1.0f, false, 1};
        case SearchProfile::MaxRecall:
            return {true, true, 2.0f, true, 5};
        case SearchProfile::MaxSpeed:
            return {true, false, 1.0f, false, 1};
    }
    return {}; // unreachable
}

} // namespace evs

#pragma once
// ──────────────────────────────────────────────────────────────
//  EdgeStoreModule — React Native TurboModule (JSI)
//
//  This is the single C++ class registered as a HostObject
//  from the native side (Android JNI / iOS ObjC++).
//  All methods receive / return JSON strings to cross the
//  JSI boundary with zero-copy overhead for bulk data.
// ──────────────────────────────────────────────────────────────
#include "../core/EdgeStore.h"
#include "StoreRegistry.h"

#include <jsi/jsi.h>
#include <memory>
#include <string>

namespace evs {

class EdgeStoreModule : public facebook::jsi::HostObject {
public:
    EdgeStoreModule() = default;

    /// Called by JS: `NativeEdgeVectorStore.init(configJson)`
    facebook::jsi::Value get(facebook::jsi::Runtime& rt,
                             const facebook::jsi::PropNameID& name) override;

    std::vector<facebook::jsi::PropNameID> getPropertyNames(
        facebook::jsi::Runtime& rt) override;

private:
    StoreRegistry registry_;

    // ── method implementations ───────────────────────────────
    facebook::jsi::Value init(facebook::jsi::Runtime& rt,
                              const facebook::jsi::Value* args, size_t count);
    facebook::jsi::Value upsertDocuments(facebook::jsi::Runtime& rt,
                                          const facebook::jsi::Value* args,
                                          size_t count);
    facebook::jsi::Value upsertVectors(facebook::jsi::Runtime& rt,
                                        const facebook::jsi::Value* args,
                                        size_t count);
    facebook::jsi::Value search(facebook::jsi::Runtime& rt,
                                 const facebook::jsi::Value* args,
                                 size_t count);
    facebook::jsi::Value remove(facebook::jsi::Runtime& rt,
                                 const facebook::jsi::Value* args,
                                 size_t count);
    facebook::jsi::Value compact(facebook::jsi::Runtime& rt,
                                  const facebook::jsi::Value* args,
                                  size_t count);
    facebook::jsi::Value importPack(facebook::jsi::Runtime& rt,
                                     const facebook::jsi::Value* args,
                                     size_t count);
    facebook::jsi::Value exportPack(facebook::jsi::Runtime& rt,
                                     const facebook::jsi::Value* args,
                                     size_t count);
    facebook::jsi::Value getStats(facebook::jsi::Runtime& rt,
                                   const facebook::jsi::Value* args,
                                   size_t count);

    // ── JSI-direct methods (no JSON for hot paths) ───────────
    facebook::jsi::Value searchDirect(facebook::jsi::Runtime& rt,
                                       const facebook::jsi::Value* args,
                                       size_t count);
    facebook::jsi::Value upsertVectorsDirect(
        facebook::jsi::Runtime& rt,
        const facebook::jsi::Value* args,
        size_t count);
};

/// Install the module into the given JSI runtime. Call once from
/// your AppDelegate / MainApplication native init.
void installEdgeStoreModule(facebook::jsi::Runtime& rt);

} // namespace evs

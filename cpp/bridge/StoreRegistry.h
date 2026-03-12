#pragma once
// ──────────────────────────────────────────────────────────────
//  StoreRegistry — shared store management for bridge layers
//
//  Used by both EdgeStoreModule (JSI HostObject) and the ObjC++
//  TurboModule to avoid duplicating the store map + resolution.
// ──────────────────────────────────────────────────────────────
#include "../core/EdgeStore.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace evs {

class StoreRegistry {
public:
    /// Create and initialise a new store from JSON config.
    /// Returns false if init fails.
    bool create(const std::string& configJson) {
        auto store = std::make_shared<EdgeStore>();
        if (!store->init(configJson)) return false;
        auto path = extractStoragePath(configJson);
        if (!path.empty()) {
            stores_[path] = store;
            lastPath_ = path;
        }
        return true;
    }

    /// Resolve a store by storagePath embedded in a JSON string.
    std::shared_ptr<EdgeStore> resolve(const std::string& json) {
        auto path = extractStoragePath(json);
        if (!path.empty()) {
            auto it = stores_.find(path);
            if (it != stores_.end()) return it->second;
        }
        return fallback();
    }

    /// Resolve a store by explicit path.
    std::shared_ptr<EdgeStore> resolveByPath(const std::string& path) {
        auto it = stores_.find(path);
        return it != stores_.end() ? it->second : nullptr;
    }

    /// Return any available store (fallback for parameterless calls).
    std::shared_ptr<EdgeStore> fallback() {
        if (!lastPath_.empty()) {
            auto it = stores_.find(lastPath_);
            if (it != stores_.end()) return it->second;
        }
        if (!stores_.empty()) return stores_.begin()->second;
        return nullptr;
    }

    bool empty() const { return stores_.empty(); }

    /// Extract "storagePath" from a JSON string without a full parse.
    static std::string extractStoragePath(const std::string& json) {
        auto pos = json.find("\"storagePath\"");
        if (pos == std::string::npos) return "";
        pos = json.find('"', pos + 13);
        if (pos == std::string::npos) return "";
        ++pos;
        auto end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    }

private:
    std::unordered_map<std::string, std::shared_ptr<EdgeStore>> stores_;
    std::string lastPath_;
};

} // namespace evs

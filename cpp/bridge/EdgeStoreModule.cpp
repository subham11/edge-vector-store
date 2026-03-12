// ──────────────────────────────────────────────────────────────
//  EdgeStoreModule — JSI implementation
// ──────────────────────────────────────────────────────────────
#include "EdgeStoreModule.h"

#include <functional>
#include <stdexcept>

namespace evs {

using namespace facebook::jsi;

// ── Helper: extract string arg ───────────────────────────────

static std::string stringArg(Runtime& rt, const Value* args, size_t count,
                              size_t idx) {
    if (idx >= count || !args[idx].isString())
        throw JSError(rt, "Expected string argument at index " +
                              std::to_string(idx));
    return args[idx].getString(rt).utf8(rt);
}

// ── Helper: create a JS function from a lambda ──────────────

using HostFn = std::function<Value(Runtime&, const Value*, size_t)>;

static Value makeFunction(Runtime& rt, const char* name, unsigned argCount,
                           HostFn fn) {
    return Function::createFromHostFunction(
        rt, PropNameID::forAscii(rt, name), argCount,
        [fn = std::move(fn)](Runtime& rt, const Value& /*thisVal*/,
                              const Value* args,
                              size_t count) -> Value {
            return fn(rt, args, count);
        });
}

// ── HostObject interface ─────────────────────────────────────

std::vector<PropNameID> EdgeStoreModule::getPropertyNames(Runtime& rt) {
    std::vector<PropNameID> names;
    const char* methods[] = {
        "init", "upsertDocuments", "upsertVectors",
        "search", "remove", "compact",
        "importPack", "exportPack", "getStats",
        "searchDirect", "upsertVectorsDirect"};
    for (auto m : methods)
        names.push_back(PropNameID::forAscii(rt, m));
    return names;
}

Value EdgeStoreModule::get(Runtime& rt, const PropNameID& name) {
    auto n = name.utf8(rt);

    if (n == "init") {
        return makeFunction(rt, "init", 1,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->init(rt, a, c);
            });
    }
    if (n == "upsertDocuments") {
        return makeFunction(rt, "upsertDocuments", 1,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->upsertDocuments(rt, a, c);
            });
    }
    if (n == "upsertVectors") {
        return makeFunction(rt, "upsertVectors", 1,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->upsertVectors(rt, a, c);
            });
    }
    if (n == "search") {
        return makeFunction(rt, "search", 1,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->search(rt, a, c);
            });
    }
    if (n == "remove") {
        return makeFunction(rt, "remove", 1,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->remove(rt, a, c);
            });
    }
    if (n == "compact") {
        return makeFunction(rt, "compact", 0,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->compact(rt, a, c);
            });
    }
    if (n == "importPack") {
        return makeFunction(rt, "importPack", 1,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->importPack(rt, a, c);
            });
    }
    if (n == "exportPack") {
        return makeFunction(rt, "exportPack", 1,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->exportPack(rt, a, c);
            });
    }
    if (n == "getStats") {
        return makeFunction(rt, "getStats", 0,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->getStats(rt, a, c);
            });
    }
    if (n == "searchDirect") {
        return makeFunction(rt, "searchDirect", 4,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->searchDirect(rt, a, c);
            });
    }
    if (n == "upsertVectorsDirect") {
        return makeFunction(rt, "upsertVectorsDirect", 4,
            [this](Runtime& rt, const Value* a, size_t c) {
                return this->upsertVectorsDirect(rt, a, c);
            });
    }
    return Value::undefined();
}

// ── method implementations ───────────────────────────────────

Value EdgeStoreModule::init(Runtime& rt, const Value* args, size_t count) {
    auto configJson = stringArg(rt, args, count, 0);
    if (!registry_.create(configJson))
        throw JSError(rt, "EdgeVectorStore: init failed");
    return Value(true);
}

Value EdgeStoreModule::upsertDocuments(Runtime& rt, const Value* args,
                                        size_t count) {
    auto json = stringArg(rt, args, count, 0);
    auto store = registry_.resolve(json);
    if (!store) throw JSError(rt, "Store not initialised");
    return Value(store->upsertDocuments(json));
}

Value EdgeStoreModule::upsertVectors(Runtime& rt, const Value* args,
                                      size_t count) {
    auto json = stringArg(rt, args, count, 0);
    auto store = registry_.resolve(json);
    if (!store) throw JSError(rt, "Store not initialised");
    return Value(store->upsertVectors(json));
}

Value EdgeStoreModule::search(Runtime& rt, const Value* args,
                               size_t count) {
    auto json = stringArg(rt, args, count, 0);
    auto store = registry_.resolve(json);
    if (!store) throw JSError(rt, "Store not initialised");
    auto result = store->search(json);
    return String::createFromUtf8(rt, result);
}

Value EdgeStoreModule::remove(Runtime& rt, const Value* args,
                               size_t count) {
    auto json = stringArg(rt, args, count, 0);
    auto store = registry_.resolve(json);
    if (!store) throw JSError(rt, "Store not initialised");
    return Value(store->remove(json));
}

Value EdgeStoreModule::compact(Runtime& rt, const Value* args,
                                size_t count) {
    auto store = registry_.fallback();
    if (!store) throw JSError(rt, "No store initialised");
    return Value(store->compact());
}

Value EdgeStoreModule::importPack(Runtime& rt, const Value* args,
                                   size_t count) {
    auto path = stringArg(rt, args, count, 0);
    auto store = registry_.fallback();
    if (!store) throw JSError(rt, "No store initialised");
    return Value(store->importPack(path));
}

Value EdgeStoreModule::exportPack(Runtime& rt, const Value* args,
                                   size_t count) {
    auto path = stringArg(rt, args, count, 0);
    auto store = registry_.fallback();
    if (!store) throw JSError(rt, "No store initialised");
    return Value(store->exportPack(path));
}

Value EdgeStoreModule::getStats(Runtime& rt, const Value* args,
                                 size_t count) {
    auto store = registry_.fallback();
    if (!store) throw JSError(rt, "No store initialised");
    auto result = store->getStats();
    return String::createFromUtf8(rt, result);
}

// ── JSI-direct methods (zero-copy hot paths) ────────────────

// Helper: extract float* from a Float32Array (zero-copy) or
// fall back to copying from a plain JS Array of numbers.
struct FloatVector {
    const float* data;
    size_t       length;
    std::vector<float> owned; // non-empty only for JS Array fallback

    FloatVector() : data(nullptr), length(0) {}
};

static FloatVector extractFloats(Runtime& rt, const Value& val) {
    FloatVector fv;
    if (!val.isObject()) return fv;

    auto obj = val.getObject(rt);

    // Path A: TypedArray (Float32Array) — zero-copy
    if (obj.hasProperty(rt, "buffer")) {
        auto bufVal = obj.getProperty(rt, "buffer");
        if (bufVal.isObject()) {
            auto bufObj = bufVal.getObject(rt);
            if (bufObj.isArrayBuffer(rt)) {
                auto ab = bufObj.getArrayBuffer(rt);
                auto byteOff = static_cast<size_t>(
                    obj.getProperty(rt, "byteOffset").asNumber());
                fv.length = static_cast<size_t>(
                    obj.getProperty(rt, "length").asNumber());
                fv.data = reinterpret_cast<const float*>(
                    ab.data(rt) + byteOff);
                return fv;
            }
        }
    }

    // Path B: plain JS Array — copy into owned buffer
    if (obj.isArray(rt)) {
        auto arr = obj.getArray(rt);
        fv.length = arr.size(rt);
        fv.owned.resize(fv.length);
        for (size_t i = 0; i < fv.length; ++i)
            fv.owned[i] = static_cast<float>(
                arr.getValueAtIndex(rt, i).asNumber());
        fv.data = fv.owned.data();
    }
    return fv;
}

static SearchProfile parseProfile(Runtime& rt, const Value& val) {
    if (!val.isString()) return SearchProfile::Balanced;
    auto s = val.getString(rt).utf8(rt);
    if (s == "memory_saver") return SearchProfile::MemorySaver;
    if (s == "max_recall")   return SearchProfile::MaxRecall;
    if (s == "max_speed")    return SearchProfile::MaxSpeed;
    return SearchProfile::Balanced;
}

Value EdgeStoreModule::searchDirect(Runtime& rt, const Value* args,
                                     size_t count) {
    // args[0] = storagePath (string)
    // args[1] = queryVector (Float32Array | number[])
    // args[2] = topK        (number)
    // args[3] = mode        (string, optional)
    auto path = stringArg(rt, args, count, 0);
    auto store = registry_.resolveByPath(path);
    if (!store) throw JSError(rt, "Store not initialised");

    auto query = extractFloats(rt, args[1]);
    if (!query.data || query.length == 0)
        throw JSError(rt, "searchDirect: invalid queryVector");

    int topK = count > 2 ? static_cast<int>(args[2].asNumber()) : 10;
    SearchProfile profile = count > 3
        ? parseProfile(rt, args[3]) : SearchProfile::Balanced;

    auto results = store->searchDirect(query.data, topK, profile);

    // Build JSI Array of Objects — no JSON serialisation
    auto jsArr = Array(rt, results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        auto obj = Object(rt);
        obj.setProperty(rt, "id",
            String::createFromUtf8(rt, results[i].id));
        obj.setProperty(rt, "distance",
            Value(static_cast<double>(results[i].distance)));
        if (!results[i].payload.empty()) {
            obj.setProperty(rt, "payload",
                String::createFromUtf8(rt, results[i].payload));
        }
        jsArr.setValueAtIndex(rt, i, std::move(obj));
    }
    return jsArr;
}

Value EdgeStoreModule::upsertVectorsDirect(Runtime& rt, const Value* args,
                                            size_t count) {
    // args[0] = storagePath (string)
    // args[1] = ids         (string[])
    // args[2] = vectors     (Float32Array — all vectors packed)
    // args[3] = dims        (number)
    auto path = stringArg(rt, args, count, 0);
    auto store = registry_.resolveByPath(path);
    if (!store) throw JSError(rt, "Store not initialised");

    // Extract ids array
    auto idsObj = args[1].getObject(rt).getArray(rt);
    size_t numVecs = idsObj.size(rt);
    std::vector<std::string> ids(numVecs);
    for (size_t i = 0; i < numVecs; ++i)
        ids[i] = idsObj.getValueAtIndex(rt, i).getString(rt).utf8(rt);

    // Extract packed float data — zero-copy from TypedArray
    auto vecs = extractFloats(rt, args[2]);
    if (!vecs.data || vecs.length == 0)
        throw JSError(rt, "upsertVectorsDirect: invalid vectors");

    size_t dims = count > 3
        ? static_cast<size_t>(args[3].asNumber())
        : vecs.length / numVecs;

    return Value(store->upsertVectorsDirect(
        ids.data(), vecs.data, numVecs, dims));
}

// ── install ──────────────────────────────────────────────────

void installEdgeStoreModule(Runtime& rt) {
    auto module = std::make_shared<EdgeStoreModule>();
    auto obj = Object::createFromHostObject(rt, module);
    rt.global().setProperty(rt, "__EdgeVectorStore", std::move(obj));
}

} // namespace evs

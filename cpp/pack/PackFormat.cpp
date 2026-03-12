// ──────────────────────────────────────────────────────────────
//  PackFormat — manifest serialisation
// ──────────────────────────────────────────────────────────────
#include "PackFormat.h"
#include "../util/JsonParser.h"
#include <cstdlib>
#include <ctime>
#include <sstream>

using evs::json::parseFlat;

namespace evs {

std::string PackManifest::toJson() const {
    auto qStr = [](Quantization q) -> const char* {
        switch (q) {
            case Quantization::F32: return "f32";
            case Quantization::F16: return "f16";
            case Quantization::I8:  return "i8";
            case Quantization::B1:  return "b1";
        }
        return "i8";
    };
    auto mStr = [](Metric m) -> const char* {
        switch (m) {
            case Metric::Cosine:       return "cosine";
            case Metric::Euclidean:    return "euclidean";
            case Metric::InnerProduct: return "ip";
        }
        return "cosine";
    };

    std::string ts = createdAt;
    if (ts.empty()) {
        time_t now = std::time(nullptr);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ",
                      std::gmtime(&now));
        ts = buf;
    }

    std::ostringstream out;
    out << "{\"version\":\"" << version << "\""
        << ",\"dimensions\":" << dimensions
        << ",\"quantization\":\"" << qStr(quantization) << "\""
        << ",\"metric\":\"" << mStr(metric) << "\""
        << ",\"vectorCount\":" << vectorCount
        << ",\"documentCount\":" << documentCount
        << ",\"createdAt\":\"" << ts << "\"}";
    return out.str();
}

bool PackManifest::fromJson(const std::string& json, PackManifest& out) {
    auto kv = parseFlat(json);
    if (kv.empty()) return false;

    auto it = kv.find("version");
    if (it != kv.end()) out.version = it->second;

    it = kv.find("dimensions");
    if (it != kv.end()) out.dimensions = std::atoi(it->second.c_str());

    it = kv.find("quantization");
    if (it != kv.end()) {
        const auto& q = it->second;
        if (q == "f32")      out.quantization = Quantization::F32;
        else if (q == "f16") out.quantization = Quantization::F16;
        else if (q == "b1")  out.quantization = Quantization::B1;
        else                 out.quantization = Quantization::I8;
    }

    it = kv.find("metric");
    if (it != kv.end()) {
        const auto& m = it->second;
        if (m == "euclidean")    out.metric = Metric::Euclidean;
        else if (m == "ip")      out.metric = Metric::InnerProduct;
        else                     out.metric = Metric::Cosine;
    }

    it = kv.find("vectorCount");
    if (it != kv.end()) out.vectorCount = std::atoll(it->second.c_str());

    it = kv.find("documentCount");
    if (it != kv.end()) out.documentCount = std::atoll(it->second.c_str());

    it = kv.find("createdAt");
    if (it != kv.end()) out.createdAt = it->second;

    return true;
}

} // namespace evs

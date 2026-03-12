// ──────────────────────────────────────────────────────────────
//  ANNEngine — implementation (wraps USearch C API)
// ──────────────────────────────────────────────────────────────
#include "ANNEngine.h"

// USearch C API
extern "C" {
#include "usearch.h"
}

#include <algorithm>
#include <cstring>

namespace evs {

// ── helpers ──────────────────────────────────────────────────

static usearch_metric_kind_t toUSearchMetric(Metric m) {
    switch (m) {
        case Metric::Cosine:       return usearch_metric_cos_k;
        case Metric::Euclidean:    return usearch_metric_l2sq_k;
        case Metric::InnerProduct: return usearch_metric_ip_k;
    }
    return usearch_metric_cos_k;
}

static usearch_scalar_kind_t toUSearchScalar(Quantization q) {
    switch (q) {
        case Quantization::F32: return usearch_scalar_f32_k;
        case Quantization::F16: return usearch_scalar_f16_k;
        case Quantization::I8:  return usearch_scalar_i8_k;
        case Quantization::B1:  return usearch_scalar_b1_k;
    }
    return usearch_scalar_i8_k;
}

// ── lifecycle ────────────────────────────────────────────────

ANNEngine::~ANNEngine() { close(); }

void ANNEngine::close() {
    if (index_) {
        usearch_error_t err = nullptr;
        usearch_free(index_, &err);
        index_ = nullptr;
    }
    dims_   = 0;
    isMmap_ = false;
}

bool ANNEngine::init(const StoreConfig& cfg) {
    usearch_init_options_t opts;
    std::memset(&opts, 0, sizeof(opts));
    opts.metric_kind     = toUSearchMetric(cfg.metric);
    opts.quantization    = toUSearchScalar(cfg.quantization);
    opts.dimensions      = static_cast<size_t>(cfg.dimensions);
    opts.connectivity    = static_cast<size_t>(cfg.connectivity);
    opts.expansion_add   = static_cast<size_t>(cfg.expansionAdd);
    opts.expansion_search = static_cast<size_t>(cfg.expansionSearch);
    opts.multi           = false;

    usearch_error_t err = nullptr;
    index_ = usearch_init(&opts, &err);
    if (err || !index_) return false;

    dims_ = cfg.dimensions;
    isMmap_ = false;
    return true;
}

bool ANNEngine::load(const std::string& path) {
    if (!index_) return false;
    usearch_error_t err = nullptr;
    usearch_load(index_, path.c_str(), &err);
    return err == nullptr;
}

bool ANNEngine::view(const std::string& path) {
    if (!index_) return false;
    usearch_error_t err = nullptr;
    usearch_view(index_, path.c_str(), &err);
    if (err) return false;
    isMmap_ = true;
    return true;
}

bool ANNEngine::save(const std::string& path) {
    if (!index_) return false;
    usearch_error_t err = nullptr;
    usearch_save(index_, path.c_str(), &err);
    return err == nullptr;
}

// ── mutations ────────────────────────────────────────────────

bool ANNEngine::add(uint64_t key, const float* data) {
    if (!index_ || isMmap_) return false;
    usearch_error_t err = nullptr;

    // Reserve space if needed (double capacity when full)
    size_t cap = usearch_capacity(index_, &err);
    size_t sz  = usearch_size(index_, &err);
    if (sz >= cap) {
        size_t newCap = (cap == 0) ? 1024 : cap * 2;
        usearch_reserve(index_, newCap, &err);
        if (err) return false;
    }

    usearch_add(index_, key, data, usearch_scalar_f32_k, &err);
    return err == nullptr;
}

bool ANNEngine::remove(uint64_t key) {
    if (!index_ || isMmap_) return false;
    usearch_error_t err = nullptr;
    usearch_remove(index_, key, &err);
    return err == nullptr;
}

// ── search ───────────────────────────────────────────────────

std::vector<ANNResult> ANNEngine::search(const float* query,
                                          size_t topK) const {
    if (!index_) return {};

    std::vector<uint64_t> keys(topK);
    std::vector<float>    dists(topK);
    usearch_error_t err = nullptr;

    size_t found = usearch_search(
        index_, query, usearch_scalar_f32_k, topK,
        keys.data(), dists.data(), &err);

    if (err) return {};

    std::vector<ANNResult> results(found);
    for (size_t i = 0; i < found; ++i) {
        results[i] = {keys[i], dists[i]};
    }
    return results;
}

// ── stats ────────────────────────────────────────────────────

size_t ANNEngine::size() const {
    if (!index_) return 0;
    usearch_error_t err = nullptr;
    return usearch_size(index_, &err);
}

size_t ANNEngine::memoryUsage() const {
    if (!index_) return 0;
    usearch_error_t err = nullptr;
    return usearch_memory_usage(index_, &err);
}

void ANNEngine::setExpansionSearch(size_t expansion) {
    if (!index_) return;
    usearch_error_t err = nullptr;
    usearch_change_expansion_search(index_, expansion, &err);
}

} // namespace evs

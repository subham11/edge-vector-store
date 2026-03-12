# Plan: `@sukshm/edge-vector-store` — Unified On-Device Vector Store

**Build one React Native package with a C++ native core that unifies USearch (compressed ANN) + SQLite (metadata) + a custom pack format (distribution). Target: Android + iOS + WASM. ~300KB–700KB native binary. Handles 1M+ vectors at 768 dims.**

## Critical Architecture Decision: Simplify Ruthlessly

**Drop ObjectBox.** Its C/C++ core is closed-source — the `objectbox-java` repo only contains Java/Kotlin + JNI wrappers around a pre-built binary. You can't vendor, modify, or compile it to WASM. Replace with **SQLite** — it's already on every phone (0 extra bytes), compiles to WASM, has FTS5 for text search, and is public domain.

**Drop the Lance runtime.** The Rust crate chain (Arrow + DataFusion + lance-*) compiles to ~15-20MB. Instead, define a minimal **`.evs` pack format** (ZIP containing a pre-built USearch index + SQLite metadata + manifest). Lance can still be used server-side to *generate* packs.

**Keep USearch as the star.** 3 header files (~5K lines), C API (400 lines), compiles everywhere including WASM, supports f32/f16/i8/b1 quantization, memory-maps indexes from disk, SIMD-accelerated on ARM.

## What You Cherry-Pick

| From | What You Take | How |
|---|---|---|
| **ObjectBox's *design*** | Hot vector cache, integrated object+vector model, offline-first | Implement via two-tier USearch (small in-RAM + large mmap'd) |
| **USearch's *code*** | HNSW ANN, i8/f16/b1 quantization, mmap disk access, SIMD | Vendor 3 headers + 1 C source file directly |
| **Lance's *concept*** | Columnar pack format, versioned snapshots, delta updates | Custom `.evs` pack format (ZIP of index + metadata + manifest) |

## Binary Size

| Component | Mobile | WASM |
|---|---|---|
| USearch (compiled) | ~200-500KB | ~300-600KB |
| SQLite | **0** (system lib) | ~1MB |
| Our C++ core | ~50-100KB | ~50-100KB |
| **Total** | **~300-700KB** | **~1.5-2MB** |

## Memory Budget (768-dim, i8 quantization, 6GB device)

| Scale | Index on disk | Active RAM (mmap'd) | Headroom |
|---|---|---|---|
| 100K vectors | ~83 MB | ~50-100 MB | ~4 GB |
| 500K vectors | ~400 MB | ~150-300 MB | ~3.5 GB |
| 1M vectors | ~930 MB | ~200-400 MB | ~3 GB |

---

## Architecture

```
JS/TS API Layer (one clean surface)
  │
  ├─ TurboModule + JSI (zero-copy vector passing)
  │
  └─ C++ Native Core
       ├── EdgeStore (orchestrator)
       ├── Journal (append-only WAL, crash recovery)
       │
       ├── MetadataStore (SQLite) ← documents, chunks, stats, tombstones
       ├── IDMapper             ← string doc IDs → uint64 USearch keys
       │
       ├── ANNEngine (USearch)  ← cold index, mmap'd from disk
       ├── HotCache (USearch)   ← small in-RAM index, LRU eviction
       ├── TieredSearch         ← merges hot+cold, normalizes scores
       │
       └── PackReader           ← imports .evs packs (ZIP of index+metadata)
```

## Search Profiles (capabilities, not engines)

| Profile | Behavior | RAM | Latency |
|---|---|---|---|
| `balanced` | Hot cache + cold mmap'd index, merge results | Medium | 1-5ms |
| `memory_saver` | Cold mmap'd index only (`usearch_view`) | Lowest | 3-10ms |
| `max_recall` | Both indexes + increased `expansion_search` + rerank | Higher | 5-15ms |
| `max_speed` | Hot cache only (recent/frequent vectors) | Medium | <1ms |

---

## Steps

### Phase 1: Foundation (sequential — everything else depends on this)

1. Initialize React Native TurboModule package with C++ template
2. Vendor USearch headers (3 files) + C API source (2 files) into `third_party/`
3. Define C++ types — `Quantization`, `Metric`, `SearchMode`, `StoreConfig`, `SearchResult`
4. Define TypeScript types — matching interfaces

### Phase 2: Storage Module (*parallel with Phase 3*)

5. Build `MetadataStore` — SQLite wrapper with WAL mode. Schema: `documents`, `vectors`, `stats`, `journal` tables
6. Build `IDMapper` — FNV-1a hash for deterministic `string→uint64` mapping, stored in SQLite

### Phase 3: ANN Module (*parallel with Phase 2*)

7. Build `ANNEngine` — USearch C API wrapper (`usearch_init`, `usearch_add`, `usearch_search`, `usearch_view`, `usearch_save`)
8. Build `HotCache` — Small in-RAM USearch instance (fixed capacity ~10K), LRU eviction to cold index
9. Build `TieredSearch` — Orchestrates hot/cold search, score normalization to [0,1], deduplication

### Phase 4: Orchestrator + Crash Recovery (*depends on Phases 2 & 3*)

10. Build `EdgeStore` — Main class owning all modules. Methods: `init`, `upsert`, `search`, `remove`, `compact`, `getStats`
11. Build `Journal` — Append-only binary WAL. Every mutation logged *before* execution. Replayed on crash recovery.

### Phase 5: Pack Format (*parallel with Phase 4*)

12. Define `.evs` format — ZIP of `manifest.json` + `index.usearch` + `metadata.db` + SHA-256 checksums
13. Build `PackReader` — Validate, extract, delta-merge into existing store (skip existing IDs)

### Phase 6: JSI Bridge (*depends on Phase 4*)

14. TurboModule codegen spec — `init`, `upsertDocuments`, `upsertVectors`, `search`, `remove`, `compact`, `importPack`, `exportPack`, `getStats`
15. JSI HostObject — `jsi::ArrayBuffer` for zero-copy vector passing, background thread for heavy ops

### Phase 7: Platform Wiring (*depends on Phase 6*)

16. Android `CMakeLists.txt` — compile USearch static lib + core, link system SQLite
17. iOS `podspec` — compile USearch + core with C++17, link `libsqlite3.tbd`
18. WASM Emscripten build — compile all + bundled SQLite amalgamation

### Verification

19. C++ unit tests for each module (MetadataStore CRUD, ANNEngine accuracy, Journal crash recovery, TieredSearch score merging)
20. React Native integration tests (full JS→C++→JS round-trip, profile switching, pack import/export)
21. Device benchmarks on mid-range Android — insert throughput, search latency at 100K/500K/1M, RAM usage, APK size delta

---

## The Hard Parts (design from day one)

| Problem | Solution |
|---|---|
| Metadata ↔ ANN index sync after crash | Append-only Journal: log before execute, replay on restart |
| Tombstones / deletions | Mark deleted in SQLite + `usearch_remove()`, rebuild on `compact()` |
| Score normalization across tiers | Normalize all distances to [0, 1] using metric-specific bounds before merging |
| Hot → cold eviction | LRU with configurable capacity; evicted vectors added to cold if not already present |
| Pack versioning across app updates | Manifest includes schema version; PackReader validates compatibility |
| Deterministic ID mapping | FNV-1a hash → collision stored in SQLite for correctness, not just speed |

---

## Decisions Made

1. **SQLite over ObjectBox** — 0 bytes extra on mobile, WASM-compatible, open-source
2. **Custom `.evs` over Lance runtime** — 15-20MB saved; Lance used server-side only
3. **C++ native core** — USearch is already C++, simplest integration path
4. **i8 default quantization** — 4× compression, best quality/size ratio
5. **Two-tier USearch** — Captures ObjectBox's hot-cache concept without closed-source dependency
6. **Profiles not engines** — `search({ mode: "balanced" })` not `searchUSearch()`

## Open Questions

1. **Embedding model** — Should the package include an on-device embedding model (ONNX + MiniLM), or expect pre-computed vectors? *Recommendation: vectors-only in v1, optional `embed()` in v2.*
2. **Cloud sync** — Design pack format for sync-friendliness now, but defer cloud implementation?

---

## Directory Structure

```
@sukshm/edge-vector-store
│
├── src/                          ← JS/TS API (React Native)
│   ├── index.ts                  — Public exports
│   ├── EdgeVectorStore.ts        — Main class
│   ├── types.ts                  — TypeScript interfaces
│   ├── profiles.ts               — Search profiles
│   └── NativeEdgeVectorStore.ts  — TurboModule codegen spec
│
├── cpp/                          ← C++ Native Core
│   ├── core/
│   │   ├── EdgeStore.h/.cpp      — Main orchestrator (owns all modules)
│   │   ├── Types.h               — Shared enums, structs, result types
│   │   ├── Config.h              — Init config, profile configs
│   │   └── Journal.h/.cpp        — Append-only WAL for crash recovery
│   ├── storage/
│   │   ├── MetadataStore.h/.cpp  — SQLite wrapper (docs, chunks, stats)
│   │   └── IDMapper.h/.cpp       — Deterministic string→uint64 mapping
│   ├── ann/
│   │   ├── ANNEngine.h/.cpp      — USearch C API wrapper
│   │   ├── HotCache.h/.cpp       — Small in-RAM index for recent vectors
│   │   └── TieredSearch.h/.cpp   — Tiered hot/cold/disk search orchestrator
│   ├── pack/
│   │   ├── PackReader.h/.cpp     — Read .evs pack files (unzip + validate)
│   │   └── PackFormat.h          — Format constants, checksums
│   └── jsi/
│       ├── EdgeStoreModule.h/.cpp — JSI TurboModule implementation
│       └── Helpers.h              — JSI ↔ C++ type marshalling
│
├── android/
│   ├── CMakeLists.txt            — NDK build (links USearch, SQLite)
│   ├── build.gradle              — React Native auto-linking
│   └── src/main/java/.../
│       └── EdgeVectorStorePackage.kt — TurboModule registration
│
├── ios/
│   ├── EdgeVectorStore.podspec   — CocoaPods spec
│   ├── EdgeVectorStore.mm        — ObjC++ TurboModule registration
│   └── CMakeLists.txt            — Build USearch + core
│
├── wasm/                         ← Web target
│   ├── CMakeLists.txt            — Emscripten build
│   └── wasm_bridge.cpp           — C++ → WASM exports
│
└── third_party/
    └── usearch/                  — Vendored USearch headers (3 files, ~5K lines)
```

## Public API Surface

```typescript
const db = await EdgeVectorStore.init({
  storagePath: "...",
  profile: "balanced",
  quantization: "i8",
  dimensions: 768,
  metric: "cosine",
})

await db.upsertDocuments([
  { id: "doc1", payload: { type: "medical" }, vector: [...] },
])

await db.upsertVectors([
  { id: "vec1", docId: "doc1", vector: [...] },
])

const results = await db.search({
  queryVector: [...],
  topK: 10,
  filter: { type: "medical" },
  mode: "balanced",
})

await db.remove(["doc1"])
await db.compact()
await db.exportPack("/path/to/export.evs")
await db.importPack("/path/to/corpus.evs")
const stats = await db.getStats()
```

## Relevant Source Files (from explored repos)

- `USearch/include/usearch/index.hpp` — Core HNSW engine, template class `index_gt<>`. Vendor this.
- `USearch/include/usearch/index_dense.hpp` — Dense index wrapper with quantization. Vendor this.
- `USearch/include/usearch/index_plugins.hpp` — Metrics, SIMD, quantization. Vendor this.
- `USearch/c/usearch.h` — C API (~400 lines). Key: `usearch_init`, `usearch_add`, `usearch_search`, `usearch_view`, `usearch_save`, `usearch_load`.
- `USearch/c/lib.cpp` — C API implementation.
- `USearch/wasm/CMakeLists.txt` — Reference for Emscripten build config.
- `objectbox-java/objectbox-java-api/.../HnswIndex.java` — Reference for HNSW config design.
- `lancedb/rust/lancedb/src/index/vector.rs` — Reference for quantization types (IvfPq, IvfSq).
- `lancedb/nodejs/lancedb/index.ts` — Reference for TypeScript API design patterns.

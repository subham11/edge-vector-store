# Edge Vector Store

> Unified on-device vector database for React Native — powered by HNSW, SIMD kernels, and zero-copy JSI.

**`react-native-edge-vector-store`** brings production-grade vector similarity search to mobile devices. It combines a USearch-backed HNSW index, in-memory metadata with binary persistence, a tiered hot/cold architecture, and custom NEON batch kernels — all accessible through a clean TypeScript API with zero-copy Float32Array transfers via JSI.

---

## Table of Contents

- [Why Edge Vector Store?](#why-edge-vector-store)
- [Features](#features)
- [Architecture](#architecture)
- [Benchmark Scores](#benchmark-scores)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Example App](#example-app)
- [Pack Format (.evs)](#pack-format-evs)
- [Configuration](#configuration)
- [Platform Support](#platform-support)
- [License](#license)

---

## Why Edge Vector Store?

Running vector search on-device unlocks RAG, semantic search, and recommendation features **without network round-trips**. Existing solutions either wrap SQLite (slow for vectors), require cloud APIs, or lack React Native support. Edge Vector Store is purpose-built for mobile:

- **Zero network dependency** — all search happens on-device
- **Sub-millisecond latency** — hot cache returns results in < 1 ms
- **Crash-safe** — binary write-ahead log protects against data loss
- **Portable** — `.evs` pack files transfer indexes between apps and platforms
- **Tiny footprint** — ~300–700 KB added to your binary (no SQLite dependency)

---

## Features

| Feature | Detail |
|---|---|
| **HNSW Index** | USearch-backed approximate nearest neighbor search |
| **Tiered Architecture** | Hot in-RAM cache (10K default) + cold mmap'd index, merged results |
| **Quantization** | F32, F16, I8, B1 — I8 default for 4× memory savings |
| **2-Stage Reranking** | Coarse quantized search → full-precision float32 rerank |
| **Custom SIMD** | Batch NEON kernels — 4 candidates simultaneously, query stays in registers |
| **Zero-Copy JSI** | Float32Array passed directly to C++ — no JSON serialisation overhead |
| **Crash Recovery** | Binary WAL (journal.bin) replays on next open |
| **Pack Import/Export** | ZIP-based `.evs` format for offline distribution |
| **Search Profiles** | `balanced`, `memory_saver`, `max_recall`, `max_speed` |
| **Multi-Store** | Multiple independent stores in a single app |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    TypeScript API                        │
│    EdgeVectorStore.init / search / upsert / compact      │
└────────────────────┬─────────────────┬───────────────────┘
                     │                 │
          ┌──────────▼──────┐  ┌───────▼──────────┐
          │   JSI HostObject │  │   TurboModule    │
          │   (zero-copy)    │  │   (JSON bridge)  │
          └──────────┬───────┘  └───────┬──────────┘
                     │                  │
                     └────────┬─────────┘
                              │
              ┌───────────────▼────────────────┐
              │         StoreRegistry          │
              │   (multi-store map + resolve)  │
              └───────────────┬────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │          EdgeStore             │
              │     (top-level orchestrator)    │
              └──┬──────┬──────┬──────┬────────┘
                 │      │      │      │
    ┌────────────▼┐ ┌───▼────┐ │  ┌───▼──────┐
    │   Journal   │ │Metadata│ │  │  PackIO   │
    │ (binary WAL)│ │ Store  │ │  │(.evs ZIP) │
    └─────────────┘ │(hash   │ │  └──────────-┘
                    │ maps)  │ │
                    └────────┘ │
                               │
           ┌───────────────────▼───────────────────┐
           │           Tiered Search               │
           │   merge hot-cache + cold-index results │
           │   optional float32 reranking           │
           └──┬─────────────┬─────────────┬────────┘
              │             │             │
    ┌─────────▼──┐  ┌───────▼──────┐  ┌──▼──────────┐
    │  HotCache  │  │  ANNEngine   │  │ VectorStore  │
    │ (in-RAM    │  │  (USearch    │  │ (flat f32    │
    │  HNSW,LRU) │  │  mmap cold)  │  │  mmap'd)     │
    └────────────┘  └──────────────┘  └──────────────┘
              │
    ┌─────────▼──────┐
    │  SIMD Kernels  │
    │ batch NEON/SSE │
    │ 4 candidates   │
    └────────────────┘
```

### Layer Breakdown

#### Core Layer (`cpp/core/`)

| Component | Purpose |
|---|---|
| **EdgeStore** | Top-level orchestrator — owns all subsystems, exposes the full public API |
| **Config** | `StoreConfig` + `ProfileParams` — dimensions, quantization, connectivity, expansion |
| **Types** | Enums (`Quantization`, `Metric`, `SearchProfile`), `SearchResult`, `StoreStats` |
| **Journal** | Binary append-only WAL for crash recovery — replays on next `init()` |

#### ANN Layer (`cpp/ann/`)

| Component | Purpose |
|---|---|
| **ANNEngine** | Thin wrapper around USearch C API — `init`, `load`, `view` (mmap), `save`, `add`, `search` |
| **HotCache** | Fixed-capacity in-RAM HNSW index with LRU eviction — evicted keys migrate to cold index |
| **TieredSearch** | Merges results from hot cache + cold mmap'd index. Optional 2-stage reranking: coarse quantized search → float32 re-score |
| **SIMDKernels** | Batch distance computation — processes 4 candidates simultaneously with query vector pinned in NEON registers. Dimension-aligned fast paths for 128/256/384/512/768 dims |

#### Storage Layer (`cpp/storage/`)

| Component | Purpose |
|---|---|
| **MetadataStore** | In-memory hash maps (`id→payload`, `docId↔numericKey`) with binary file persistence. O(1) lookups, zero SQLite overhead |
| **VectorStore** | Flat float32 vector file with mmap — used for 2-stage reranking |
| **MmapFile** | POSIX mmap wrapper — zero heap allocation, read-only memory-mapped access |
| **IDMapper** | Deterministic `string→uint64` via FNV-1a hash with linear-probe collision resolution |

#### Pack Layer (`cpp/pack/`)

| Component | Purpose |
|---|---|
| **PackFormat** | `.evs` manifest: version, vector count, dimensions, quantization, timestamps |
| **PackIO** | ZIP-based reader/writer — bundles `manifest.json` + `cold.usearch` + `metadata.db` |

#### Bridge Layer (`cpp/bridge/`)

| Component | Purpose |
|---|---|
| **EdgeStoreModule** | JSI `HostObject` installed as `global.__EdgeVectorStore` — exposes all methods + zero-copy `searchDirect`/`upsertVectorsDirect` |
| **StoreRegistry** | Multi-store management — maps `storagePath → EdgeStore` instances, shared by both bridge paths |

#### Utility (`cpp/util/`)

| Component | Purpose |
|---|---|
| **JsonParser** | Minimal hand-rolled JSON parser — no external dependency. `parseFlat`, `parseFloatArray`, `parseObjectArray` |

#### Benchmark (`cpp/bench/`)

| Component | Purpose |
|---|---|
| **BenchmarkEngine** | Builds in-memory HNSW index, sweeps multiple EF values, measures recall/latency/QPS |

### Third-Party Libraries (Vendored)

| Library | Purpose |
|---|---|
| **USearch** | HNSW approximate nearest neighbor engine (C API) |
| **miniz** | ZIP I/O for `.evs` pack format |
| **SimSIMD** | SIMD distance functions (present but disabled — custom kernels used instead) |

---

## Benchmark Scores

Benchmarks run on a 100K vector dataset (384 dimensions, cosine distance) on iOS Simulator / Apple Silicon:

### Core Performance

| Metric | Edge Vector Store | USearch (raw) | Notes |
|---|---|---|---|
| **Insert Throughput** | 1,323 vec/s | 2,861 vec/s | EVS includes metadata + WAL |
| **Search Mean Latency** | 0.587 ms | 0.320 ms | Full stack with persistence |
| **Search P99 Latency** | 0.851 ms | 0.369 ms | |
| **Recall@10** | 0.834 | 0.854 | Near-parity |
| **Memory** | 20.99 MB | 20.99 MB | Equal index footprint |
| **Cold Start** | 0.84 ms | — | mmap, near-instant |
| **Disk (meta)** | 0.53 MB | — | Metadata persistence |

### Overhead Breakdown

| Source | Cost |
|---|---|
| JSON parse (vector arrays) | ~0.08 ms |
| JSON serialise (results) | ~0.15 ms |
| Metadata lookups (hash map) | ~0.02 ms |
| Journal flush (WAL append) | ~0.50 ms/batch |

> **Tip:** Use `searchDirect()` with Float32Array to bypass JSON overhead entirely on hot paths.

### Memory Budget by Scale (768d, I8 quantization)

| Vectors | Index on Disk | Active RAM (mmap'd) |
|---|---|---|
| 100K | ~83 MB | ~50–100 MB |
| 500K | ~400 MB | ~150–300 MB |
| 1M | ~930 MB | ~200–400 MB |

### Binary Size Impact

| Component | Mobile (iOS/Android) | WASM |
|---|---|---|
| USearch (compiled) | ~200–500 KB | ~300–600 KB |
| C++ core + bridge | ~50–100 KB | ~50–100 KB |
| **Total added** | **~300–700 KB** | **~1.5–2 MB** |

---

## Installation

```sh
# npm
npm install react-native-edge-vector-store

# yarn
yarn add react-native-edge-vector-store
```

### iOS

```sh
cd ios && bundle exec pod install
```

The podspec automatically compiles all C++ sources and links USearch + miniz. Requires iOS 13.0+.

### Android

No extra steps — the Gradle plugin builds the native library via CMake automatically. Requires minSdk 24.

---

## Quick Start

```typescript
import { EdgeVectorStore } from 'react-native-edge-vector-store';
import type { StoreConfig, Document, SearchOptions } from 'react-native-edge-vector-store';

// 1. Initialise a store
const config: StoreConfig = {
  storagePath: '/path/to/store',
  dimensions: 384,
  quantization: 'i8',    // 4× memory savings vs f32
  metric: 'cosine',
};
const store = await EdgeVectorStore.init(config);

// 2. Insert documents with embeddings
await store.upsertDocuments([
  {
    id: 'doc-1',
    payload: { title: 'Introduction to ML', category: 'ai' },
    vector: new Float32Array(384), // your embedding
  },
  {
    id: 'doc-2',
    payload: { title: 'React Native Guide', category: 'mobile' },
    vector: new Float32Array(384),
  },
]);

// 3. Search
const results = await store.search({
  queryVector: new Float32Array(384), // query embedding
  topK: 5,
  mode: 'balanced',
});

console.log(results);
// [{ id: 'doc-1', distance: 0.12, payload: { title: '...' } }, ...]

// 4. Export as portable .evs pack
await store.exportPack('/path/to/export.evs');
```

### Zero-Copy Hot Path (Advanced)

For latency-critical paths, use `Float32Array` directly — the vector data crosses the JSI bridge with zero copies:

```typescript
// Bulk upsert with packed Float32Array
const ids = ['vec-0', 'vec-1', 'vec-2'];
const packed = new Float32Array(3 * 384); // all vectors contiguous
await store.upsertVectors(
  ids.map((id, i) => ({
    id,
    docId: id,
    vector: packed.subarray(i * 384, (i + 1) * 384),
  }))
);

// Direct search — bypasses JSON serialisation entirely
const results = await store.search({
  queryVector: new Float32Array(384),
  topK: 10,
  mode: 'max_speed', // hot cache only, < 1ms
});
```

---

## API Reference

### `EdgeVectorStore`

#### `static async init(config: StoreConfig): Promise<EdgeVectorStore>`

Creates or opens a vector store at the given path.

#### `async upsertDocuments(docs: Document[]): Promise<void>`

Insert or update documents with optional embeddings and JSON payloads.

#### `async upsertVectors(entries: VectorEntry[]): Promise<void>`

Insert or update raw vector entries. Uses zero-copy Float32Array transfer via JSI when available.

#### `async search(options: SearchOptions): Promise<SearchResult[]>`

Find the nearest neighbours. Uses the zero-copy `searchDirect` JSI path internally.

#### `async remove(ids: string[]): Promise<void>`

Delete documents and their associated vectors by ID.

#### `async compact(): Promise<void>`

Merge hot cache into the cold index and reclaim space.

#### `async importPack(path: string): Promise<void>`

Import a `.evs` pack file into the store.

#### `async exportPack(path: string): Promise<void>`

Export the store as a portable `.evs` pack file.

#### `async getStats(): Promise<StoreStats>`

Returns store statistics — document count, vector count, hot cache size, memory usage, quantization, dimensions.

#### `static async benchmarkRawANN(config): Promise<Array<Record<string, unknown>>>`

Run a raw HNSW benchmark with EF sweeps. Returns an array of results with recall, latency, and QPS for each EF value.

### Types

```typescript
type Quantization = 'f32' | 'f16' | 'i8' | 'b1';
type Metric = 'cosine' | 'euclidean' | 'inner_product';
type SearchProfile = 'balanced' | 'memory_saver' | 'max_recall' | 'max_speed';
type VectorLike = Float32Array | number[];

interface StoreConfig {
  storagePath: string;
  dimensions: number;
  profile?: SearchProfile;      // default: 'balanced'
  quantization?: Quantization;  // default: 'i8'
  metric?: Metric;              // default: 'cosine'
  hotCacheCapacity?: number;    // default: 10000
  connectivity?: number;        // HNSW M, default: 16
  expansionAdd?: number;        // default: 128
  expansionSearch?: number;     // default: 64
}

interface Document {
  id: string;
  payload?: Record<string, unknown>;
  vector?: VectorLike;
}

interface VectorEntry {
  id: string;
  docId: string;
  vector: VectorLike;
}

interface SearchOptions {
  queryVector: VectorLike;
  topK?: number;      // default: 10
  filter?: Record<string, unknown>;
  mode?: SearchProfile;
}

interface SearchResult {
  id: string;
  distance: number;
  payload?: Record<string, unknown>;
}

interface StoreStats {
  documentCount: number;
  vectorCount: number;
  hotCacheCount: number;
  memoryUsageBytes: number;
  coldIndexSizeBytes: number;
  quantization: Quantization;
  dimensions: number;
}
```

### Search Profiles

| Profile | Behaviour | Typical Latency |
|---|---|---|
| `balanced` | Hot cache + cold mmap'd index, merged results | ~0.5 ms |
| `memory_saver` | Cold mmap'd index only (no hot cache RAM) | ~0.8 ms |
| `max_recall` | Both tiers + increased expansion + float32 reranking | ~1.5 ms |
| `max_speed` | Hot cache only | < 1 ms |

---

## Example App

The `example/` directory contains a full React Native 0.84 app demonstrating vector search and RAG (Retrieval-Augmented Generation).

### Running the Example

```sh
cd example
yarn install
cd ios && bundle exec pod install && cd ..
yarn ios
```

### App Structure

```
example/
├── App.tsx                    # Navigation setup (4 screens)
├── src/
│   ├── screens/
│   │   ├── HomeScreen.tsx     # Landing page with navigation cards
│   │   ├── ChatScreen.tsx     # RAG-powered semantic chat
│   │   ├── BenchmarkScreen.tsx# 5-phase benchmark harness
│   │   └── ResultsScreen.tsx  # Detailed benchmark results viewer
│   └── services/
│       ├── ChatService.ts     # RAG pipeline orchestration
│       ├── VectorSearchService.ts  # EdgeVectorStore wrapper
│       ├── QueryEmbedder.ts   # ONNX MiniLM-L6-v2 (384d)
│       ├── PromptTemplates.ts # Gemma 2 prompt formatting
│       └── BenchmarkRunner.ts # 5-phase benchmark suite
├── models/
│   ├── all-MiniLM-L6-v2.onnx # Embedding model (384 dims)
│   └── gemma-2-2b-it-Q4_K_M.gguf # Gemma 2B for generation
```

### Screens

#### Home Screen
Landing page with navigation cards to the Chat and Benchmark screens.

#### Chat Screen — RAG Pipeline
A full retrieval-augmented generation demo:

1. **Embed** — User query is embedded using MiniLM-L6-v2 (ONNX Runtime, 384 dimensions)
2. **Search** — Embedding is used to search the vector store for relevant documents
3. **Generate** — Retrieved contexts are injected into a Gemma 2B prompt, which generates the response via llama.rn

```
User Query → QueryEmbedder (MiniLM ONNX) → EdgeVectorStore.search()
                                                    ↓
                                            Top-K documents
                                                    ↓
                                 PromptTemplates.buildRAGPrompt()
                                                    ↓
                                      llama.rn (Gemma 2B Q4_K_M)
                                                    ↓
                                              AI Response
```

#### Benchmark Screen — 5-Phase Harness
Runs a comprehensive benchmark suite with toggleable phases, progress bars, and log output:

| Phase | Description | Parameters |
|---|---|---|
| **1. Sanity** | Smoke test — 10K vectors, 384d, F32 | Default HNSW build |
| **2. Core Parity** | 100K vectors, 384d, F32 | Light / default / heavy builds, matched recall bands |
| **3. Mobile Reality** | 100K vectors, 384d, persistent mode | Cold start vs warm start timing |
| **4. Compression** | 100K vectors, 384d | F32 vs I8 vs I8+rerank comparison |
| **5. Scale** | 250K vectors, 384d + 768d | High-dimensionality stress test |

Each phase calls `EdgeVectorStore.benchmarkRawANN()` with varying EF sweep values, measuring **recall@10**, **mean/P95/P99 latency**, and **queries per second**.

#### Results Screen
Displays detailed per-phase results with recall, latency, throughput, and resource usage — grouped by phase for easy comparison.

### Key Services

#### `VectorSearchService`
Wraps `EdgeVectorStore` for the chat use case — handles initialisation, pack loading, and search with the `balanced` profile.

#### `QueryEmbedder`
Runs the MiniLM-L6-v2 ONNX model on-device using `onnxruntime-react-native`. Includes a simple tokeniser, mean pooling, and L2 normalisation. Falls back to random embeddings if the model is unavailable.

#### `ChatService`
Orchestrates the full RAG pipeline — chains QueryEmbedder → VectorSearchService → llama.rn (Gemma 2B). Falls back to template responses if the generation model is not found.

---

## Pack Format (.evs)

The `.evs` format is a standard ZIP file containing:

| Entry | Purpose |
|---|---|
| `manifest.json` | Version, vector count, dimensions, quantization, timestamps |
| `cold.usearch` | Serialised HNSW index |
| `metadata.db` | Binary metadata store (document payloads + ID mappings) |

Use `exportPack()` to create portable index files and `importPack()` to load them anywhere:

```typescript
// On build machine — prepare the pack
await store.exportPack('/output/knowledge-base.evs');

// On device — load the pre-built pack
await store.importPack('/assets/knowledge-base.evs');
```

---

## Configuration

### HNSW Tuning

| Parameter | Default | Effect |
|---|---|---|
| `connectivity` (M) | 16 | Higher → better recall, more memory |
| `expansionAdd` (efConstruction) | 128 | Higher → better index quality, slower inserts |
| `expansionSearch` (ef) | 64 | Higher → better recall, slower search |
| `hotCacheCapacity` | 10,000 | Max vectors in the fast in-RAM index |

### Quantization Trade-offs

| Level | Bytes/dim | Recall Impact | Speed |
|---|---|---|---|
| `f32` | 4 | Baseline | Baseline |
| `f16` | 2 | Negligible | ~1.5× faster |
| `i8` | 1 | ~1–3% drop | ~3× faster |
| `b1` | 0.125 | ~5–10% drop | ~10× faster |

> Use `i8` (default) for the best balance. Use `max_recall` profile with I8 to get reranking — coarse quantized search followed by full float32 re-scoring.

---

## Platform Support

| Platform | Status | Mechanism |
|---|---|---|
| **iOS** | ✅ | ObjC++ TurboModule + JSI HostObject |
| **Android** | ✅ | JNI + JSI HostObject |
| **WASM** | ✅ | Emscripten + embind |
| **Desktop (test)** | ✅ | CMake static lib + test executables |

**Requirements:**
- React Native ≥ 0.73 (New Architecture / TurboModules)
- iOS 13.0+
- Android minSdk 24
- C++17

---

## License

MIT

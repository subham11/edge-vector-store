// ─── Quantization ────────────────────────────────────────────
/** Scalar quantization level for stored vectors. */
export type Quantization = 'f32' | 'f16' | 'i8' | 'b1';

// ─── Distance metric ────────────────────────────────────────
export type Metric = 'cosine' | 'euclidean' | 'inner_product';

// ─── Search profiles ────────────────────────────────────────
/**
 * Search profiles expose *capabilities*, not engines.
 *
 * - `balanced`     – hot cache + cold mmap'd index, merged results
 * - `memory_saver` – cold mmap'd index only
 * - `max_recall`   – both tiers, increased expansion, rerank
 * - `max_speed`    – hot cache only (<1 ms)
 */
export type SearchProfile =
  | 'balanced'
  | 'memory_saver'
  | 'max_recall'
  | 'max_speed';

// ─── Configuration ──────────────────────────────────────────
export interface StoreConfig {
  /** Filesystem path for the store directory. */
  storagePath: string;
  /** Embedding dimensionality (e.g. 768). */
  dimensions: number;
  /** Default search profile. */
  profile?: SearchProfile;
  /** Quantization level (default: `'i8'`). */
  quantization?: Quantization;
  /** Distance metric (default: `'cosine'`). */
  metric?: Metric;
  /** Max vectors kept in the hot (in-RAM) cache (default: 10 000). */
  hotCacheCapacity?: number;
  /** HNSW connectivity parameter M (default: 16). */
  connectivity?: number;
  /** HNSW expansion during insertion (default: 128). */
  expansionAdd?: number;
  /** HNSW expansion during search (default: 64). */
  expansionSearch?: number;
}

// ─── Vector data ────────────────────────────────────────────
/** Float vector data — Float32Array for zero-copy JSI, or number[] for convenience. */
export type VectorLike = Float32Array | number[];

// ─── Documents ──────────────────────────────────────────────
export interface Document {
  /** Application-level string identifier. */
  id: string;
  /** Arbitrary JSON-serialisable metadata attached to the document. */
  payload?: Record<string, unknown>;
  /**
   * Embedding vector.
   * Length must equal `StoreConfig.dimensions`.
   * Values are float32; on-disk they are stored at the configured quantization.
   */
  vector?: VectorLike;
}

// ─── Vectors ────────────────────────────────────────────────
export interface VectorEntry {
  /** Unique vector key (maps 1-to-1 to a document). */
  id: string;
  /** Associated document id. */
  docId: string;
  /** Float32 embedding. */
  vector: VectorLike;
}

// ─── Search ─────────────────────────────────────────────────
export interface SearchOptions {
  /** Float32 query vector. */
  queryVector: VectorLike;
  /** Number of nearest neighbours to return (default: 10). */
  topK?: number;
  /** Optional metadata filter (key-value equality). */
  filter?: Record<string, unknown>;
  /** Override the store's default profile for this query. */
  mode?: SearchProfile;
}

export interface SearchResult {
  /** Document id. */
  id: string;
  /** Distance / score (lower = closer for metric distances). */
  distance: number;
  /** Document payload (if requested / stored). */
  payload?: Record<string, unknown>;
}

// ─── Stats ──────────────────────────────────────────────────
export interface StoreStats {
  /** Total documents in metadata store. */
  documentCount: number;
  /** Total vectors across hot + cold indexes. */
  vectorCount: number;
  /** Vectors currently in the hot cache. */
  hotCacheCount: number;
  /** Approximate native memory usage in bytes. */
  memoryUsageBytes: number;
  /** On-disk size of the cold index in bytes. */
  coldIndexSizeBytes: number;
  /** Quantization in use. */
  quantization: Quantization;
  /** Configured dimensionality. */
  dimensions: number;
}

// ─── Pack manifest ──────────────────────────────────────────
export interface PackManifest {
  /** Pack format version. */
  version: number;
  /** Number of vectors in the pack. */
  vectorCount: number;
  /** Dimensions of each vector. */
  dimensions: number;
  /** Quantization used in the index. */
  quantization: Quantization;
  /** Distance metric used. */
  metric: Metric;
  /** SHA-256 hex digest of `index.usearch` + `metadata.db`. */
  checksum: string;
}

/**
 * @sukshm/edge-vector-store
 *
 * Unified on-device vector store for React Native.
 * USearch (HNSW ANN) · SQLite (metadata) · .evs packs (distribution)
 */
export {
  type Quantization,
  type Metric,
  type SearchProfile,
  type VectorLike,
  type StoreConfig,
  type Document,
  type VectorEntry,
  type SearchOptions,
  type SearchResult,
  type StoreStats,
  type PackManifest,
} from './types';

export { EdgeVectorStore } from './EdgeVectorStore';

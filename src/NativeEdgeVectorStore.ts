import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

/**
 * Native TurboModule specification for EdgeVectorStore.
 *
 * All methods mirror the public JS API but accept / return
 * plain JSON-serialisable types so codegen can bridge them.
 */
export interface Spec extends TurboModule {
  /** Install the JSI HostObject (global.__EdgeVectorStore). */
  install(): Promise<boolean>;

  /** Initialise (or re-open) a store at the given path. */
  init(configJson: string): Promise<boolean>;

  /** Insert or update documents (with optional embedded vectors). */
  upsertDocuments(docsJson: string): Promise<void>;

  /** Insert or update raw vectors keyed to existing documents. */
  upsertVectors(entriesJson: string): Promise<void>;

  /**
   * Approximate nearest-neighbour search.
   * `queryVector` is a base-64 encoded Float32Array for zero-copy.
   * Returns JSON-encoded SearchResult[].
   */
  search(optionsJson: string): Promise<string>;

  /** Delete documents (and their vectors) by id. */
  remove(idsJson: string): Promise<void>;

  /** Compact: rebuild cold index, flush hot cache, vacuum metadata. */
  compact(): Promise<void>;

  /** Import a `.evs` pack file, delta-merging into the store. */
  importPack(path: string): Promise<void>;

  /** Export the current store as a `.evs` pack file. */
  exportPack(path: string): Promise<void>;

  /** Return store statistics as JSON. */
  getStats(): Promise<string>;

  /** Run a raw USearch-only benchmark (no SQLite metadata). */
  benchmarkRawANN(configJson: string): Promise<string>;
}

export default TurboModuleRegistry.getEnforcing<Spec>(
  'EdgeVectorStore',
);

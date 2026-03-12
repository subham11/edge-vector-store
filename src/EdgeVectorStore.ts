import NativeEdgeVectorStore from './NativeEdgeVectorStore';
import type {
  StoreConfig,
  Document,
  VectorEntry,
  VectorLike,
  SearchOptions,
  SearchResult,
  StoreStats,
} from './types';

// ── JSI HostObject (installed as global by native init) ──────

interface EVSHostObject {
  init(configJson: string): boolean;
  upsertDocuments(docsJson: string): boolean;
  upsertVectors(entriesJson: string): boolean;
  search(optionsJson: string): string;
  remove(idsJson: string): boolean;
  compact(): boolean;
  importPack(path: string): boolean;
  exportPack(path: string): boolean;
  getStats(): string;
  // JSI-direct hot-path methods (Phase 1)
  searchDirect(
    storagePath: string,
    queryVector: Float32Array,
    topK: number,
    mode?: string,
  ): Array<{ id: string; distance: number; payload?: string }>;
  upsertVectorsDirect(
    storagePath: string,
    ids: string[],
    vectors: Float32Array,
    dims: number,
  ): boolean;
}

let _jsiInstalled = false;

/** Ensure the JSI HostObject is installed. Call once at startup. */
async function ensureJSI(): Promise<void> {
  if (_jsiInstalled) return;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if ((global as any).__EdgeVectorStore) {
    _jsiInstalled = true;
    return;
  }
  try {
    // TurboModule's getTurboModule: installs the HostObject on iOS;
    // Android's install() calls nativeInstall via JNI.
    await NativeEdgeVectorStore.install();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    _jsiInstalled = !!(global as any).__EdgeVectorStore;
  } catch {
    _jsiInstalled = false;
  }
}

/** True if the JSI HostObject with direct methods is available. */
function hasJSI(): boolean {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return _jsiInstalled && !!(global as any).__EdgeVectorStore;
}

/** Access the JSI HostObject installed on the JS global. */
function getJSI(): EVSHostObject {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (global as any).__EdgeVectorStore as EVSHostObject;
}

/** Ensure the vector is a Float32Array for zero-copy JSI transfer. */
function toFloat32(v: VectorLike): Float32Array {
  return v instanceof Float32Array ? v : new Float32Array(v);
}

/**
 * Unified on-device vector store.
 *
 * Uses JSI HostObject for zero-copy hot paths when available,
 * falls back to TurboModule (JSON) otherwise.
 */
export class EdgeVectorStore {
  private storagePath: string;

  private constructor(storagePath: string) {
    this.storagePath = storagePath;
  }

  // ── lifecycle ──────────────────────────────────────────────

  static async init(config: StoreConfig): Promise<EdgeVectorStore> {
    // Ensure JSI HostObject is installed before first use
    await ensureJSI();

    if (hasJSI()) {
      const ok = getJSI().init(JSON.stringify(config));
      if (!ok) throw new Error('EdgeVectorStore: native init failed');
    } else {
      const ok = await NativeEdgeVectorStore.init(JSON.stringify(config));
      if (!ok) throw new Error('EdgeVectorStore: native init failed');
    }
    return new EdgeVectorStore(config.storagePath);
  }

  // ── writes ─────────────────────────────────────────────────

  async upsertDocuments(docs: Document[]): Promise<void> {
    if (hasJSI()) {
      getJSI().upsertDocuments(JSON.stringify(docs));
    } else {
      await NativeEdgeVectorStore.upsertDocuments(JSON.stringify(docs));
    }
  }

  /**
   * Insert/update vectors.
   * JSI path: packs into a single Float32Array for zero-copy transfer.
   * TurboModule path: falls back to JSON serialization.
   */
  async upsertVectors(entries: VectorEntry[]): Promise<void> {
    if (entries.length === 0) return;

    if (hasJSI()) {
      const dims = entries[0].vector.length;
      const ids = new Array<string>(entries.length);
      const packed = new Float32Array(entries.length * dims);

      for (let i = 0; i < entries.length; i++) {
        ids[i] = entries[i].id;
        const vec = entries[i].vector;
        if (vec instanceof Float32Array) {
          packed.set(vec, i * dims);
        } else {
          for (let d = 0; d < dims; d++) packed[i * dims + d] = vec[d];
        }
      }
      getJSI().upsertVectorsDirect(this.storagePath, ids, packed, dims);
    } else {
      // TurboModule JSON path — convert VectorLike to number[]
      const jsonEntries = entries.map((e) => ({
        ...e,
        vector: e.vector instanceof Float32Array ? Array.from(e.vector) : e.vector,
      }));
      await NativeEdgeVectorStore.upsertVectors(JSON.stringify(jsonEntries));
    }
  }

  // ── search ─────────────────────────────────────────────────

  async search(options: SearchOptions): Promise<SearchResult[]> {
    if (hasJSI()) {
      const queryVector = toFloat32(options.queryVector);
      const raw = getJSI().searchDirect(
        this.storagePath,
        queryVector,
        options.topK ?? 10,
        options.mode,
      );
      return raw.map((r) => ({
        id: r.id,
        distance: r.distance,
        payload: r.payload ? JSON.parse(r.payload) : undefined,
      }));
    } else {
      const jsonOpts = {
        ...options,
        queryVector:
          options.queryVector instanceof Float32Array
            ? Array.from(options.queryVector)
            : options.queryVector,
      };
      const resultJson = await NativeEdgeVectorStore.search(
        JSON.stringify(jsonOpts),
      );
      return JSON.parse(resultJson) as SearchResult[];
    }
  }

  // ── deletes ────────────────────────────────────────────────

  async remove(ids: string[]): Promise<void> {
    if (hasJSI()) {
      getJSI().remove(JSON.stringify(ids));
    } else {
      await NativeEdgeVectorStore.remove(JSON.stringify(ids));
    }
  }

  // ── maintenance ────────────────────────────────────────────

  async compact(): Promise<void> {
    if (hasJSI()) {
      getJSI().compact();
    } else {
      await NativeEdgeVectorStore.compact();
    }
  }

  async importPack(path: string): Promise<void> {
    if (hasJSI()) {
      getJSI().importPack(path);
    } else {
      await NativeEdgeVectorStore.importPack(path);
    }
  }

  async exportPack(path: string): Promise<void> {
    if (hasJSI()) {
      getJSI().exportPack(path);
    } else {
      await NativeEdgeVectorStore.exportPack(path);
    }
  }

  async getStats(): Promise<StoreStats> {
    if (hasJSI()) {
      const json = getJSI().getStats();
      return JSON.parse(json) as StoreStats;
    } else {
      const json = await NativeEdgeVectorStore.getStats();
      return JSON.parse(json) as StoreStats;
    }
  }

  // ── benchmarks ───────────────────────────────────────────

  static async benchmarkRawANN(config: {
    dims: number;
    numVectors: number;
    numQueries: number;
    topK: number;
    quantization?: string;
    ef?: number;
    sweepEf?: string;
    rerank?: string;
    oversample?: number;
    warmup?: number;
    connectivity?: number;
    expansionAdd?: number;
  }): Promise<Array<Record<string, number | string | boolean>>> {
    const json = await NativeEdgeVectorStore.benchmarkRawANN(
      JSON.stringify(config),
    );
    return JSON.parse(json);
  }
}

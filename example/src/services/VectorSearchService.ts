// VectorSearchService — wraps EdgeVectorStore for RAG search

import RNFS from 'react-native-fs';
import { EdgeVectorStore } from '@sukshm/edge-vector-store';

const DIMS = 384;

export class VectorSearchService {
  private store: any = null;

  async initialize(onStatus?: (msg: string) => void): Promise<void> {
    onStatus?.('Initializing vector store...');

    const storeDir = `${RNFS.DocumentDirectoryPath}/evs_farmer_store`;
    const exists = await RNFS.exists(storeDir);
    if (!exists) {
      await RNFS.mkdir(storeDir);
    }

    this.store = await EdgeVectorStore.init({
      storagePath: storeDir,
      dimensions: DIMS,
      quantization: 'i8',
      metric: 'cosine',
    });

    const stats = await this.store.getStats();
    onStatus?.(
      `Vector store ready — ${stats.vectorCount} vectors, ${stats.documentCount} docs`,
    );
  }

  async loadPack(packPath: string, onStatus?: (msg: string) => void): Promise<void> {
    if (!this.store) throw new Error('Store not initialized');

    onStatus?.('Importing .evs pack...');
    await this.store.importPack(packPath);

    const stats = await this.store.getStats();
    onStatus?.(
      `Pack loaded — ${stats.vectorCount} vectors, ${stats.documentCount} docs`,
    );
  }

  async search(
    queryVector: number[],
    topK: number = 5,
  ): Promise<any[]> {
    if (!this.store) throw new Error('Store not initialized');

    return this.store.search({
      queryVector,
      topK,
      mode: 'balanced',
    });
  }

  async getStats() {
    if (!this.store) return null;
    return this.store.getStats();
  }

  dispose() {
    this.store = null;
  }
}

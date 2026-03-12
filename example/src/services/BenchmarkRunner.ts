import RNFS from 'react-native-fs';
import { EdgeVectorStore } from 'react-native-edge-vector-store';

// ─── Benchmark Suite ────────────────────────────────────────
//
// Phase 1: Sanity   — 10K, 384d, raw, F32
// Phase 2: Core     — 100K, 384d, raw, matched recall bands
// Phase 3: Mobile   — 100K, 384d, persistent, cold/warm
// Phase 4: Compress — 100K, 384d, I8+rerank vs F32
// Phase 5: Scale    — 250K, 384d+768d, persistent

export interface PhaseConfig {
  id: number;
  name: string;
  enabled: boolean;
}

export const PHASES: PhaseConfig[] = [
  { id: 1, name: 'Sanity (10K raw)', enabled: true },
  { id: 2, name: 'Core parity (100K raw)', enabled: true },
  { id: 3, name: 'Mobile reality (100K persist)', enabled: false },
  { id: 4, name: 'Compression (I8 vs F32)', enabled: false },
  { id: 5, name: 'Scale (250K)', enabled: false },
];

// ─── Result types ───────────────────────────────────────────

export interface BenchResult {
  engine: string;
  mode: string;
  phase: number;
  datasetSize: number;
  dimensions: number;
  metricType: string;
  buildConfig: string;
  searchConfig: string;
  recallAt10: number;
  meanLatency: number;
  p50: number;
  p95: number;
  p99: number;
  qps: number;
  insertThroughput: number;
  buildTimeSec: number;
  memoryDeltaMB: number;
  diskUsageMB: number;
  coldStartMs: number;
  warmStartMs: number;
}

// ─── Constants ──────────────────────────────────────────────

const TOP_K = 10;
const NUM_QUERIES = 100;
const WARMUP = 10;

// ef sweep: covers low → very high search breadth
const EF_SWEEP = '16,32,48,64,96,128,200,300,500';

// Build quality configs
const BUILD_CONFIGS = [
  { label: 'light',   connectivity: 8,  expansionAdd: 64 },
  { label: 'default', connectivity: 16, expansionAdd: 128 },
  { label: 'heavy',   connectivity: 32, expansionAdd: 256 },
];

// Target recall bands for matched-recall comparison
const RECALL_BANDS = [0.40, 0.60, 0.75, 0.85, 0.90];

// ─── Helpers ────────────────────────────────────────────────

function randomVec(dims: number): number[] {
  const v = new Array(dims);
  for (let i = 0; i < dims; i++) v[i] = Math.random() * 2 - 1;
  let norm = 0;
  for (let i = 0; i < dims; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < dims; i++) v[i] /= norm;
  return v;
}

interface RawSweepResult {
  ef: number;
  recallAtK: number;
  searchMean: number;
  searchP50: number;
  searchP95: number;
  searchP99: number;
  qps: number;
  buildTimeSec: number;
  insertThroughput: number;
  memoryMB: number;
  buildConfig: string;
  quantization: string;
  rerank: boolean;
  oversample: number;
  numVectors: number;
  dims: number;
}

// Find the sweep point closest to a target recall band
function findNearestRecall(
  sweepPoints: RawSweepResult[],
  target: number,
  tolerance = 0.05,
): RawSweepResult | null {
  let best: RawSweepResult | null = null;
  let bestDist = Infinity;
  for (const p of sweepPoints) {
    const dist = Math.abs(p.recallAtK - target);
    if (dist < bestDist) {
      bestDist = dist;
      best = p;
    }
  }
  return best && bestDist <= tolerance ? best : null;
}

function rawToBenchResult(
  r: RawSweepResult,
  phase: number,
  mode: string,
  engine: string,
): BenchResult {
  return {
    engine,
    mode,
    phase,
    datasetSize: r.numVectors,
    dimensions: r.dims,
    metricType: 'cosine',
    buildConfig: r.buildConfig,
    searchConfig: `ef=${r.ef}${r.rerank ? ` +rerank×${r.oversample}` : ''} ${r.quantization}`,
    recallAt10: r.recallAtK,
    meanLatency: r.searchMean,
    p50: r.searchP50,
    p95: r.searchP95,
    p99: r.searchP99,
    qps: r.qps,
    insertThroughput: r.insertThroughput,
    buildTimeSec: r.buildTimeSec,
    memoryDeltaMB: r.memoryMB,
    diskUsageMB: 0,
    coldStartMs: 0,
    warmStartMs: 0,
  };
}

// ─── Runner ─────────────────────────────────────────────────

export class BenchmarkRunner {
  async run(
    phases: PhaseConfig[],
    onLog: (msg: string) => void,
    onProgress?: (fraction: number, label: string) => void,
  ): Promise<BenchResult[]> {
    const results: BenchResult[] = [];
    const enabled = phases.filter((p) => p.enabled);
    const total = enabled.length;
    let done = 0;

    for (const phase of enabled) {
      const base = done / total;
      const range = 1 / total;
      const progress = (frac: number, label: string) =>
        onProgress?.(base + frac * range, `P${phase.id}: ${label}`);

      onLog(`━━━ Phase ${phase.id}: ${phase.name} ━━━`);

      let phaseResults: BenchResult[] = [];
      switch (phase.id) {
        case 1:
          phaseResults = await this.phaseSanity(onLog, progress);
          break;
        case 2:
          phaseResults = await this.phaseCoreParity(onLog, progress);
          break;
        case 3:
          phaseResults = await this.phaseMobileReality(onLog, progress);
          break;
        case 4:
          phaseResults = await this.phaseCompression(onLog, progress);
          break;
        case 5:
          phaseResults = await this.phaseScale(onLog, progress);
          break;
      }
      results.push(...phaseResults);
      done++;
    }

    onLog('━━━ Benchmark suite complete ━━━');
    onProgress?.(1.0, 'Done');
    return results;
  }

  // ── Phase 1: Sanity baseline ──────────────────────────────

  private async phaseSanity(
    onLog: (msg: string) => void,
    onProgress: (frac: number, label: string) => void,
  ): Promise<BenchResult[]> {
    const results: BenchResult[] = [];

    // F32, default build, sweep ef
    onLog('Raw F32 default build — sweep ef');
    onProgress(0.1, 'F32 sweep...');
    const sweep = await this.rawSweep({
      numVectors: 10_000,
      dims: 384,
      quantization: 'f32',
      build: BUILD_CONFIGS[1], // default
      rerank: false,
    });

    for (const r of sweep) {
      results.push(rawToBenchResult(r, 1, 'raw', 'EVS-core'));
      onLog(
        `  ef=${r.ef}: recall=${r.recallAtK.toFixed(3)} ` +
          `p50=${r.searchP50.toFixed(3)}ms qps=${r.qps.toFixed(0)}`,
      );
    }

    onProgress(1.0, 'Done');
    return results;
  }

  // ── Phase 2: Core parity ─────────────────────────────────

  private async phaseCoreParity(
    onLog: (msg: string) => void,
    onProgress: (frac: number, label: string) => void,
  ): Promise<BenchResult[]> {
    const results: BenchResult[] = [];
    const builds = BUILD_CONFIGS;
    let step = 0;
    const totalSteps = builds.length;

    for (const build of builds) {
      onLog(`Build: ${build.label} (M=${build.connectivity} A=${build.expansionAdd})`);
      onProgress(step / totalSteps, `${build.label} build...`);

      const sweep = await this.rawSweep({
        numVectors: 100_000,
        dims: 384,
        quantization: 'f32',
        build,
        rerank: false,
      });

      for (const r of sweep) {
        results.push(rawToBenchResult(r, 2, 'raw', 'EVS-core'));
      }

      // Report matched recall bands
      for (const band of RECALL_BANDS) {
        const match = findNearestRecall(sweep, band);
        if (match) {
          onLog(
            `  [${build.label}] recall≈${band}: ef=${match.ef} ` +
              `actual=${match.recallAtK.toFixed(3)} ` +
              `p50=${match.searchP50.toFixed(3)}ms p99=${match.searchP99.toFixed(3)}ms ` +
              `qps=${match.qps.toFixed(0)}`,
          );
        }
      }

      step++;
    }

    onProgress(1.0, 'Done');
    return results;
  }

  // ── Phase 3: Mobile reality ──────────────────────────────

  private async phaseMobileReality(
    onLog: (msg: string) => void,
    onProgress: (frac: number, label: string) => void,
  ): Promise<BenchResult[]> {
    const results: BenchResult[] = [];
    const numVectors = 100_000;
    const dims = 384;

    // Generate vectors once for all persistent tests
    onLog('Generating 100K vectors...');
    onProgress(0.0, 'Generating vectors...');
    const vectors: number[][] = [];
    for (let i = 0; i < numVectors; i++) vectors.push(randomVec(dims));
    const queries: number[][] = [];
    for (let i = 0; i < NUM_QUERIES; i++) queries.push(randomVec(dims));
    onProgress(0.1, 'Vectors ready');

    // Sweep expansionSearch values for persistent mode
    const efValues = [32, 64, 128, 200];
    let step = 0;

    for (const ef of efValues) {
      onLog(`Persistent ef=${ef}...`);
      onProgress(0.1 + (step / efValues.length) * 0.8, `persist ef=${ef}`);

      const r = await this.benchPersistent(
        vectors, queries, dims, ef, onLog,
      );
      results.push({ ...r, phase: 3 });

      onLog(
        `  ef=${ef}: recall=${r.recallAt10.toFixed(3)} ` +
          `p50=${r.p50.toFixed(3)}ms p99=${r.p99.toFixed(3)}ms ` +
          `cold=${r.coldStartMs.toFixed(0)}ms warm=${r.warmStartMs.toFixed(1)}ms ` +
          `disk=${r.diskUsageMB.toFixed(1)}MB`,
      );
      step++;
    }

    onProgress(1.0, 'Done');
    return results;
  }

  // ── Phase 4: Compression advantage ───────────────────────

  private async phaseCompression(
    onLog: (msg: string) => void,
    onProgress: (frac: number, label: string) => void,
  ): Promise<BenchResult[]> {
    const results: BenchResult[] = [];
    const build = BUILD_CONFIGS[1]; // default

    // 1. F32 baseline sweep
    onLog('F32 baseline sweep...');
    onProgress(0.0, 'F32 baseline...');
    const f32Sweep = await this.rawSweep({
      numVectors: 100_000, dims: 384,
      quantization: 'f32', build, rerank: false,
    });
    for (const r of f32Sweep) {
      results.push(rawToBenchResult(r, 4, 'compression', 'F32'));
    }

    // 2. I8 sweep (no rerank)
    onLog('I8 sweep (no rerank)...');
    onProgress(0.33, 'I8 sweep...');
    const i8Sweep = await this.rawSweep({
      numVectors: 100_000, dims: 384,
      quantization: 'i8', build, rerank: false,
    });
    for (const r of i8Sweep) {
      results.push(rawToBenchResult(r, 4, 'compression', 'I8'));
    }

    // 3. I8 + F32 rerank sweep
    onLog('I8+F32 rerank sweep...');
    onProgress(0.66, 'I8+rerank...');
    const rerankSweep = await this.rawSweep({
      numVectors: 100_000, dims: 384,
      quantization: 'i8', build, rerank: true, oversample: 3,
    });
    for (const r of rerankSweep) {
      results.push(rawToBenchResult(r, 4, 'compression', 'I8+rerank'));
    }

    // Report matched bands
    for (const band of [0.60, 0.75, 0.85]) {
      const f32Match = findNearestRecall(f32Sweep, band);
      const i8Match = findNearestRecall(i8Sweep, band);
      const rrMatch = findNearestRecall(rerankSweep, band);

      onLog(`  recall≈${band}:`);
      if (f32Match)
        onLog(`    F32:       ef=${f32Match.ef} p50=${f32Match.searchP50.toFixed(3)}ms mem=${f32Match.memoryMB.toFixed(1)}MB`);
      if (i8Match)
        onLog(`    I8:        ef=${i8Match.ef} p50=${i8Match.searchP50.toFixed(3)}ms mem=${i8Match.memoryMB.toFixed(1)}MB`);
      if (rrMatch)
        onLog(`    I8+rerank: ef=${rrMatch.ef} p50=${rrMatch.searchP50.toFixed(3)}ms mem=${rrMatch.memoryMB.toFixed(1)}MB`);
    }

    onProgress(1.0, 'Done');
    return results;
  }

  // ── Phase 5: Scale pressure ──────────────────────────────

  private async phaseScale(
    onLog: (msg: string) => void,
    onProgress: (frac: number, label: string) => void,
  ): Promise<BenchResult[]> {
    const results: BenchResult[] = [];
    const build = BUILD_CONFIGS[1]; // default

    const configs = [
      { n: 250_000, d: 384 },
      { n: 250_000, d: 768 },
    ];

    let step = 0;
    for (const { n, d } of configs) {
      onLog(`Scale: ${n / 1000}K × ${d}d`);
      onProgress(step / configs.length, `${n / 1000}K×${d}d...`);

      const sweep = await this.rawSweep({
        numVectors: n, dims: d,
        quantization: 'f32', build, rerank: false,
      });

      for (const r of sweep) {
        results.push(rawToBenchResult(r, 5, 'raw', 'EVS-core'));
      }

      const match75 = findNearestRecall(sweep, 0.75);
      if (match75) {
        onLog(
          `  recall≈0.75: ef=${match75.ef} p50=${match75.searchP50.toFixed(3)}ms ` +
            `build=${match75.buildTimeSec.toFixed(1)}s mem=${match75.memoryMB.toFixed(1)}MB`,
        );
      }

      step++;
    }

    onProgress(1.0, 'Done');
    return results;
  }

  // ── Raw sweep helper ─────────────────────────────────────

  private async rawSweep(opts: {
    numVectors: number;
    dims: number;
    quantization: string;
    build: { label: string; connectivity: number; expansionAdd: number };
    rerank: boolean;
    oversample?: number;
  }): Promise<RawSweepResult[]> {
    const raw = await EdgeVectorStore.benchmarkRawANN({
      dims: opts.dims,
      numVectors: opts.numVectors,
      numQueries: NUM_QUERIES,
      topK: TOP_K,
      quantization: opts.quantization,
      connectivity: opts.build.connectivity,
      expansionAdd: opts.build.expansionAdd,
      sweepEf: EF_SWEEP,
      rerank: opts.rerank ? 'true' : 'false',
      oversample: opts.oversample ?? 3,
      warmup: WARMUP,
    });

    return raw.map((r: any) => ({
      ef: r.ef as number,
      recallAtK: r.recallAtK as number,
      searchMean: r.searchMean as number,
      searchP50: r.searchP50 as number,
      searchP95: r.searchP95 as number,
      searchP99: r.searchP99 as number,
      qps: r.qps as number,
      buildTimeSec: r.buildTimeSec as number,
      insertThroughput: r.insertThroughput as number,
      memoryMB: r.memoryMB as number,
      buildConfig: `${opts.build.label} M${opts.build.connectivity}_A${opts.build.expansionAdd}`,
      quantization: opts.quantization,
      rerank: opts.rerank,
      oversample: opts.oversample ?? 3,
      numVectors: opts.numVectors,
      dims: opts.dims,
    }));
  }

  // ── Persistent mode benchmark ────────────────────────────

  private async benchPersistent(
    vectors: number[][],
    queries: number[][],
    dims: number,
    expansionSearch: number,
    onLog: (msg: string) => void,
  ): Promise<BenchResult> {
    const storeDir = `${RNFS.DocumentDirectoryPath}/evs_bench_persist`;
    if (await RNFS.exists(storeDir)) await RNFS.unlink(storeDir);
    await RNFS.mkdir(storeDir);

    // Build: init + insert + compact
    const tBuild = performance.now();
    const store = await EdgeVectorStore.init({
      storagePath: storeDir,
      dimensions: dims,
      quantization: 'f32',
      metric: 'cosine',
      expansionSearch,
    });

    const BATCH = 500;
    for (let start = 0; start < vectors.length; start += BATCH) {
      const end = Math.min(start + BATCH, vectors.length);
      const docs = [];
      const vecs = [];
      for (let i = start; i < end; i++) {
        docs.push({ id: `v${i}`, payload: { idx: i } });
        vecs.push({ id: `v${i}`, docId: `v${i}`, vector: vectors[i] });
      }
      await store.upsertDocuments(docs);
      await store.upsertVectors(vecs);
    }
    await store.compact();
    const buildTimeSec = (performance.now() - tBuild) / 1000;

    // Stats after build
    const stats = await store.getStats();
    const memoryMB = stats.memoryUsageBytes / 1e6;
    const diskMB = stats.coldIndexSizeBytes / 1e6;

    // Cold start: re-init from disk
    const tCold = performance.now();
    const store2 = await EdgeVectorStore.init({
      storagePath: storeDir,
      dimensions: dims,
      quantization: 'f32',
      metric: 'cosine',
      expansionSearch,
    });
    const coldStartMs = performance.now() - tCold;

    // Warmup
    for (let w = 0; w < WARMUP && w < queries.length; w++) {
      await store2.search({ queryVector: queries[w], topK: TOP_K });
    }

    // Warm start: first search after warmup
    const tWarm = performance.now();
    await store2.search({ queryVector: queries[0], topK: TOP_K });
    const warmStartMs = performance.now() - tWarm;

    // Timed search
    const latencies: number[] = [];
    const annResults: number[][] = [];
    for (let q = 0; q < queries.length; q++) {
      const ts = performance.now();
      const hits = await store2.search({ queryVector: queries[q], topK: TOP_K });
      latencies.push(performance.now() - ts);
      annResults.push(
        hits.map((h) => {
          const m = h.id.match(/\d+/);
          return m ? parseInt(m[0], 10) : -1;
        }),
      );
    }

    // Recall@K via brute-force
    let totalHits = 0;
    for (let q = 0; q < queries.length; q++) {
      const scores: { idx: number; dot: number }[] = [];
      const qv = queries[q];
      for (let i = 0; i < vectors.length; i++) {
        let dot = 0;
        for (let d = 0; d < dims; d++) dot += qv[d] * vectors[i][d];
        scores.push({ idx: i, dot });
      }
      scores.sort((a, b) => b.dot - a.dot);
      const gtSet = new Set(scores.slice(0, TOP_K).map((s) => s.idx));
      for (const annIdx of annResults[q]) {
        if (gtSet.has(annIdx)) totalHits++;
      }
    }
    const recallAt10 = totalHits / (queries.length * TOP_K);

    latencies.sort((a, b) => a - b);
    const mean = latencies.reduce((s, v) => s + v, 0) / latencies.length;
    const p50 = latencies[Math.floor(latencies.length * 0.5)];
    const p95 = latencies[Math.floor(latencies.length * 0.95)];
    const p99 = latencies[Math.floor(latencies.length * 0.99)];

    await RNFS.unlink(storeDir);

    return {
      engine: 'EVS-persistent',
      mode: 'persistent',
      phase: 3,
      datasetSize: vectors.length,
      dimensions: dims,
      metricType: 'cosine',
      buildConfig: 'default',
      searchConfig: `ef=${expansionSearch} f32`,
      recallAt10,
      meanLatency: mean,
      p50,
      p95,
      p99,
      qps: mean > 0 ? 1000 / mean : 0,
      insertThroughput: vectors.length / buildTimeSec,
      buildTimeSec,
      memoryDeltaMB: memoryMB,
      diskUsageMB: diskMB,
      coldStartMs,
      warmStartMs,
    };
  }
}

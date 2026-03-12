#!/usr/bin/env python3
"""
Benchmark LanceDB against the same 100K farmer dataset.

Usage:
    pip install lancedb numpy pyarrow
    python tools/bench_lancedb.py

Input:
    data/farmers_100k.npy             — float32 vectors (100000 x 384)
    data/farmers_100k_meta.json       — metadata records
    data/farmers_100k_queries.bin     — 1000 query vectors
    data/farmers_100k_groundtruth.bin — ground-truth top-10 indices

Output:
    data/bench_lancedb.json — benchmark results
"""

import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

try:
    import lancedb
    import pyarrow as pa
except ImportError:
    print("Install: pip install lancedb numpy pyarrow")
    sys.exit(1)


DIMS = 384
TOP_K = 10
NUM_QUERIES = 1000


def measure_rss_mb():
    """Get current RSS in MB (macOS/Linux)."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return ru.ru_maxrss / (1024 * 1024)  # macOS returns bytes
    except Exception:
        return 0.0


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"

    # Load data
    print("Loading vectors...")
    vectors = np.load(data_dir / "farmers_100k.npy")
    n = vectors.shape[0]

    print("Loading metadata...")
    with open(data_dir / "farmers_100k_meta.json") as f:
        meta = json.load(f)

    print("Loading queries...")
    queries = np.fromfile(data_dir / "farmers_100k_queries.bin", dtype=np.float32)
    queries = queries.reshape(NUM_QUERIES, DIMS)

    print("Loading ground-truth...")
    gt = np.fromfile(data_dir / "farmers_100k_groundtruth.bin", dtype=np.uint64)
    gt = gt.reshape(NUM_QUERIES, TOP_K)

    # Setup LanceDB
    db_path = data_dir / "bench_lancedb_store"
    if db_path.exists():
        shutil.rmtree(db_path)

    rss_before = measure_rss_mb()

    # ── Insert benchmark ──────────────────────────────────────
    print(f"\n[INSERT] {n} vectors into LanceDB...")
    db = lancedb.connect(str(db_path))

    # Build pyarrow table
    data = []
    for i in range(n):
        row = {"id": i, "vector": vectors[i].tolist()}
        for field in ["state", "crop", "season", "soilType"]:
            row[field] = meta[i].get(field, "")
        data.append(row)

    t0 = time.perf_counter()
    table = db.create_table("farmers", data)
    insert_time = time.perf_counter() - t0
    insert_throughput = n / insert_time

    print(f"  Insert: {insert_time:.2f}s ({insert_throughput:.0f} vec/s)")

    # Create IVF_PQ index
    print("  Creating IVF_PQ index...")
    t0 = time.perf_counter()
    table.create_index(
        metric="cosine",
        num_partitions=256,
        num_sub_vectors=48,
    )
    index_time = time.perf_counter() - t0
    print(f"  Index build: {index_time:.2f}s")

    rss_after = measure_rss_mb()

    # ── Search benchmark ──────────────────────────────────────
    print(f"\n[SEARCH] {NUM_QUERIES} queries, top-{TOP_K}...")
    latencies = []

    for i in range(NUM_QUERIES):
        q = queries[i].tolist()
        t0 = time.perf_counter()
        results = table.search(q).limit(TOP_K).to_list()
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed * 1000)  # ms

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean_lat = statistics.mean(latencies)

    print(f"  Latency — mean: {mean_lat:.2f}ms, p50: {p50:.2f}ms, "
          f"p95: {p95:.2f}ms, p99: {p99:.2f}ms")

    # ── Recall benchmark ──────────────────────────────────────
    print(f"\n[RECALL] Computing recall@{TOP_K}...")
    recall_sum = 0.0
    for i in range(NUM_QUERIES):
        q = queries[i].tolist()
        results = table.search(q).limit(TOP_K).to_list()
        result_ids = {r["id"] for r in results}
        gt_ids = set(gt[i].tolist())
        recall_sum += len(result_ids & gt_ids) / TOP_K

    recall = recall_sum / NUM_QUERIES
    print(f"  Recall@{TOP_K}: {recall:.4f}")

    # ── Disk usage ────────────────────────────────────────────
    disk_bytes = sum(
        f.stat().st_size
        for f in db_path.rglob("*")
        if f.is_file()
    )

    # ── Cold start ────────────────────────────────────────────
    print("\n[COLD START] Reopening database...")
    del table
    del db

    t0 = time.perf_counter()
    db2 = lancedb.connect(str(db_path))
    table2 = db2.open_table("farmers")
    _ = table2.search(queries[0].tolist()).limit(1).to_list()
    cold_start = (time.perf_counter() - t0) * 1000  # ms
    print(f"  Cold start: {cold_start:.2f}ms")

    # ── Results ───────────────────────────────────────────────
    results = {
        "engine": "LanceDB",
        "version": lancedb.__version__,
        "numVectors": n,
        "dimensions": DIMS,
        "insertTimeSec": round(insert_time, 3),
        "insertThroughput": round(insert_throughput, 0),
        "indexBuildTimeSec": round(index_time, 3),
        "searchLatencyMs": {
            "mean": round(mean_lat, 3),
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
        },
        "recallAt10": round(recall, 4),
        "diskUsageBytes": disk_bytes,
        "diskUsageMB": round(disk_bytes / 1e6, 2),
        "memoryDeltaMB": round(rss_after - rss_before, 2),
        "coldStartMs": round(cold_start, 2),
    }

    out_path = data_dir / "bench_lancedb.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")
    print(json.dumps(results, indent=2))

    # Cleanup
    del table2
    del db2
    shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert numpy vectors + metadata JSON to flat binary files for C++ benchmark.

Usage:
    python tools/convert_vectors.py

Input:
    data/farmers_100k.npy          — float32 array (N x 384)
    data/farmers_100k_meta.json    — records with id field

Output:
    data/farmers_100k_vectors.bin  — raw float32 (N * 384 * 4 bytes)
    data/farmers_100k_ids.bin      — raw uint64 sequential keys (N * 8 bytes)
    data/farmers_100k_queries.bin  — 1000 random query vectors (1000 * 384 * 4 bytes)
"""

import struct
from pathlib import Path

import numpy as np


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"

    # Load vectors
    npy_path = data_dir / "farmers_100k.npy"
    vectors = np.load(npy_path)
    n, dims = vectors.shape
    print(f"Loaded {n} vectors, {dims} dims from {npy_path}")

    # Write raw vectors
    vec_path = data_dir / "farmers_100k_vectors.bin"
    vectors.astype(np.float32).tofile(vec_path)
    print(f"Wrote vectors: {vec_path} ({vec_path.stat().st_size / 1e6:.1f} MB)")

    # Write sequential uint64 keys
    ids_path = data_dir / "farmers_100k_ids.bin"
    ids = np.arange(n, dtype=np.uint64)
    ids.tofile(ids_path)
    print(f"Wrote IDs: {ids_path} ({ids_path.stat().st_size / 1e3:.1f} KB)")

    # Generate 1000 query vectors (random samples from the dataset)
    rng = np.random.default_rng(42)
    query_indices = rng.choice(n, size=1000, replace=False)
    queries = vectors[query_indices].copy()
    # Add small noise so queries aren't identical to stored vectors
    queries += rng.normal(0, 0.01, queries.shape).astype(np.float32)
    # Re-normalize (cosine metric)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / np.clip(norms, 1e-8, None)

    query_path = data_dir / "farmers_100k_queries.bin"
    queries.astype(np.float32).tofile(query_path)
    print(f"Wrote queries: {query_path} ({query_path.stat().st_size / 1e3:.1f} KB)")

    # Save ground-truth indices for recall calculation
    # (brute-force cosine similarity for each query against all vectors)
    print("Computing ground-truth top-10 for recall...")
    gt_path = data_dir / "farmers_100k_groundtruth.bin"
    k = 10
    gt_indices = np.zeros((1000, k), dtype=np.uint64)
    for i in range(1000):
        # cosine similarity = dot product (vectors are normalized)
        sims = vectors @ queries[i]
        top_k = np.argpartition(sims, -k)[-k:]
        top_k = top_k[np.argsort(-sims[top_k])]
        gt_indices[i] = top_k.astype(np.uint64)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/1000 queries done")

    gt_indices.tofile(gt_path)
    print(f"Wrote ground-truth: {gt_path} ({gt_path.stat().st_size / 1e3:.1f} KB)")

    print(f"\n✓ Conversion complete")


if __name__ == "__main__":
    main()

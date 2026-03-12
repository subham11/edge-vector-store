#!/usr/bin/env python3
"""
Export pre-computed farmer dataset as an .evs pack file.

This creates an .evs pack that the React Native app can import
via EdgeVectorStore.importPack().

Usage:
    python tools/export_evs_pack.py

Input:
    data/farmers_100k.json — records with id, fields, vector
Output:
    data/farmers_100k.evs  — zip-based pack (manifest + usearch index + sqlite db)

NOTE: This script uses the compiled desktop binary (evs_test) approach.
      It generates a JSON config, upserts all data via the C++ EdgeStore,
      then exports the pack. Requires building the desktop target first.
      Alternatively, it writes a script that the C++ benchmark can invoke.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    project_root = Path(__file__).resolve().parent.parent

    json_path = data_dir / "farmers_100k.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run generate_dataset.py first.")
        sys.exit(1)

    # Check if the desktop test binary exists
    build_dir = project_root / "build" / "desktop"
    test_bin = build_dir / "evs_test"
    if not test_bin.exists():
        print(f"ERROR: {test_bin} not found. Build desktop target first:")
        print(f"  cd {build_dir} && cmake ../.. -DEVS_DESKTOP_TEST=ON && make -j")
        sys.exit(1)

    print("Loading dataset...")
    with open(json_path) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records")

    # We'll create a Python-based packer that directly writes the .evs
    # format using the same structure the C++ code expects.
    # For now, generate chunked JSON files that a C++ ingest tool can consume.

    pack_dir = data_dir / "pack_staging"
    pack_dir.mkdir(exist_ok=True)

    # Write documents (without vectors) in chunks for memory efficiency
    chunk_size = 5000
    doc_chunks_dir = pack_dir / "docs"
    doc_chunks_dir.mkdir(exist_ok=True)

    vec_chunks_dir = pack_dir / "vecs"
    vec_chunks_dir.mkdir(exist_ok=True)

    for chunk_idx in range(0, len(records), chunk_size):
        chunk = records[chunk_idx : chunk_idx + chunk_size]
        chunk_num = chunk_idx // chunk_size

        # Documents (metadata only)
        docs = []
        for r in chunk:
            doc = {k: v for k, v in r.items() if k != "vector"}
            doc["payload"] = {
                "state": doc.pop("state"),
                "farmerName": doc.pop("farmerName"),
                "soilType": doc.pop("soilType"),
                "season": doc.pop("season"),
                "crop": doc.pop("crop"),
                "warnings": doc.pop("warnings"),
                "areaAcres": doc.pop("areaAcres"),
                "irrigationType": doc.pop("irrigationType"),
                "yieldPerAcre": doc.pop("yieldPerAcre"),
                "weather": doc.pop("weather"),
                "dailyWaterLiters": doc.pop("dailyWaterLiters"),
                "fertilizers": doc.pop("fertilizers"),
            }
            docs.append(doc)

        doc_path = doc_chunks_dir / f"chunk_{chunk_num:04d}.json"
        with open(doc_path, "w") as f:
            json.dump(docs, f, separators=(",", ":"))

        # Vectors
        vecs = [{"id": r["id"], "vector": r["vector"]} for r in chunk]
        vec_path = vec_chunks_dir / f"chunk_{chunk_num:04d}.json"
        with open(vec_path, "w") as f:
            json.dump(vecs, f, separators=(",", ":"))

        print(f"  Chunk {chunk_num}: {len(chunk)} records")

    print(f"\nStaging files written to {pack_dir}")
    print(f"  Doc chunks: {len(list(doc_chunks_dir.iterdir()))}")
    print(f"  Vec chunks: {len(list(vec_chunks_dir.iterdir()))}")

    # Generate the ingest script for the C++ benchmark tool
    ingest_script = pack_dir / "ingest_commands.txt"
    with open(ingest_script, "w") as f:
        f.write(f"STORAGE_PATH={data_dir / 'evs_store'}\n")
        f.write(f"PACK_OUTPUT={data_dir / 'farmers_100k.evs'}\n")
        f.write(f"DIMS=384\n")
        for p in sorted(doc_chunks_dir.iterdir()):
            f.write(f"DOC_CHUNK={p}\n")
        for p in sorted(vec_chunks_dir.iterdir()):
            f.write(f"VEC_CHUNK={p}\n")

    print(f"\nIngest script: {ingest_script}")
    print("The C++ benchmark or a dedicated ingest tool will consume these.")
    print("To create the .evs pack, build and run: ./evs_bench --export-pack")


if __name__ == "__main__":
    main()

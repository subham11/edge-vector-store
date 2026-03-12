#!/usr/bin/env python3
"""
Aggregate benchmark results from all engines into a comparison report.

Usage:
    python tools/compare_results.py

Input:
    data/bench_lancedb.json
    data/bench_cpp.json  (output of C++ benchmark)

Output:
    data/benchmark_report.md
"""

import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  SKIP: {path.name} (not found)")
        return None
    with open(path) as f:
        return json.load(f)


def format_throughput(v):
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:.0f}"


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"

    print("Loading benchmark results...")
    files = {
        "LanceDB": data_dir / "bench_lancedb.json",
        "C++ Engines": data_dir / "bench_cpp.json",
    }

    results = {}
    for name, path in files.items():
        data = load_json(path)
        if data:
            # C++ benchmark file contains multiple engines
            if isinstance(data, list):
                for engine_data in data:
                    results[engine_data["engine"]] = engine_data
            elif "engines" in data:
                for engine_data in data["engines"]:
                    results[engine_data["engine"]] = engine_data
            else:
                results[data.get("engine", name)] = data

    if not results:
        print("No benchmark results found. Run the benchmarks first.")
        sys.exit(1)

    # Build markdown report
    engines = list(results.keys())
    lines = [
        "# Vector Database Benchmark Report",
        "",
        f"**Engines tested:** {', '.join(engines)}",
        f"**Dataset:** 100K farmer records, 384 dimensions, cosine similarity",
        "",
        "## Summary Table",
        "",
        "| Metric | " + " | ".join(engines) + " |",
        "| --- | " + " | ".join(["---"] * len(engines)) + " |",
    ]

    # Row helper
    def row(label, key_fn):
        vals = []
        for e in engines:
            try:
                vals.append(str(key_fn(results[e])))
            except (KeyError, TypeError):
                vals.append("—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    row("Insert throughput (vec/s)", lambda d: format_throughput(d.get("insertThroughput", 0)))
    row("Insert time (s)", lambda d: f"{d.get('insertTimeSec', '—')}")
    row("Search p50 (ms)", lambda d: f"{d['searchLatencyMs']['p50']:.3f}")
    row("Search p95 (ms)", lambda d: f"{d['searchLatencyMs']['p95']:.3f}")
    row("Search p99 (ms)", lambda d: f"{d['searchLatencyMs']['p99']:.3f}")
    row("Search mean (ms)", lambda d: f"{d['searchLatencyMs']['mean']:.3f}")
    row("Recall@10", lambda d: f"{d.get('recallAt10', '—')}")
    row("Disk usage (MB)", lambda d: f"{d.get('diskUsageMB', '—')}")
    row("Memory delta (MB)", lambda d: f"{d.get('memoryDeltaMB', '—')}")
    row("Cold start (ms)", lambda d: f"{d.get('coldStartMs', '—')}")

    lines.extend([
        "",
        "## Notes",
        "",
        "- **Flat Search**: Brute-force baseline (ground truth for recall calculation)",
        "- **USearch (raw)**: HNSW index only, no metadata/SQLite overhead",
        "- **EdgeVectorStore**: Full stack — USearch HNSW + SQLite metadata + tiered hot/cold + journal",
        "- **LanceDB**: IVF_PQ index via Python binding (Node.js/Rust backend)",
        "- All search latencies measured over 1000 queries, single-threaded",
        "- Recall computed against brute-force ground truth",
        "",
    ])

    report = "\n".join(lines)

    out_path = data_dir / "benchmark_report.md"
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\n✓ Report saved to {out_path}")


if __name__ == "__main__":
    main()

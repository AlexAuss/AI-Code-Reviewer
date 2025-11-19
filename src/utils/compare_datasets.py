"""Compare schema and samples across three JSONL datasets.

This script reads three JSONL files, summarizes their keys, counts, and
writes a human-readable comparison report to an output text file.

Usage:
  python src/utils/compare_datasets.py \
    --code_ref Datasets/Code_Refinement/ref-train.jsonl \
    --comment_gen Datasets/Comment_Generation/msg-train.jsonl \
    --quality Datasets/Diff_Quality_Estimation/cls-train-chunk-0.jsonl \
    --out report.txt --sample 5

The script streams files (memory-efficient). By default it collects the
first `sample` JSON objects from each file to include in the report and
computes key frequency statistics for the whole file.
"""
from pathlib import Path
import json
from collections import Counter
import argparse
from typing import Dict, Any, List


def analyze_file(path: Path, sample_limit: int = 5) -> Dict[str, Any]:
    """Stream a JSONL file and collect schema info and samples.

    Returns a dict with:
      - total_lines: int
      - key_counts: Counter of keys occurrence across records
      - samples: list of first `sample_limit` parsed JSON objects
      - sample_keys: list of key sets for first samples
    """
    key_counts = Counter()
    samples: List[Dict[str, Any]] = []
    sample_keys: List[List[str]] = []
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                # skip malformed lines but count them
                continue

            # update key counts
            for k in obj.keys():
                key_counts[k] += 1

            # save sample
            if len(samples) < sample_limit:
                samples.append(obj)
                sample_keys.append(list(obj.keys()))

    return {
        "total_lines": total,
        "key_counts": key_counts,
        "samples": samples,
        "sample_keys": sample_keys,
    }


def write_report(out_path: Path, analyses: Dict[str, Dict[str, Any]]):
    with out_path.open("w", encoding="utf-8") as out:
        out.write("Dataset comparison report\n")
        out.write("========================\n\n")

        # Summary per dataset
        for name, info in analyses.items():
            out.write(f"Dataset: {name}\n")
            out.write(f"Path: {info['path']}\n")
            out.write(f"Total lines (records): {info['analysis']['total_lines']}\n")
            out.write("Top keys and frequencies:\n")
            for k, v in info['analysis']['key_counts'].most_common(20):
                out.write(f"  {k}: {v}\n")
            out.write("\nSamples (first records):\n")
            for i, sample in enumerate(info['analysis']['samples']):
                out.write(f"--- Sample {i+1} ---\n")
                # pretty print json with indentation
                out.write(json.dumps(sample, ensure_ascii=False, indent=2))
                out.write("\n\n")
            out.write("\n\n")

        # Cross-dataset key comparison
        out.write("Cross-dataset key summary\n")
        out.write("-------------------------\n")
        # collect all keys
        all_keys = Counter()
        dataset_keys = {}
        for name, info in analyses.items():
            ks = set(info['analysis']['key_counts'].keys())
            dataset_keys[name] = ks
            for k in ks:
                all_keys[k] += 1

        out.write("Keys present in how many datasets (key: count):\n")
        for k, v in all_keys.most_common():
            out.write(f"  {k}: {v}\n")

        out.write("\nKeys unique to each dataset:\n")
        for name, ks in dataset_keys.items():
            unique = ks - set().union(*(s for n, s in dataset_keys.items() if n != name))
            out.write(f"  {name} unique keys ({len(unique)}): {sorted(list(unique))}\n")

    print(f"Report written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare three JSONL datasets and write a human readable report.")
    parser.add_argument("--code_ref", required=True, help="Path to Code_Refinement JSONL (e.g., ref-train.jsonl)")
    parser.add_argument("--comment_gen", required=True, help="Path to Comment_Generation JSONL (e.g., msg-train.jsonl)")
    parser.add_argument("--quality", required=True, help="Path to Diff_Quality_Estimation JSONL (e.g., cls-train-chunk-0.jsonl)")
    parser.add_argument("--out", required=True, help="Output report text file")
    parser.add_argument("--sample", type=int, default=5, help="Number of sample records to include per file")

    args = parser.parse_args()

    paths = {
        "Code_Refinement": Path(args.code_ref),
        "Comment_Generation": Path(args.comment_gen),
        "Diff_Quality_Estimation": Path(args.quality),
    }

    analyses = {}
    for name, p in paths.items():
        if not p.exists():
            print(f"File not found: {p}")
            return
        analysis = analyze_file(p, sample_limit=args.sample)
        analyses[name] = {"path": str(p), "analysis": analysis}

    write_report(Path(args.out), analyses)


if __name__ == "__main__":
    main()

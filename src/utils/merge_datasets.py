"""Merge multiple JSONL code-review datasets into a single deduplicated JSONL.

This script streams input files (or directories) and stores merged records in a
SQLite database keyed by the SHA256 of a normalized `diff` string. Using SQLite
lets us merge fields from different datasets without keeping everything in RAM.

Usage examples:
  # Merge specific files and write merged.jsonl (default DB: merged.db)
  python src/utils/merge_datasets.py \
    --inputs Datasets/Comment_Generation/msg-train.jsonl Datasets/Code_Refinement/ref-train.jsonl Datasets/Diff_Quality_Estimation/cls-train-chunk-0.jsonl \
    --output merged.jsonl

  # Merge all jsonl files under a directory (recursively), with progress
  python src/utils/merge_datasets.py --inputs Datasets --output merged.jsonl

Notes:
 - The script expects each line in inputs to be a JSON object containing a
   textual code diff (commonly the key is `diff`, but other field names are
   handled heuristically). Merged record fields will include: `diff`,
   `comment`, `review`, `label`, `sources` and `meta`.
 - The SQLite DB used for merging is persisted (default: `merged.db`) so you
   can resume or inspect the intermediate state.
"""
import argparse
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Iterable, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merge_datasets")


def find_jsonl_files(paths: Iterable[Path]) -> List[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        elif p.is_file():
            files.append(p)
    return files


def normalize_diff(diff: str) -> str:
    # Basic normalization: strip leading/trailing whitespace and collapse
    # multiple spaces to single. You can extend this for language-specific
    # normalization (remove timestamps, repo-specific paths, etc.).
    if diff is None:
        return ""
    s = diff.strip()
    # normalize newlines and trailing spaces on each line
    lines = [line.rstrip() for line in s.splitlines()]
    # collapse multiple blank lines to a single blank line
    out_lines = []
    prev_blank = False
    for L in lines:
        is_blank = len(L) == 0
        if is_blank and prev_blank:
            continue
        out_lines.append(L)
        prev_blank = is_blank
    return "\n".join(out_lines)


def diff_hash(diff: str) -> str:
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def init_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS merged (
            diff_hash TEXT PRIMARY KEY,
            diff TEXT,
            comment TEXT,
            review TEXT,
            label TEXT,
            sources TEXT,
            meta TEXT
        )
        """
    )
    conn.commit()
    return conn


def upsert_record(conn: sqlite3.Connection, key: str, record: Dict[str, Any]):
    cur = conn.cursor()
    # Fetch existing
    cur.execute("SELECT diff, comment, review, label, sources, meta FROM merged WHERE diff_hash = ?", (key,))
    row = cur.fetchone()

    # Return True if a new row was inserted (useful for preview mode)
    if row is None:
        # Insert new
        cur.execute(
            "INSERT INTO merged(diff_hash, diff, comment, review, label, sources, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                record.get("diff"),
                record.get("comment"),
                record.get("review"),
                record.get("label"),
                json.dumps(record.get("sources", [])),
                json.dumps(record.get("meta", {})),
            ),
        )
        conn.commit()
        return True
    else:
        # Merge: keep whichever field is non-empty, append sources and merge meta
        existing = {
            "diff": row[0],
            "comment": row[1],
            "review": row[2],
            "label": row[3],
            "sources": json.loads(row[4]) if row[4] else [],
            "meta": json.loads(row[5]) if row[5] else {},
        }

        merged_comment = record.get("comment") or existing.get("comment")
        merged_review = record.get("review") or existing.get("review")
        merged_label = record.get("label") or existing.get("label")
        merged_sources = list(dict.fromkeys(existing.get("sources", []) + record.get("sources", [])))
        merged_meta = {**existing.get("meta", {}), **record.get("meta", {})}

        cur.execute(
            "UPDATE merged SET comment = ?, review = ?, label = ?, sources = ?, meta = ? WHERE diff_hash = ?",
            (
                merged_comment,
                merged_review,
                merged_label,
                json.dumps(merged_sources),
                json.dumps(merged_meta),
                key,
            ),
        )

    conn.commit()
    return False


def extract_fields_from_json(item: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    # Heuristic extraction: try common field names
    # Prioritize known keys across your datasets (from data_comparison_report)
    diff = (
        item.get("hunk")
        or item.get("old_hunk")
        or item.get("patch")
        or item.get("diff")
        or item.get("changes")
        or item.get("code_diff")
        or ""
    )

    # comment / message fields (human written review / msg)
    comment = (
        item.get("comment")
        or item.get("msg")
        or item.get("message")
        or item.get("review_comment")
        or item.get("msg_text")
        or ""
    )

    # refined code suggestion or new content
    review = item.get("new") or item.get("after") or item.get("refined_code") or item.get("patch_after") or ""

    # old/new code snippets
    old_code = item.get("old") or item.get("before") or ""
    new_code = item.get("new") or review or ""

    # quality label normalization (many files use `y`)
    label = None
    if "y" in item:
        label = item.get("y")
    else:
        label = item.get("label") or item.get("quality") or item.get("score")

    # language / repo / id metadata
    lang = item.get("lang") or item.get("language") or None
    repo = item.get("repo") or item.get("proj") or item.get("project") or None
    rec_id = item.get("id") or item.get("ghid") or item.get("idx") or None

    # meta: preserve everything that's not part of the canonical fields
    reserved = {
        "hunk",
        "old_hunk",
        "patch",
        "diff",
        "changes",
        "code_diff",
        "comment",
        "msg",
        "message",
        "review_comment",
        "review",
        "after",
        "refined_code",
        "label",
        "quality",
        "score",
        "new",
        "old",
        "before",
        "id",
        "ghid",
        "idx",
        "proj",
        "project",
        "lang",
        "language",
        "oldf",
        "oldfpath",
    }

    meta = {k: v for k, v in item.items() if k not in reserved}

    return {
        "id": rec_id,
        "diff": diff,
        "old_code": old_code,
        "new_code": new_code,
        "comment": comment,
        "review": review,
        "label": label,
        "lang": lang,
        "repo": repo,
        "file_text": item.get("oldf") or item.get("old_file") or None,
        "sources": [source_name],
        "meta": meta,
    }


def merge_files(file_paths: List[Path], db_path: Path, preview: int = 0):
    conn = init_db(db_path)
    processed = 0
    inserted_new = 0
    for file in file_paths:
        logger.info(f"Processing {file}")
        source_name = file.name
        with file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception as e:
                    logger.debug(f"Skipping invalid JSON line in {file}: {e}")
                    continue

                rec = extract_fields_from_json(item, source_name)
                norm = normalize_diff(rec["diff"])
                key = diff_hash(norm)
                rec["diff"] = norm
                was_new = upsert_record(conn, key, rec)
                if was_new:
                    inserted_new += 1
                processed += 1

                # If preview requested, stop when we've gathered that many unique merged records
                if preview and inserted_new >= preview:
                    logger.info(f"Preview target reached: {inserted_new} unique records inserted")
                    conn.close()
                    return

                if processed % 10000 == 0:
                    logger.info(f"Processed {processed} records so far (unique inserted: {inserted_new})")

    logger.info(f"Done processing. Total records processed: {processed} (unique inserted: {inserted_new})")
    conn.close()


def export_merged(db_path: Path, out_path: Path, limit: int = 0):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    query = "SELECT diff, comment, review, label, sources, meta FROM merged"
    if limit and isinstance(limit, int) and limit > 0:
        query = query + f" LIMIT {limit}"

    cur.execute(query)
    with out_path.open("w") as out_f:
        for r in cur.fetchall():
            out_obj = {
                "diff": r[0],
                "comment": r[1],
                "review": r[2],
                "label": r[3],
                "sources": json.loads(r[4]) if r[4] else [],
                "meta": json.loads(r[5]) if r[5] else {},
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    conn.close()
    logger.info(f"Exported merged dataset to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL code-review datasets into one deduplicated JSONL using a SQLite DB")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input files or directories (space separated)")
    parser.add_argument("--output", required=True, help="Output merged JSONL file")
    parser.add_argument("--db", default="merged.db", help="Path to sqlite DB used during merging (default: merged.db)")
    parser.add_argument("--preview", type=int, default=0, help="Preview mode: stop after inserting N unique merged records and export only those N records (useful for quick inspection)")

    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    files = find_jsonl_files(paths)
    if not files:
        logger.error("No input JSONL files found. Provide files or directories containing .jsonl files")
        return

    db_path = Path(args.db)
    out_path = Path(args.output)

    merge_files(files, db_path, preview=args.preview)
    export_merged(db_path, out_path, limit=args.preview)


if __name__ == "__main__":
    main()

import pandas as pd
import re
import os

NUM_ROWS = 100000  # optional limit for testing
CHUNK_SIZE = 5000  # number of rows to process at a time (tune for memory)
CODE_REF_TRAIN_PATH = "Datasets/Code_Refinement/ref-train.jsonl"
COMMENT_GEN_TRAIN_PATH = "Datasets/Comment_Generation/msg-train.jsonl"
COMBINED_OUTPUT_PATH = "Datasets/Unified_Dataset/train.jsonl"

def infer_language_from_code(code: str) -> str | None:
    if not isinstance(code, str) or not code.strip():
        return None

    if re.search(r'\bdef\s+\w+\s*\(', code) or re.search(r'\bimport\s+\w+', code) or ("print(" in code and "{" not in code):
        return "python"
    if re.search(r'\bfunction\s+\w+\s*\(', code) or "console.log(" in code or re.search(r'=>\s*{', code):
        return "javascript"
    if re.search(r'\bpublic\s+(class|static|void|int|String)\b', code) or "System.out.println" in code:
        return "java"
    if re.search(r'#include\s*<\w+>', code) or re.search(r'\bint\s+main\s*\(', code) or "std::" in code:
        return "cpp"
    if re.search(r'using\s+System;', code) or re.search(r'\bnamespace\s+\w+', code):
        return "csharp"
    if re.search(r'\bfunc\s+\w+\s*\(', code):
        return "go"
    if re.search(r'<\?php', code):
        return "php"
    if re.search(r'\bdef\s+self\.', code) or ("end" in code and "do" in code):
        return "ruby"
    if re.search(r'\bprintf\s*\(', code):
        return "c"
    return None


def stream_and_merge_jsonl(ref_path, gen_path, output_path, chunksize=1000, max_rows=None):
    """
    Stream two JSONL files, apply transformations and language inference, 
    and write combined dataset to a JSONL file line by line.
    """
    # Ensure output file is empty
    if os.path.exists(output_path):
        os.remove(output_path)

    total_written = 0

    # --- Process Code_Refinement ---
    for chunk in pd.read_json(ref_path, lines=True, chunksize=chunksize):
        if max_rows is not None and total_written >= max_rows:
            break

        # Transform
        chunk = chunk.rename(columns={
            'oldf': 'original_file',
            'old_hunk': 'original_patch',
            'hunk': 'refined_patch',
            'comment': 'review_comment',
            'lang': 'language'
        })
        chunk['quality_label'] = 0
        chunk['source_dataset'] = 'Code_Refinement'
        chunk = chunk[['original_file', 'language', 'original_patch',
                       'refined_patch', 'review_comment', 'quality_label', 'source_dataset']]

        # Write to JSONL
        chunk.to_json(output_path, orient='records', lines=True, force_ascii=False,
                      mode='a', index=False)
        total_written += len(chunk)
        if max_rows is not None:
            total_written = min(total_written, max_rows)

    # --- Process Comment_Generation ---
    for chunk in pd.read_json(gen_path, lines=True, chunksize=chunksize):
        if max_rows is not None and total_written >= max_rows:
            break

        # Transform
        chunk = chunk.rename(columns={
            'oldf': 'original_file',
            'patch': 'original_patch',
            'msg': 'review_comment',
            'y': 'quality_label'
        })
        chunk['refined_patch'] = None
        chunk['language'] = None
        chunk['source_dataset'] = 'Comment_Generation'
        chunk = chunk[['original_file', 'language', 'original_patch',
                       'refined_patch', 'review_comment', 'quality_label', 'source_dataset']]

        # Language inference
        mask = chunk['language'].isna()
        chunk.loc[mask, 'language'] = chunk.loc[mask, 'original_file'].apply(infer_language_from_code)

        # Write to JSONL
        chunk.to_json(output_path, orient='records', lines=True, force_ascii=False,
                      mode='a', index=False)
        total_written += len(chunk)
        if max_rows is not None:
            total_written = min(total_written, max_rows)

    print(f"✅ Combined dataset written to {output_path}, total rows: {total_written}")


if __name__ == "__main__":
    stream_and_merge_jsonl(
        CODE_REF_TRAIN_PATH,
        COMMENT_GEN_TRAIN_PATH,
        COMBINED_OUTPUT_PATH,
        CHUNK_SIZE,  # tune for memory
        # max_rows=NUM_ROWS
    )
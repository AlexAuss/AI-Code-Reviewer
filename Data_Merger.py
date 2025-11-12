import pandas as pd
import os
import time
import hashlib

# Limits the number of rows to read for testing purposes
NUM_ROWS = 1000

def load_or_cache_jsonl(path, num_rows, cache_path=None, use_parquet=True):
    """
    Load a .jsonl file and cache it for faster future loading.
    """
    if cache_path is None:
        ext = "parquet" if use_parquet else "pkl"
        cache_path = os.path.splitext(path)[0] + f".{ext}"
    
    # If cached file exists, load that
    if os.path.exists(cache_path):
        print(f"🔄 Loading cached data from {cache_path} ...")
        start = time.time()
        df = (pd.read_parquet(cache_path) if use_parquet 
              else pd.read_pickle(cache_path))
        print(f"✅ Loaded cache in {time.time() - start:.2f}s")
        return df
    
    # Otherwise, read the JSONL and cache it
    print(f"📂 Reading JSONL from {path} ...")
    start = time.time()
    df = pd.read_json(path, lines=True, encoding='utf-8', nrows=num_rows)
    print(f"✅ Loaded JSONL in {time.time() - start:.2f}s, caching...")
    
    # Cache for next time
    if use_parquet:
        df.to_parquet(cache_path, index=False)
    else:
        df.to_pickle(cache_path)
    print(f"💾 Cached to {cache_path}")
    
    return df



if __name__ == "__main__":
    ref_train_df = load_or_cache_jsonl("data/Code_Refinement/ref-train.jsonl", NUM_ROWS)
    gen_train_df = load_or_cache_jsonl("data/Comment_Generation/msg-train.jsonl", NUM_ROWS)

    # Assume your two DataFrames are named:
    # df_refine  -> code_refinement dataframe
    # df_gen     -> comment_generation dataframe

    # --- Transform code_refinement dataframe ---
    refine_renamed = ref_train_df.rename(columns={
        'oldf': 'original_file',
        'old_hunk': 'original_patch',
        'hunk': 'refined_patch',
        'comment': 'review_comment',
        'lang': 'language'
    })

    refine_renamed['quality_label'] = 0
    refine_renamed['source_dataset'] = 'Code_Refinement'

    refine_final = refine_renamed[[
        'original_file', 'language', 'original_patch',
        'refined_patch', 'review_comment',
        'quality_label', 'source_dataset'
    ]]


    # --- Transform comment_generation dataframe ---
    gen_renamed = gen_train_df.rename(columns={
        'oldf': 'original_file',
        'patch': 'original_patch',
        'msg': 'review_comment',
        'y': 'quality_label'
    })

    gen_renamed['refined_patch'] = None
    gen_renamed['language'] = None  # or fill from metadata if available
    gen_renamed['source_dataset'] = 'Comment_Generation'

    gen_final = gen_renamed[[
        'original_file', 'language', 'original_patch',
        'refined_patch', 'review_comment',
        'quality_label', 'source_dataset'
    ]]


    # --- Combine both datasets ---
    unified_df = pd.concat([refine_final, gen_final], ignore_index=True)

    # --- Optional: shuffle or reset index ---
    unified_df = unified_df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = "data/combined_dataset.jsonl"
    unified_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"✅ Combined dataset written to {output_path}")
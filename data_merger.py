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

def hash_value(val):
    """Return MD5 hash of a single value (used for key matching)."""
    if pd.isna(val):
        return None
    return hashlib.md5(str(val).encode()).hexdigest()


def categorize_row(row):
    """Categorize a row based on annotation completeness."""
    has_comment = 'comment' in row and pd.notna(row['comment'])
    has_msg = 'msg' in row and pd.notna(row['msg'])
    has_y = 'y' in row and pd.notna(row['y'])
    
    # Fully annotated: text + label
    if (has_comment or has_msg) and has_y:
        return 'fully_annotated'
    # Partially annotated: only text
    elif has_comment or has_msg:
        return 'partially_annotated'
    # Only labeled: only y
    elif has_y:
        return 'only_labeled'
    else:
        return 'unlabeled'



if __name__ == "__main__":
    ref_train_df = load_or_cache_jsonl("data/Code_Refinement/ref-train.jsonl", NUM_ROWS)
    gen_train_df = load_or_cache_jsonl("data/Comment_Generation/msg-train.jsonl", NUM_ROWS)
    diff_train_chunk_0_df = load_or_cache_jsonl("data/Diff_Quality_Estimation/cls-train-chunk-0.jsonl", NUM_ROWS)
    diff_train_chunk_1_df = load_or_cache_jsonl("data/Diff_Quality_Estimation/cls-train-chunk-1.jsonl", NUM_ROWS)
    diff_train_chunk_2_df = load_or_cache_jsonl("data/Diff_Quality_Estimation/cls-train-chunk-2.jsonl", NUM_ROWS)
    diff_train_chunk_3_df = load_or_cache_jsonl("data/Diff_Quality_Estimation/cls-train-chunk-3.jsonl", NUM_ROWS)

    # List of all your datasets
    dfs = [
        ref_train_df, gen_train_df, diff_train_chunk_0_df, diff_train_chunk_1_df,
        diff_train_chunk_2_df, diff_train_chunk_3_df
    ]

    # Step 1: Choose the column common to all dataframes (e.g., 'id' or 'oldf')
    common_col = 'oldf'  # change if needed

    # Step 2: Add MD5 hashes of that column to each dataframe
    for i, df in enumerate(dfs):
        if common_col in df.columns:
            dfs[i] = df.copy()
            dfs[i]['md5_' + common_col] = df[common_col].apply(hash_value)
        else:
            print(f"⚠️ Warning: {common_col} not found in DataFrame {i}")

    # Step 3: Concatenate all datasets
    combined_df = pd.concat(dfs, ignore_index=True, join='outer')

    # Step 4: Group rows with the same hash together (these represent duplicates across datasets)
    grouped = combined_df.groupby(f'md5_{common_col}', dropna=True)

    # Step 5 (optional): For each group, merge non-null values into a single row
    merged_rows = []
    for _, group in grouped:
        merged_row = {}
        for col in combined_df.columns:
            if col in group:
                non_null_values = group[col].dropna()
                merged_row[col] = non_null_values.iloc[0] if len(non_null_values) > 0 else None
            else:
                merged_row[col] = None
        merged_rows.append(merged_row)

    # Step 6: Create the master merged dataset
    combined_df = pd.DataFrame(merged_rows).reset_index(drop=True)

    # Apply categorization to combined_df
    combined_df['annotation_category'] = combined_df.apply(categorize_row, axis=1)

    output_path = "data/combined_dataset.jsonl"
    combined_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"✅ Combined dataset written to {output_path}")
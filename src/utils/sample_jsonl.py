"""
Extract first N samples from a JSONL file and write to a new JSONL file.
Usage:
  python3 src/utils/sample_jsonl.py --input Datasets/Unified_Dataset/train.jsonl --output Datasets/Unified_Dataset/train_100.jsonl --num 100
"""
import argparse
import json

parser = argparse.ArgumentParser(description="Extract first N samples from a JSONL file.")
parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
parser.add_argument('--num', type=int, default=100, help='Number of samples to extract')
args = parser.parse_args()

count = 0
with open(args.input, 'r', encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
    for line in fin:
        if count >= args.num:
            break
        try:
            obj = json.loads(line)
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            count += 1
        except Exception:
            continue
print(f"Wrote {count} samples to {args.output}")

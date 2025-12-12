# ==============================================================
# ROUGE-L + BERTScore + pass@k + CodeBLEU Evaluation with TQDM
# ==============================================================

from rouge_score import rouge_scorer
from bert_score import BERTScorer
from codebleu import calc_codebleu  # pip install codebleu
import nltk

import os
import json
from tqdm import tqdm
import random
import ctypes
import logging as std_logging
std_logging.getLogger().setLevel(std_logging.ERROR)
import codebleu.utils
from tree_sitter import Language
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, logging
import torch
logging.set_verbosity_error()

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize scorers
rouge_scorer_inst = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scorer_bert = BERTScorer(lang="en", rescale_with_baseline=True)

# Local DeepSeek model setup with progress bar for download
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
print("Loading smaller DeepSeek model... (this may take time on first run)")
try:
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if device == "mps":
        # MPS often works best when explicitly loading to device, device_map="auto" can be flaky without accelerate support
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device if device != "cuda" and device != "cpu" else None)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load training samples
train_path = "Datasets/Unified_Dataset/train.jsonl"
train_samples = []
with open(train_path, "r", encoding="utf-8") as fin:
    for i, line in enumerate(fin):
        if i >= 3:  # Reduced to 3 for smaller prompts
            break
        train_samples.append(json.loads(line.strip()))

# Load test samples
test_path = "Datasets/Unified_Dataset/test_filtered.jsonl"
test_samples = []
with open(test_path, "r", encoding="utf-8") as fin:
    for i, line in enumerate(fin):
        if i >= 5000:  # Limit to first 10 for testing
            break
        test_samples.append(json.loads(line.strip()))

# ==========================================================
# Helper functions
# ==========================================================

def build_prompt(samples, user_query):
    samples_sorted = sorted(samples, key=lambda x: x["quality_label"])
    results = []
    results.append(
        "Act as a code reviewer. Use examples below to write a review comment for the User Query. Limit to 10 sentences. If refinement needed, provide refined code after. Start with 'Review Comment:' and 'Refined Patch:'.\n\n"
    )
    for idx, sample in enumerate(samples_sorted, start=1):
        block = []
        block.append(f"Result {idx}")
        block.append(f"Original Patch:\n{sample['original_patch']}")
        block.append(f"Review Comment:\n{sample['review_comment']}")
        if sample["quality_label"] == 0:
            block.append(f"Refined Patch:\n{sample['refined_patch']}")
        results.append("\n".join(block))
    prompt = "\n\n".join(results) + f"\n\nUser Query:\n\n{user_query}"
    return prompt

def extract_review_and_refined(generated_text):
    review_comment = ""
    refined_code = None
    if "Review Comment:" in generated_text:
        parts = generated_text.split("Review Comment:", 1)
        remainder = parts[1].strip()
        if "Refined Patch:" in remainder:
            review_comment_part, refined_code_part = remainder.split("Refined Patch:", 1)
            review_comment = review_comment_part.strip()
            refined_code = refined_code_part.strip()
        else:
            review_comment = remainder.strip()
    return review_comment, refined_code

def normalize_language(lang):
    """Normalize language names to match CodeBLEU supported languages."""
    if not lang:
        return "python"  # default
    lang = lang.lower().strip()
    if lang in ['.cs', 'csharp']:
        return 'c_sharp'
    elif lang == 'js':
        return 'javascript'
    elif lang == 'py':
        return 'python'
    elif lang == 'rb':
        return 'ruby'
    else:
        return lang


# ==========================================================
# Evaluation loop with ROUGE-L, BERTScore, and CodeBLEU
# ==========================================================

batch_size = 8  # Process in batches for GPU efficiency

rouge_l_scores = []
bertscore_f1_scores = []

codebleu_scores = []



# Monkeypatch codebleu to support tree-sitter 0.22+
def patched_get_tree_sitter_language(lang):
    capsule = None
    if lang == 'python':
        import tree_sitter_python
        capsule = tree_sitter_python.language()
    elif lang == 'java':
         import tree_sitter_java
         capsule = tree_sitter_java.language()
    elif lang == 'javascript':
         import tree_sitter_javascript
         capsule = tree_sitter_javascript.language()
    elif lang == 'ruby':
         import tree_sitter_ruby
         capsule = tree_sitter_ruby.language()
    elif lang == 'php':
         import tree_sitter_php
         capsule = tree_sitter_php.language_php()
    elif lang == 'go':
         import tree_sitter_go
         capsule = tree_sitter_go.language()
    elif lang == 'c_sharp':
         try:
            import tree_sitter_c_sharp
            capsule = tree_sitter_c_sharp.language()
         except ImportError:
            pass
    elif lang == 'cpp':
         import tree_sitter_cpp
         capsule = tree_sitter_cpp.language()
    elif lang == 'c':
         import tree_sitter_c
         capsule = tree_sitter_c.language()
    
    if capsule:
        if isinstance(capsule, int):
            return Language(capsule)
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"tree_sitter.Language")
        return Language(ptr)
    
    return None

codebleu.utils.get_tree_sitter_language = patched_get_tree_sitter_language
import codebleu.codebleu
import codebleu.syntax_match
import codebleu.dataflow_match
codebleu.codebleu.get_tree_sitter_language = patched_get_tree_sitter_language
codebleu.syntax_match.get_tree_sitter_language = patched_get_tree_sitter_language
codebleu.dataflow_match.get_tree_sitter_language = patched_get_tree_sitter_language


# Process in batches
for i in tqdm(range(0, len(test_samples), batch_size), desc="Processing Batches"):
    batch = test_samples[i:i + batch_size]
    batch_prompts = []
    batch_ground_reviews = []
    batch_ground_refined_codes = []
    batch_user_queries = []
    batch_langs = []

    for test_sample in batch:
        user_query = test_sample["original_patch"]
        ground_review = test_sample["review_comment"]
        ground_refined_code = test_sample.get("refined_patch")
        lang = normalize_language(test_sample.get("language"))

        prompt_string = build_prompt(train_samples, user_query)
        batch_prompts.append(prompt_string)
        batch_ground_reviews.append(ground_review)
        batch_ground_refined_codes.append(ground_refined_code)
        batch_user_queries.append(user_query)
        batch_langs.append(lang)

    # Batch generate review comments
    input_texts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in batch_prompts]
    generated_batch = generator(input_texts, max_new_tokens=500, do_sample=False)

    for idx_in_batch, (user_query, ground_review, ground_refined_code, lang, generated) in enumerate(zip(batch_user_queries, batch_ground_reviews, batch_ground_refined_codes, batch_langs, generated_batch)):
        global_idx = i + idx_in_batch + 1
        # print(f"\n--- Processing sample {global_idx}/{len(test_samples)} ---")
        # print(f"Prompt length: {len(batch_prompts[idx_in_batch])} characters")

        generated_review_full = generated[0]['generated_text'][len(input_texts[idx_in_batch]):].strip()
        generated_review_comment, generated_refined_code = extract_review_and_refined(generated_review_full)
        # print(f"Generated review comment length: {len(generated_review_comment)} characters")

        if not generated_review_comment.strip():
             print(f"Skipping sample {global_idx}: Empty generated review comment.")
             continue

        # ROUGE-L
        rouge_scores = rouge_scorer_inst.score(ground_review, generated_review_comment)
        rouge_l = rouge_scores['rougeL'].fmeasure
        rouge_l = rouge_scores['rougeL'].fmeasure

        # BERTScore
        P, R, F1 = scorer_bert.score([generated_review_comment], [ground_review])
        bertscore_f1 = F1.item()
        bertscore_f1 = F1.item()



        # CodeBLEU
        codebleu_score = 0.0
        codebleu_exception = False
        if ground_refined_code is not None and generated_refined_code is not None:
            try:
                codebleu_result = calc_codebleu([[ground_refined_code]], [generated_refined_code], lang=lang)
                codebleu_score = codebleu_result['codebleu']
            except Exception as e:
                print(f"CodeBLEU failed for sample {global_idx} (lang={lang}): {e}")
                codebleu_exception = True
        
        if not codebleu_exception:
            rouge_l_scores.append(rouge_l)
            bertscore_f1_scores.append(bertscore_f1)
            codebleu_scores.append(codebleu_score)

        # print(f"Sample {global_idx} completed. ROUGE-L: {rouge_l:.4f}, BERTScore: {bertscore_f1:.4f}, CodeBLEU: {codebleu_score:.4f}")

        # Update and write running averages to a file
        if rouge_l_scores:
            current_average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
            current_average_bertscore = sum(bertscore_f1_scores) / len(bertscore_f1_scores)

            if codebleu_scores:
                current_average_codebleu = sum(codebleu_scores) / len(codebleu_scores)
            else:
                current_average_codebleu = 0.0

            with open("running_averages_deepseek.txt", "w", encoding="utf-8") as f:
                f.write(f"Samples processed: {len(rouge_l_scores)}\n")
                f.write(f"Average ROUGE-L F1 Score: {current_average_rouge_l:.4f}\n")
                f.write(f"Average BERTScore F1: {current_average_bertscore:.4f}\n")

                f.write(f"Average CodeBLEU: {current_average_codebleu:.4f}\n")



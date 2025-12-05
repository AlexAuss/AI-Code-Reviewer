# ==============================================================
# ROUGE-L + BERTScore + pass@k + CodeBLEU Evaluation with TQDM
# ==============================================================

from rouge_score import rouge_scorer
from bert_score import BERTScorer
from codebleu import calc_codebleu  # pip install codebleu
import os
from openai import OpenAI
import json
from tqdm import tqdm
import random

# Initialize scorers
rouge_scorer_inst = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scorer_bert = BERTScorer(lang="en", rescale_with_baseline=True)

# HuggingFace Qwen client setup
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="API_KEY_PLACEHOLDER"  # Replace with your actual API key
)

# Load training samples
train_path = "Datasets/Unified_Dataset/train.jsonl"
train_samples = []
with open(train_path, "r", encoding="utf-8") as fin:
    for i, line in enumerate(fin):
        if i >= 5:
            break
        train_samples.append(json.loads(line.strip()))

# Load test samples
test_path = "Datasets/Unified_Dataset/test.jsonl"
test_samples = []
with open(test_path, "r", encoding="utf-8") as fin:
    for line in fin:
        test_samples.append(json.loads(line.strip()))

# ==========================================================
# Helper functions
# ==========================================================

def build_prompt(samples, user_query):
    samples_sorted = sorted(samples, key=lambda x: x["quality_label"])
    results = []
    results.append(
        "Directions:\n\nAct as a code reviewer. Using the results as examples below, write a "
        "review comment for the User Query and limit the response to "
        "10 sentences. If you determine it needs to be refined, "
        "provide the refined version of the code after the review comment. "
        "The review comment section must start with a header, \"Review Comment:\", on its "
        "own line and \"Refined Code:\" correspondingly.\n\n"
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
        if "Refined Code:" in remainder:
            review_comment_part, refined_code_part = remainder.split("Refined Code:", 1)
            review_comment = review_comment_part.strip()
            refined_code = refined_code_part.strip()
        else:
            review_comment = remainder.strip()
    return review_comment, refined_code

def generate_k_completions(prompt, k=5):
    """Generate k candidate completions for the same prompt"""
    candidates = []
    for _ in range(k):
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        )
        generated_full = completion.choices[0].message.content
        _, refined_code = extract_review_and_refined(generated_full)
        if refined_code is not None:
            candidates.append(refined_code.strip())
    return candidates

# ==========================================================
# Evaluation loop with pass@k and CodeBLEU (writing to file)
# ==========================================================

k = 5  # number of completions to consider for pass@k

rouge_l_scores = []
bertscore_f1_scores = []
pass_at_k_scores = []
codebleu_scores = []

for idx, test_sample in enumerate(tqdm(test_samples, desc="Processing test samples"), start=1):
    user_query = test_sample["original_patch"]
    ground_review = test_sample["review_comment"]
    ground_refined_code = test_sample.get("refined_patch")
    ground_lang = test_sample.get("language", "python")  # default python

    prompt_string = build_prompt(train_samples, user_query)

    # ---------------------- Review Comment metrics ----------------------
    completion = client.chat.completions.create(
        model="Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt_string}]}]
    )
    generated_review_full = completion.choices[0].message.content
    generated_review_comment, generated_refined_code = extract_review_and_refined(generated_review_full)

    # ROUGE-L
    rouge_scores = rouge_scorer_inst.score(ground_review, generated_review_comment)
    rouge_l = rouge_scores['rougeL'].fmeasure
    rouge_l_scores.append(rouge_l)

    # BERTScore
    P, R, F1 = scorer_bert.score([generated_review_comment], [ground_review])
    bertscore_f1 = F1.item()
    bertscore_f1_scores.append(bertscore_f1)

    # ---------------------- Refined Code metrics ----------------------
    pass_at_k = 0.0
    codebleu_score = 0.0

    if ground_refined_code is not None:
        candidates = generate_k_completions(prompt_string, k=k)

        # pass@k: if any candidate matches ground truth
        if any(cand.strip() == ground_refined_code.strip() for cand in candidates):
            pass_at_k = 1.0
        pass_at_k_scores.append(pass_at_k)

        # Placeholder for CodeBLEU (currently 0.0)
        codebleu_scores.append(codebleu_score)
    else:
        pass_at_k_scores.append(0)
        codebleu_scores.append(0.0)

    # ---------------------- Write evaluation to file ----------------------
    output_file = f"sample_eval_{idx}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("================ Sample Evaluation ================\n\n")
        f.write("Original Patch (User Query):\n")
        f.write(user_query + "\n\n")
        f.write("Ground Truth Review Comment:\n")
        f.write(ground_review + "\n\n")
        f.write("Generated Review Comment:\n")
        f.write(generated_review_comment + "\n\n")
        f.write("Ground Truth Refined Code:\n")
        f.write(str(ground_refined_code) + "\n\n")
        f.write("Generated Refined Code:\n")
        f.write(str(generated_refined_code) + "\n\n")
        f.write(f"ROUGE-L F1 Score: {rouge_l:.4f}\n")
        f.write(f"BERTScore F1: {bertscore_f1:.4f}\n")
        f.write(f"pass@{k}: {pass_at_k}\n")
        f.write(f"CodeBLEU: {codebleu_score:.4f}\n")
        f.write("\n==================================================\n")


# ==========================================================
# Average metrics
# ==========================================================
average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
average_bertscore = sum(bertscore_f1_scores) / len(bertscore_f1_scores)
average_pass_at_k = sum(pass_at_k_scores) / len(pass_at_k_scores)
average_codebleu = sum(codebleu_scores) / len(codebleu_scores)

print(f"\nAverage ROUGE-L F1 Score: {average_rouge_l:.4f}")
print(f"Average BERTScore F1: {average_bertscore:.4f}")
print(f"Average pass@{k}: {average_pass_at_k:.4f}")
print(f"Average CodeBLEU: {average_codebleu:.4f}")

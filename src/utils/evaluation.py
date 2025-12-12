# ==============================================================
# ROUGE-L + BERTScore + pass@k + CodeBLEU Evaluation with TQDM
# ==============================================================

from rouge_score import rouge_scorer
from bert_score import BERTScorer
from codebleu import calc_codebleu  # pip install codebleu
import os
import json
from tqdm import tqdm
import random
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize scorers
rouge_scorer_inst = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scorer_bert = BERTScorer(lang="en", rescale_with_baseline=True)

# Local DeepSeek model setup with progress bar for download
model_name = "deepseek-ai/DeepSeek-coder-1.3b-instruct"
print("Loading DeepSeek model... (this may take time on first run)")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load training samples
train_path = "Datasets/Unified_Dataset/train.jsonl"
train_samples = []
with open(train_path, "r", encoding="utf-8") as fin:
    for i, line in enumerate(fin):
        if i >= 5:  # Reduced to 3 for smaller prompts
            break
        train_samples.append(json.loads(line.strip()))

# Load test samples
test_path = "Datasets/Unified_Dataset/test.jsonl"
test_samples = []
with open(test_path, "r", encoding="utf-8") as fin:
    for i, line in enumerate(fin):
        # if i >= 10:  # Limit to first 10 for testing
        #     break
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
        "own line, as well as the \"Refined Patch:\" section. Do not include additional text before or after these sections.\n\n"
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

def generate_k_completions(prompt, k=5):
    """Generate k candidate completions for the same prompt"""
    candidates = []
    for _ in range(k):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generator(input_text, max_new_tokens=1000, do_sample=True)
        generated_full = generated[0]['generated_text'][len(input_text):].strip()
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
    messages = [{"role": "user", "content": prompt_string}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    generated = generator(input_text, max_new_tokens=1000, do_sample=False)
    generated_review_full = generated[0]['generated_text'][len(input_text):].strip()
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
        f.write("Original Patch (User Query):\n")
        f.write(user_query + "\n\n")
        f.write("Review Comment:\n")
        f.write(ground_review + "\n\n")
        f.write("Refined Patch:\n")
        f.write(str(ground_refined_code) + "\n\n")
        f.write("Generated Review Comment:\n")
        f.write(generated_review_comment + "\n\n")
        f.write("Generated Refined Patch:\n")
        f.write(str(generated_refined_code) + "\n")

    # Update and write running averages to a file
    if rouge_l_scores:
        current_average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
        current_average_bertscore = sum(bertscore_f1_scores) / len(bertscore_f1_scores)
        current_average_pass_at_k = sum(pass_at_k_scores) / len(pass_at_k_scores)
        current_average_codebleu = sum(codebleu_scores) / len(codebleu_scores)

        with open("running_averages.txt", "w", encoding="utf-8") as f:
            f.write(f"Samples processed: {len(rouge_l_scores)}\n")
            f.write(f"Average ROUGE-L F1 Score: {current_average_rouge_l:.4f}\n")
            f.write(f"Average BERTScore F1: {current_average_bertscore:.4f}\n")
            f.write(f"Average pass@{k}: {current_average_pass_at_k:.4f}\n")
            f.write(f"Average CodeBLEU: {current_average_codebleu:.4f}\n")

# ==============================================================
# Part 1 – Setup
# ==============================================================

import os
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Local DeepSeek model setup
model_name = "deepseek-ai/DeepSeek-coder-1.3b-instruct"
print("Loading DeepSeek model... (this may take time on first run)")

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
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device if device != "cuda" and device != "cpu" else None)

# ==============================================================
# Part 2 – Build prompt
# ==============================================================

def build_prompt(samples, user_query, output_file="prompt_output.txt"):
    samples_sorted = sorted(samples, key=lambda x: x["quality_label"])

    results = []
    results.append(
        "Directions:\n\nAct as a code reviewer. Using the results as examples below, write a "
        "review comment for the User Query and limit the response to "
        "10 sentences. If you determine it needs to be refined, "
        "provide the refined version of the code after the review comment. "
        "The review comment section must start with a header, \"Review Comment:\", on its "
        "own line, as well as the \"Refined Code:\" section. Do not include additional text before or after these sections.\n\n"
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
    
    # Save prompt to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    
    return prompt

def generate_qwen_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    generated = generator(input_text, max_new_tokens=500, do_sample=False)
    response = generated[0]['generated_text'][len(input_text):].strip()
    return response

# Load training samples
train_path = "Datasets/Unified_Dataset/train.jsonl"
samples = []
with open(train_path, "r", encoding="utf-8") as fin:
    for i, line in enumerate(fin):
        if i >= 5:  # Use 3 examples
            break
        samples.append(json.loads(line.strip()))

# Example usage
user_query = "print(Hello World"
prompt_string = build_prompt(samples, user_query)
response_text = generate_qwen_response(prompt_string)

with open("qwen_response.txt", "w", encoding="utf-8") as f:
    f.write(response_text)

print("DeepSeek response saved to qwen_response.txt")

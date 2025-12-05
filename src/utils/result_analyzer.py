# ==============================================================
# Part 1 – Setup
# ==============================================================

import os
from openai import OpenAI
import json

# ===================== STUDENT TODO START =====================
# Make sure you set your HF token as an environment variable
# export HF_TOKEN="YOUR_HF_API_KEY_HERE"
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="API_KEY_PLACEHOLDER"  # Replace with your actual API key
)
# ===================== STUDENT TODO END =====================

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
        "provide the refined verison of the code after the review comment. The review "
        "comment section must start with a header, \"Review Comment:\", on its "
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

    # Access the text properly using the .content attribute
    # completion.choices[0].message is a ChatCompletionMessage
    # Its content is a string
    return completion.choices[0].message.content

# Example usage
user_query = "print(Hello World)"
prompt_string = build_prompt(samples, user_query)
response_text = generate_qwen_response(prompt_string)

with open("qwen_response.txt", "w", encoding="utf-8") as f:
    f.write(response_text)

print("Qwen response saved to qwen_response.txt")

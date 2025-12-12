# ==============================================================
# Qwen LLM Integration Module
# ==============================================================
"""
Qwen Code Review LLM Module

Provides functions to:
- Load Qwen model
- Build prompts from retrieved examples
- Generate code review responses

Usage:
    from src.utils.llm_model import load_deepseek_model, generate_deepseek_response, build_review_prompt
    
    # Load model once
    tokenizer, generator = load_deepseek_model()
    
    # Generate review
    prompt = build_review_prompt(retrieved_examples, user_patch)
    review = generate_deepseek_response(prompt, tokenizer, generator)
"""

import os
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, logging as transformers_logging
import torch

# Suppress transformer warnings
transformers_logging.set_verbosity_error()

# Global variables for model (loaded once)
_tokenizer = None
_generator = None
_model = None
_device = None

def load_deepseek_model(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct"):
    """
    Load Qwen model (call once at startup).
    
    Args:
        model_name: HuggingFace model name (default: Qwen 1.5B Coder - optimized for M1 Pro)
        
    Returns:
        tuple: (tokenizer, generator) for generating responses
    """
    global _tokenizer, _generator, _model, _device
    
    if _tokenizer is not None and _generator is not None:
        print(f"Qwen model already loaded on {_device}")
        return _tokenizer, _generator
    
    print(f"Loading Qwen model ({model_name})... (this may take time on first run)")
    
    # Force CPU for M1 Pro to avoid MPS 4GB tensor limit
    # MPS has issues with models > 1B parameters due to tensor size limits
    if torch.cuda.is_available():
        _device = "cuda"
        print("Using NVIDIA CUDA GPU")
    else:
        _device = "cpu"
        print("Using CPU (M1 Pro CPU is still fast with optimizations)")
    
    print(f"Device selected: {_device}")
    
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        if _device == "cuda":
            # CUDA device with auto device mapping
            print("Loading model for CUDA GPU with float16...")
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto", 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            # CPU with optimizations for M1 Pro
            print("Loading model on CPU with optimizations for Apple Silicon...")
            # Use bfloat16 if available (better on M1), otherwise float32
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or hasattr(torch.backends, 'mps') else torch.float32
            print(f"Using dtype: {dtype}")
            
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            _model = _model.to(_device)
            
    except Exception as e:
        print(f"⚠️  Warning: Failed to load with optimized settings: {e}")
        print("Falling back to basic CPU loading with float32...")
        _device = "cpu"
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(_device)
    
    # Create pipeline with appropriate device setting
    device_param = 0 if _device == "cuda" else -1  # -1 for CPU
    
    _generator = pipeline(
        "text-generation", 
        model=_model, 
        tokenizer=_tokenizer, 
        device=device_param
    )
    
    print(f"✅ Qwen 1.5B model loaded successfully on {_device}")
    print(f"   Model parameters: ~1.5B (~3GB memory)")
    print(f"   Note: Running on CPU is optimized for M1 Pro performance cores")
    return _tokenizer, _generator

# ==============================================================
# Prompt Building and Response Generation
# ==============================================================

def build_review_prompt(retrieved_examples, user_patch, save_to_file=None):
    """
    Build prompt for code review from retrieved examples.
    
    Args:
        retrieved_examples: List of dicts with keys: patch, review_comment, refined_patch, quality_label
        user_patch: The code patch to review
        save_to_file: Optional path to save prompt for debugging
        
    Returns:
        str: Formatted prompt for LLM
    """
    # Sort examples by quality (best first)
    examples_sorted = sorted(
        retrieved_examples, 
        key=lambda x: x.get("quality_label", 1),
        reverse=False  # 0 = needs refinement, 1 = good quality
    )

    # Build the instruction part
    instruction = (
        "You are an expert code reviewer. Below are examples of code patches (showing changes made) with their review comments.\n\n"
        "IMPORTANT: You are reviewing the CHANGES (lines with + and - symbols), NOT fixing the old code.\n"
        "- Lines with '-' are REMOVED code (old/before)\n"
        "- Lines with '+' are ADDED code (new/after)\n"
        "- Your job is to review whether these changes are good or bad\n\n"
        "Your response MUST follow this exact format:\n\n"
        "Review Comment:\n"
        "[Evaluate the CHANGES: Are they improvements? Do they introduce bugs/security issues? "
        "Mention specific problems. Be concise, 2-3 sentences]\n\n"
        "Refined Patch:\n"
        "[ONLY if the NEW code (+ lines) has issues, provide COMPLETE fixed code that addresses ALL problems you mentioned. "
        "Use proper best practices (e.g., parameterized queries for SQL, proper error handling, etc.). "
        "If the changes are already good, OMIT this entire section]\n\n"
        "Do NOT include anything before 'Review Comment:' or after the refined patch.\n"
        "---\n\n"
    )
    
    # Format examples
    examples_text = []
    for idx, example in enumerate(examples_sorted, start=1):
        ex_parts = [f"EXAMPLE {idx}:"]
        
        # Get patch (support different field names)
        patch = example.get('original_patch') or example.get('patch', '')
        ex_parts.append(f"Code Patch:\n{patch}")
        
        # Review comment
        review = example.get('review_comment', '')
        ex_parts.append(f"\nReview Comment:\n{review}")
        
        # Refined patch if available and quality label is 0
        if example.get("quality_label", 1) == 0 and example.get('refined_patch'):
            ex_parts.append(f"\nRefined Patch:\n{example['refined_patch']}")
        
        examples_text.append('\n'.join(ex_parts))
    
    # Combine all parts
    prompt = (
        instruction +
        '\n\n'.join(examples_text) +
        f"\n\n{'='*80}\n\n"
        f"NEW PATCH TO REVIEW (- means removed code, + means added code):\n\n{user_patch}\n\n"
        f"YOUR TASK:\n"
        f"1. Review the CHANGES: Are they improvements or do they introduce issues?\n"
        f"2. If issues exist, provide a COMPLETE fix in 'Refined Patch' that solves ALL problems\n"
        f"3. Use actual best practices in your fixes (e.g., parameterized queries, proper validation)\n\n"
        f"YOUR RESPONSE (start with 'Review Comment:'):\n\n"
    )
    
    # Save prompt to file for debugging
    if save_to_file:
        with open(save_to_file, "w", encoding="utf-8") as f:
            f.write(prompt)
    
    return prompt

def generate_deepseek_response(prompt, tokenizer=None, generator=None, max_tokens=500):
    """
    Generate response from Qwen model.
    
    Args:
        prompt: The prompt string
        tokenizer: Tokenizer instance (if None, uses global)
        generator: Generator pipeline (if None, uses global)
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: Generated response text
    """
    global _tokenizer, _generator
    
    # Use global instances if not provided
    if tokenizer is None:
        tokenizer = _tokenizer
    if generator is None:
        generator = _generator
    
    if tokenizer is None or generator is None:
        raise ValueError("Model not loaded. Call load_deepseek_model() first.")
    
    # Log the prompt being sent to LLM
    print("\n" + "="*80)
    print("🤖 PROMPT SENT TO LLM")
    print("="*80)
    print(prompt)
    print("="*80)
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Max tokens to generate: {max_tokens}")
    print("="*80 + "\n")
    
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("⏳ Generating response from Qwen model...\n")
    
    generated = generator(input_text, max_new_tokens=max_tokens, do_sample=False)
    response = generated[0]['generated_text'][len(input_text):].strip()
    
    # Log the response
    print("\n" + "="*80)
    print("✅ RESPONSE FROM LLM")
    print("="*80)
    print(response)
    print("="*80 + "\n")
    
    return response

def extract_review_and_code(response_text):
    """
    Extract review comment and refined code from LLM response.
    Handles cases where model generates extra text.
    
    Args:
        response_text: Raw response from LLM
        
    Returns:
        tuple: (review_comment, refined_code)
    """
    review_comment = ""
    refined_code = None
    
    # Find the first occurrence of "Review Comment:"
    if "Review Comment:" in response_text:
        # Split on the first occurrence only
        parts = response_text.split("Review Comment:", 1)
        remainder = parts[1].strip()
        
        # Check for refined patch/code sections
        if "Refined Patch:" in remainder:
            split_marker = "Refined Patch:"
            review_part, code_part = remainder.split(split_marker, 1)
            review_comment = review_part.strip()
            
            # Extract only the code, stop at next "Review Comment:" if it appears
            if "Review Comment:" in code_part:
                refined_code = code_part.split("Review Comment:")[0].strip()
            else:
                refined_code = code_part.strip()
                
        elif "Refined Code:" in remainder:
            split_marker = "Refined Code:"
            review_part, code_part = remainder.split(split_marker, 1)
            review_comment = review_part.strip()
            
            # Extract only the code, stop at next "Review Comment:" if it appears
            if "Review Comment:" in code_part:
                refined_code = code_part.split("Review Comment:")[0].strip()
            else:
                refined_code = code_part.strip()
        else:
            # No refined code section, just review
            # Stop at any subsequent "Review Comment:" header
            if "\nReview Comment:" in remainder:
                review_comment = remainder.split("\nReview Comment:")[0].strip()
            else:
                review_comment = remainder.strip()
    else:
        # Fallback: no "Review Comment:" header found
        # Look for any code block or just use the text
        review_comment = response_text.strip()
    
    # Clean up: Remove any trailing headers or artifacts
    if review_comment:
        # Stop at common section markers that shouldn't be in review
        for marker in ["\n\nEXAMPLE", "\n\nNEW PATCH", "\n\nCode Patch:", "\n---"]:
            if marker in review_comment:
                review_comment = review_comment.split(marker)[0].strip()
    
    if refined_code:
        # Clean up refined code similarly
        for marker in ["\n\nEXAMPLE", "\n\nNEW PATCH", "\nReview Comment:", "\n---"]:
            if marker in refined_code:
                refined_code = refined_code.split(marker)[0].strip()
    
    return review_comment, refined_code


# ==============================================================
# Example/Test Usage (optional)
# ==============================================================

if __name__ == "__main__":
    # Example usage when running this file directly
    print("Testing Qwen LLM module...")
    
    # Load model
    tokenizer, generator = load_deepseek_model()
    
    # Load sample training examples
    train_path = "Datasets/Unified_Dataset/train.jsonl"
    samples = []
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                if i >= 3:  # Use 3 examples for testing
                    break
                samples.append(json.loads(line.strip()))
        
        # Test user query
        user_query = "print(Hello World"
        
        # Build prompt
        prompt_string = build_review_prompt(samples, user_query, save_to_file="prompt_output.txt")
        print(f"\nPrompt built ({len(prompt_string)} chars)")
        
        # Generate response
        print("\nGenerating response...")
        response_text = generate_deepseek_response(prompt_string, tokenizer, generator)
        
        # Extract parts
        review, refined = extract_review_and_code(response_text)
        
        # Save output
        with open("qwen_response.txt", "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("QWEN RESPONSE\n")
            f.write("=" * 80 + "\n\n")
            f.write("Review Comment:\n")
            f.write(review + "\n\n")
            if refined:
                f.write("Refined Code:\n")
                f.write(refined + "\n")
        
        print("✅ Qwen response saved to qwen_response.txt")
        print(f"\nReview preview: {review[:200]}...")
    else:
        print(f"⚠️  Training data not found at {train_path}")

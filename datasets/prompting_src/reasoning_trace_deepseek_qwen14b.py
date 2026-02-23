import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json
import argparse
from make_prompt import iter_prompts
from datetime import datetime
import sys
sys.stdout.flush()

# ---------------------------
# Reasoning Extractor
# ---------------------------
def extract_reasoning(text):
    """
    Extract the first reasoning span <Rstart> ... <Rend> 
    that appears after the </think> tag.
    """
    # 1. Find </think> tag
    think_end_idx = text.find("</think>")
    if think_end_idx == -1:
        return "[NONE]"

    # 2. Slice text after </think>
    post_think_text = text[think_end_idx + len("</think>"):]

    # 3. Search for the first <Rstart> ... <Rend> pair
    match = re.search(r"<Rstart>(.*?)<Rend>", post_think_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "[NONE]"

def is_reasoning_valid(reasoning_trace):
    """
    Validates a reasoning trace based on:
    1. Starts with 'R0:' followed by an alphabet/number.
    2. Has 'R1:' followed by an alphabet/number somewhere after R0.
    3. Last non-empty line contains 'Final Verdict:' followed by an alphabet/number.
    """
    lines = [line.strip() for line in reasoning_trace.strip().split("\n") if line.strip()]
    if not lines:
        return False

    # Check R0:
    if not re.match(r"^R0:\s*[A-Za-z0-9]", lines[0]):
        return False

    # Check R1: (anywhere after R0)
    if not any(re.match(r"^R1:\s*[A-Za-z0-9]", line) for line in lines[1:]):
        return False

    # Check Final Verdict in last line
    last_line = lines[-1]
    if not re.match(r"^Final Verdict:\s*[A-Za-z0-9]", last_line):
        return False

    return True

if __name__ == "__main__":
    # ---------------------------
    # Argument Parser
    # ---------------------------
    parser = argparse.ArgumentParser(description="Run reasoning extraction for claim dataset.")

    parser.add_argument(
        "--gpu_index",
        type=int,
        default=99,
        help="Index of the GPU to use (default: 0)"
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=2,
        help="Number of shots for prompt file (e.g., 2-shot-prompt.txt)"
    )

    args = parser.parse_args()

    # ---------------------------
    # Configuration
    # ---------------------------
    GPU_INDEX = args.gpu_index
    MODEL_PATH = "/home/models/DeepSeek-R1-Distill-Qwen-14B"
    print("using" , MODEL_PATH)
    N_SHOT = args.n_shot

    # ---------------------------
    # GPU Setup
    # ---------------------------
    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------
    # Load Model & Tokenizer Once
    # ---------------------------
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    model.eval()
    model.to(device)

    # ---------------------------
    # Prompting
    # ---------------------------
    output_dir = f"/home/ojas/scripts/datasets/RTs/RT_Deepseek_Qwen_14B_{N_SHOT}Shot"
    os.makedirs(output_dir, exist_ok=True)
    BATCH_SIZE = 100
    file_counter = 1
    results = []

    # Read target claim IDs from file
    with open("claim_ids_deepseek4shot.txt", "r") as f:
        target_claim_ids = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(target_claim_ids)} target claim IDs from file")
    print(target_claim_ids)


    for (prompt,claim_id) in iter_prompts(n_shot=N_SHOT): # from make_prompt.py
        # Skip if claim_id is not in the target list
        if claim_id not in target_claim_ids:
            print(f"Skipping claim {claim_id}")
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        num_tokens = inputs["input_ids"].shape[1]
        print(f"Prompt token count: {num_tokens}")

        # ---------------------------
        # Run Generation
        # ---------------------------
        print(f"\n--- Claim {claim_id} (using {MODEL_PATH}) --- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Prompt token count: {num_tokens}")

        reasoning_trace = "[NONE]"
        attempt = 1

        while reasoning_trace.startswith("[NONE]"):
            print(f"Attempt {attempt}...")
            with torch.no_grad():
                outputs = model.generate(
                **inputs,
                max_new_tokens=2000,
                temperature=0.6
            )

            # Decode only new tokens
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Try to extract reasoning
            reasoning_trace = extract_reasoning(response)
            
            # Final reasoning validation check
            if not is_reasoning_valid(reasoning_trace):
                reasoning_trace = "[NONE]"

            if attempt >= 5:
                print(f"!! [EXCEEDED 5 ATTEMPTS] !!")
                reasoning_trace = "[EXCEEDED 5 ATTEMPTS]"

            attempt += 1
 
        print(f"Extracted reasoning length: {len(reasoning_trace.split())} words")

        # Append result to list
        results.append({
            # "prompt": prompt,
            "shots": N_SHOT,
            "claim_id": claim_id,
            "reasoning_trace": reasoning_trace
        })

        # Save every BATCH_SIZE results
        if len(results) == BATCH_SIZE:
            output_json_path = os.path.join(output_dir, f"traces_1_{file_counter}.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(results)} entries to {output_json_path}")

            # Reset batch
            results = []
            file_counter += 1

    # Save any remaining results from batches at the end
    if results:
        output_json_path = os.path.join(output_dir, f"traces_1_{file_counter}.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} entries to {output_json_path}")


# python -u reasoning_trace_deepseek_qwen14b.py --gpu_index 2 --n_shot 4 >> LOGS_deepseek_qwen14b_4shot.txt

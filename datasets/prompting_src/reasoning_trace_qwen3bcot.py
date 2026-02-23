import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json
import argparse
from datetime import datetime
from make_prompt import iter_prompts
import sys
sys.stdout.flush()

# ---------------------------
# Reasoning Extractor
# ---------------------------
def extract_reasoning(text):
    """
    Extract the first valid reasoning trace from text, based on:
    1. No "..." placeholders (likely prompt echo)
    2. Must start with R0:
    3. Must contain 'Final Verdict:' somewhere
    4. Must meet a minimal length requirement
    """
    # Find all candidate <Rstart>...</Rend> blocks
    blocks = re.findall(r"<Rstart>(.*?)<Rend>", text, re.DOTALL)

    for block in blocks:
        cleaned = block.strip()

        # --- 1. Skip if contains "..." (prompt echo) ---
        if "..." in cleaned:
            continue

        # --- 2. Must start with R0: ---
        if not re.match(r"^\s*R0:", cleaned):
            continue

        # --- 3. Must contain Final Verdict: somewhere ---
        # This check ensures it's a complete reasoning trace, not partial thoughts
        if "Final Verdict:" not in cleaned:
            continue

        # --- 4. Must be reasonably long to avoid junk ---
        if len(cleaned) < 50:  # tweak threshold if needed
            continue

        return cleaned  # Return the first valid block
    
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
    MODEL_PATH = "/home/models/Qwen-2.5-3b-Evol-CoT"
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
    output_dir = f"/home/ojas/scripts/datasets/RTs/RT_Qwen_3B_CoT_{N_SHOT}Shot"
    os.makedirs(output_dir, exist_ok=True)
    BATCH_SIZE = 100
    file_counter = 1
    results = []

    # Read claim IDs from text file
    with open("/home/ojas/scripts/reasoning_codes/claim_ids_cot_1shot_short.txt", "r") as f:
        target_claim_ids = [line.strip() for line in f if line.strip()]
    
    print("count of todo ids" , len(target_claim_ids))

    print(f"Loaded {len(target_claim_ids)} target claim IDs from file")

    for (prompt, claim_id) in iter_prompts(n_shot=N_SHOT): # from make_prompt.py
        # Skip if claim_id is not in the target list
        if claim_id not in target_claim_ids:
            print(f"Skipping claim {claim_id} (not in target list)")
            continue
            
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        num_tokens = inputs["input_ids"].shape[1]

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
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.1,
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
            output_json_path = os.path.join(output_dir, f"traces_finalrunszz2_{file_counter}.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(results)} entries to {output_json_path}")

            # Reset batch
            results = []
            file_counter += 1

    # Save any remaining results from batches at the end
    if results:
        output_json_path = os.path.join(output_dir, f"traces_finalrunszz2_{file_counter}.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} entries to {output_json_path}")

    # # Save ALL results to JSON
    # output_json_path = f"FinalTest_DeepSeek-R1-Distill-Qwen-7B_{N_SHOT}shot.json"
    # with open(output_json_path, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)

    # print(f"Saved {len(results)} entries to {output_json_path}")

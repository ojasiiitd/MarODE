import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json
import argparse
from datetime import datetime
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

# ---------------------------
# Custom Prompt Generator
# ---------------------------
CLAIMS_DATASET = "/home/ojas/scripts/datasets/claims_dataset_1200.json"

def iter_prompts_filtered(
    target_claim_ids,
    json_path=CLAIMS_DATASET,
    base_prompt_dir="/home/ojas/scripts/prompts",
    n_shot=6
):
    """
    Generator that yields prompts only for specified claim_ids
    """
    # Select prompt file
    base_prompt_file = os.path.join(base_prompt_dir, f"{n_shot}-shot-prompt.txt")
    print(f"Using Dataset File: {CLAIMS_DATASET}")
    print(f"Using Prompt File: {base_prompt_file}")
    print(f"Filtering for {len(target_claim_ids)} target claim IDs")
    
    if not os.path.exists(base_prompt_file):
        raise FileNotFoundError(f"Base prompt file not found: {base_prompt_file}")
    
    # Load base prompt
    with open(base_prompt_file, "r", encoding="utf-8") as f:
        base_prompt_content = f.read().strip()

    # Load dataset
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter data for target claim_ids
    filtered_data = [entry for entry in data if entry.get("unique_claim_id") in target_claim_ids]
    print(f"Found {len(filtered_data)} matching entries in dataset")

    # Yield one prompt at a time
    for entry in filtered_data:
        claim_id = entry.get("unique_claim_id", "")
        claim = entry.get("claim", "")
        label = entry.get("mapped_verdict", "")
        evidence_text = entry.get("evidence_text", [])

        formatted_str = (
            f'Claim: "{claim}"\n'
            f'Label: "{label}"\n'
            f"Evidence Text: [\n"
            + ",\n".join(f'      "{et}"' for et in evidence_text)
            + "\n ]\n"
        )

        yield (
            base_prompt_content
            + "\n\n"
            + formatted_str
            + "\nPlease provide the reasoning traces. Ensure that the reasoning output is between <Rstart> and <Rend>."
        ), claim_id

def process_claim_ids(claim_ids_string, gpu_index=0, n_shot=2, model_path="ABCD"):
    """
    Process newline-separated claim IDs string
    """
    # Parse newline-separated claim IDs
    TARGET_CLAIM_IDS = [cid.strip() for cid in claim_ids_string.split('\n') if cid.strip()]
    print(f"Parsed {len(TARGET_CLAIM_IDS)} claim IDs from input string")
    
    # ---------------------------
    # GPU Setup
    # ---------------------------
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------
    # Load Model & Tokenizer Once
    # ---------------------------
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model.eval()
    model.to(device)

    # ---------------------------
    # Prompting with Filtered Claims
    # ---------------------------

    # CHANGE AS NEEED
    checkdir = "RT_GPT_OSS_20B_2Shot"
    output_dir = f"/home/ojas/scripts/datasets/RTs/{checkdir}" + "_FIXINGRUNS"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    file_counter = 1
    processed_count = 0
    
    print(f"Processing {len(TARGET_CLAIM_IDS)} target claim IDs: {TARGET_CLAIM_IDS}")
    
    for prompt, claim_id in iter_prompts_filtered(TARGET_CLAIM_IDS, n_shot=n_shot):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        num_tokens = inputs["input_ids"].shape[1]

        # ---------------------------
        # Run Generation
        # ---------------------------
        print(f"\n--- Claim {claim_id} (using {model_path}) --- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Prompt token count: {num_tokens}")

        reasoning_trace = "[NONE]"
        attempt = 1

        while reasoning_trace.startswith("[NONE]"):
            print(f"Attempt {attempt}...")
            with torch.no_grad():
                # CHANGE SAMPLING PARAMETERS AS NEEDED GPT OSS
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
                reasoning_trace = "[EXCEEDED 3 ATTEMPTS]"
                break
            
            attempt += 1

        print(f"Extracted reasoning length: {len(reasoning_trace.split())} words")

        # Append result to list
        results.append({
            "shots": n_shot,
            "claim_id": claim_id,
            "reasoning_trace": reasoning_trace
        })
        processed_count += 1

        # Save every 100 entries
        if processed_count % 1 == 0:
            output_filename = f"rt_traces_{file_counter}.json"
            output_json_path = os.path.join(output_dir, output_filename)
            
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(results)} entries to {output_json_path}")
            
            # Reset for next batch
            results = []
            file_counter += 1

    # Save any remaining results
    if results:
        output_filename = f"rt_traces_{file_counter}.json"
        output_json_path = os.path.join(output_dir, output_filename)
        
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved final {len(results)} entries to {output_json_path}")

    print(f"Processing complete! Processed {processed_count} total entries")
    return processed_count

if __name__ == "__main__":
    # ---------------------------
    # Argument Parser
    # ---------------------------
    parser = argparse.ArgumentParser(description="Run reasoning extraction for specific claim IDs.")

    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="Index of the GPU to use (default: 0)"
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=2,
        help="Number of shots for prompt file (e.g., 2-shot-prompt.txt)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        # CHANGE MODEL NAME AS NEEDED
        default="ABCD",
        help="Path to the model (default: ABCD)"
    )
    parser.add_argument(
        "--claim_ids_file",
        type=str,
        required=True,
        help="Path to file containing newline-separated claim IDs"
    )

    args = parser.parse_args()

    # Read claim IDs from file
    try:
        with open(args.claim_ids_file, 'r') as f:
            claim_ids_string = f.read()
    except FileNotFoundError:
        print(f"Error: Claim IDs file not found: {args.claim_ids_file}")
        sys.exit(1)

    # Process the claim IDs
    process_claim_ids(
        claim_ids_string=claim_ids_string,
        gpu_index=args.gpu_index,
        n_shot=args.n_shot,
        model_path=args.model_path
    )

# python only_claim_json.py --claim_ids_file claim_ids_new.txt --gpu_index 3 --n_shot 2 --model_path "/home/models/gpt-oss-20b"

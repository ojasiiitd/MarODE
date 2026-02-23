import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json
import argparse
from make_prompt import iter_prompts
import sys
sys.stdout.flush()

# ---------------------------
# Reasoning Extractor
# ---------------------------
def extract_reasoning(text):
    """
    Extract the reasoning between <Rstart> and <Rend> 
    that appears immediately after </think>.
    """
    # 1. Find where </think> ends
    think_end_idx = text.find("</think>")
    if think_end_idx == -1:
        return "[NONE]"

    # 2. Slice text after </think>
    post_think_text = text[think_end_idx + len("</think>"):]

    # 3. Search for <Rstart>...</Rend> only in post_think_text
    match = re.search(r"<Rstart>(.*?)<Rend>", post_think_text, re.DOTALL)
    return match.group(1).strip() if match else "[NONE]"

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
    MODEL_PATH = "/home/models/models_ojas/DeepSeek-R1-Distill-Qwen-7B"
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
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    model.eval()
    model.to(device)

    # ---------------------------
    # Prompting
    # ---------------------------
    output_dir = f"/home/ojas/scripts/datasets/RTs/RT_Deepseek_Qwen_7B_{N_SHOT}Shot"
    os.makedirs(output_dir, exist_ok=True)
    BATCH_SIZE = 100
    file_counter = 1
    results = []
    for (prompt,claim_id) in iter_prompts(n_shot=N_SHOT): # from make_prompt.py
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        num_tokens = inputs["input_ids"].shape[1]

        # ---------------------------
        # Run Generation
        # ---------------------------
        print(f"\n--- Claim {claim_id} (using {MODEL_PATH})---")
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
            output_json_path = os.path.join(output_dir, f"traces_{file_counter}.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(results)} entries to {output_json_path}")

            # Reset batch
            results = []
            file_counter += 1

    # Save any remaining results from batches at the end
    if results:
        output_json_path = os.path.join(output_dir, f"traces_{file_counter}.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} entries to {output_json_path}")

    # # Save ALL results to JSON
    # output_json_path = f"FinalTest_DeepSeek-R1-Distill-Qwen-7B_{N_SHOT}shot.json"
    # with open(output_json_path, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)

    # print(f"Saved {len(results)} entries to {output_json_path}")

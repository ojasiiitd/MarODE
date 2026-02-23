import json
from itertools import combinations
from transformers import pipeline
from tqdm import tqdm
import argparse
from typing import List
import re

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(description='Process reasoning traces with evidence flag')
parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file')
parser.add_argument('--output', '-o', required=True, help='Path to the output JSON file')
parser.add_argument('--evidence', '-e', type=lambda x: x.lower() == 'true', default=True, help='Whether to use evidence (true/false)')
parser.add_argument('--gpu', '-g', required=True, type=int, default=-10, help='GPU id to use (e.g. 0 for first GPU)')
args = parser.parse_args()
GPU_USE = args.gpu
EVIDENCE = args.evidence

# -------------------------------
# Load an NLI model with batching
# -------------------------------
nli = pipeline("text-classification", 
               model="/home/models/deberta-xlarge-mnli", 
               device=GPU_USE, 
               top_k=None,
               batch_size=16,  # Enable batch processing
               truncation=True,
               padding=True)

# -------------------------------
# Batch NLI helper function
# -------------------------------
def nli_relation_batch(premises, hypotheses):
    """Run NLI on multiple premise-hypothesis pairs in batch."""
    # Prepare inputs for batch processing
    inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    
    # Process in batch
    results = nli(inputs)
    
    relations = []
    for result in results:
        # Handle different return formats
        if isinstance(result, list) and len(result) > 0:
            # List of dictionaries
            if isinstance(result[0], dict) and 'label' in result[0]:
                scores = {r["label"].lower(): r["score"] for r in result}
                relations.append(max(scores, key=scores.get))
            # List of strings
            elif isinstance(result[0], str):
                relations.append(result[0].lower())
        # Single dictionary
        elif isinstance(result, dict) and 'label' in result:
            relations.append(result['label'].lower())
        # Single string
        elif isinstance(result, str):
            relations.append(result.lower())
        # Fallback
        else:
            relations.append("neutral")
    
    return relations

def normalize_reasoning_steps(reasoning_trace: str) -> List[str]:
    """
    Extracts and normalizes reasoning steps from a given string.

    This function is more robust as it handles multiple types of prefixes (R<num>:, sent<num>:, int<num>:)
    and removes them to return a clean list of reasoning step strings.
    """
    steps = []
    for line in reasoning_trace.splitlines():
        line = line.strip()
        if not line:
            continue

        # First remove R<number>: prefix
        line = re.sub(r'^\s*R\d+\s*:\s*', '', line)
        
        # Then remove sent<number>: or int<number>: prefixes
        line = re.sub(r'^\s*(sent\d+|int\d+)\s*:\s*', '', line)
        
        if line:  # Only add non-empty lines
            steps.append(line)
    return steps

# -------------------------------
# Evaluate one entry with batch optimization
# -------------------------------
def evaluate_entry(entry):
    claim = entry["claim"]
    reasoning_trace = entry.get("reasoning_trace", "") 
    # Use the robust normalization function to extract reasoning steps
    reasoning_steps = normalize_reasoning_steps(reasoning_trace)

    # --- Strong Global Coherence (SGC) ---
    # Batch process all step->claim relations
    if reasoning_steps:
        step_claim_relations = nli_relation_batch(reasoning_steps, [claim] * len(reasoning_steps))
        sgc = all(rel == "entailment" for rel in step_claim_relations)
        wgc = all(rel != "contradiction" for rel in step_claim_relations)
    else:
        sgc, wgc = False, False

    # --- Local Coherence (LC) ---
    if len(reasoning_steps) > 1:
        # Prepare all pairwise combinations for batch processing
        premises1, premises2 = [], []
        for e1, e2 in combinations(reasoning_steps, 2):
            premises1.append(e1)
            premises2.append(e2)
            premises1.append(e2)  # For bidirectional check
            premises2.append(e1)
        
        # Batch process all pairwise relations
        if premises1:
            pairwise_relations = nli_relation_batch(premises1, premises2)
            # Check that no pair has contradiction in either direction
            lc = all(rel != "contradiction" for rel in pairwise_relations)
        else:
            lc = True
    else:
        lc = True  # Single step is always locally coherent

    return int(sgc), int(wgc), int(lc)

# -------------------------------
# Evaluate entire dataset and save with scores
# -------------------------------
def evaluate_and_save_dataset(input_json_path, output_json_path):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    sgc_scores, wgc_scores, lc_scores = [], [], []
    
    # Evaluate each entry with progress bar
    for i, entry in enumerate(tqdm(data, desc="Processing entries")):
        sgc, wgc, lc = evaluate_entry(entry)
        
        # Add scores to the entry
        entry["coherence_scores"] = {
            "sgc": sgc,
            "wgc": wgc,
            "lc": lc,
            "mean_coherence": (sgc + wgc + lc) / 3
        }
        
        sgc_scores.append(sgc)
        wgc_scores.append(wgc)
        lc_scores.append(lc)

    # Calculate overall averages
    avg_sgc = sum(sgc_scores) / len(sgc_scores) if sgc_scores else 0
    avg_wgc = sum(wgc_scores) / len(wgc_scores) if wgc_scores else 0
    avg_lc = sum(lc_scores) / len(lc_scores) if lc_scores else 0
    baseline = (avg_sgc + avg_wgc + avg_lc) / 3 if (avg_sgc + avg_wgc + avg_lc) > 0 else 0

    # Save only the enhanced entries
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data, avg_sgc, avg_wgc, avg_lc, baseline

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    input_path = args.input
    output_path = args.output

    print("Evaluating dataset with batch processing...")
    enhanced_data, avg_sgc, avg_wgc, avg_lc, baseline = evaluate_and_save_dataset(input_path, output_path)
    
    print("\n=== Evaluation Results ===")
    print(f"Average Strong Global Coherence (SGC): {avg_sgc:.3f}")
    print(f"Average Weak Global Coherence   (WGC): {avg_wgc:.3f}")
    print(f"Average Local Coherence        (LC):  {avg_lc:.3f}")
    print(f"Baseline Score (mean of 3):         {baseline:.3f}")
    print(f"\nResults saved to: {output_path}")
    print(f"Total entries processed: {len(enhanced_data)}")


# python Kottoyama_processor.py --input /home/ojas/scripts/datasets/Human_Eval/sample_exps/T_final_main.json --output /home/ojas/scripts/datasets/Human_Eval/sample_exps/T_final_main_KT.json --evidence true --gpu 2



# python Kottoyama_processor.py --input /home/ojas/scripts/datasets/Human_Eval/sample_exps/T_final_main_perturbed.json --output /home/ojas/scripts/datasets/Human_Eval/sample_exps/T_final_main_perturbed_KT.json --evidence true --gpu 2

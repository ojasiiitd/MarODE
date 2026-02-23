import re
import numpy as np
import json
import torch
from transformers import pipeline
import argparse
from tqdm import tqdm

# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(description='Calculate RECEval scores for reasoning traces in batch.')
parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file')
parser.add_argument('--output', '-o', required=True, help='Path to the output JSON file')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id to use (e.g., 0 for the first GPU). Use -1 for CPU.')
args = parser.parse_args()
GPU_USE = args.gpu
BATCH_SIZE = 64  # You can adjust this based on your GPU memory
args = parser.parse_args()

# ---------------------------------------------------
# Helper Functions (Optimized for Batch Processing)
# ---------------------------------------------------

def extract_steps_batch(reasoning_traces):
    """
    Extracts reasoning steps from a batch of reasoning traces.
    Preserves the original logic to handle R<n>:, sent<n>:, and int<n>: prefixes.
    """
    all_steps = []
    for reasoning_trace in reasoning_traces:
        steps = []
        for line in reasoning_trace.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove prefixes
            line = re.sub(r'^\s*R\d+\s*:\s*', '', line)
            line = re.sub(r'^\s*(sent\d+|int\d+)\s*:\s*', '', line)
            if line:
                steps.append(line)
        all_steps.append(steps)
    return all_steps

def nli_score_batch(nli_pipeline, premises, hypotheses):
    """
    Computes NLI scores for a batch of premise-hypothesis pairs.
    Returns separate lists for entailment and contradiction scores.
    """
    if not premises:
        return [], []
        
    inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    results = nli_pipeline(inputs)
    
    entailment_scores = []
    contradiction_scores = []
    
    for result_labels in results:
        entail_score = 0.0
        contra_score = 0.0
        # The pipeline returns a list of label dicts for each input
        for label_dict in result_labels:
            if label_dict["label"].lower() == "entailment":
                entail_score = label_dict["score"]
            elif label_dict["label"].lower() == "contradiction":
                contra_score = label_dict["score"]
        entailment_scores.append(entail_score)
        contradiction_scores.append(contra_score)
        
    return entailment_scores, contradiction_scores

# ---------------------------------------------------
# Batched RECEval Metric Functions
# ---------------------------------------------------

def metric_intra_step_batch(all_steps, nli_pipeline):
    """
    Calculates Intra-Step Correctness for a batch of entries.
    This corrected version perfectly matches the logic of the simple function.
    """
    batch_premises, batch_hypotheses = [], []
    # This list will store the final scores for each entry.
    final_scores = []
    # This list tracks which scores in the batch correspond to which entry.
    # We use it to map the NLI results back correctly.
    entry_mapping = []

    for entry_idx, steps in enumerate(all_steps):
        # We start with a list of scores for this specific entry.
        entry_specific_scores = []

        if len(steps) < 2:
            final_scores.append(0.0)
            continue
        
        for i in range(len(steps) - 1):
            premise = " ".join(steps[0:i])
            hypothesis = steps[i+1]
            
            # This is the crucial logic change.
            if not premise.strip() or not hypothesis.strip():
                # If premise is empty (like when i=0), the score is 0.0 by definition.
                entry_specific_scores.append(0.0)
            else:
                # Otherwise, add the pair to the batch to be scored by the model.
                batch_premises.append(premise)
                batch_hypotheses.append(hypothesis)
                # Mark that this score belongs to the current entry.
                entry_mapping.append(entry_idx)
                # Add a placeholder that we will fill in later.
                entry_specific_scores.append(None)
        
        final_scores.append(entry_specific_scores)
    
    # Get all non-zero scores in a single model call
    if batch_premises:
        all_entailment_scores, _ = nli_score_batch(nli_pipeline, batch_premises, batch_hypotheses)
        
        # Distribute the calculated scores back to their original entries
        for i, score in enumerate(all_entailment_scores):
            original_entry_idx = entry_mapping[i]
            # Find the first 'None' placeholder in the target entry's score list and replace it.
            placeholder_idx = final_scores[original_entry_idx].index(None)
            final_scores[original_entry_idx][placeholder_idx] = score

    # Finally, calculate the mean for each entry's list of scores
    results = [np.mean(scores) if scores else 0.0 for scores in final_scores]
    
    return results

def metric_inter_step_batch(all_steps, nli_pipeline):
    """Calculates Inter-Step Correctness for a batch of entries."""
    batch_premises, batch_hypotheses = [], []
    pairs_per_entry = []

    for steps in all_steps:
        entry_pairs_count = 0
        if len(steps) >= 2:
            for i in range(1, len(steps)):
                premise = steps[i-1]
                hypothesis = steps[i]
                if premise.strip() and hypothesis.strip():
                    batch_premises.append(premise)
                    batch_hypotheses.append(hypothesis)
                    entry_pairs_count += 1
        pairs_per_entry.append(entry_pairs_count)
        
    _, all_contradiction_scores = nli_score_batch(nli_pipeline, batch_premises, batch_hypotheses)
    
    final_scores = []
    current_idx = 0
    for count in pairs_per_entry:
        if count == 0:
            # If no contradictions can be calculated, score is perfect (1.0)
            final_scores.append(1.0)
            continue
        entry_scores = all_contradiction_scores[current_idx : current_idx + count]
        avg_contra = np.mean(entry_scores) if entry_scores else 0.0
        final_scores.append(1.0 - avg_contra)
        current_idx += count
        
    return final_scores

# ---------------------------------------------------
# Main Calculation Function
# ---------------------------------------------------

def calculate_receval_batch(entries, nli_pipeline):
    """Orchestrates RECEval calculation for a batch of entries."""
    reasoning_traces = [entry["reasoning_trace"] for entry in entries]
    all_steps = extract_steps_batch(reasoning_traces)
    
    intra_scores = metric_intra_step_batch(all_steps, nli_pipeline)
    inter_scores = metric_inter_step_batch(all_steps, nli_pipeline)
    
    for i, entry in enumerate(entries):
        intra = intra_scores[i]
        inter = inter_scores[i]
        composite = (intra + inter) / 2
        
        entry["receval_scores"] = {
            "Intra-Step Correctness": intra,
            "Inter-Step Correctness": inter,
        }
        entry["mean_RECEval"] = float(composite)
    
    return entries

# ---------------------------------------------------
# Main Execution
# ---------------------------------------------------
def main():
    print(f"Loading data from {args.input}...")
    with open(args.input, "r") as f:
        entries = json.load(f)

    print("Loading NLI model...")
    nli_pipeline = pipeline(
        "text-classification",
        model="/home/models/deberta-xlarge-mnli",
        tokenizer="/home/models/deberta-xlarge-mnli",
        device=GPU_USE,
        top_k=None, # Return all labels (entailment, neutral, contradiction)
        batch_size= BATCH_SIZE, # Tune based on your GPU memory
        truncation=True
    )
    
    results = []
    batch_size = BATCH_SIZE # Can be the same as in pipeline or different
    total_batches = (len(entries) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(entries), batch_size), desc="Processing batches", total=total_batches):
        batch_entries = entries[i : i + batch_size]
        processed_batch = calculate_receval_batch(batch_entries, nli_pipeline)
        results.extend(processed_batch)

    print(f"Saving results with RECEval scores to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    valid_scores = [x["mean_RECEval"] for x in results]
    print("\nProcessing complete!")
    print(f"Total entries processed: {len(results)}")
    print(f"Overall mean RECEval score: {np.mean(valid_scores):.4f}")

if __name__ == "__main__":
    main()

# python RECEVAL.py --input /home/ojas/scripts/datasets/Final_Runs/DeepLlama_8B_1shot_PTRB_BASELINES.json --output /home/ojas/scripts/datasets/Final_Runs/DeepLlama_8B_1shot_PTRB_BASELINES_receval.json --gpu 1



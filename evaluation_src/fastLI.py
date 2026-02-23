import json
import argparse
import numpy as np
from transformers import pipeline
from datasets import Dataset
from tqdm import tqdm
import torch

# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(description='Process reasoning traces with evidence flag')
parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file')
parser.add_argument('--output', '-o', required=True, help='Path to the output JSON file')
parser.add_argument('--evidence', '-e', type=lambda x: x.lower() == 'true', default=True, help='Whether to use evidence (true/false)')
parser.add_argument('--gpu', '-g', required=True, type=int, default=-10, help='GPU id to use (e.g. 0 for first GPU)')
args = parser.parse_args()
GPU_USE = args.gpu

# ---------------------------------------------------
# Model Initialization
# ---------------------------------------------------
torch.set_float32_matmul_precision('high')

nli_model = pipeline(
    "text-classification",
    model="/home/models/deberta-xlarge-mnli",
    device=GPU_USE,
    top_k=None,
    batch_size=16,
    truncation=True,
    max_length=512,
    torch_dtype=torch.float16
)

EVIDENCE = args.evidence
REFERENCE = False

# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------
def extract_steps(reasoning_trace):
    """Extract reasoning steps from the trace."""
    steps = []
    for line in reasoning_trace.split("\n"):
        if ":" in line:
            step_text = line.split(":", 1)[-1].strip()
            if step_text:
                steps.append(step_text)
    return steps


def get_contradiction_score(result):
    """Robust extraction of contradiction probability."""
    if isinstance(result, list):
        for r in result:
            if r.get("label", "").lower() == "contradiction":
                return float(r.get("score", 0.0))
    elif isinstance(result, dict):
        if result.get("label", "").lower() == "contradiction":
            return float(result.get("score", 0.0))
    return 0.0


def get_pcontr_batch(premises, hypotheses):
    """Compute contradiction probabilities for all pairs."""
    if not premises:
        return []
    try:
        inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
        results = nli_model(inputs)

        # The pipeline may return:
        # - list[dict] if batch_size == 1
        # - list[list[dict]] for normal batches
        contradiction_scores = []
        for res in results:
            if isinstance(res, list):
                contradiction_scores.append(get_contradiction_score(res))
            elif isinstance(res, dict):
                contradiction_scores.append(get_contradiction_score(res))
            else:
                contradiction_scores.append(0.0)

        return contradiction_scores
    except Exception as e:
        print(f"[get_pcontr_batch] Error: {e}")
        return [0.0] * len(premises)

# ---------------------------------------------------
# ROSCOE-LI Components
# ---------------------------------------------------
def self_consistency_batch(samples):
    """Self-Consistency (hi ↔ hj)."""
    all_premises, all_hypotheses, idx_map = [], [], []
    scores = []

    for s_idx, sample in enumerate(samples):
        steps = extract_steps(sample["reasoning_trace"])
        if len(steps) < 2:
            scores.append(1.0)
            continue
        for i in range(1, len(steps)):
            for j in range(i):
                all_premises.append(steps[i])
                all_hypotheses.append(steps[j])
                idx_map.append(s_idx)
        scores.append(None)

    if not all_premises:
        return [1.0 for _ in samples]

    pcontr_scores = get_pcontr_batch(all_premises, all_hypotheses)

    per_sample_scores = [[] for _ in samples]
    for s_idx, contr in zip(idx_map, pcontr_scores):
        per_sample_scores[s_idx].append(contr)

    for i, contrs in enumerate(per_sample_scores):
        scores[i] = 1 - max(contrs) if contrs else 1.0
    return scores


def source_consistency_batch(samples):
    """Source-Consistency (h ↔ s)."""
    all_premises, all_hypotheses, idx_map = [], [], []
    scores = []

    for s_idx, sample in enumerate(samples):
        steps = extract_steps(sample["reasoning_trace"])
        context = sample.get("evidence_text", [])
        if not context or not steps:
            scores.append(1.0)
            continue
        for step in steps:
            for sent in context:
                all_premises.append(step)
                all_hypotheses.append(sent)
                idx_map.append(s_idx)
        scores.append(None)

    if not all_premises:
        return [1.0 for _ in samples]

    pcontr_scores = get_pcontr_batch(all_premises, all_hypotheses)
    per_sample_scores = [[] for _ in samples]
    for s_idx, contr in zip(idx_map, pcontr_scores):
        per_sample_scores[s_idx].append(contr)

    for i, contrs in enumerate(per_sample_scores):
        scores[i] = 1 - max(contrs) if contrs else 1.0
    return scores


def calculate_roscoe_li(sample):
    """Calculate ROSCOE-LI metrics for a single sample."""
    try:
        if EVIDENCE and not REFERENCE:
            self_score = self_consistency_batch([sample])[0]
            src_score = source_consistency_batch([sample])[0]
            scores = {
                "Self-Consistency": self_score,
                "Source-Consistency": src_score
            }
        else:
            scores = {
                "Self-Consistency": self_consistency_batch([sample])[0]
            }

        mean_score = sum(scores.values()) / len(scores)
        sample["roscoe_li_scores"] = scores
        sample["mean_ROSCOE_LI"] = float(mean_score)
        return sample

    except Exception as e:
        print(f"[calculate_roscoe_li] Error: {e}")
        sample["roscoe_li_scores"] = {"Self-Consistency": -1, "Source-Consistency": -1}
        sample["mean_ROSCOE_LI"] = 0.0
        return sample

# ---------------------------------------------------
# Batch Processing
# ---------------------------------------------------
def process_batch(batch):
    samples = [{key: batch[key][i] for key in batch} for i in range(len(batch["claim"]))]
    processed = [calculate_roscoe_li(s) for s in samples]
    output_batch = {k: [s[k] for s in processed] for k in processed[0].keys()}
    return output_batch

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main(input_file, output_file):
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        entries = json.load(f)

    dataset = Dataset.from_list(entries)
    print(f"Processing {len(dataset)} entries with ROSCOE-LI metrics...")

    # Disable multiprocessing for GPU stability
    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=8,
        desc="Calculating ROSCOE-LI metrics"
    )

    print(f"Saving results to {output_file}...")
    data_list = dataset.to_list()
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)

    valid_scores = [x for x in dataset["mean_ROSCOE_LI"] if x > 0]
    print(f"\nProcessing complete!")
    print(f"Total entries processed: {len(dataset)}")
    print(f"Entries with valid scores: {len(valid_scores)}")

# ---------------------------------------------------
if __name__ == "__main__":
    main(args.input, args.output)

# python fastLI.py --input /home/ojas/scripts/datasets/Human_Eval/sample_exps/test.json --output /home/ojas/scripts/datasets/Human_Eval/sample_exps/test_li_fast.json --evidence true --gpu 1

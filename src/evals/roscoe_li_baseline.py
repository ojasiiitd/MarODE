import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import pipeline
import torch


# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(
    description="Calculate ROSCOE-LI baseline metrics."
)

parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", required=True)
parser.add_argument("--model-path", required=True, help="Path to MNLI model")
parser.add_argument("--gpu", "-g", type=int, default=-1)
parser.add_argument("--evidence", "-e", type=lambda x: x.lower() == "true", default=True)

args = parser.parse_args()

DEVICE = args.gpu
USE_EVIDENCE = args.evidence

torch.set_float32_matmul_precision("high")


# ---------------------------------------------------
# Load NLI Model
# ---------------------------------------------------
nli_model = pipeline(
    "text-classification",
    model=args.model_path,
    device=DEVICE,
    top_k=None,
    batch_size=16,
    truncation=True,
    max_length=512,
    torch_dtype=torch.float16 if DEVICE >= 0 else torch.float32,
)


# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------
def extract_steps(reasoning_trace):
    steps = []
    for line in reasoning_trace.split("\n"):
        if ":" in line:
            step_text = line.split(":", 1)[-1].strip()
            if step_text:
                steps.append(step_text)
    return steps


def get_contradiction_score(result):
    if isinstance(result, list):
        for r in result:
            if r.get("label", "").lower() == "contradiction":
                return float(r.get("score", 0.0))
    elif isinstance(result, dict):
        if result.get("label", "").lower() == "contradiction":
            return float(result.get("score", 0.0))
    return 0.0


def get_pcontr_batch(premises, hypotheses):
    if not premises:
        return []

    inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    results = nli_model(inputs)

    contradiction_scores = []
    for res in results:
        contradiction_scores.append(get_contradiction_score(res))

    return contradiction_scores


# ---------------------------------------------------
# ROSCOE-LI Components
# ---------------------------------------------------
def self_consistency_batch(samples):
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

    per_sample = [[] for _ in samples]
    for s_idx, contr in zip(idx_map, pcontr_scores):
        per_sample[s_idx].append(contr)

    for i, contrs in enumerate(per_sample):
        scores[i] = 1 - max(contrs) if contrs else 1.0

    return scores


def source_consistency_batch(samples):
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

    per_sample = [[] for _ in samples]
    for s_idx, contr in zip(idx_map, pcontr_scores):
        per_sample[s_idx].append(contr)

    for i, contrs in enumerate(per_sample):
        scores[i] = 1 - max(contrs) if contrs else 1.0

    return scores


def calculate_roscoe_li(sample):
    self_score = self_consistency_batch([sample])[0]

    if USE_EVIDENCE:
        src_score = source_consistency_batch([sample])[0]
        scores = {
            "Self-Consistency": self_score,
            "Source-Consistency": src_score,
        }
    else:
        scores = {
            "Self-Consistency": self_score,
        }

    mean_score = sum(scores.values()) / len(scores)
    sample["roscoe_li_scores"] = scores
    sample["mean_ROSCOE_LI"] = float(mean_score)

    return sample


# ---------------------------------------------------
# Batch Processing
# ---------------------------------------------------
def process_batch(batch):
    samples = [{k: batch[k][i] for k in batch} for i in range(len(batch["claim"]))]
    processed = [calculate_roscoe_li(s) for s in samples]
    return {k: [s[k] for s in processed] for k in processed[0].keys()}


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    with open(args.input, "r") as f:
        entries = json.load(f)

    dataset = Dataset.from_list(entries)

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=8,
        desc="Calculating ROSCOE-LI",
    )

    data_list = dataset.to_list()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)

    valid_scores = [x for x in dataset["mean_ROSCOE_LI"] if x > 0]

    print("\n=== ROSCOE-LI Results ===")
    print(f"Total entries: {len(dataset)}")
    print(f"Valid entries: {len(valid_scores)}")


if __name__ == "__main__":
    main()
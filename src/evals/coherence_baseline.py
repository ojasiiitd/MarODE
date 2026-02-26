import json
import argparse
import re
from itertools import combinations
from typing import List
from tqdm import tqdm
from transformers import pipeline


# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(
    description="Evaluate reasoning traces using Local and Global Coherence baselines."
)

parser.add_argument("--input", "-i", required=True, help="Path to input JSON file")
parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
parser.add_argument("--model-path", required=True, help="Path to MNLI model")
parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU id (default: CPU)")

args = parser.parse_args()


# -------------------------------
# Load NLI model
# -------------------------------
nli = pipeline(
    "text-classification",
    model=args.model_path,
    device=args.gpu,
    top_k=None,
    batch_size=16,
    truncation=True,
    padding=True,
)


# -------------------------------
# Batch NLI helper
# -------------------------------
def nli_relation_batch(premises: List[str], hypotheses: List[str]) -> List[str]:
    inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    results = nli(inputs)

    relations = []

    for result in results:
        if isinstance(result, list) and result:
            scores = {r["label"].lower(): r["score"] for r in result}
            relations.append(max(scores, key=scores.get))
        elif isinstance(result, dict):
            relations.append(result["label"].lower())
        else:
            relations.append("neutral")

    return relations


# -------------------------------
# Normalize reasoning steps
# -------------------------------
def normalize_reasoning_steps(reasoning_trace: str) -> List[str]:
    steps = []

    for line in reasoning_trace.splitlines():
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"^\s*R\d+\s*:\s*", "", line)
        line = re.sub(r"^\s*(sent\d+|int\d+)\s*:\s*", "", line)

        if line:
            steps.append(line)

    return steps


# -------------------------------
# Evaluate single entry
# -------------------------------
def evaluate_entry(entry):
    claim = entry["claim"]
    reasoning_trace = entry.get("reasoning_trace", "")
    steps = normalize_reasoning_steps(reasoning_trace)

    # --- Strong & Weak Global Coherence ---
    if steps:
        relations = nli_relation_batch(steps, [claim] * len(steps))
        sgc = int(all(r == "entailment" for r in relations))
        wgc = int(all(r != "contradiction" for r in relations))
    else:
        sgc, wgc = 0, 0

    # --- Local Coherence ---
    if len(steps) > 1:
        p1, p2 = [], []
        for s1, s2 in combinations(steps, 2):
            p1.extend([s1, s2])
            p2.extend([s2, s1])

        relations = nli_relation_batch(p1, p2)
        lc = int(all(r != "contradiction" for r in relations))
    else:
        lc = 1

    return sgc, wgc, lc


# -------------------------------
# Evaluate dataset
# -------------------------------
def evaluate_dataset(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    sgc_scores, wgc_scores, lc_scores = [], [], []

    for entry in tqdm(data, desc="Evaluating"):
        sgc, wgc, lc = evaluate_entry(entry)

        entry["coherence_scores"] = {
            "sgc": sgc,
            "wgc": wgc,
            "lc": lc,
            "mean_coherence": (sgc + wgc + lc) / 3,
        }

        sgc_scores.append(sgc)
        wgc_scores.append(wgc)
        lc_scores.append(lc)

    avg_sgc = sum(sgc_scores) / len(sgc_scores) if sgc_scores else 0
    avg_wgc = sum(wgc_scores) / len(wgc_scores) if wgc_scores else 0
    avg_lc = sum(lc_scores) / len(lc_scores) if lc_scores else 0
    baseline = (avg_sgc + avg_wgc + avg_lc) / 3

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n=== Coherence Baseline Results ===")
    print(f"Average SGC: {avg_sgc:.3f}")
    print(f"Average WGC: {avg_wgc:.3f}")
    print(f"Average LC : {avg_lc:.3f}")
    print(f"Baseline   : {baseline:.3f}")
    print(f"\nSaved to: {output_path}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    evaluate_dataset(args.input, args.output)
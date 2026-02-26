import re
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import pipeline


# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(
    description="Calculate RECEval scores for reasoning traces (batched)."
)

parser.add_argument("--input", "-i", required=True, help="Input JSON file")
parser.add_argument("--output", "-o", required=True, help="Output JSON file")
parser.add_argument("--model-path", required=True, help="Path to MNLI model")
parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU id (default: CPU)")
parser.add_argument("--batch-size", type=int, default=64)

args = parser.parse_args()


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def extract_steps_batch(reasoning_traces):
    all_steps = []

    for reasoning_trace in reasoning_traces:
        steps = []
        for line in reasoning_trace.splitlines():
            line = line.strip()
            if not line:
                continue

            line = re.sub(r"^\s*R\d+\s*:\s*", "", line)
            line = re.sub(r"^\s*(sent\d+|int\d+)\s*:\s*", "", line)

            if line:
                steps.append(line)

        all_steps.append(steps)

    return all_steps


def nli_score_batch(nli_pipeline, premises, hypotheses):
    if not premises:
        return [], []

    inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    results = nli_pipeline(inputs)

    entailment_scores = []
    contradiction_scores = []

    for labels in results:
        entail_score = 0.0
        contra_score = 0.0

        for label_dict in labels:
            if label_dict["label"].lower() == "entailment":
                entail_score = label_dict["score"]
            elif label_dict["label"].lower() == "contradiction":
                contra_score = label_dict["score"]

        entailment_scores.append(entail_score)
        contradiction_scores.append(contra_score)

    return entailment_scores, contradiction_scores


# ---------------------------------------------------
# RECEval Metrics
# ---------------------------------------------------
def metric_intra_step_batch(all_steps, nli_pipeline):
    batch_premises, batch_hypotheses = [], []
    entry_mapping = []
    final_scores = []

    for entry_idx, steps in enumerate(all_steps):
        entry_scores = []

        if len(steps) < 2:
            final_scores.append([0.0])
            continue

        for i in range(len(steps) - 1):
            premise = " ".join(steps[0:i])
            hypothesis = steps[i + 1]

            if not premise.strip() or not hypothesis.strip():
                entry_scores.append(0.0)
            else:
                batch_premises.append(premise)
                batch_hypotheses.append(hypothesis)
                entry_mapping.append(entry_idx)
                entry_scores.append(None)

        final_scores.append(entry_scores)

    if batch_premises:
        entailment_scores, _ = nli_score_batch(
            nli_pipeline, batch_premises, batch_hypotheses
        )

        for i, score in enumerate(entailment_scores):
            original_idx = entry_mapping[i]
            placeholder_idx = final_scores[original_idx].index(None)
            final_scores[original_idx][placeholder_idx] = score

    return [float(np.mean(scores)) if scores else 0.0 for scores in final_scores]


def metric_inter_step_batch(all_steps, nli_pipeline):
    batch_premises, batch_hypotheses = [], []
    pairs_per_entry = []

    for steps in all_steps:
        count = 0
        if len(steps) >= 2:
            for i in range(1, len(steps)):
                premise = steps[i - 1]
                hypothesis = steps[i]

                if premise.strip() and hypothesis.strip():
                    batch_premises.append(premise)
                    batch_hypotheses.append(hypothesis)
                    count += 1

        pairs_per_entry.append(count)

    _, contradiction_scores = nli_score_batch(
        nli_pipeline, batch_premises, batch_hypotheses
    )

    final_scores = []
    idx = 0

    for count in pairs_per_entry:
        if count == 0:
            final_scores.append(1.0)
            continue

        entry_scores = contradiction_scores[idx : idx + count]
        avg_contra = np.mean(entry_scores) if entry_scores else 0.0
        final_scores.append(float(1.0 - avg_contra))
        idx += count

    return final_scores


# ---------------------------------------------------
# Main RECEval Calculation
# ---------------------------------------------------
def calculate_receval_batch(entries, nli_pipeline):
    reasoning_traces = [e["reasoning_trace"] for e in entries]
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
    with open(args.input, "r") as f:
        entries = json.load(f)

    nli_pipeline = pipeline(
        "text-classification",
        model=args.model_path,
        tokenizer=args.model_path,
        device=args.gpu,
        top_k=None,
        batch_size=args.batch_size,
        truncation=True,
    )

    results = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(entries), batch_size), desc="Processing"):
        batch = entries[i : i + batch_size]
        results.extend(calculate_receval_batch(batch, nli_pipeline))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    valid_scores = [x["mean_RECEval"] for x in results]
    print("\n=== RECEval Results ===")
    print(f"Total entries: {len(results)}")
    print(f"Overall mean RECEval: {np.mean(valid_scores):.4f}")


if __name__ == "__main__":
    main()
import json
import argparse
import numpy as np
import re
from tqdm import tqdm
from datasets import Dataset
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(
    description="Calculate ROSCOE-SS baseline metrics."
)

parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", required=True)
parser.add_argument("--model-path", required=True, help="Path to sentence-transformer model")
parser.add_argument("--gpu", "-g", type=int, default=-1)
parser.add_argument("--evidence", "-e", type=lambda x: x.lower() == "true", default=True)

args = parser.parse_args()

DEVICE = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
USE_EVIDENCE = args.evidence

# ---------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------
embedding_model = SentenceTransformer(args.model_path, device=DEVICE)


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def extract_steps(reasoning_trace):
    return [
        match.group(1)
        for line in reasoning_trace.split("\n")
        if (match := re.match(r"R\d+:\s*(.+)", line.strip()))
    ]


def cosine_sim(a, b):
    embeddings = embedding_model.encode(
        [a, b],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return float(np.dot(embeddings[0], embeddings[1]))


# ---------------------------------------------------
# ROSCOE-SS Components
# ---------------------------------------------------
def info_chain(sample):
    source_text = " ".join(sample.get("evidence_text", []))
    reasoning_chain = " ".join(extract_steps(sample["reasoning_trace"]))

    if not source_text or not reasoning_chain:
        return 0.0

    sim = cosine_sim(reasoning_chain, source_text)
    return round((1 + sim) / 2, 4)


def repetition_step(sample):
    steps = extract_steps(sample["reasoning_trace"])

    if len(steps) < 2:
        return 0.5

    embeddings = embedding_model.encode(
        steps,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    sims = np.matmul(embeddings, embeddings.T)
    np.fill_diagonal(sims, -1)
    max_sim = np.max(sims)

    return round((1 - max_sim) / 2, 4)


def semantic_coverage_chain(sample):
    reasoning_chain = " ".join(extract_steps(sample["reasoning_trace"]))
    claim = sample.get("reference_text", "")

    if not claim or not reasoning_chain:
        return 0.0

    sim = cosine_sim(reasoning_chain, claim)
    return round((1 + sim) / 2, 4)


def calculate_roscoe_ss(entry):
    scores = {}

    if USE_EVIDENCE:
        scores["Info-Chain"] = info_chain(entry)
        scores["Repetition-Step"] = repetition_step(entry)
    else:
        scores["Repetition-Step"] = repetition_step(entry)

    mean_score = round(sum(scores.values()) / len(scores), 4)

    entry["roscoe_ss_scores"] = scores
    entry["mean_ROSCOE_SS"] = float(mean_score)

    return entry


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    with open(args.input, "r") as f:
        entries = json.load(f)

    dataset = Dataset.from_list(entries)

    def process_batch(batch):
        results = [
            calculate_roscoe_ss({key: batch[key][i] for key in batch})
            for i in range(len(batch["claim"]))
        ]
        return {k: [r[k] for r in results] for k in results[0]}

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=64,
        desc="Calculating ROSCOE-SS",
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)

    valid_scores = [x for x in dataset["mean_ROSCOE_SS"] if x > 0]

    print("\n=== ROSCOE-SS Results ===")
    print(f"Total entries: {len(dataset)}")
    print(f"Valid entries: {len(valid_scores)}")
    if valid_scores:
        print(f"Overall mean ROSCOE-SS: {np.mean(valid_scores):.4f}")


if __name__ == "__main__":
    main()
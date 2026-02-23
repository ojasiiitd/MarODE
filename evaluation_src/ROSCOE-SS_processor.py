import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from tqdm import tqdm
import re

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

# Load a sentence embedding model
embedding_model = SentenceTransformer("/home/models/sentence-transformers--all-MiniLM-L6-v2" , device=f"cuda:{GPU_USE}")

EVIDENCE = args.evidence
REFERENCE = False

def cosine_sim_batch(vecs_a, vecs_b):
    """Compute pairwise cosine similarity with pre-normalized embeddings."""
    # Normalize for cosine sim
    a = vecs_a / np.linalg.norm(vecs_a, axis=1, keepdims=True)
    b = vecs_b / np.linalg.norm(vecs_b, axis=1, keepdims=True)
    return np.matmul(a, b.T)

def cosine_sim(a, b):
    """Single cosine similarity between two texts (fast version)."""
    embeddings = embedding_model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True, dtype=np.float32)
    return float(np.dot(embeddings[0], embeddings[1]))

def extract_steps(reasoning_trace):
    """Extract reasoning steps from the trace, removing R0: prefixes."""
    return [match.group(1) for line in reasoning_trace.split("\n")
            if (match := re.match(r"R\d+: (.+)", line.strip()))]

def info_chain(sample):
    """ROSCOE-SS: Informativness-Chain (h→s)."""
    source_text = " ".join(sample["evidence_text"])
    reasoning_chain_clean = " ".join(extract_steps(sample["reasoning_trace"]))
    sim = cosine_sim(reasoning_chain_clean, source_text)
    return round((1 + sim) / 2, 4)

def repetition_step(sample):
    """ROSCOE-SS: Repetition-Step (hi ↔ hj)."""
    steps = extract_steps(sample["reasoning_trace"])
    if len(steps) < 2:
        return 0.5  # neutral score if only one step
    
    embeddings = embedding_model.encode(steps, convert_to_numpy=True, normalize_embeddings=True, dtype=np.float32)
    sims = np.matmul(embeddings, embeddings.T)
    np.fill_diagonal(sims, -1)  # ignore self-similarity
    max_sim = np.max(sims)
    return round((1 - max_sim) / 2, 4)

def semantic_coverage_chain(sample):
    """ROSCOE-SS: Semantic Coverage-Chain (r ↔ h)."""
    reasoning_chain_clean = " ".join(extract_steps(sample["reasoning_trace"]))
    claim = sample.get("reference_text", "")
    sim = cosine_sim(reasoning_chain_clean, claim)
    return round((1 + sim) / 2, 4)

def calculate_roscoe_ss(entry):
    """Calculate all ROSCOE-SS metrics for a single sample"""
    try:
        scores = {}
        if EVIDENCE:
            scores["Info-Chain"] = info_chain(entry)
            scores["Repetition-Step"] = repetition_step(entry)
        elif not EVIDENCE:
            scores["Repetition-Step"] = repetition_step(entry)

        mean_score = round(sum(scores.values()) / len(scores), 4)
        entry["roscoe_ss_scores"] = scores
        entry["mean_ROSCOE_SS"] = float(mean_score)
        return entry
    except Exception as e:
        print(f"Error processing sample: {e}")
        entry["roscoe_ss_scores"] = {"Info-Chain": -1, "Repetition-Step": -1, "Semantic Coverage-Chain": -1}
        entry["mean_ROSCOE_SS"] = 0.0
        return entry

def main(input_file, output_file):
    """Main function to process JSON file with ROSCOE-SS metrics"""
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        entries = json.load(f)

    dataset = Dataset.from_list(entries)
    print(f"Processing {len(dataset)} entries with ROSCOE-SS metrics...")

    def process_batch(batch):
        results = [calculate_roscoe_ss({key: batch[key][i] for key in batch}) for i in range(len(batch["claim"]))]
        return {k: [r[k] for r in results] for k in results[0]}

    dataset = dataset.map(process_batch, batched=True, batch_size=64, desc="Calculating ROSCOE-SS metrics")

    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)

    valid_scores = [x for x in dataset["mean_ROSCOE_SS"] if x > 0]
    print(f"\nProcessing complete!")
    print(f"Total entries processed: {len(dataset)}")
    print(f"Entries with valid scores: {len(valid_scores)}")
    if valid_scores:
        print(f"Overall mean ROSCOE-SS: {np.mean(valid_scores):.4f}")

if __name__ == "__main__":
    main(args.input, args.output)

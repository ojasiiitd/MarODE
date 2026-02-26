import re
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(
    description="Calculate ROSCOE-LC baseline metrics."
)

parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", required=True)
parser.add_argument("--ppl-model", required=True, help="Path to causal LM for perplexity")
parser.add_argument("--cola-model", required=True, help="Path to CoLA model")
parser.add_argument("--gpu", "-g", type=int, default=-1)

args = parser.parse_args()

DEVICE = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"


# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
ppl_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model)
if ppl_tokenizer.pad_token is None:
    ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

ppl_model = AutoModelForCausalLM.from_pretrained(args.ppl_model).to(DEVICE)
ppl_model.eval()

cola_tokenizer = AutoTokenizer.from_pretrained(args.cola_model)
if cola_tokenizer.pad_token is None:
    cola_tokenizer.pad_token = cola_tokenizer.eos_token

cola_model = AutoModelForSequenceClassification.from_pretrained(args.cola_model).to(DEVICE)
cola_model.eval()


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def extract_steps(reasoning_trace):
    return [
        line.split(":", 1)[-1].strip()
        for line in reasoning_trace.split("\n")
        if ":" in line and line.split(":", 1)[-1].strip()
    ]


@torch.no_grad()
def compute_perplexity_batch(texts):
    encodings = ppl_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    max_length = ppl_model.config.n_positions
    stride = 512
    batch_ppls = []

    for input_ids in encodings.input_ids:
        input_ids = input_ids.unsqueeze(0)
        nlls = []

        for i in range(0, input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i

            chunk_ids = input_ids[:, begin_loc:end_loc]
            target_ids = chunk_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = ppl_model(chunk_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc) if nlls else torch.tensor(float("inf"))
        batch_ppls.append(round(ppl.item(), 4))

    return batch_ppls


@torch.no_grad()
def compute_grammar_score_batch(texts):
    inputs = cola_tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(DEVICE)

    outputs = cola_model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    return np.round(probs[:, 1].cpu().numpy(), 4).tolist()


# ---------------------------------------------------
# ROSCOE-LC Metrics
# ---------------------------------------------------
def perplexity_chain_batch(entries):
    texts = [" ".join(extract_steps(e["reasoning_trace"])) for e in entries]
    ppls = compute_perplexity_batch(texts)
    return [round(1.0 / p if p > 0 else 0.0, 4) for p in ppls]


def perplexity_step_batch(entries):
    all_steps, step_indices = [], []

    for i, e in enumerate(entries):
        steps = extract_steps(e["reasoning_trace"])
        all_steps.extend(steps)
        step_indices.extend([i] * len(steps))

    if not all_steps:
        return [0.0] * len(entries)

    step_ppls = compute_perplexity_batch(all_steps)
    step_inv = [1.0 / p if p > 0 else 0.0 for p in step_ppls]

    entry_scores = np.zeros(len(entries))
    entry_counts = np.zeros(len(entries))

    for idx, score in zip(step_indices, step_inv):
        entry_scores[idx] += score
        entry_counts[idx] += 1

    return np.round(np.divide(entry_scores, np.maximum(entry_counts, 1)), 4).tolist()


def grammar_score_batch(entries):
    all_steps, step_indices = [], []

    for i, e in enumerate(entries):
        steps = extract_steps(e["reasoning_trace"])
        all_steps.extend(steps)
        step_indices.extend([i] * len(steps))

    if not all_steps:
        return [0.0] * len(entries)

    step_scores = compute_grammar_score_batch(all_steps)

    entry_scores = np.zeros(len(entries))
    entry_counts = np.zeros(len(entries))

    for idx, score in zip(step_indices, step_scores):
        entry_scores[idx] += score
        entry_counts[idx] += 1

    return np.round(np.divide(entry_scores, np.maximum(entry_counts, 1)), 4).tolist()


def calculate_roscoe_lc_batch(entries):
    ppl_chain = perplexity_chain_batch(entries)
    ppl_step = perplexity_step_batch(entries)
    grammar = grammar_score_batch(entries)

    for i, e in enumerate(entries):
        scores = {
            "Perplexity-Chain": ppl_chain[i],
            "Perplexity-Step": ppl_step[i],
            "Grammar": grammar[i],
        }

        e["roscoe_lc_scores"] = scores
        e["mean_ROSCOE_LC"] = float(round(sum(scores.values()) / 3, 4))

    return entries


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    with open(args.input, "r") as f:
        entries = json.load(f)

    dataset = Dataset.from_list(entries)

    def process_batch(batch):
        batch_entries = [
            {k: batch[k][i] for k in batch}
            for i in range(len(batch["claim"]))
        ]
        results = calculate_roscoe_lc_batch(batch_entries)
        return {k: [r[k] for r in results] for k in results[0]}

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=32,
        desc="Calculating ROSCOE-LC",
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)

    scores = [x for x in dataset["mean_ROSCOE_LC"] if x > 0]

    print("\n=== ROSCOE-LC Results ===")
    print(f"Total entries: {len(dataset)}")
    if scores:
        print(f"Overall mean ROSCOE-LC: {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
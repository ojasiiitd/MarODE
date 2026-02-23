import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import Dataset
import torch.nn.functional as F
import numpy as np
import json
import argparse
from tqdm import tqdm

GPU_USE = "cuda:"

# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(description='Process reasoning traces with evidence flag')
parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file')
parser.add_argument('--output', '-o', required=True, help='Path to the output JSON file')
parser.add_argument('--evidence', '-e', type=lambda x: x.lower() == 'true', default=True, help='Whether to use evidence (true/false)')
parser.add_argument('--gpu', '-g', required=True, type=int, default=-10, help='GPU id to use (e.g. 0 for first GPU)')
args = parser.parse_args()
GPU_USE = GPU_USE + str(args.gpu)

EVIDENCE = args.evidence

# ----------------------------
# Load Models
# ----------------------------
ppl_model_name = "/home/models/gpt2-large"
# ppl_model_name = "/home/models/gpt2"
ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
if ppl_tokenizer.pad_token is None:
    ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_name).to(GPU_USE)
ppl_model.eval()

# cola_model_name = "/home/models/textattack--roberta-base-CoLA" # slow
# cola_model_name = "/home/models/textattack--distilbert-base-cased-CoLA" # medium
cola_model_name = "/home/models/tinybertcola" # fast
cola_tokenizer = AutoTokenizer.from_pretrained(cola_model_name)
if cola_tokenizer.pad_token is None:
    cola_tokenizer.pad_token = cola_tokenizer.eos_token
cola_model = AutoModelForSequenceClassification.from_pretrained(cola_model_name).to(GPU_USE)
cola_model.eval()

# ----------------------------
# Helper functions
# ----------------------------
def extract_steps(reasoning_trace):
    """Extract reasoning steps from the trace."""
    return [line.split(":", 1)[-1].strip()
            for line in reasoning_trace.split("\n")
            if ":" in line and line.split(":", 1)[-1].strip()]

@torch.no_grad()
def compute_perplexity_batch(texts):
    """Compute perplexity for a batch of texts using GPT-2 LM."""
    encodings = ppl_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(GPU_USE)
    max_length = ppl_model.config.n_positions
    stride = 512

    batch_ppls = []
    for input_ids in encodings.input_ids:
        input_ids = input_ids.unsqueeze(0)  # keep batch dim
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

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc) if nlls else torch.tensor(float('inf'))
        batch_ppls.append(round(ppl.item(), 4))
    return batch_ppls

@torch.no_grad()
def compute_grammar_score_batch(texts):
    """Compute grammatical acceptability scores for a batch of texts."""
    inputs = cola_tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(GPU_USE)
    outputs = cola_model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    return np.round(probs[:, 1].cpu().numpy(), 4).tolist()  # label 1 = grammatically acceptable

# ----------------------------
# Batched ROSCOE-LC Metrics
# ----------------------------
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
    step_inv_ppls = [1.0 / p if p > 0 else 0.0 for p in step_ppls]

    # aggregate with numpy for speed
    entry_scores = np.zeros(len(entries))
    entry_counts = np.zeros(len(entries))
    for idx, inv_ppl in zip(step_indices, step_inv_ppls):
        entry_scores[idx] += inv_ppl
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
    if EVIDENCE or not EVIDENCE:
        ppl_chain_scores = perplexity_chain_batch(entries)
        ppl_step_scores = perplexity_step_batch(entries)
        grammar_scores = grammar_score_batch(entries)

    results = []
    for i, e in enumerate(entries):
        try:
            scores = {
                "Perplexity-Chain": ppl_chain_scores[i],
                "Perplexity-Step": ppl_step_scores[i],
                "Grammar": grammar_scores[i]
            }
            mean_score = round(sum(scores.values()) / len(scores), 4)
            e["roscoe_lc_scores"] = scores
            e["mean_ROSCOE_LC"] = float(mean_score)
            results.append(e)
        except Exception as ex:
            print(f"Error processing entry: {ex}")
            e["roscoe_lc_scores"] = {"Perplexity-Chain": -1, "Perplexity-Step": -1, "Grammar": -1}
            e["mean_ROSCOE_LC"] = 0.0
            results.append(e)
    return results

# ----------------------------
# Main
# ----------------------------
def main(input_file, output_file):
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        entries = json.load(f)

    dataset = Dataset.from_list(entries)
    print(f"Processing {len(dataset)} entries with ROSCOE-LC metrics...")

    def process_batch(batch):
        entries_batch = [{k: batch[k][i] for k in batch} for i in range(len(batch["claim"]))]
        results = calculate_roscoe_lc_batch(entries_batch)
        return {k: [r[k] for r in results] for k in results[0]}

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=32,   # increased for throughput
        desc="Calculating ROSCOE-LC metrics"
    )

    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)

    valid_scores = [x for x in dataset["mean_ROSCOE_LC"] if x > 0]
    print(f"\nProcessing complete!")
    print(f"Total entries processed: {len(dataset)}")
    print(f"Entries with valid scores: {len(valid_scores)}")
    if valid_scores:
        print(f"Overall mean ROSCOE-LC: {np.mean(valid_scores):.4f}")
        print(f"Perplexity-Chain mean: {np.mean([x['Perplexity-Chain'] for x in dataset['roscoe_lc_scores'] if x['Perplexity-Chain'] > 0]):.4f}")
        print(f"Perplexity-Step mean: {np.mean([x['Perplexity-Step'] for x in dataset['roscoe_lc_scores'] if x['Perplexity-Step'] > 0]):.4f}")
        print(f"Grammar mean: {np.mean([x['Grammar'] for x in dataset['roscoe_lc_scores'] if x['Grammar'] > 0]):.4f}")

if __name__ == "__main__":
    main(args.input, args.output)


# python ROSCOE-LC_processor.py --input /home/ojas/scripts/datasets/Human_Eval/sample_exps/DeepLlama_8B_2shot_sample_ptrb_baselines.json --output /home/ojas/scripts/datasets/Human_Eval/sample_exps/DeepLlama_8B_2shot_sample_ptrb_baselines.json --evidence true --gpu 0

# python ROSCOE-LC_processor.py --input /home/ojas/scripts/datasets/Human_Eval/math_random_150_R4-R10_BASELINES.json --output /home/ojas/scripts/datasets/Human_Eval/math_random_150_R4-R10_BASELINES.json --evidence false --gpu 2

import re
import numpy as np
import json
from transformers import pipeline
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import argparse

# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(description='Process reasoning traces with evidence flag')
parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file')
parser.add_argument('--output', '-o', required=True, help='Path to the output JSON file')
parser.add_argument('--evidence', '-e', type=lambda x: x.lower() == 'true', default=True,
                    help='Whether to use evidence (true/false)')
parser.add_argument('--gpu', '-g', required=True, type=int, default=-10,
                    help='GPU id to use (e.g. 0 for first GPU)')
args = parser.parse_args()
GPU_USE = args.gpu
BATCH_SIZE = 8  # You can adjust this based on your GPU memory
device = torch.device(f"cuda:{GPU_USE}" if torch.cuda.is_available() and GPU_USE >= 0 else "cpu")

model_path = "/home/models/deberta-xlarge-mnli"

EVIDENCE = args.evidence  # Set from command line argument
REFERENCE = False

# ---------------------------------------------------
# Helper functions with batch optimization
# ---------------------------------------------------
def extract_steps_batch(reasoning_traces):
    all_steps = []
    for reasoning_trace in reasoning_traces: 
        steps = []
        for line in reasoning_trace.split("\n"):
            match = re.match(r"R\d+: (.+)", line.strip())
            if match:
                steps.append(match.group(1))
        all_steps.append(steps)
    return all_steps

def alignment_score_batch(nli_pipeline, hypotheses, premises):
    inputs = [{"text": p, "text_pair": h} for h, p in zip(hypotheses, premises)]
    results = nli_pipeline(inputs)
    entailment_scores = []
    for result in results:
        scores = result if isinstance(result, list) else [result]
        entail_score = 0.0
        for r in scores:
            if isinstance(r, dict) and r.get("label", "").lower() == "entailment":
                entail_score = r.get("score", 0.0)
                break
        entailment_scores.append(entail_score)
    return entailment_scores

def alignment_score_token_batch(hypotheses, premises, tokenizer, model):
    h_inputs = tokenizer(hypotheses, return_tensors="pt", truncation=True, padding=True, max_length=128)
    s_inputs = tokenizer(premises, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Move to same device as model
    h_inputs = {k: v.to(device) for k, v in h_inputs.items()}
    s_inputs = {k: v.to(device) for k, v in s_inputs.items()}

    with torch.no_grad():
        h_outputs = model(**h_inputs)
        s_outputs = model(**s_inputs)

    scores = []
    for i in range(len(hypotheses)):
        h_embeds = h_outputs.last_hidden_state[i].cpu().numpy()
        s_embeds = s_outputs.last_hidden_state[i].cpu().numpy()
        pair_scores = []
        for h_vec in h_embeds:
            sims = cosine_similarity([h_vec], s_embeds)[0]
            pair_scores.append(np.max(sims))
        scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)
    return scores

# ---------------------------------------------------
# Batched ROSCOE–SA metric functions
# ---------------------------------------------------
def faithfulness_step_batch(entries, nli_pipeline):
    scores = []
    all_steps = extract_steps_batch([entry["reasoning_trace"] for entry in entries])
    all_evidence = [entry["evidence_text"] for entry in entries]

    for steps, evidence in zip(all_steps, all_evidence):
        if not steps or not evidence:
            scores.append(0.0)
            continue
        hypotheses, premises = [], []
        for h in steps:
            for s in evidence:
                hypotheses.append(h)
                premises.append(s)
        if not hypotheses:
            scores.append(0.0)
            continue
        entail_scores = alignment_score_batch(nli_pipeline, hypotheses, premises)
        step_scores = []
        step_idx = 0
        for h in steps:
            step_entail_scores = entail_scores[step_idx:step_idx + len(evidence)]
            step_scores.append(max(step_entail_scores) if step_entail_scores else 0.0)
            step_idx += len(evidence)
        scores.append(np.mean(step_scores))
    return scores

def faithfulness_token_batch(entries, tokenizer, model):
    scores = []
    all_steps = extract_steps_batch([entry["reasoning_trace"] for entry in entries])
    all_evidence = [entry["evidence_text"] for entry in entries]

    for steps, evidence in zip(all_steps, all_evidence):
        if not steps or not evidence:
            scores.append(0.0)
            continue
        hypotheses, premises = [], []
        for h in steps:
            for s in evidence:
                hypotheses.append(h)
                premises.append(s)
        if not hypotheses:
            scores.append(0.0)
            continue
        token_scores = alignment_score_token_batch(hypotheses, premises, tokenizer, model)
        step_scores = []
        step_idx = 0
        for h in steps:
            step_token_scores = token_scores[step_idx:step_idx + len(evidence)]
            step_scores.append(max(step_token_scores) if step_token_scores else 0.0)
            step_idx += len(evidence)
        scores.append(np.mean(step_scores))
    return scores

def info_step_batch(entries, nli_pipeline):
    scores = []
    all_steps = extract_steps_batch([entry["reasoning_trace"] for entry in entries])
    all_evidence = [entry["evidence_text"] for entry in entries]

    for steps, evidence in zip(all_steps, all_evidence):
        if not steps or not evidence:
            scores.append(0.0)
            continue
        hypotheses, premises = [], []
        for h in steps:
            for s in evidence:
                hypotheses.append(h)
                premises.append(s)
        if not hypotheses:
            scores.append(0.0)
            continue
        entail_scores = alignment_score_batch(nli_pipeline, hypotheses, premises)
        step_scores = []
        step_idx = 0
        for h in steps:
            step_entail_scores = entail_scores[step_idx:step_idx + len(evidence)]
            step_scores.append(np.mean(step_entail_scores) if step_entail_scores else 0.0)
            step_idx += len(evidence)
        scores.append(np.mean(step_scores))
    return scores

def repetition_token_batch(entries, tokenizer, model):
    scores = []
    all_steps = extract_steps_batch([entry["reasoning_trace"] for entry in entries])

    for steps in all_steps:
        if len(steps) < 2:
            scores.append(1.0)
            continue
        hypotheses, premises = [], []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                hypotheses.append(steps[i])
                premises.append(steps[j])
        if not hypotheses:
            scores.append(1.0)
            continue
        token_scores = alignment_score_token_batch(hypotheses, premises, tokenizer, model)
        max_sim = max(token_scores) if token_scores else 0.0
        scores.append(1 - max_sim)
    return scores

def calculate_roscoe_sa_batch(entries, nli_pipeline, tokenizer, model):
    results = []
    if EVIDENCE and not REFERENCE:
        faith_step_scores = faithfulness_step_batch(entries, nli_pipeline)
        faith_token_scores = faithfulness_token_batch(entries, tokenizer, model)
        info_scores = info_step_batch(entries, nli_pipeline)
        repetition_scores = repetition_token_batch(entries, tokenizer, model)
        for i, entry in enumerate(entries):
            try:
                scores = {
                    "Faithfulness-Step": faith_step_scores[i],
                    "Faithfulness-Token": faith_token_scores[i],
                    "Info-Step": info_scores[i],
                    "Repetition-Token": repetition_scores[i],
                }
                mean_score = sum(scores.values()) / len(scores)
                entry["roscoe_sa_scores"] = scores
                entry["mean_ROSCOE_SA"] = float(mean_score)
                results.append(entry)
            except Exception:
                entry["roscoe_sa_scores"] = {
                    "Faithfulness-Step": -1,
                    "Faithfulness-Token": -1,
                    "Info-Step": -1,
                    "Repetition-Token": -1,
                }
                entry["mean_ROSCOE_SA"] = 0.0
                results.append(entry)
    elif not EVIDENCE and not REFERENCE:
        repetition_scores = repetition_token_batch(entries, tokenizer, model)
        for i, entry in enumerate(entries):
            try:
                scores = {
                    "Repetition-Token": repetition_scores[i],
                }
                mean_score = sum(scores.values()) / len(scores)
                entry["roscoe_sa_scores"] = scores
                entry["mean_ROSCOE_SA"] = float(mean_score)
                results.append(entry)
            except Exception:
                entry["roscoe_sa_scores"] = {"Repetition-Token": -1}
                entry["mean_ROSCOE_SA"] = 0.0
                results.append(entry)
    return results

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main(input_file, output_file):
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        entries = json.load(f)

    print("Loading models...")
    tok = AutoTokenizer.from_pretrained(model_path)
    nli_model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
    nli_model.eval()
    nli_model.to(device)  # <<< move model to GPU/CPU

    nli_pipeline = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=GPU_USE,
        top_k=None,
        truncation=True,
        max_length=128,
        batch_size=BATCH_SIZE,
        padding=True
    )

    batch_size = BATCH_SIZE
    total_batches = (len(entries) + batch_size - 1) // batch_size
    results = []
    for i in tqdm(range(0, len(entries), batch_size), desc="Processing batches", total=total_batches):
        batch_entries = entries[i:i + batch_size]
        batch_results = calculate_roscoe_sa_batch(batch_entries, nli_pipeline, tok, nli_model)
        results.extend(batch_results)

    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    mean_scores = [x["mean_ROSCOE_SA"] for x in results if x["mean_ROSCOE_SA"] > 0]
    print(f"\nProcessing complete!")
    print(f"Total entries processed: {len(results)}")
    print(f"Entries with valid scores: {len(mean_scores)}")
    print(f"Overall mean ROSCOE-SA: {np.mean(mean_scores):.5f}" if mean_scores else "No valid scores")

if __name__ == "__main__":
    main(args.input, args.output)

# python ROSCOE-SA_processor.py --input "/home/ojas/scripts/datasets/Human_Eval/sample_exps/test.json" --output "/home/ojas/scripts/datasets/Human_Eval/sample_exps/test_sa_org.json" --evidence "True" --gpu "0"

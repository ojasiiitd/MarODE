import re
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(
    description="Calculate ROSCOE-SA baseline metrics."
)

parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", required=True)
parser.add_argument("--model-path", required=True, help="Path to MNLI model")
parser.add_argument("--gpu", "-g", type=int, default=-1)
parser.add_argument("--evidence", "-e", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--batch-size", type=int, default=8)

args = parser.parse_args()

DEVICE = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
)

USE_EVIDENCE = args.evidence
BATCH_SIZE = args.batch_size


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def extract_steps_batch(reasoning_traces):
    all_steps = []
    for reasoning_trace in reasoning_traces:
        steps = []
        for line in reasoning_trace.split("\n"):
            match = re.match(r"R\d+:\s*(.+)", line.strip())
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
            if r.get("label", "").lower() == "entailment":
                entail_score = r.get("score", 0.0)
                break
        entailment_scores.append(entail_score)

    return entailment_scores


def alignment_score_token_batch(hypotheses, premises, tokenizer, model):
    h_inputs = tokenizer(
        hypotheses, return_tensors="pt", truncation=True,
        padding=True, max_length=128
    ).to(DEVICE)

    s_inputs = tokenizer(
        premises, return_tensors="pt", truncation=True,
        padding=True, max_length=128
    ).to(DEVICE)

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
# ROSCOE-SA Components
# ---------------------------------------------------
def faithfulness_step_batch(entries, nli_pipeline):
    scores = []
    all_steps = extract_steps_batch([e["reasoning_trace"] for e in entries])
    all_evidence = [e.get("evidence_text", []) for e in entries]

    for steps, evidence in zip(all_steps, all_evidence):
        if not steps or not evidence:
            scores.append(0.0)
            continue

        hypotheses, premises = [], []
        for h in steps:
            for s in evidence:
                hypotheses.append(h)
                premises.append(s)

        entail_scores = alignment_score_batch(nli_pipeline, hypotheses, premises)

        step_scores = []
        idx = 0
        for _ in steps:
            step_entails = entail_scores[idx:idx + len(evidence)]
            step_scores.append(max(step_entails) if step_entails else 0.0)
            idx += len(evidence)

        scores.append(np.mean(step_scores))

    return scores


def faithfulness_token_batch(entries, tokenizer, model):
    scores = []
    all_steps = extract_steps_batch([e["reasoning_trace"] for e in entries])
    all_evidence = [e.get("evidence_text", []) for e in entries]

    for steps, evidence in zip(all_steps, all_evidence):
        if not steps or not evidence:
            scores.append(0.0)
            continue

        hypotheses, premises = [], []
        for h in steps:
            for s in evidence:
                hypotheses.append(h)
                premises.append(s)

        token_scores = alignment_score_token_batch(hypotheses, premises, tokenizer, model)

        step_scores = []
        idx = 0
        for _ in steps:
            step_tokens = token_scores[idx:idx + len(evidence)]
            step_scores.append(max(step_tokens) if step_tokens else 0.0)
            idx += len(evidence)

        scores.append(np.mean(step_scores))

    return scores


def info_step_batch(entries, nli_pipeline):
    scores = []
    all_steps = extract_steps_batch([e["reasoning_trace"] for e in entries])
    all_evidence = [e.get("evidence_text", []) for e in entries]

    for steps, evidence in zip(all_steps, all_evidence):
        if not steps or not evidence:
            scores.append(0.0)
            continue

        hypotheses, premises = [], []
        for h in steps:
            for s in evidence:
                hypotheses.append(h)
                premises.append(s)

        entail_scores = alignment_score_batch(nli_pipeline, hypotheses, premises)

        step_scores = []
        idx = 0
        for _ in steps:
            step_entails = entail_scores[idx:idx + len(evidence)]
            step_scores.append(np.mean(step_entails) if step_entails else 0.0)
            idx += len(evidence)

        scores.append(np.mean(step_scores))

    return scores


def repetition_token_batch(entries, tokenizer, model):
    scores = []
    all_steps = extract_steps_batch([e["reasoning_trace"] for e in entries])

    for steps in all_steps:
        if len(steps) < 2:
            scores.append(1.0)
            continue

        hypotheses, premises = [], []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                hypotheses.append(steps[i])
                premises.append(steps[j])

        token_scores = alignment_score_token_batch(hypotheses, premises, tokenizer, model)
        max_sim = max(token_scores) if token_scores else 0.0
        scores.append(1 - max_sim)

    return scores


def calculate_roscoe_sa_batch(entries, nli_pipeline, tokenizer, model):
    results = []

    if USE_EVIDENCE:
        faith_step = faithfulness_step_batch(entries, nli_pipeline)
        faith_token = faithfulness_token_batch(entries, tokenizer, model)
        info = info_step_batch(entries, nli_pipeline)
        repetition = repetition_token_batch(entries, tokenizer, model)

        for i, entry in enumerate(entries):
            scores = {
                "Faithfulness-Step": faith_step[i],
                "Faithfulness-Token": faith_token[i],
                "Info-Step": info[i],
                "Repetition-Token": repetition[i],
            }
            entry["roscoe_sa_scores"] = scores
            entry["mean_ROSCOE_SA"] = float(sum(scores.values()) / 4)
            results.append(entry)

    else:
        repetition = repetition_token_batch(entries, tokenizer, model)
        for i, entry in enumerate(entries):
            scores = {"Repetition-Token": repetition[i]}
            entry["roscoe_sa_scores"] = scores
            entry["mean_ROSCOE_SA"] = float(repetition[i])
            results.append(entry)

    return results


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    with open(args.input, "r") as f:
        entries = json.load(f)

    print("Loading models...")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    nli_model_embed = AutoModel.from_pretrained(args.model_path, output_hidden_states=True).to(DEVICE)
    nli_model_embed.eval()

    nli_pipeline = pipeline(
        "text-classification",
        model=args.model_path,
        tokenizer=args.model_path,
        device=args.gpu,
        top_k=None,
        truncation=True,
        max_length=128,
        batch_size=BATCH_SIZE,
        padding=True,
    )

    results = []

    total_batches = (len(entries) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(entries), BATCH_SIZE), total=total_batches, desc="Processing"):
        batch = entries[i:i + BATCH_SIZE]
        batch_results = calculate_roscoe_sa_batch(batch, nli_pipeline, tok, nli_model_embed)
        results.extend(batch_results)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    valid_scores = [x["mean_ROSCOE_SA"] for x in results if x["mean_ROSCOE_SA"] > 0]

    print("\n=== ROSCOE-SA Results ===")
    print(f"Total entries: {len(results)}")
    print(f"Valid entries: {len(valid_scores)}")
    if valid_scores:
        print(f"Overall mean ROSCOE-SA: {np.mean(valid_scores):.5f}")


if __name__ == "__main__":
    main()

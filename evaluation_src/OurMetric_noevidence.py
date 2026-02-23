from __future__ import annotations
import re
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import argparse
import json
from datasets import Dataset
import math

model_path = "/home/models/deberta-xlarge-mnli"

# ---------------------------------------------------
# Configuration (Evidence removed)
# ---------------------------------------------------
@dataclass
class BlockWeights:
    coherence: float = 0.5
    quality: float = 0.5   # evidence weight removed

@dataclass
class RandomWalkConfig:
    order_n: int = 2
    n_walks: int = 256
    max_len: Optional[int] = None
    temperature: float = 0.5
    self_loops: bool = False

@dataclass
class GranularityConfig:
    ideal_steps_low: int = 4
    ideal_steps_high: int = 5
    max_sent_per_step: int = 3
    max_seq_len: int = 256

@dataclass
class EvaluatorConfig:
    embed_model_name: str = "/home/models/sentence-transformers--all-MiniLM-L6-v2"
    nli_model_name: str = "/home/models/deberta-xlarge-mnli"
    random_walk: RandomWalkConfig = field(default_factory=RandomWalkConfig)
    granularity: GranularityConfig = field(default_factory=GranularityConfig)
    weights: BlockWeights = field(default_factory=BlockWeights)

# ---------------------------------------------------
# Embedding + NLI helpers
# ---------------------------------------------------
class EmbeddingBackend:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name).to(f"cuda:{GPU_USE}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts, convert_to_numpy=True,
            normalize_embeddings=True, batch_size=32
        )

    def tokenize(self, text: str) -> List[str]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.convert_ids_to_tokens(ids)

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b.T)

class NLICrossEncoder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(f"cuda:{GPU_USE}")
        self.model.eval()
        self.label_ix = {"contradiction": 0, "entailment": 1, "neutral": 2}

    @torch.no_grad()
    def probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        tok = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512)
        tok = {k: v.to(f"cuda:{GPU_USE}") for k, v in tok.items()}
        out = self.model(**tok)
        pr = F.softmax(out.logits, dim=-1).detach().cpu().numpy()[0]
        return {
            "contradiction": float(pr[self.label_ix["contradiction"]]),
            "entailment": float(pr[self.label_ix["entailment"]]),
            "neutral": float(pr[self.label_ix["neutral"]]),
        }

# ---------------------------------------------------
# Block 1: Coherence
# ---------------------------------------------------
def build_transition_matrix(step_embs: np.ndarray, order_n: int, temperature: float, self_loops: bool) -> np.ndarray:
    S = EmbeddingBackend.cosine_sim(step_embs, step_embs)
    np.fill_diagonal(S, -1.0 if not self_loops else S.diagonal())
    logits = S / max(temperature, 1e-6)
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True)

def simulate_walks(steps: List[str], P: np.ndarray, cfg: RandomWalkConfig) -> float:
    k = len(steps)
    indices = list(range(k))
    walk_scores = []

    start_pool = [i for i in indices if i < max(1, k // 3)]

    for _ in range(cfg.n_walks):
        curr = random.choice(start_pool)
        path = [curr]
        max_len = cfg.max_len or k

        while len(path) < max_len:
            row = P[curr].copy()
            for j in indices:
                if j in path:
                    row[j] *= 0.75
            row = np.maximum(row, 0)
            if row.sum() <= 0:
                break
            row /= row.sum()
            nxt = np.random.choice(indices, p=row)
            path.append(nxt)
            curr = nxt

        if len(path) > 1:
            rewards = 0
            for i in range(len(path) - 1):
                step = path[i+1] - path[i]
                if step in (-1, 1):
                    if i > 0:
                        prev_step = path[i] - path[i-1]
                        if prev_step == -step:
                            continue
                    rewards += 1
            score = rewards / (len(path) - 1)
        else:
            score = 0.0
        walk_scores.append(score)

    return float(np.mean(walk_scores)) if walk_scores else 0.0

# ---------------------------------------------------
# Block 2: Redundancy (Quality part 1)
# ---------------------------------------------------
def redundancy_score(steps, embed_backend=None, cfg=None, device="cuda:0"):
    def regex_tokens(text: str):
        return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))

    if not steps or len(steps) < 2:
        return 1.0

    tokens = [regex_tokens(s) for s in steps]

    penalties = []
    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = tokens[i], tokens[j]
            diff = ti.symmetric_difference(tj)
            d = len(diff)

            if d == 0:
                penalty = 1.0
            else:
                penalty = max(0.0, round(np.exp(-d / 4), 2))

            penalties.append(penalty)

    avg_penalty = sum(penalties) / len(penalties) if penalties else 0.0
    return (1.0 - avg_penalty)

# ---------------------------------------------------
# Block 3 REMOVED (Evidence Matching)
# ---------------------------------------------------

# ---------------------------------------------------
# Block 4: Verdict Score
# ---------------------------------------------------
class VerdictScorerDifferential:
    def __init__(self, nli_model: NLICrossEncoder, embed: EmbeddingBackend, cfg=None):
        self.nli = nli_model
        self.embed = embed
        if cfg is None:
            class C: pass
            cfg = C()
        self.r = getattr(cfg, "ode_r", 2.0)
        self.dt = getattr(cfg, "ode_dt", 1.0)
        self.eps = 1e-6

    def _compute_per_step_signals(self, steps: List[str]):
        n = len(steps)
        entail_list, contra_list = [], []
        for i in range(n):
            if i == 0:
                entail_list.append(0.0)
                contra_list.append(0.0)
            else:
                probs = self.nli.probs(steps[i-1], steps[i])
                entail_list.append(float(probs.get("entailment", 0.0)))
                contra_list.append(float(probs.get("contradiction", 0.0)))
        return entail_list, contra_list

    def _ode_rhs(self, p: float, S: float) -> float:
        return self.r * (S - 0.5) * p * (1.0 - p)

    def _rk4_step(self, p: float, S: float, dt: float) -> float:
        k1 = self._ode_rhs(p, S)
        k2 = self._ode_rhs(p + 0.5*dt*k1, S)
        k3 = self._ode_rhs(p + 0.5*dt*k2, S)
        k4 = self._ode_rhs(p + dt*k3, S)
        p_next = p + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return float(max(self.eps, min(1.0 - self.eps, p_next)))

    def score(self, reasoning_steps: List[str], claim: str, gold_label: str, integrator: str = "rk4") -> float:
        if not reasoning_steps or len(reasoning_steps) < 1:
            return 0.0

        entail_list, contra_list = self._compute_per_step_signals(reasoning_steps)
        p = 0.5

        for i in range(len(reasoning_steps)):
            S_raw = entail_list[i] - contra_list[i]
            S = max(0.0, min(1.0, (S_raw + 1.0) / 2.0))
            if integrator == "rk4":
                p = self._rk4_step(p, S, self.dt)
            else:
                dp = self._ode_rhs(p, S) * self.dt
                p = float(max(self.eps, min(1.0 - self.eps, p + dp)))

        return float(max(0.0, min(1.0, p)))

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def normalize_reasoning_steps(reasoning_trace: str) -> List[str]:
    steps = []
    for line in reasoning_trace.splitlines():
        line = line.strip()
        if not line:
            continue

        line = re.sub(r'^\s*R\d+\s*:\s*', '', line)
        line = re.sub(r'^\s*(sent\d+|int\d+)\s*:\s*', '', line)

        if line:
            steps.append(line)
    return steps

def coherence_batch(entries, embed_backend, cfg):
    scores = []
    for entry in entries:
        steps = normalize_reasoning_steps(entry["reasoning_trace"])
        if len(steps) < 2:
            scores.append(1.0)
            continue
        step_embs = embed_backend.encode(steps)
        P = build_transition_matrix(
            step_embs,
            cfg.random_walk.order_n,
            cfg.random_walk.temperature,
            cfg.random_walk.self_loops
        )
        coherence_score = simulate_walks(steps, P, cfg.random_walk)
        scores.append(coherence_score)
    return scores

def concision_batch(entries, embed_backend, cfg, device):
    scores = []
    for entry in entries:
        steps = normalize_reasoning_steps(entry["reasoning_trace"])
        if len(steps) <= 1:
            scores.append(1.0)
            continue

        redundancy_value = redundancy_score(
            steps, embed_backend, cfg.granularity, device
        )
        scores.append(redundancy_value)
    return scores

def verdict_batch(entries, nli_backend, embed_backend, cfg):
    scores = []
    scorer = VerdictScorerDifferential(nli_backend, embed_backend, cfg)

    for entry in entries:
        steps = normalize_reasoning_steps(entry["reasoning_trace"])
        verdict_score = scorer.score(
            steps, entry["claim"], entry["label"]
        )
        scores.append(verdict_score)
    return scores

# ---------------------------------------------------
# Final Metric (Evidence removed completely)
# ---------------------------------------------------
def calculate_ourmetric_batch(entries, embed_backend, nli_backend, cfg):
    results = []

    coherence_scores = coherence_batch(entries, embed_backend, cfg)
    redundancy_scores = concision_batch(entries, embed_backend, cfg, device=f"cuda:{GPU_USE}")
    verdict_scores = verdict_batch(entries, nli_backend, embed_backend, cfg)

    for i, entry in enumerate(entries):
        block2_weight = 0.5
        block4_weight = 0.5

        quality = (block2_weight * redundancy_scores[i]) + (block4_weight * verdict_scores[i])

        ourmetric = {
            "coherence_score": float(coherence_scores[i]),
            "quality_score": float(quality),
            "total_score": float(
                (cfg.weights.coherence * coherence_scores[i])
                + (cfg.weights.quality * quality)
            )
        }

        entry["ourmetric"] = ourmetric
        results.append(entry)

    return results

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main(input_file, output_file):
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        entries = json.load(f)

    dataset = Dataset.from_list(entries)

    config = EvaluatorConfig()

    embed_backend = EmbeddingBackend(config.embed_model_name)
    nli_backend = NLICrossEncoder(config.nli_model_name)

    def process_batch(batch):
        entries_batch = []
        for i in range(len(batch["claim"])):
            entry = {key: batch[key][i] for key in batch}
            entries_batch.append(entry)

        results = calculate_ourmetric_batch(entries_batch, embed_backend, nli_backend, config)

        out = {}
        for key in results[0].keys():
            out[key] = [r[key] for r in results]
        return out

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=16,
        desc="Calculating our metrics"
    )

    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)

    print("\nProcessing complete!")
    print(f"Total entries processed: {len(dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input file and generate output file with our metrics.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file')
    parser.add_argument('--output', '-o', required=True, help='Path to the output JSON file')
    parser.add_argument('--gpu', '-g', required=True, type=int, default=-10, help='GPU id to use (e.g. 0 for first GPU)')
    args = parser.parse_args()
    GPU_USE = args.gpu

    if GPU_USE >= 0 and torch.cuda.is_available():
        print(f"Setting CUDA device to {GPU_USE}")
        torch.cuda.set_device(GPU_USE)

    main(args.input, args.output)

# python OurMetric_noevidence.py --input /home/ojas/scripts/datasets/Human_Eval/math_random_150_R4-R10_BASELINES.json --output /home/ojas/scripts/datasets/Human_Eval/math_random_150_R4-R10_BASELINES_check.json --gpu 0

# python OurMetric_noevidence.py --input /home/ojas/scripts/datasets/Human_Eval/pubhealth_random_150_R4-10_BASELINES.json --output /home/ojas/scripts/datasets/Human_Eval/pubhealth_random_150_R4-10_BASELINES.json --gpu 0
from __future__ import annotations
import re
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import argparse
import json
from datasets import Dataset
import math

# ---------------------------------------------------
# Device Helper
# ---------------------------------------------------
def get_device(gpu_index: int) -> torch.device:
    if gpu_index >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
@dataclass
class BlockWeights:
    coherence: float = 0.333333
    quality: float = 0.333333
    evidence: float = 0.3333333

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
class EvidenceConfig:
    sim_threshold: float = 0.45
    entail_weight: float = 0.5
    contra_weight: float = 0.5

@dataclass
class EvaluatorConfig:
    embed_model_name: str = "/home/models/sentence-transformers--all-MiniLM-L6-v2"
    nli_model_name: str = "/home/models/deberta-xlarge-mnli"
    random_walk: RandomWalkConfig = field(default_factory=RandomWalkConfig)
    granularity: GranularityConfig = field(default_factory=GranularityConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    weights: BlockWeights = field(default_factory=BlockWeights)

# ---------------------------------------------------
# Embedding + NLI Backends
# ---------------------------------------------------
class EmbeddingBackend:
    def __init__(self, model_name: str, device: torch.device):
        self.model = SentenceTransformer(model_name, device=str(device))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts, convert_to_numpy=True,
            normalize_embeddings=True, batch_size=32
        )

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b.T)

class NLICrossEncoder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.label_ix = {"contradiction": 0, "entailment": 1, "neutral": 2}

    @torch.no_grad()
    def probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        tok = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True,
            padding=True, max_length=256
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.model(**tok)
        pr = F.softmax(out.logits, dim=-1).cpu().numpy()[0]
        return {
            "contradiction": float(pr[0]),
            "entailment": float(pr[1]),
            "neutral": float(pr[2]),
        }

# ---------------------------------------------------
# Block 1: Coherence
# ---------------------------------------------------
def build_transition_matrix(step_embs, temperature, self_loops):
    S = EmbeddingBackend.cosine_sim(step_embs, step_embs)
    np.fill_diagonal(S, -1.0 if not self_loops else S.diagonal())
    logits = S / max(temperature, 1e-6)
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True)

def simulate_walks(steps, P, cfg):
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
                        prev = path[i] - path[i-1]
                        if prev == -step:
                            continue
                    rewards += 1
            score = rewards / (len(path) - 1)
        else:
            score = 0.0
        walk_scores.append(score)

    return float(np.mean(walk_scores)) if walk_scores else 0.0

# ---------------------------------------------------
# Block 2: Redundancy
# ---------------------------------------------------
def redundancy_score(steps):
    if len(steps) < 2:
        return 1.0

    def tokens(t):
        return set(re.findall(r"\b[a-zA-Z0-9]+\b", t.lower()))

    token_sets = [tokens(s) for s in steps]
    penalties = []

    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            d = len(token_sets[i].symmetric_difference(token_sets[j]))
            penalty = 1.0 if d == 0 else max(0.0, round(np.exp(-d / 4), 2))
            penalties.append(penalty)

    avg_penalty = sum(penalties) / len(penalties) if penalties else 0.0
    return 1.0 - avg_penalty

# ---------------------------------------------------
# Block 3: Evidence
# ---------------------------------------------------
def evidence_alignment_score(steps, evidences, embed, nli, cfg):
    if not evidences:
        return 0.5

    step_embs = embed.encode(steps)
    ev_embs = embed.encode(evidences)
    sim_matrix = EmbeddingBackend.cosine_sim(step_embs, ev_embs)

    scores = []
    for i, step in enumerate(steps):
        j = np.argmax(sim_matrix[i])
        sim_val = sim_matrix[i, j]
        score = 0.0
        if sim_val >= cfg.sim_threshold:
            score = 0.5
            probs = nli.probs(evidences[j], step)
            score += cfg.entail_weight * probs["entailment"]
            score -= cfg.contra_weight * probs["contradiction"]
            score = max(0.0, min(1.0, score))
        scores.append(score)

    return sum(scores) / len(scores)

# ---------------------------------------------------
# Block 4: ODE Verdict
# ---------------------------------------------------
class VerdictScorerDifferential:
    def __init__(self, nli_model):
        self.nli = nli_model
        self.r = 2.0
        self.dt = 1.0
        self.eps = 1e-6

    def _ode_rhs(self, p, S):
        return self.r * (S - 0.5) * p * (1 - p)

    def _rk4_step(self, p, S):
        k1 = self._ode_rhs(p, S)
        k2 = self._ode_rhs(p + 0.5*self.dt*k1, S)
        k3 = self._ode_rhs(p + 0.5*self.dt*k2, S)
        k4 = self._ode_rhs(p + self.dt*k3, S)
        p_next = p + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return float(max(self.eps, min(1 - self.eps, p_next)))

    def score(self, steps):
        if len(steps) < 1:
            return 0.0

        p = 0.5
        for i in range(1, len(steps)):
            probs = self.nli.probs(steps[i-1], steps[i])
            S_raw = probs["entailment"] - probs["contradiction"]
            S = max(0.0, min(1.0, (S_raw + 1)/2))
            p = self._rk4_step(p, S)

        return float(max(0.0, min(1.0, p)))

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def normalize_reasoning_steps(text):
    steps = []
    for line in text.splitlines():
        line = re.sub(r'^\s*(R\d+|sent\d+|int\d+)\s*:\s*', '', line.strip())
        if line:
            steps.append(line)
    return steps

def main(input_file, output_file, device):
    with open(input_file, "r") as f:
        entries = json.load(f)

    config = EvaluatorConfig()
    embed = EmbeddingBackend(config.embed_model_name, device)
    nli = NLICrossEncoder(config.nli_model_name, device)
    verdict_model = VerdictScorerDifferential(nli)

    for entry in entries:
        steps = normalize_reasoning_steps(entry["reasoning_trace"])

        if len(steps) < 2:
            coherence = 1.0
        else:
            embs = embed.encode(steps)
            P = build_transition_matrix(
                embs,
                config.random_walk.temperature,
                config.random_walk.self_loops
            )
            coherence = simulate_walks(steps, P, config.random_walk)

        redundancy = redundancy_score(steps)
        evidence = evidence_alignment_score(
            steps, entry["evidence_text"], embed, nli, config.evidence
        )
        verdict = verdict_model.score(steps)

        quality = 0.5 * redundancy + 0.5 * verdict
        total = (
            config.weights.coherence * coherence +
            config.weights.quality * quality +
            config.weights.evidence * evidence
        )

        entry["ourmetric"] = {
            "coherence_score": float(coherence),
            "quality_score": float(quality),
            "evidence_score": float(evidence),
            "total_score": float(total)
        }

    with open(output_file, "w") as f:
        json.dump(entries, f, indent=2)

    print("MarODE scoring complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    device = get_device(args.gpu)
    main(args.input, args.output, device)
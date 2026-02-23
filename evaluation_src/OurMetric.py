from __future__ import annotations
import re
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import argparse
import json
from datasets import Dataset
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model_path = "/home/models/deberta-xlarge-mnli"

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
        tok = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=256)
        tok = {k: v.to(f"cuda:{GPU_USE}") for k, v in tok.items()}
        out = self.model(**tok)
        pr = F.softmax(out.logits, dim=-1).detach().cpu().numpy()[0]
        return {
            "contradiction": float(pr[self.label_ix["contradiction"]]),
            "entailment": float(pr[self.label_ix["entailment"]]),
            "neutral": float(pr[self.label_ix["neutral"]]),
        }

# ---------------------------------------------------
# Block 1: Coherence functions
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

    # restrict starting pool to first third
    start_pool = [i for i in indices if i < max(1, k // 3)]

    for _ in range(cfg.n_walks):
        curr = random.choice(start_pool)
        path = [curr]
        max_len = cfg.max_len or k

        while len(path) < max_len:
            row = P[curr].copy()
            for j in indices:
                if j in path:
                    row[j] *= 0.75   # discourage repeats
            row = np.maximum(row, 0)
            if row.sum() <= 0:
                break
            row /= row.sum()
            nxt = np.random.choice(indices, p=row)
            path.append(nxt)
            curr = nxt

        # ---- scoring: reward i -> i±1 transitions, ignore oscillations ----
        if len(path) > 1:
            rewards = 0
            for i in range(len(path) - 1):
                step = path[i+1] - path[i]
                if step in (-1, 1):  # forward or backward
                    if i > 0:
                        prev_step = path[i] - path[i-1]
                        if prev_step == -step:
                            continue  # ignore oscillation
                    rewards += 1
            score = rewards / (len(path) - 1)
        else:
            score = 0.0
        walk_scores.append(score)

    return float(np.mean(walk_scores)) if walk_scores else 0.0

# # ---------------------------------------------------
# # Block 2: Concision functions
# # ---------------------------------------------------
# def concision_score_only(steps: List[str], cfg: GranularityConfig, nli_model=None, tokenizer=None, device="cuda:0"):
#     if nli_model is None or tokenizer is None:
#         return "Please Load NLI/Tokenizer model"

#     if len(steps) < 2:
#         return 1.0

#     scores = []
#     for i in range(len(steps) - 1):
#         premise, hypothesis = steps[i], steps[i+1]

#         # Forward
#         inputs_fw = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len).to(device)
#         with torch.no_grad():
#             probs_fw = torch.softmax(nli_model(**inputs_fw).logits, dim=-1).cpu().numpy()[0]
#         contra_fw, entail_fw, neut_fw = probs_fw[0], probs_fw[1], probs_fw[2]

#         # Backward
#         inputs_bw = tokenizer(hypothesis, premise, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len).to(device)
#         with torch.no_grad():
#             probs_bw = torch.softmax(nli_model(**inputs_bw).logits, dim=-1).cpu().numpy()[0]
#         contra_bw, entail_bw, neut_bw = probs_bw[0], probs_bw[1], probs_bw[2]

#         # assign concision quality scores
#         if contra_fw > 0.5 or contra_bw > 0.5:
#             scores.append(0.001)        # contradiction = low quality
#         elif entail_fw > 0.7 and entail_bw > 0.7:
#             scores.append(0.002)        # paraphrase = still low quality
#         elif neut_fw > 0.5 and neut_bw > 0.5:
#             scores.append(0.003)        # unrelated = moderate
#         else:
#             scores.append(1.000)        # mixed = pretty good concision
    
#     return sum(scores) / len(scores) , scores

# ---------------------------------------------------
# Block 2: Redundancy Checker
# ---------------------------------------------------
def redundancy_score(steps, embed_backend=None, cfg=None, device="cuda:0"):
    """
    Penalize redundancy based on how many words differ between steps.
    Uses regex-based tokenization (no model tokenizer).
    - Exact same steps -> max penalty
    - Few word differences -> moderate penalty
    - Many word differences -> low/no penalty
    Only upper triangle of pairwise comparisons is considered.
    """
    def regex_tokens(text: str):
        """Extract alphanumeric words, lowercase them."""
        return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))

    if not steps or len(steps) < 2:
        return 1.0  # no redundancy with 0 or 1 step

    # tokenize each step into sets of regex tokens
    tokens = [regex_tokens(s) for s in steps]

    penalties = []
    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, n):  # upper triangle only
            ti, tj = tokens[i], tokens[j]
            diff = ti.symmetric_difference(tj)
            d = len(diff)

            # define graded penalties
            if d == 0:  
                penalty = 1.0   # identical
            else:
                # Exponential decay: penalty decreases as difference grows
                penalty = max(0.0, round(np.exp(-d / 4), 2))   # update this value 5 ... 60

            penalties.append(penalty)
    
    # print(penalties)

    avg_penalty = sum(penalties) / len(penalties) if penalties else 0.0
    return (1.0 - avg_penalty)

# ---------------------------------------------------
# Block 3: Evidence Matching
# ---------------------------------------------------
def evidence_alignment_score(steps: List[str], evidences: List[str],
                             embed: EmbeddingBackend,
                             nli: Optional[NLICrossEncoder],
                             cfg: EvidenceConfig) -> float:
    if not evidences:
        return 0.5  # neutral if no evidence given

    step_embs = embed.encode(steps)
    ev_embs = embed.encode(evidences)
    sim_matrix = EmbeddingBackend.cosine_sim(step_embs, ev_embs)

    # For each step, find max matching evidence
    align_scores = []
    for i, step in enumerate(steps):
        j = np.argmax(sim_matrix[i])
        sim_val = sim_matrix[i, j]
        score = 0.0
        if sim_val >= cfg.sim_threshold:
            score = 0.5  # base score for similarity
            if nli is not None:
                nli_probs = nli.probs(evidences[j], step)
                score += cfg.entail_weight * nli_probs["entailment"]
                score -= cfg.contra_weight * nli_probs["contradiction"]
                score = max(0.0, min(1.0, score))
        align_scores.append(score)

    return sum(align_scores) / max(len(align_scores), 1)


# ---------------------------------------------------
# Block 4: Verdict Scorer (OLD before 23 Oct 2025)
# ---------------------------------------------------

# class VerdictScorerDifferential:
#     def __init__(self, nli_model: NLICrossEncoder, embed: EmbeddingBackend, cfg=None):
#         self.nli = nli_model
#         self.embed = embed
#         if cfg is None:
#             class C: pass
#             cfg = C()
#         self.r = getattr(cfg, "ode_r", 2.0)
#         self.dt = getattr(cfg, "ode_dt", 1.0)
#         self.eps = 1e-6

#     def _compute_per_step_signals(self, steps: List[str]):
#         n = len(steps)
#         entail_list, contra_list = [], []
#         for i in range(n):
#             if i == 0:
#                 entail_list.append(0.0)
#                 contra_list.append(0.0)
#             else:
#                 probs = self.nli.probs(steps[i-1], steps[i])
#                 entail_list.append(float(probs.get("entailment", 0.0)))
#                 contra_list.append(float(probs.get("contradiction", 0.0)))
#         return entail_list, contra_list

#     def _ode_rhs(self, p: float, S: float) -> float:
#         return self.r * (S - 0.5) * p * (1.0 - p)

#     def _rk4_step(self, p: float, S: float, dt: float) -> float:
#         k1 = self._ode_rhs(p, S)
#         k2 = self._ode_rhs(p + 0.5*dt*k1, S)
#         k3 = self._ode_rhs(p + 0.5*dt*k2, S)
#         k4 = self._ode_rhs(p + dt*k3, S)
#         p_next = p + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
#         return float(max(self.eps, min(1.0 - self.eps, p_next)))

#     def score(self, reasoning_steps: List[str], claim: str, evidences: List[str], gold_label: str, integrator: str = "rk4") -> float:
#         if not reasoning_steps or len(reasoning_steps) < 1:
#             return 0.0

#         entail_list, contra_list = self._compute_per_step_signals(reasoning_steps)
#         p = 0.5  # initial belief

#         for i in range(len(reasoning_steps)):
#             S_raw = entail_list[i] - contra_list[i]
#             S = max(0.0, min(1.0, (S_raw + 1.0) / 2.0))
#             if integrator == "rk4":
#                 p = self._rk4_step(p, S, self.dt)
#             else:
#                 dp = self._ode_rhs(p, S) * self.dt
#                 p = float(max(self.eps, min(1.0 - self.eps, p + dp)))

#         final_probs = self.nli.probs(reasoning_steps[-1], claim) if self.nli else {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
#         p_e = final_probs.get("entailment", 0.0)
#         p_c = final_probs.get("contradiction", 0.0)
#         p_n = final_probs.get("neutral", 0.0)
        
#         label_mapping = {
#             "SUPPORTED": p_e,
#             "REFUTED": p_c,
#             "PARTLY SUPPORTED": 1 - math.tanh(2 * abs(p_e - 0.5)), # updated arghodeep 27/9/2025
#             "PARTLY REFUTED":1 - math.tanh(2 * abs(p_c - 0.5)), # updated arghodeep 27/9/2025
#             "UNVERIFIABLE": p_n,
#         }

#         scores = {lbl: p * val for lbl, val in label_mapping.items()}
#         final_score = scores.get(gold_label, 0.0)
#         return float(max(0.0, min(1.0, final_score)))

# Removed Label Mapping
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

    def score(self, reasoning_steps: List[str], claim: str, evidences: List[str], gold_label: str, integrator: str = "rk4") -> float:
        if not reasoning_steps or len(reasoning_steps) < 1:
            return 0.0

        entail_list, contra_list = self._compute_per_step_signals(reasoning_steps)
        p = 0.5  # initial belief

        for i in range(len(reasoning_steps)):
            S_raw = entail_list[i] - contra_list[i]
            S = max(0.0, min(1.0, (S_raw + 1.0) / 2.0))
            if integrator == "rk4":
                p = self._rk4_step(p, S, self.dt)
            else:
                dp = self._ode_rhs(p, S) * self.dt
                p = float(max(self.eps, min(1.0 - self.eps, p + dp)))

        # Directly use the final integrated probability as the verdict score
        final_score = float(max(0.0, min(1.0, p)))

        return final_score

# ---------------------------------------------------
# Batch Processing Functions (continued)
# ---------------------------------------------------
def normalize_reasoning_steps(reasoning_trace: str) -> List[str]:
    steps = []
    for line in reasoning_trace.splitlines():
        line = line.strip()
        if not line:
            continue

        # First remove R<number>: prefix
        line = re.sub(r'^\s*R\d+\s*:\s*', '', line)
        
        # Then remove sent<number>: or int<number>: prefixes
        line = re.sub(r'^\s*(sent\d+|int\d+)\s*:\s*', '', line)
        
        if line:  # Only add non-empty lines
            steps.append(line)
    return steps


def coherence_batch(entries, embed_backend, cfg):
    """Batch version of coherence scoring"""
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
    """Batch version of concision scoring"""
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

def evidence_batch(entries, embed_backend, nli_backend, cfg):
    """Batch version of evidence scoring"""
    scores = []
    
    for entry in entries:
        steps = normalize_reasoning_steps(entry["reasoning_trace"])
        evidences = entry["evidence_text"]
        
        evidence_score = evidence_alignment_score(
            steps, evidences, embed_backend, nli_backend, cfg.evidence
        )
        scores.append(evidence_score)
    
    return scores

def verdict_batch(entries, nli_backend, embed_backend, cfg):
    """Batch version of verdict scoring"""
    scores = []
    
    verdict_scorer = VerdictScorerDifferential(nli_backend, embed_backend, cfg)
    
    for entry in entries:
        steps = normalize_reasoning_steps(entry["reasoning_trace"])
        
        verdict_score = verdict_scorer.score(
            steps, entry["claim"], entry["evidence_text"], entry["label"]
        )
        scores.append(verdict_score)
    
    return scores

def calculate_ourmetric_batch(entries, embed_backend, nli_backend, cfg):
    """Batch version of ourmetric calculation"""
    results = []
    # Calculate all metrics in batch
    coherence_scores = coherence_batch(entries, embed_backend, cfg)
    redundancy_scores = concision_batch(entries, embed_backend, cfg , device=f"cuda:{GPU_USE}")
    evidence_scores = evidence_batch(entries, embed_backend, nli_backend, cfg)
    verdict_scores = verdict_batch(entries, nli_backend, embed_backend, cfg)

    for i, entry in enumerate(entries):
        # print("DEBUG" , coherence_scores[i] , "(" , redundancy_scores[i] , verdict_scores[i] , ")" , evidence_scores[i])
        block2_weight = 0.5
        block4_weight = 0.5
        ourmetric = {
            "coherence_score": float(coherence_scores[i]),
            "quality_score": ( (block2_weight * redundancy_scores[i]) + (block4_weight * verdict_scores[i]) ),
            "evidence_score": float(evidence_scores[i]),
            "b1_b2": float(coherence_scores[i]*.5) + ((block2_weight * redundancy_scores[i]) + (block4_weight * verdict_scores[i]))*.5,
            "b2_b3": ((block2_weight * redundancy_scores[i]) + (block4_weight * verdict_scores[i]))*.5 + float(evidence_scores[i]*.5),
            "b3_b1": float(evidence_scores[i]*.5) + float(coherence_scores[i]*.5),
            "total_score": float(
                cfg.weights.coherence * coherence_scores[i] +
                cfg.weights.quality * ((block2_weight * redundancy_scores[i]) + (block4_weight * verdict_scores[i])) +
                cfg.weights.evidence * evidence_scores[i]
            )
        }
        
        entry["ourmetric"] = ourmetric
        results.append(entry)
            
    return results

# ---------------------------------------------------
# Main Batch Processing Function
# ---------------------------------------------------
def main(input_file, output_file):
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        entries = json.load(f)

    # Create dataset
    dataset = Dataset.from_list(entries)

    # Configuration
    config = EvaluatorConfig(
        embed_model_name="/home/models/sentence-transformers--all-MiniLM-L6-v2",    # sentence embedding model
        nli_model_name="/home/models/deberta-xlarge-mnli",                          # NLI model
    )

    # Initialize backends
    embed_backend = EmbeddingBackend(config.embed_model_name)
    nli_backend = NLICrossEncoder(config.nli_model_name)

    def process_batch(batch):
        """Process a batch of entries"""
        # Convert batch to list of entries
        entries_batch = []
        for i in range(len(batch["claim"])):
            entry = {key: batch[key][i] for key in batch}
            entries_batch.append(entry)
        
        # Process the entire batch
        results = calculate_ourmetric_batch(entries_batch, embed_backend, nli_backend, config)
        
        # Convert list of dicts → dict of lists
        out = {}
        for key in results[0].keys():
            out[key] = [r[key] for r in results]
        return out

    # Process in batches
    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=16,  # Batch size for memory efficiency
        desc="Calculating our metrics"
    )

    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)

    # Calculate statistics for printing
    if EVIDENCE:
        valid_scores = [x for x in dataset["ourmetric"] if isinstance(x, dict) and x.get("total_score", 0) > 0]
        
        if valid_scores:
            avg_coherence = np.mean([x.get("coherence_score", 0) for x in valid_scores])
            avg_redundancy = np.mean([x.get("redundancy_score", 0) for x in valid_scores])
            avg_evidence = np.mean([x.get("evidence_score", 0) for x in valid_scores])
            avg_verdict = np.mean([x.get("verdict_score", 0) for x in valid_scores])
            avg_quality = np.mean([x.get("quality_score", 0) for x in valid_scores])
            avg_total = np.mean([x.get("total_score", 0) for x in valid_scores])
        else:
            avg_coherence = avg_redundancy = avg_evidence = avg_verdict = avg_total = 0.0

        print(f"\nProcessing complete!")
        print(f"Total entries processed: {len(dataset)}")
        print(f"Entries with valid scores: {len(valid_scores)}")
        print(f"Average Coherence: {avg_coherence:.4f}")
        print(f"Average Quality: {avg_quality:.4f}")
        print(f"Average Evidence: {avg_evidence:.4f}")
        print(f"Average Total Score: {avg_total:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input file and generate output file with our metrics.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file')
    parser.add_argument('--output', '-o', required=True, help='Path to the output JSON file')
    parser.add_argument('--evidence', '-e', type=lambda x: x.lower() == 'true', default=True, help='Whether to use evidence (true/false)')
    parser.add_argument('--gpu', '-g', required=True, type=int, default=-10, help='GPU id to use (e.g. 0 for first GPU)')
    args = parser.parse_args()
    GPU_USE = args.gpu
    EVIDENCE = args.evidence

    # --- ADD THIS BLOCK ---
    if GPU_USE >= 0 and torch.cuda.is_available():
        print(f"Setting default CUDA device to {GPU_USE}")
        torch.cuda.set_device(GPU_USE)
    # ----------------------

    main(args.input, args.output)


# python OurMetric.py --input /home/ojas/scripts/datasets/Human_Eval/sample_exps/DeepLlama_8B_2shot_sample_ptrb_baselines.json --output /home/ojas/scripts/datasets/Human_Eval/sample_exps/DeepLlama_8B_2shot_sample_ptrb_baselines.json --gpu 0 --evidence true

# python OurMetric.py --input /home/ojas/scripts/datasets/Human_Eval/sample_exps/GPTOSS_4Shot_sample_ptrb_baselines.json --output /home/ojas/scripts/datasets/Human_Eval/sample_exps/GPTOSS_4Shot_sample_ptrb_baselines.json --gpu 0 --evidence true

# python OurMetric.py --input /home/ojas/scripts/datasets/Human_Eval/sample_exps/Qwen_3B_CoT_1shot_sample_ptrb_baselines.json --output /home/ojas/scripts/datasets/Human_Eval/sample_exps/Qwen_3B_CoT_1shot_sample_ptrb_baselines.json --gpu 2 --evidence true

# python OurMetric.py --input /home/ojas/scripts/datasets/Human_Eval/sample_exps/test.json --output /home/ojas/scripts/datasets/Human_Eval/sample_exps/test_baselines_bnbm.json --gpu 0 --evidence true


# python OurMetric.py --input /home/ojas/scripts/datasets/Final_Runs/Qwen_CoT_3B_4shot_PTRB_BASELINES.json --output /home/ojas/scripts/datasets/Final_Runs/Qwen_CoT_3B_4shot_PTRB_BASELINES_newourmetric.json --gpu 1 --evidence true

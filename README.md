# Markovian ODE-Guided Scoring for Offline Reasoning Trace Evaluation
![alt text](MarODE_Method.png "MarODE Methodology")

## Install via PyPI
We have made our evaluation pipeline publicly available as a Python package on PyPI, and it can be freely installed using:
```bash
pip install marode
```

**Quick Code Implementation:**
```python
from marode.evaluator import MarODEEvaluator, EvaluatorConfig, get_device

# Initialize evaluator
config = EvaluatorConfig()
device = get_device(0)
evaluator = MarODEEvaluator(config, device)

entry = {
  "id": "example_1",
  "reasoning_trace": "R0: ...\nR1: ...\nR2: ...",
  "evidence_text": ["evidence 1", "evidence 2"]
}

# Score reasoning trace
scored_entry = evaluator.score_entry(entry)

# Print MarODE scores
for k, v in scored_entry["ourmetric"].items():
    print(f"{k}: {v:.4f}")
```

## Dataset Preparation
We provide preprocessing scripts to construct enriched versions of the PolitiFact and LIAR datasets by scraping full article evidence from fact-check pages. All required dependencies for running our experiments are specified in the `requirements.txt` file. 
```python
python scripts/prepare_liar_with_politifact_evidence.py \
    --input data/LIAR_train.tsv \
    --output data/LIAR_extracted.json
```
```python
python src/dataset/prepare_politifact_with_evidence.py \
    --input data/politifact_factcheck_data.json \
    --output data/politifact_extracted.json

```
Both scripts generate a JSON file with the following structure:
```json
{
  "claim": "...",
  "label": "...",
  "evidence_text": ["paragraph 1", "paragraph 2", ...]
}
```

## Generating Reasoning Traces
We generate reasoning traces using multiple open-source and instruction-tuned language models over the prepared claim datasets.
The following models have been used by us:
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B
- Qwen-2.5-3b-Evol-CoT
- GPT-OSS-20B

```python
python -m src/reasoning/run_generation.py \
  --model-path <PATH_TO_MODEL> \
  --dataset data/claims_dataset_1200.json \
  --prompt-dir prompts \
  --output-dir outputs/<MODEL_NAME> \
  --n-shot 6 \
  --gpu-index 0
```
Each run produces batched JSON files of the form:
```json
{
  "shots": 4,
  "claim_id": "...",
  "reasoning_trace": "R0: ...\nR1: ...\n...\nFinal Verdict: ..."
}
```

## MarODE Metric
MarODE (Markov + ODE Reasoning Evaluator) is our proposed metric for evaluating reasoning traces.
It decomposes reasoning quality into three principled components:
- Coherence – Structural consistency of reasoning steps via random-walk transition dynamics over sentence embeddings.
- Quality – Internal step quality, combining redundancy penalties and a differential ODE-based belief update mechanism.
- Evidence Alignment – Semantic and NLI-based alignment between reasoning steps and supporting evidence.
The final score is a weighted combination.

**Running MarODE:**
```python
python src/evals/MarODE.py \
  --input path/to/your_input.json \
  --output path/to/output_with_marode.json \
  --gpu 0
```

**Expected Input Format**:
```json
{
  "claim": "...",
  "label": "...",
  "reasoning_trace": "...",
  "evidence_text": ["...", "..."]
}
```
**Output Format: **MarODE appends an ourmetric field to each entry:
```json
"ourmetric": {
  "coherence_score": ...,
  "quality_score": ...,
  "evidence_score": ...,
  "total_score": ...
}
```
The `total_score` represents the final MarODE evaluation score in the range [0, 1].

## Evaluation
We evaluate generated reasoning traces using multiple established baselines alongside our proposed metric. All evaluation scripts operate on JSON files containing reasoning traces and append metric-specific scores to each entry.

### Running Baseline Evaluations
Given a file of generated traces (e.g., outputs/deepseek_llama8b/traces_1.json), the following commands compute each evaluation metric:
**Coherence (SGC, WGC, LC)**
```python
python src/evals/coherence_baseline.py \
  --input outputs/deepseek_llama8b/traces_1.json \
  --output outputs/deepseek_llama8b/traces_1_coherence.json \
  --model-path /home/models/deberta-xlarge-mnli \
  --gpu 0
```
**LLM-as-a-Judge (Prometheus)**
```python
python src/evals/llm_judge_prometheus.py \
  --input outputs/deepseek_llama8b/traces_1.json \
  --output outputs/deepseek_llama8b/traces_1_judged.json \
  --model-path /home/models/prometheus-eval--prometheus-7b-v2.0 \
  --gpu 0 \
  --evidence true
```
**RECEval**
```python
python src/evals/receval_baseline.py \
  --input outputs/deepseek_llama8b/traces_1.json \
  --output outputs/deepseek_llama8b/traces_1_receval.json \
  --model-path /home/models/deberta-xlarge-mnli \
  --gpu 0 \
  --batch-size 64
```
**ROSCOE-LC**
```python
python src/evals/roscoe_lc_baseline.py \
  --input outputs/deepseek_llama8b/traces_1.json \
  --output outputs/deepseek_llama8b/traces_1_roscoe.json \
  --ppl-model /home/models/gpt2-large \
  --cola-model /home/models/tinybertcola \
  --gpu 0
```
**ROSCOE-LI**
```python
python src/evals/roscoe_li_baseline.py \
  --input outputs/deepseek_llama8b/traces_1.json \
  --output outputs/deepseek_llama8b/traces_1_roscoe_li.json \
  --model-path /home/models/deberta-xlarge-mnli \
  --gpu 0 \
  --evidence true
```
**ROSCOE-SS**
```python
python src/evals/roscoe_sa_baseline.py \
  --input outputs/deepseek_llama8b/traces_1.json \
  --output outputs/deepseek_llama8b/traces_1_roscoe_sa.json \
  --model-path /home/models/deberta-xlarge-mnli \
  --gpu 0 \
  --evidence true
```
**ROSCOE-SA**
```python
python src/evals/roscoe_ss_baseline.py \
  --input outputs/deepseek_llama8b/traces_1.json \
  --output outputs/deepseek_llama8b/traces_1_roscoe_ss.json \
  --model-path /home/models/sentence-transformers--all-MiniLM-L6-v2 \
  --gpu 0 \
  --evidence true
```
To compute correlations **(Somers’ D)** between perturbation scores and evaluation metrics across filtered result files:
```python
python src/evals/correlation_analysis.py \
  --dir outputs/final_runs \
  --pattern "filtered_*.json" \
  --save correlation_results.csv
```

Somers’ D correlations measuring the association between human-centric perturbation scores and evaluation metrics across different backbone models and prompting settings are given in the following table. For each column, the three highest correlations are highlighted in **bold**.

| Metric | Qwen-3B-CoT | DeepSeek-Qwen-7B | Deepseek-Qwen-14B | Deepseek-LLaMA-8B | GPT-OSS-20B | LIAR (1-shot) | LIAR (2-shot) | LIAR (4-shot) | PolitiFact (1-shot) | PolitiFact (2-shot) | PolitiFact (4-shot) |
|--------|-------------|------------------|-------------------|-------------------|-------------|---------------|---------------|---------------|----------------------|----------------------|----------------------|
| ROSCOE-SA | 0.1187 | 0.1053 | 0.1111 | 0.1117 | 0.1048 | 0.1004 | 0.0983 | 0.1075 | 0.1145 | 0.1152 | **0.1281** |
| ROSCOE-SS | 0.1294 | 0.1242 | 0.1301 | 0.1256 | 0.1284 | 0.1167 | 0.1146 | **0.1386** | 0.1264 | 0.1350 | 0.1331 |
| ROSCOE-LI | 0.0318 | 0.0163 | 0.0496 | 0.0172 | 0.0146 | 0.0253 | 0.0279 | 0.0286 | 0.0283 | 0.0256 | 0.0200 |
| ROSCOE-LC | -0.0189 | -0.0009 | -0.0223 | 0.0228 | -0.0173 | -0.0073 | -0.0076 | -0.0184 | 0.0002 | -0.0047 | -0.0066 |
| **ROSCOE_MEAN** | 0.0819 | 0.0686 | 0.0958 | 0.0689 | 0.0622 | 0.0691 | 0.0728 | 0.0776 | 0.0793 | 0.0785 | 0.0742 |
| LLM_as_a_Judge | 0.0436 | 0.0271 | 0.0421 | 0.0375 | 0.0330 | 0.0367 | 0.0282 | 0.0140 | 0.0516 | 0.0455 | 0.0407 |
| Local_and_Global | 0.0458 | 0.0192 | 0.0516 | 0.0321 | 0.0316 | 0.0417 | 0.0397 | 0.0347 | 0.0296 | 0.0379 | 0.0330 |
| ReCEval | 0.0457 | 0.0049 | 0.0311 | -0.0018 | 0.0072 | 0.0082 | 0.0084 | 0.0211 | 0.0253 | 0.0243 | 0.0177 |
| **MarODE** | **0.2937** | **0.2371** | **0.2882** | **0.2921** | **0.2634** | **0.2618** | **0.2636** | **0.2604** | **0.2895** | **0.2840** | **0.2792** |
| MarODE_COHERENCE (α) | **0.2857** | 0.1747 | **0.2798** | 0.2826 | 0.2014 | 0.2395 | 0.2357 | 0.2427 | 0.2165 | 0.2284 | 0.2296 |
| MarODE_QUALITY (β) | 0.0286 | 0.0639 | 0.0331 | 0.0082 | 0.0467 | 0.0181 | 0.0342 | 0.0358 | 0.0374 | 0.0405 | 0.0410 |
| MarODE_EVIDENCE (γ) | 0.2353 | 0.1955 | 0.2253 | 0.2367 | 0.2139 | 0.2075 | 0.2122 | 0.2082 | 0.2398 | 0.2341 | 0.2279 |
| **MarODE (αβ)** | **0.3272** | **0.2093** | **0.3279** | **0.3118** | **0.2309** | **0.2762** | **0.2743** | **0.2769** | **0.2583** | **0.2652** | **0.2639** |
| MarODE (βγ) | 0.2403 | 0.1992 | 0.2309 | 0.2392 | 0.2173 | 0.2103 | 0.2164 | 0.2129 | 0.2444 | 0.2379 | 0.2322 |
| MarODE (αγ) | 0.2857 | **0.2303** | 0.2791 | **0.2857** | **0.2570** | **0.2550** | **0.2560** | **0.2527** | **0.2799** | **0.2763** | **0.2714** |


To evaluate whether increasing the number of in-context examples (1-shot, 2-shot, 4-shot) leads to statistically significant differences in evaluation metrics, we perform paired Wilcoxon signed-rank tests across aligned claim instances.
```python
python wilcoxon_shot_analysis.py \
  --shot1 RT_Original_1shot.json \
  --shot2 RT_Original_2shot.json \
  --shot4 RT_Original_4shot.json \
  --output Shot_Difference_Wilcoxon_Results.csv
```

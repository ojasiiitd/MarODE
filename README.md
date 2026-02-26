# Markovian ODE-Guided Scoring for Offline Reasoning Trace Evaluation
![alt text](MarODE_Method.png "MarODE Methodology")

## Dataset Preparation
We provide preprocessing scripts to construct enriched versions of the PolitiFact and LIAR datasets by scraping full article evidence from fact-check pages.
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
  "shots": 6,
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
To evaluate whether increasing the number of in-context examples (1-shot, 2-shot, 4-shot) leads to statistically significant differences in evaluation metrics, we perform paired Wilcoxon signed-rank tests across aligned claim instances.
```python
python wilcoxon_shot_analysis.py \
  --shot1 RT_Original_1shot.json \
  --shot2 RT_Original_2shot.json \
  --shot4 RT_Original_4shot.json \
  --output Shot_Difference_Wilcoxon_Results.csv
```

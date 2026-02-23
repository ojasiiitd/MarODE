import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(description='Process reasoning traces with evidence flag')
parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file')
parser.add_argument('--output', '-o', required=True, help='Path to the output JSON file')
parser.add_argument('--evidence', '-e', type=lambda x: x.lower() == 'true', default=True, help='Whether to use evidence (true/false)')
parser.add_argument('--gpu', '-g', required=True, type=int, default=-10, help='GPU id to use (e.g. 0 for first GPU)')
args = parser.parse_args()
GPU_USE = args.gpu

# -------------------------------
# PrometheusJudge class
# -------------------------------
class PrometheusJudge:
    def __init__(self, model_path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map=device if device != "cuda" else "auto"
        )
        if device != "cuda":
            self.model = self.model.to(device)
        
    def generate_score(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        if GPU_USE >= 0:
            inputs = {k: v.to(f"cuda:{GPU_USE}") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,  # Increased slightly for decimal precision
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the new generated text (after the prompt)
        generated_text = response[len(prompt):].strip()

        print(generated_text)  # Debugging line to see the raw output
        
        # Enhanced extraction for 2-decimal-place scores
        try:
            import re
            # Look for patterns like 0.xx, .xx, or xx.xx
            patterns = [
                r"\b\d\.\d{2}\b",  # 0.75, 1.00, etc.
                r"\b\.\d{2}\b",    # .75, .25, etc.
                r"\b\d\.\d{1,2}\b", # fallback for 1-2 decimals
                r"\b\d{1,2}\b"      # integers that can be normalized
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, generated_text)
                if matches:
                    score_str = matches[0]
                    # Handle cases like ".75" by adding leading 0
                    if score_str.startswith('.'):
                        score_str = '0' + score_str
                    
                    score = float(score_str)
                    
                    # Normalize if needed
                    if score > 1 and score <= 100:
                        score = score / 100.0
                    elif score > 10:
                        score = score / 10.0
                    
                    # Clamp to 0-1 range and round to 2 decimal places
                    score = max(0.0, min(1.0, score))
                    score = round(score, 2)
                    return score
                    
            # If no pattern found, return default with some variation
            return -1
            
        except:
            return -1  # Default score on error

# -------------------------------
# Create prompt for judging
# -------------------------------
def create_judge_prompt(claim, label, evidence_text, reasoning_trace, use_evidence=True):
    evidence_str = "\n".join(evidence_text) if evidence_text else "No evidence provided."
    
    if use_evidence:
        prompt = f"""Evaluate the reasoning trace using the scoring rubric below. Assume the evidence to be true.
Scoring Rubric
1.00 (Outstanding)
0.75 (Good)
0.50 (Adequate)
0.25 (Poor)
0.00 (Critically Flawed)

CLAIM: {claim}
LABEL: {label}
EVIDENCE: {evidence_str}
REASONING TRACE: {reasoning_trace}

Provide ONLY a numerical score between 0.00 and 1.00 with exactly two decimal places. Do not include any text, explanation, or formatting.

Score: """
    else:
        prompt = f"""Evaluate the reasoning trace using the scoring rubric below.
Scoring Rubric
1.00 (Outstanding)
0.75 (Good)
0.50 (Adequate)
0.25 (Poor)
0.00 (Critically Flawed)

CLAIM: {claim}
LABEL: {label}
REASONING TRACE: {reasoning_trace}

Provide ONLY a numerical score between 0.00 and 1.00 with exactly two decimal places. Do not include any text, explanation, or formatting.

Score: """
    
    return prompt
# -------------------------------
# Process dataset with judge scoring
# -------------------------------
def process_with_judge(input_json_path, output_json_path, use_evidence=True):
    # Load judge model
    print("Loading PrometheusJudge model...")
    judge = PrometheusJudge(
        model_path="/home/models/prometheus-eval--prometheus-7b-v2.0",
        device=f"cuda:{GPU_USE}" if GPU_USE >= 0 else "cpu"
    )
    
    # Load data
    with open(input_json_path, "r") as f:
        data = json.load(f)
    
    # Process each entry
    for entry in tqdm(data, desc="Judging reasoning traces"):
        claim = entry.get("claim", "")
        label = entry.get("label", "")
        evidence_text = entry.get("evidence_text", [])
        reasoning_trace = entry.get("reasoning_trace", "")
        
        # Create prompt
        prompt = create_judge_prompt(claim, label, evidence_text, reasoning_trace, use_evidence)
        
        # Get score from judge
        score = judge.generate_score(prompt)
        
        # Add score to entry
        entry["judge_score"] = score
    
    # Save results
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    input_path = args.input
    output_path = args.output
    use_evidence = args.evidence
    
    print(f"Processing {input_path} with PrometheusJudge...")
    print(f"Using evidence: {use_evidence}")
    print(f"Using GPU: {GPU_USE}")
    
    result_data = process_with_judge(input_path, output_path, use_evidence)
    
    # Calculate statistics
    scores = [entry.get("judge_score", -1) for entry in result_data]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    print(f"\n=== Judge Results ===")
    print(f"Average judge score: {avg_score:.3f}")
    print(f"Total entries processed: {len(result_data)}")
    print(f"Results saved to: {output_path}")

# python Prometheus_LLM_JUDGE.py --input /home/ojas/scripts/datasets/Final_Runs/DeepQwen_7B_2shot_BASELINES.json --output /home/ojas/scripts/datasets/DeepQwen_7B_2shot_BASELINES_JUDGE.json --gpu 3 --evidence true

# python Prometheus_LLM_JUDGE.py --input /home/ojas/scripts/datasets/Human_Eval/sample_exps/T_final_main_perturbed.json --output /home/ojas/scripts/datasets/Human_Eval/sample_exps/T_final_main_perturbed_JUDGE.json --gpu 2 --evidence true

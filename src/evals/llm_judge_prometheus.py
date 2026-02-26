import json
import argparse
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(
    description="Evaluate reasoning traces using Prometheus LLM-as-a-Judge."
)

parser.add_argument("--input", "-i", required=True, help="Path to input JSON file")
parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
parser.add_argument("--model-path", required=True, help="Path to Prometheus judge model")
parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU id (default: CPU)")
parser.add_argument("--evidence", "-e", type=lambda x: x.lower() == "true", default=True)

args = parser.parse_args()


# -------------------------------
# Prometheus Judge
# -------------------------------
class PrometheusJudge:
    def __init__(self, model_path: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        ).to(device)

        self.model.eval()
        self.device = device

    def generate_score(self, prompt: str) -> float:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return self._extract_score(generated_text)

    @staticmethod
    def _extract_score(text: str) -> float:
        patterns = [
            r"\b\d\.\d{2}\b",
            r"\b\.\d{2}\b",
            r"\b\d\.\d{1,2}\b",
            r"\b\d{1,2}\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                score_str = matches[0]
                if score_str.startswith("."):
                    score_str = "0" + score_str

                score = float(score_str)

                if score > 1 and score <= 100:
                    score /= 100.0
                elif score > 10:
                    score /= 10.0

                return round(max(0.0, min(1.0, score)), 2)

        return -1.0


# -------------------------------
# Prompt builder
# -------------------------------
def create_judge_prompt(claim, label, evidence_text, reasoning_trace, use_evidence=True):
    evidence_str = "\n".join(evidence_text) if evidence_text else "No evidence provided."

    rubric = """Scoring Rubric
1.00 (Outstanding)
0.75 (Good)
0.50 (Adequate)
0.25 (Poor)
0.00 (Critically Flawed)
"""

    if use_evidence:
        return f"""Evaluate the reasoning trace using the scoring rubric below. Assume the evidence to be true.

{rubric}

CLAIM: {claim}
LABEL: {label}
EVIDENCE:
{evidence_str}

REASONING TRACE:
{reasoning_trace}

Provide ONLY a numerical score between 0.00 and 1.00 with exactly two decimal places.

Score: """
    else:
        return f"""Evaluate the reasoning trace using the scoring rubric below.

{rubric}

CLAIM: {claim}
LABEL: {label}

REASONING TRACE:
{reasoning_trace}

Provide ONLY a numerical score between 0.00 and 1.00 with exactly two decimal places.

Score: """


# -------------------------------
# Process dataset
# -------------------------------
def process_dataset(input_path, output_path, model_path, device, use_evidence):
    judge = PrometheusJudge(model_path=model_path, device=device)

    with open(input_path, "r") as f:
        data = json.load(f)

    for entry in tqdm(data, desc="Judging reasoning traces"):
        prompt = create_judge_prompt(
            entry.get("claim", ""),
            entry.get("label", ""),
            entry.get("evidence_text", []),
            entry.get("reasoning_trace", ""),
            use_evidence,
        )

        entry["judge_score"] = judge.generate_score(prompt)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"

    print(f"Running Prometheus Judge on: {args.input}")
    print(f"Device: {device}")
    print(f"Using evidence: {args.evidence}")

    results = process_dataset(
        args.input,
        args.output,
        args.model_path,
        device,
        args.evidence,
    )

    scores = [e.get("judge_score", -1) for e in results if e.get("judge_score", -1) >= 0]
    avg_score = sum(scores) / len(scores) if scores else 0

    print("\n=== Judge Results ===")
    print(f"Average score: {avg_score:.3f}")
    print(f"Entries processed: {len(results)}")
    print(f"Saved to: {args.output}")
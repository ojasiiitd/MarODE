# -------------------------------
# Import the evaluator
# -------------------------------
from marode.evaluator import MarODEEvaluator, EvaluatorConfig, get_device

# -------------------------------
# Initialize evaluator
# -------------------------------
config = EvaluatorConfig()
device = get_device(0)  # CPU; use 0 if you want GPU
evaluator = MarODEEvaluator(config, device)

# -------------------------------
# Example reasoning trace
# -------------------------------
reasoning_trace = """
R0: The suspect was seen near the crime scene at 9 PM.
R1: Surveillance footage confirms someone matching the suspect's description.
R2: Surveillance footage confirms someone matching the suspect's description.
R3: Surveillance footage confirms someone matching the suspect's description.
R4: The suspect has no alibi for that time.
R5: Fingerprints from the crime scene match the suspect.
"""

entry = {
    "id": "colab_test_1",
    "reasoning_trace": reasoning_trace,
    "evidence_text": [
        "A person resembling the suspect was caught on CCTV at 9 PM.",
        "Fingerprints collected at the scene match the suspect's fingerprints.",
        "No other witnesses have confirmed the suspect's whereabouts."
    ]
}

# -------------------------------
# Score the reasoning trace
# -------------------------------
scored_entry = evaluator.score_entry(entry)

# -------------------------------
# Print the scores
# -------------------------------
scores = scored_entry["ourmetric"]
for k, v in scores.items():
    print(f"{k}: {v:.4f}")

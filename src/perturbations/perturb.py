import json
import copy
import random
import re
import argparse
from typing import List, Tuple

# Import perturbation functions
from perturbations.perturb_fxns import *

# -----------------------
# Argument Parsing
# -----------------------
parser = argparse.ArgumentParser(description="Apply controlled perturbations to reasoning traces.")
parser.add_argument("--input", "-i", required=True, help="Path to input JSON file")
parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output

print(f"Using input: {INPUT_FILE}")
print(f"Saving output: {OUTPUT_FILE}")

# -----------------------
# Perturbation Registry
# -----------------------
PERTURB_FUNCS: List[Tuple[str, callable]] = [
    ("penultimate_ambiguity", perturb_penultimate_ambiguity),
    ("temporal_confusion", perturb_temporal_confusion),
    ("unsupported_conclusion", perturb_unsupported_conclusion),
    ("irrelevant_elaboration", perturb_irrelevant_elaboration),
    ("duplication", perturb_duplication),
    ("underspecification", perturb_underspecification),
    ("antonym_insertion", perturb_antonym_insertion),
    ("definitional_redundancy", perturb_definitional_redundancy),
    ("modal_logic_confusion", perturb_modal_logic_confusion),
    ("ordering", perturb_ordering),
    ("quantifier_abuse", perturb_quantifier_abuse),
    ("random_hyphenation", perturb_random_hyphenation),
    ("key_concept_swap", perturb_key_concept_swap),
    ("final_verdict_insertion", perturb_final_verdict_insertion),
]

# -----------------------
# Score Mapping (0–6 perturbations)
# -----------------------
score_map = {i: round(1.0 - i * 0.1, 1) for i in range(7)}
# {0:1.0, 1:0.9, ..., 6:0.4}

# -----------------------
# Load Data
# -----------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries")

if not isinstance(data, list):
    raise ValueError("Input JSON must contain a list of entries.")

random.shuffle(data)

# -----------------------
# Balanced Assignment
# -----------------------
n = len(data)
group_size = max(1, n // 7)  # 7 bins: 0–6 perturbations

balanced_data = []

for idx, item in enumerate(data):
    # Determine planned perturbation count (0–6)
    planned = min(idx // group_size, 6)

    # Extract reasoning steps
    steps = [
        m.strip()
        for m in re.findall(
            r"R\d+:\s*(.*?)(?=(?:\nR\d+:)|\Z)",
            item["reasoning_trace"],
            flags=re.S,
        )
    ]

    pert_steps = copy.deepcopy(steps)

    applied = 0
    applied_names = []
    unused_funcs = PERTURB_FUNCS.copy()

    while applied < planned and unused_funcs:
        func_name, func = random.choice(unused_funcs)
        unused_funcs.remove((func_name, func))

        try:
            success = func(pert_steps)
            if success:
                applied += 1
                applied_names.append(func_name)
        except Exception:
            continue  # skip failed perturbations safely

    score = score_map.get(applied, 0.0)

    # Reconstruct reasoning trace
    if pert_steps and pert_steps[-1].strip().startswith("Final Verdict"):
        final_verdict = pert_steps[-1]
        steps_only = pert_steps[:-1]
    else:
        final_verdict = None
        steps_only = pert_steps

    numbered_steps = [f"R{i}: {s}" for i, s in enumerate(steps_only)]
    full_trace = "\n".join(numbered_steps + ([final_verdict] if final_verdict else []))

    # Build output entry
    item_out = copy.deepcopy(item)
    item_out["reasoning_trace"] = full_trace
    item_out["perturbation_score"] = score
    item_out["perturbations_applied"] = applied_names

    balanced_data.append(item_out)

# -----------------------
# Save Results
# -----------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(balanced_data, f, indent=2, ensure_ascii=False)

print("Perturbation process complete.")
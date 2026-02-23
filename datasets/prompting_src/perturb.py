# python perturb.py -i /home/ojas/scripts/datasets/Final_Runs/Qwen_CoT_3B_4shot.json -o /home/ojas/scripts/datasets/Final_Runs/Qwen_CoT_3B_4shot.json

import json
import copy
import random
import re
import os
from typing import List
from collections import Counter

# Load perturbation functions
from perturb_fxns import *
import argparse

# -----------------------
# Parameterized arguments
# -----------------------
parser = argparse.ArgumentParser(description="Process JSON files.")
parser.add_argument("--input", "-i", required=True, help="Path to input JSON file")
parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")

args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output

print("Using input:", INPUT_FILE)
print("Saving output:", OUTPUT_FILE)

# -----------------------
# Perturbation functions
# -----------------------
PERTURB_FUNCS = [
    ("penultimate_ambiguity", perturb_penultimate_ambiguity),
    ("temporal_confusion", perturb_temporal_confusion),
    ("unsupported_conclusion", perturb_unsupported_conclusion),
    ("irrelevant_elaboration", perturb_irrelevant_elaboration),
    # ("redundant", perturb_redundant),
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
# Score mapping (0–6 perturbations)
# -----------------------
score_map = {i: round(1.0 - i * 0.1, 1) for i in range(7)}
# {0:1.0, 1:0.9, ..., 6:0.4}

# -----------------------
# Load original data
# -----------------------
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries")

# -----------------------
# Force balanced assignment
# -----------------------
random.shuffle(data)  # shuffle before splitting
group_size = len(data) // 6  # 6 groups
balanced_data = []

for i, item in enumerate(data):
    planned = i // group_size  # how many perturbations this entry should get (0–6 max)

    # Cap planned at 6
    planned = min(planned, 6)

    # extract reasoning steps
    steps = [
        m.strip()
        for m in re.findall(r"R\d+:\s*(.*?)(?=(?:\nR\d+:)|\Z)", item["reasoning_trace"], flags=re.S)
    ]
    pert_steps = copy.deepcopy(steps)
    
    applied = 0
    applied_names = []
    # choose perturbations until we get exactly `planned` successes or run out
    unused_funcs = PERTURB_FUNCS[:]  # copy full list

    while applied < planned and unused_funcs:
        func_name, func = random.choice(unused_funcs)
        unused_funcs.remove((func_name, func))  # don’t reuse same perturbation

        try:
            success = func(pert_steps)
            if success:
                applied += 1
                applied_names.append(func_name)
        except Exception:
            continue  # ignore errors, just skip
    
    # lookup score based on how many actually succeeded
    score = score_map.get(applied, 0.0)

    # reconstruct reasoning trace
    if pert_steps and pert_steps[-1].strip().startswith("Final Verdict"):
        final_verdict = pert_steps[-1]
        steps_only = pert_steps[:-1]
    else:
        final_verdict = None
        steps_only = pert_steps

    numbered_steps = [f"R{i}: {s}" for i, s in enumerate(steps_only)]
    if final_verdict:
        full_trace = "\n".join(numbered_steps + [final_verdict])
    else:
        full_trace = "\n".join(numbered_steps)

    # update entry
    item_out = copy.deepcopy(item)
    item_out["reasoning_trace"] = full_trace
    item_out["perturbation_score"] = score
    item_out["perturbations_applied"] = applied_names  # NEW FIELD

    balanced_data.append(item_out)

# -----------------------
# Save results
# -----------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(balanced_data, f, indent=2, ensure_ascii=False)




# -----------------------
# Perturbation functions list
# -----------------------
# PERTURB_FUNCS = [
#     ALL 35
#     ("ordering", perturb_ordering),
#     ("deletion", perturb_deletion),
#     ("duplication", perturb_duplication),
#     ("negation", perturb_negation),
#     ("hedge", perturb_hedge),
#     ("number", perturb_number),
#     ("entity", perturb_entity),
#     ("pronoun", perturb_pronoun),
#     ("template", perturb_template),
#     ("circular", perturb_circular),
#     ("contradiction", perturb_contradiction),
#     ("redundant", perturb_redundant),
#     ("irrelevant_elaboration", perturb_irrelevant_elaboration),
#     ("overgeneralization", perturb_overgeneralization),
#     ("underspecification", perturb_underspecification),
#     ("temporal_confusion", perturb_temporal_confusion),
#     ("cause_effect_reversal", perturb_cause_effect_reversal),
#     ("connector_abuse", perturb_connector_abuse),
#     ("unsupported_conclusion", perturb_unsupported_conclusion),
#     ("quantifier_abuse", perturb_quantifier_abuse),
#     ("unjustified_emphasis", perturb_unjustified_emphasis),
#     ("lexical_substitution", perturb_lexical_substitution),
#     ("random_hyphenation" , perturb_random_hyphenation),
#     ("word_truncation" , perturb_word_truncation),
#     ("domain_shift" , perturb_domain_shift),
#     ("verb_aspect" , perturb_verb_aspect),
#     ("adjective_intensity" , perturb_adjective_intensity),
#     ("meronym_confusion" , perturb_meronym_confusion),
#     ("antonym_insertion" , perturb_antonym_insertion),
#     ("key_concept_swap" , perturb_key_concept_swap),
#     ("modal_logic_confusion" , perturb_modal_logic_confusion),
#     ("reasoning_genre_mixing" , perturb_reasoning_genre_mixing),
#     ("final_verdict_insertion" , perturb_final_verdict_insertion),
#     ("definitional_redundancy" , perturb_definitional_redundancy),
#     ("entailment_manipulation" , perturb_entailment_manipulation),
#     ("embedding_space_manipulation" , perturb_embedding_space_manipulation),
#     ("penultimate_ambiguity" , perturb_penultimate_ambiguity),

#     SELECTED PERTURBATIONS
#     ("penultimate_ambiguity" , perturb_penultimate_ambiguity),
#     ("temporal_confusion", perturb_temporal_confusion),
#     ("unsupported_conclusion", perturb_unsupported_conclusion),
#     ("irrelevant_elaboration", perturb_irrelevant_elaboration),
#     ("redundant", perturb_redundant),
#     ("duplication", perturb_duplication),
#     ("underspecification", perturb_underspecification),
#     ("antonym_insertion" , perturb_antonym_insertion),
#     ("definitional_redundancy" , perturb_definitional_redundancy),
#     ("modal_logic_confusion" , perturb_modal_logic_confusion),
#     ("ordering", perturb_ordering),
#     ("quantifier_abuse", perturb_quantifier_abuse),
#     ("random_hyphenation" , perturb_random_hyphenation),
#     ("key_concept_swap" , perturb_key_concept_swap),
#     ("final_verdict_insertion" , perturb_final_verdict_insertion),
# ]

# -----------------------
# Load original data
# -----------------------
# with open(INPUT_FILE, 'r') as f:
#     data = json.load(f)

# print(f"Loaded {len(data)} entries from {INPUT_FILE}")

# # -----------------------
# # Process each perturbation function individually
# # -----------------------
# for func_name, perturb_func in PERTURB_FUNCS:
#     print(f"\nProcessing perturbation: {func_name}")
    
#     perturbed_data = []
    
#     for item in data:
#         # Create a deep copy of the item
#         item_out = copy.deepcopy(item)
        
#         # Extract reasoning steps
#         steps = [m.strip() for m in re.findall(r'R\d+:\s*(.*?)(?=(?:\nR\d+:)|\Z)', item["reasoning_trace"], flags=re.S)]
#         pert_steps = copy.deepcopy(steps)
        
#         # Apply exactly one perturbation
#         success = perturb_func(pert_steps)
        
#         if not success:
#             # If perturbation fails, use original steps
#             pert_steps = copy.deepcopy(steps)
        
#         # Reconstruct the reasoning trace
#         if pert_steps and pert_steps[-1].strip().startswith("Final Verdict"):
#             final_verdict = pert_steps[-1]
#             steps_only = pert_steps[:-1]
#         else:
#             final_verdict = None
#             steps_only = pert_steps
        
#         # Add R0:, R1:, ... numbering
#         numbered_steps = [f"R{i}: {s}" for i, s in enumerate(steps_only)]
        
#         if final_verdict:
#             full_trace = "\n".join(numbered_steps + [final_verdict])
#         else:
#             full_trace = "\n".join(numbered_steps)
        
#         item_out["reasoning_trace"] = full_trace
#         item_out["perturbation_type"] = func_name
#         item_out["perturbation_success"] = success
        
#         perturbed_data.append(item_out)
    
#     # Save results for this perturbation
#     output_file = os.path.join(OUTPUT_DIR, f"T_sample_main_{func_name}.json")
#     with open(output_file, "w") as f:
#         json.dump(perturbed_data, f, indent=2, ensure_ascii=False)
    
#     print(f"Saved {len(perturbed_data)} entries to {output_file}")
    
#     # Print statistics for this perturbation
#     success_count = sum(1 for item in perturbed_data if item["perturbation_success"])
#     print(f"Perturbation success rate: {success_count}/{len(perturbed_data)} ({success_count/len(perturbed_data)*100:.1f}%)")

# # -----------------------
# # Also create a version with all perturbations combined (optional)
# # -----------------------
# print(f"\nCreating combined perturbation file...")
# combined_data = []

# for i, item in enumerate(data):
#     # For each original entry, create multiple versions with different perturbations
#     for func_name, perturb_func in PERTURB_FUNCS:
#         item_out = copy.deepcopy(item)
        
#         # Extract reasoning steps
#         steps = [m.strip() for m in re.findall(r'R\d+:\s*(.*?)(?=(?:\nR\d+:)|\Z)', item["reasoning_trace"], flags=re.S)]
#         pert_steps = copy.deepcopy(steps)
        
#         # Apply perturbation
#         success = perturb_func(pert_steps)
        
#         if not success:
#             pert_steps = copy.deepcopy(steps)
        
#         # Reconstruct reasoning trace
#         if pert_steps and pert_steps[-1].strip().startswith("Final Verdict"):
#             final_verdict = pert_steps[-1]
#             steps_only = pert_steps[:-1]
#         else:
#             final_verdict = None
#             steps_only = pert_steps
        
#         numbered_steps = [f"R{i}: {s}" for i, s in enumerate(steps_only)]
        
#         if final_verdict:
#             full_trace = "\n".join(numbered_steps + [final_verdict])
#         else:
#             full_trace = "\n".join(numbered_steps)
        
#         item_out["reasoning_trace"] = full_trace
#         item_out["perturbation_type"] = func_name
#         item_out["perturbation_success"] = success
#         item_out["original_index"] = i
        
#         combined_data.append(item_out)

# # Save combined results
# # combined_output_file = os.path.join(OUTPUT_DIR, "T_sample_main_all_perturbations.json")
# # with open(combined_output_file, "w") as f:
# #     json.dump(combined_data, f, indent=2, ensure_ascii=False)

# # print(f"Saved {len(combined_data)} entries to {combined_output_file}")
# print("All perturbation tests completed!")






# - good perturbations

# underspecification
# ordering
# final_verdict_insertion
# definitional_redundancy
# irrelevant_elaboration
# penultimate_ambiguity

# - bad perturbations

# modal_logic_confusion
# unsupported_conclusion
# key_concept_swap
# duplication
# temporal_confusion
# quantifier_abuse
# random_hyphenation

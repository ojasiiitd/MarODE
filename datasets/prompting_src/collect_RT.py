import json
import random
import argparse

# # --------------------------------------------
# # Command-line arguments
# # --------------------------------------------
# parser = argparse.ArgumentParser(description="Merge reasoning traces with claims dataset")
# parser.add_argument('--input', '-i', required=True, help='Path to the reasoning traces JSON file')
# parser.add_argument('--output', '-o', required=True, help='Path to save the merged JSON file')
# parser.add_argument('--model', '-m', required=True, help='MODEL NAME')
# args = parser.parse_args()

# INPUT_FILE = args.input
# OUTPUT_FILE = args.output
# MODEL = args.model

# # --------------------------------------------
# # Functions
# # --------------------------------------------
# def remove_final_verdict(entry):
#     trace = entry.get("reasoning_trace", "")
#     lines = trace.strip().split("\n")
#     if lines and lines[-1].strip().startswith("Final Verdict:"):
#         lines = lines[:-1]  # drop last line
#     entry["reasoning_trace"] = "\n".join(lines)
#     entry["reasoning_trace"] = entry["reasoning_trace"].strip()
#     return entry

# # --------------------------------------------
# # Main Logic (unchanged)
# # --------------------------------------------
# with open(INPUT_FILE, 'r') as f:
#     reasoning_traces = json.load(f)

# # Load the claims dataset
# with open('/home/ojas/scripts/datasets/claims_dataset_1200.json', 'r') as f:
#     claims_data = json.load(f)

# # Create a mapping from claim_id to claim details for faster lookup
# claims_dict = {}
# for claim in claims_data:
#     unique_id = claim.get('unique_claim_id')
#     if unique_id:
#         claims_dict[unique_id] = claim

# # Create the merged dataset
# merged_data = []

# for trace_entry in reasoning_traces:
#     claim_id = trace_entry.get('claim_id')
    
#     # Find the corresponding claim in the claims dataset
#     if claim_id in claims_dict:
#         claim_info = claims_dict[claim_id]
        
#         # Determine dataset based on claim_id prefix
#         if claim_id.startswith('03'):
#             dataset_name = "LIAR"
#         elif claim_id.startswith('04'):
#             dataset_name = "POLITIFACT"
#         else:
#             dataset_name = "UNKNOWN"
        
#         # Remove final verdict line before merging
#         trace_entry = remove_final_verdict(trace_entry)
        
#         # Create merged entry
#         merged_entry = {
#             "model": MODEL,
#             "dataset": dataset_name,
#             "claim_id": claim_info.get('unique_claim_id'),
#             "shots": trace_entry.get('shots'),
#             "claim": claim_info.get('claim'),
#             "label": claim_info.get('mapped_verdict'),
#             "evidence_text": claim_info.get('evidence_text', []),
#             "reasoning_trace": trace_entry.get('reasoning_trace')
#         }
        
#         merged_data.append(merged_entry)
#     else:
#         print(f"Warning: Claim ID {claim_id} not found in claims dataset")

# # Save the merged dataset
# with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#     json.dump(merged_data, f, indent=2, ensure_ascii=False)

# print(f"Successfully created {OUTPUT_FILE}")
# print(f"Total entries processed: {len(reasoning_traces)}")
# print(f"Successfully merged entries: {len(merged_data)}")
# print(f"Missing claims: {len(reasoning_traces) - len(merged_data)}")

# python collect_RT.py --input "/home/ojas/scripts/datasets/RTs/RT_Qwen_3B_CoT_4Shot/qwen_3b_cot_4shot_combined.json"  --output "/home/ojas/scripts/datasets/RTs/Qwen_CoT_3B_4shot.json" --model Qwen_CoT_3B






# GET RANDOM ENTRIES
input_file = "/home/ojas/scripts/datasets/RTs/Qwen_CoT_3B_4shot.json"
output_file = "/home/ojas/scripts/datasets/RTs/Qwen_CoT_3B_4shot_sample.json"

# Load JSON file
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Pick 300 random entries
sampled_data = random.sample(data, 800)  # ensure len(data) >= 300

# Save to new file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(sampled_data)} random entries to {output_file}")
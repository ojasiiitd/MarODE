import json
import os
import glob
import argparse
from collections import defaultdict

def combine_json_preserving_order(rt_dir, output_file):
    """
    Combine JSON files while preserving the original order from claims_dataset_1200.json
    
    Args:
        rt_dir: Path to directory containing reasoning trace JSON files
        output_file: Path for the output combined JSON file
    """
    # Load the original dataset to get the order of unique_claim_ids
    dataset_path = "/home/ojas/scripts/datasets/claims_dataset_1200.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Get all unique_claim_ids in original order
    original_claim_ids = [item['unique_claim_id'] for item in original_data]
    print(f"Total unique_claim_ids in original dataset: {len(original_claim_ids)}")
    
    # Create a mapping from claim_id to entry from all RT files
    claim_id_to_entry = {}
    json_files = glob.glob(os.path.join(rt_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in RT directory")
    
    # Read all entries from RT files and create mapping
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                rt_data = json.load(f)
            
            if isinstance(rt_data, list):
                for entry in rt_data:
                    claim_id = entry.get('claim_id')
                    if claim_id and claim_id not in claim_id_to_entry:
                        claim_id_to_entry[claim_id] = entry
            elif isinstance(rt_data, dict):
                claim_id = rt_data.get('claim_id')
                if claim_id and claim_id not in claim_id_to_entry:
                    claim_id_to_entry[claim_id] = rt_data
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"Found {len(claim_id_to_entry)} unique claim_ids in RT files")
    
    # Create the combined output in original order
    combined_entries = []
    missing_claim_ids = []
    
    for unique_claim_id in original_claim_ids:
        if unique_claim_id in claim_id_to_entry:
            combined_entries.append(claim_id_to_entry[unique_claim_id])
        else:
            missing_claim_ids.append(unique_claim_id)
            # Create a placeholder entry for missing claim_id
            placeholder_entry = {
                "shots": "UNKNOWN",
                "claim_id": unique_claim_id,
                "reasoning_trace": "[MISSING - NOT GENERATED]"
            }
            combined_entries.append(placeholder_entry)
    
    # Save the combined entries
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_entries, f, indent=2, ensure_ascii=False)
    
    # Report statistics
    print(f"\nCombination complete!")
    print(f"Total entries in output: {len(combined_entries)}")
    print(f"Successfully matched entries: {len(combined_entries) - len(missing_claim_ids)}")
    print(f"Missing claim_ids: {len(missing_claim_ids)}")
    
    if missing_claim_ids:
        print(f"\nMissing claim_ids: {missing_claim_ids[:10]}{'...' if len(missing_claim_ids) > 10 else ''}")
        # Save missing claim_ids to file
        missing_file = output_file.replace('.json', '_missing.txt')
        with open(missing_file, 'w') as f:
            f.write("Missing claim_ids:\n")
            for claim_id in missing_claim_ids:
                f.write(f"{claim_id}\n")
        print(f"Missing claim_ids saved to: {missing_file}")
    
    return combined_entries, missing_claim_ids

def combine_json_strict(rt_dir, output_file):
    """
    Strict version that only includes entries that exist in RT files
    (No placeholder entries for missing claim_ids)
    """
    # Load the original dataset to get the order
    dataset_path = "/home/ojas/scripts/datasets/claims_dataset_1200.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    original_claim_ids = [item['unique_claim_id'] for item in original_data]
    
    # Create mapping from RT files
    claim_id_to_entry = {}
    json_files = glob.glob(os.path.join(rt_dir, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                rt_data = json.load(f)
            
            if isinstance(rt_data, list):
                for entry in rt_data:
                    claim_id = entry.get('claim_id')
                    if claim_id and claim_id not in claim_id_to_entry:
                        claim_id_to_entry[claim_id] = entry
            elif isinstance(rt_data, dict):
                claim_id = rt_data.get('claim_id')
                if claim_id and claim_id not in claim_id_to_entry:
                    claim_id_to_entry[claim_id] = rt_data
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Create output with only existing entries, preserving order
    combined_entries = []
    missing_claim_ids = []
    
    for unique_claim_id in original_claim_ids:
        if unique_claim_id in claim_id_to_entry:
            combined_entries.append(claim_id_to_entry[unique_claim_id])
        else:
            missing_claim_ids.append(unique_claim_id)
    
    # Save the combined entries
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_entries, f, indent=2, ensure_ascii=False)
    
    print(f"Strict combination complete!")
    print(f"Entries in output: {len(combined_entries)}")
    print(f"Missing claim_ids: {len(missing_claim_ids)}")
    
    return combined_entries, missing_claim_ids

def main():
    parser = argparse.ArgumentParser(description="Combine JSON files preserving original order")
    parser.add_argument(
        "rt_dir",
        help="Path to directory containing reasoning trace JSON files"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="combined_output_ordered.json",
        help="Output file path (default: combined_output_ordered.json)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict mode (exclude missing entries instead of creating placeholders)"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.rt_dir):
        print(f"Error: Directory not found: {args.rt_dir}")
        return
    
    if args.strict:
        combine_json_strict(args.rt_dir, args.output)
    else:
        combine_json_preserving_order(args.rt_dir, args.output)

if __name__ == "__main__":
    main()

# python combine_json.py "/home/ojas/scripts/datasets/RTs/RT_Deepseek_Llama_8B_1Shot" --output "Deepseek_Llama_8B_1shot_combined.json"


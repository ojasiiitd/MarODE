'''
check if all claim ids are present or not, and if any reasoning trace is too short.
also save the problematic claim ids to a file.
'''

import json
import os
import glob
import shutil
from collections import defaultdict

def find_and_clean_problematic_claim_ids(dir):
    # Load the original dataset
    dataset_path = "/home/ojas/scripts/datasets/claims_dataset_1200.json"
    # Find all JSON files in the RT directory
    rt_dir = dir
    
    with open(dataset_path, 'r') as f:
        original_data = json.load(f)
    
    # Get all claim IDs from original dataset
    original_claim_ids = {item['unique_claim_id'] for item in original_data}
    print(f"Total claim IDs in original dataset: {len(original_claim_ids)}")
    
    json_files = glob.glob(os.path.join(rt_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in RT directory")
    
    # Collect all claim IDs from RT files and check reasoning trace length
    rt_claim_ids = set()
    short_trace_claim_ids = []
    duplicate_claim_ids = []
    claim_id_counter = defaultdict(int)
    
    # First pass: identify problematic claim IDs
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                rt_data = json.load(f)
            
            for item in rt_data:
                claim_id = item.get('claim_id')
                reasoning_trace = item.get('reasoning_trace', '')
                
                if claim_id:
                    # Count occurrences for duplicate detection
                    claim_id_counter[claim_id] += 1
                    
                    # Check if reasoning trace is too short (less than 10 words)
                    if len(reasoning_trace.split()) < 10:
                        short_trace_claim_ids.append(claim_id)
                        
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_file}")
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
    
    # Identify duplicate claim IDs (appearing more than once)
    duplicate_claim_ids = [claim_id for claim_id, count in claim_id_counter.items() if count > 1]
    rt_claim_ids = set(claim_id_counter.keys())  # All unique claim IDs found
    
    print(f"Total claim IDs found in RT files: {len(rt_claim_ids)}")
    print(f"Claim IDs with short reasoning traces (<10 words): {len(short_trace_claim_ids)}")
    print(f"Duplicate claim IDs: {len(duplicate_claim_ids)}")
    
    # Find missing claim IDs
    missing_claim_ids = original_claim_ids - rt_claim_ids
    print(f"Missing claim IDs: {len(missing_claim_ids)}")
    
    # Combine all problematic claim IDs (remove duplicates)
    problematic_claim_ids = list(missing_claim_ids) + short_trace_claim_ids + duplicate_claim_ids
    problematic_claim_ids = list(set(problematic_claim_ids))
    
    print(f"Total problematic claim IDs: {len(problematic_claim_ids)}")
    
    # Second pass: clean up JSON files by removing entries with problematic claim IDs
    cleaned_files_count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                rt_data = json.load(f)
            
            # Filter out entries with problematic claim IDs
            cleaned_data = [
                item for item in rt_data 
                if item.get('claim_id') not in problematic_claim_ids
            ]
            
            # Only create updated file if changes were made
            if len(cleaned_data) != len(rt_data):
                # Create updated filename
                base_name = os.path.basename(json_file)
                name_without_ext = os.path.splitext(base_name)[0]
                updated_file = os.path.join(rt_dir, f"{name_without_ext}_updated.json")
                
                # Save cleaned data
                with open(updated_file, 'w') as f:
                    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
                
                print(f"Created cleaned file: {updated_file} "
                      f"(removed {len(rt_data) - len(cleaned_data)} entries)")
                cleaned_files_count += 1
            else:
                print(f"No changes needed for: {json_file}")
                
        except Exception as e:
            print(f"Error processing file {json_file} for cleanup: {e}")

    # Save results to a file
    output_path = "/home/ojas/scripts/datasets/AA_problematic_claim_ids.txt"
    with open(output_path, 'w') as f:
        f.write("PROBLEMATIC CLAIM IDS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Original dataset claims: {len(original_claim_ids)}\n")
        f.write(f"Found in RT files: {len(rt_claim_ids)}\n")
        f.write(f"Missing from RT files: {len(missing_claim_ids)}\n")
        f.write(f"Short reasoning traces (<10 words): {len(short_trace_claim_ids)}\n")
        f.write(f"Duplicate claim IDs: {len(duplicate_claim_ids)}\n")
        f.write(f"Total problematic claims: {len(problematic_claim_ids)}\n")
        f.write(f"Cleaned files created: {cleaned_files_count}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MISSING CLAIM IDs:\n")
        f.write("-" * 30 + "\n")
        for claim_id in sorted(missing_claim_ids):
            f.write(f"{claim_id}\n")
        
        f.write("\nCLAIM IDs WITH SHORT REASONING TRACES:\n")
        f.write("-" * 40 + "\n")
        for claim_id in sorted(short_trace_claim_ids):
            f.write(f"{claim_id}\n")
        
        f.write("\nDUPLICATE CLAIM IDs:\n")
        f.write("-" * 30 + "\n")
        for claim_id in sorted(duplicate_claim_ids):
            count = claim_id_counter[claim_id]
            f.write(f"{claim_id} (appears {count} times)\n")
    
    print(f"Detailed report saved to: {output_path}")
    print(f"Created {cleaned_files_count} updated JSON files with problematic entries removed")
    
    return problematic_claim_ids

# Additional function to just analyze without cleaning
def analyze_claim_ids(dir):
    """
    Analyze claim IDs without cleaning files
    """
    dataset_path = "/home/ojas/scripts/datasets/claims_dataset_1200.json"
    rt_dir = dir
    
    with open(dataset_path, 'r') as f:
        original_data = json.load(f)
    
    original_claim_ids = {item['unique_claim_id'] for item in original_data}
    json_files = glob.glob(os.path.join(rt_dir, "*.json"))
    
    claim_id_counter = defaultdict(int)
    short_trace_claim_ids = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                rt_data = json.load(f)
            
            for item in rt_data:
                claim_id = item.get('claim_id')
                reasoning_trace = item.get('reasoning_trace', '')
                
                if claim_id:
                    claim_id_counter[claim_id] += 1
                    if len(reasoning_trace.split()) < 10:
                        short_trace_claim_ids.append(claim_id)
                        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    rt_claim_ids = set(claim_id_counter.keys())
    duplicate_claim_ids = [claim_id for claim_id, count in claim_id_counter.items() if count > 1]
    missing_claim_ids = original_claim_ids - rt_claim_ids
    
    return {
        'original_count': len(original_claim_ids),
        'found_count': len(rt_claim_ids),
        'missing': sorted(missing_claim_ids),
        'short_traces': sorted(short_trace_claim_ids),
        'duplicates': sorted(duplicate_claim_ids),
        'duplicate_details': {cid: count for cid, count in claim_id_counter.items() if count > 1}
    }

if __name__ == "__main__":
    rt_directory = "/home/ojas/scripts/datasets/RTs/RT_Qwen_3B_CoT_4Shot"
    
    # Run full analysis and cleaning
    problematic_ids = find_and_clean_problematic_claim_ids(rt_directory)
    print(f"\nProblematic claim IDs found and removed: {len(problematic_ids)}")
    
    # Optional: Quick analysis only
    # analysis = analyze_claim_ids(rt_directory)
    # print(f"Missing: {len(analysis['missing'])}")
    # print(f"Short traces: {len(analysis['short_traces'])}")
    # print(f"Duplicates: {len(analysis['duplicates'])}")
    
import json
import os

CLAIMS_DATASET = "/home/ojas/scripts/datasets/claims_dataset_1200.json"

def iter_prompts(
    json_path=CLAIMS_DATASET, # directory with the dataset
    base_prompt_dir="/home/ojas/scripts/prompts",  # directory with 2-shot-prompt.txt, 6-shot-prompt.txt, etc.
    n_shot = 6
):
    """
    Generator that yields one prompt at a time by appending claims to a base prompt file
    based on n_shot, with final reasoning instruction.
    """
    # Select prompt file
    base_prompt_file = os.path.join(base_prompt_dir, f"{n_shot}-shot-prompt.txt")
    print(f"using Dataset File {CLAIMS_DATASET}")
    print(f"using Prompt File {base_prompt_file}")
    if not os.path.exists(base_prompt_file):
        raise FileNotFoundError(f"Base prompt file not found: {base_prompt_file}")
    
    # Load base prompt
    with open(base_prompt_file, "r", encoding="utf-8") as f:
        base_prompt_content = f.read().strip()

    # Load dataset
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Yield one prompt at a time
    for entry in data:
        claim_id = entry.get("unique_claim_id", "")
        claim = entry.get("claim", "")
        label = entry.get("mapped_verdict", "")
        evidence_text = entry.get("evidence_text", [])

        formatted_str = (
            f'Claim: "{claim}"\n'
            f'Label: "{label}"\n'
            f"Evidence Text: [\n"
            + ",\n".join(f'      "{et}"' for et in evidence_text)
            + "\n ]\n"
        )

        yield (
            base_prompt_content
            + "\n\n"
            + formatted_str
            + "\nPlease provide the reasoning traces. Ensure that the reasoning output is between <Rstart> and <Rend>."
        ) , claim_id

# Example usage:
# for prompt,claim_id in iter_prompts(n_shot=6):
#     print(prompt , claim_id)
    # break  # Uncomment if you want only the first prompt
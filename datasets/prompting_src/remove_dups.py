import json

# Load your JSON file
with open("/home/ojas/scripts/datasets/claims_dataset_1200.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Track duplicates
seen_claims = set()
duplicates = []

for entry in data:
    claim = entry["claim"].strip().lower()  # normalize spacing/case
    if claim in seen_claims:
        duplicates.append(entry)
    else:
        seen_claims.add(claim)

# Show duplicates
print(f"Found {len(duplicates)} duplicate entries.")
# for d in duplicates:
#     print(d["claim"])

# Optional: Remove duplicates and save
unique_entries = []
seen = set()
for entry in data:
    claim_norm = entry["claim"].strip().lower()
    if claim_norm not in seen:
        seen.add(claim_norm)
        unique_entries.append(entry)

print(len(unique_entries))

with open("/home/ojas/scripts/datasets/claims_dataset_1200_unique.json", "w", encoding="utf-8") as f:
    json.dump(unique_entries, f, ensure_ascii=False, indent=2)

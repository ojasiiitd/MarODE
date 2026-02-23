#COMBINE
# import json
# from pathlib import Path

# # ==========================
# # INPUT FILES
# # ==========================
# FILES = [
#     "entbank_random_150_R4-10_E_BASELINES.json",
#     "math_random_150_R4-R10_BASELINES.json",
#     "proof_random_150_R4-10_E_BASELINES.json",
#     "pubhealth_random_150_R4-10_BASELINES.json",
#     "strat_random_150_R3+_E_BASELINES.json",
# ]

# OUTPUT_FILE = "HumanEvals_combined.json"

# # ==========================
# # MERGE LOGIC
# # ==========================
# combined = []

# for fname in FILES:
#     path = Path(fname)
#     if not path.exists():
#         raise FileNotFoundError(f"Missing file: {fname}")

#     print(f"Loading {fname}...")
#     with path.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     if not isinstance(data, list):
#         raise ValueError(f"{fname} does not contain a JSON list")

#     combined.extend(data)
#     print(f"  ➜ Added {len(data)} entries")

# # ==========================
# # SAVE OUTPUT
# # ==========================
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(combined, f, indent=2, ensure_ascii=False)

# print("\n✅ Merge complete")
# print(f"Total entries combined: {len(combined)}")
# print(f"Saved to: {OUTPUT_FILE}")







# ADD HUMANEVAL SCORE

# import json
# import pandas as pd

# # ==========================
# # FILE PATHS
# # ==========================
# JSON_FILE = "HumanEvals_combined.json"
# CSV_FILE = "eval_humaneval_combined.csv"
# OUTPUT_FILE = "HumanEvals_combined_allscores.json"

# # ==========================
# # LOAD CSV (HUMAN EVALS)
# # ==========================
# print(f"Loading CSV: {CSV_FILE}")
# df = pd.read_csv(CSV_FILE)

# # Build lookup dict by claim_id
# human_eval_map = {}

# for _, row in df.iterrows():
#     cid = row["claim_id"]
#     human_eval_map[cid] = {
#         "Q1": row.get("Q1"),
#         "Q2": row.get("Q2"),
#         "Q3": row.get("Q3"),
#         "Q4": row.get("Q4"),
#         "Q5": row.get("Q5"),
#         "Q6": row.get("Q6"),
#         "combined": row.get("Human_Evaluation_Score"),
#     }

# print(f"Loaded human evals for {len(human_eval_map)} claim_ids")

# # ==========================
# # LOAD JSON
# # ==========================
# print(f"Loading JSON: {JSON_FILE}")
# with open(JSON_FILE, "r", encoding="utf-8") as f:
#     data = json.load(f)

# if not isinstance(data, list):
#     raise ValueError("JSON file must contain a list of entries")

# print(f"Total JSON entries: {len(data)}")

# # ==========================
# # ATTACH HUMAN EVAL SCORES
# # ==========================
# missing_claim_ids = []

# for entry in data:
#     cid = entry.get("claim_id")

#     if cid in human_eval_map:
#         entry["HumanEval_Scores"] = human_eval_map[cid]
#     else:
#         missing_claim_ids.append(cid)

# # ==========================
# # SAVE OUTPUT
# # ==========================
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)

# print("\n✅ Human evaluation scores attached")
# print(f"Saved output to: {OUTPUT_FILE}")

# # ==========================
# # REPORT MISSING CLAIMS
# # ==========================
# if missing_claim_ids:
#     print("\n⚠️ Claim IDs with NO matching human evaluation:")
#     for cid in sorted(set(missing_claim_ids)):
#         print(cid)
#     print(f"\nTotal missing: {len(set(missing_claim_ids))}")
# else:
#     print("\n🎉 All claim_ids matched successfully!")




# --------------------------------------------------------------------------
# SOMERSD

import json
import pandas as pd
from scipy.stats import somersd

# -------------------------------
# Load data
# -------------------------------
file_path = "/home/ojas/scripts/datasets/Human_Eval/HumanEvals_combined_allscores.json"

with open(file_path, "r") as f:
    data = json.load(f)

records = []
for entry in data:

    roscoe_means = [
        entry.get("mean_ROSCOE_SA"),
        entry.get("mean_ROSCOE_SS"),
        entry.get("mean_ROSCOE_LI"),
        entry.get("mean_ROSCOE_LC")
    ]
    roscoe_means = [x for x in roscoe_means if x is not None]
    total_roscoe_mean = sum(roscoe_means) / len(roscoe_means) if roscoe_means else None

    records.append({
        "claim_id": entry.get("claim_id"),
        "dataset_prefix": entry.get("claim_id", "").rsplit("_", 1)[0],
        "humaneval_combined": entry.get("HumanEval_Scores", {}).get("combined"),

        "roscoe-sa": entry.get("mean_ROSCOE_SA"),
        "roscoe-ss": entry.get("mean_ROSCOE_SS"),
        "roscoe-li": entry.get("mean_ROSCOE_LI"),
        "roscoe-lc": entry.get("mean_ROSCOE_LC"),
        "total_roscoe_mean": total_roscoe_mean,
        "judge_score": entry.get("judge_score"),
        "mean_coherence": entry.get("coherence_scores", {}).get("mean_coherence"),
        "mean_RECEval": entry.get("mean_RECEval"),

        "ourmetric_coherence_score": entry.get("ourmetric", {}).get("coherence_score"),
        "ourmetric_quality_score": entry.get("ourmetric", {}).get("quality_score"),
        "ourmetric_evidence_score": entry.get("ourmetric", {}).get("evidence_score"),
        "ourmetric_b1b2": entry.get("ourmetric", {}).get("b1_b2"),
        "ourmetric_b2b3": entry.get("ourmetric", {}).get("b2_b3"),
        "ourmetric_b3b1": entry.get("ourmetric", {}).get("b3_b1"),
        "ourmetric_total_score": entry.get("ourmetric", {}).get("total_score"),
    })

df = pd.DataFrame(records)

# -------------------------------
# Dataset buckets
# -------------------------------
datasets = [
    "HUMAN_ENTBANK",
    "HUMAN_MATH",
    "HUMAN_PROOF",
    "HUMAN_PUBHEALTH",
    "HUMAN_STRATEGYQA"
]

# -------------------------------
# Ordered metrics
# -------------------------------
metric_order = [
    ("ROSCOE-SA", "roscoe-sa"),
    ("ROSCOE-SS", "roscoe-ss"),
    ("ROSCOE-LI", "roscoe-li"),
    ("ROSCOE-LC", "roscoe-lc"),
    ("ROSCOE_MEAN", "total_roscoe_mean"),
    ("LLM_AS_A_JUDGE", "judge_score"),
    ("KOTONYA_AND_TONI", "mean_coherence"),
    ("RECEval", "mean_RECEval"),
    ("OM_COHORENCE (B1)", "ourmetric_coherence_score"),
    ("OM_QUALITY (B2)", "ourmetric_quality_score"),
    ("OM_EVIDENCE (B3)", "ourmetric_evidence_score"),
    ("OM_B1_B2", "ourmetric_b1b2"),
    ("OM_B2_B3", "ourmetric_b2b3"),
    ("OM_B1_B3", "ourmetric_b3b1"),
    ("OM_MEAN", "ourmetric_total_score"),
]

# -------------------------------
# Compute Somers’ D
# -------------------------------
rows = []

for ds in datasets:
    df_ds = df[df["dataset_prefix"].str.startswith(ds)]

    for out_name, col in metric_order:
        tmp = df_ds[["humaneval_combined", col]].dropna()
        if len(tmp) < 2:
            continue

        d = somersd(tmp["humaneval_combined"], tmp[col])

        rows.append({
            "Dataset": ds,
            "Metric": out_name,
            "SomersD": d.statistic,
            "pValue": d.pvalue,
            "N": len(tmp)
        })

results_df = pd.DataFrame(rows)

# Preserve metric order in output
results_df["Metric"] = pd.Categorical(
    results_df["Metric"],
    categories=[m[0] for m in metric_order],
    ordered=True
)

results_df = results_df.sort_values(["Dataset", "Metric"])

print(results_df.to_string(index=False))
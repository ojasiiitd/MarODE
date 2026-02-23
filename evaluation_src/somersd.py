import json
import pandas as pd
from scipy.stats import somersd   # assuming your scipy has this
import glob

# path to directory
dir_path = "/home/ojas/scripts/datasets/Final_Runs/"

# get all matching files
file_paths = glob.glob(dir_path + "filtered_*.json")

for file_path in file_paths:
    print("\nProcessing File:", file_path, "\n")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract rowsP
    records = []
    for entry in data:

        roscoe_means = [
            entry.get("mean_ROSCOE_SA"),
            entry.get("mean_ROSCOE_SS"),
            entry.get("mean_ROSCOE_LI"),
            entry.get("mean_ROSCOE_LC"),
        ]
        roscoe_means = [x for x in roscoe_means if x is not None]

        total_roscoe_mean = sum(roscoe_means) / len(roscoe_means) if roscoe_means else None

        records.append({
            "claim_id": entry.get("claim_id"),
            "perturbation_score": entry.get("perturbation_score"),
            "roscoe-sa": entry.get("mean_ROSCOE_SA"),
            "roscoe-ss": entry.get("mean_ROSCOE_SS"),
            "roscoe-li": entry.get("mean_ROSCOE_LI"),
            "roscoe-lc": entry.get("mean_ROSCOE_LC"),
            "total_roscoe_mean": total_roscoe_mean,
            "judge_score": entry.get("judge_score"),
            "mean_coherence": entry.get("coherence_scores", {}).get("mean_coherence"),
            "mean_RECEval": entry.get("mean_RECEval"),
            "ourmetric_total_score": entry.get("ourmetric", {}).get("total_score"),
            "ourmetric_coherence_score": entry.get("ourmetric", {}).get("coherence_score"),
            "ourmetric_evidence_score": entry.get("ourmetric", {}).get("evidence_score"),
            "ourmetric_quality_score": entry.get("ourmetric", {}).get("quality_score"),
            "ourmetric_b1b2": entry.get("ourmetric", {}).get("b1_b2"),
            "ourmetric_b2b3": entry.get("ourmetric", {}).get("b2_b3"),
            "ourmetric_b3b1": entry.get("ourmetric", {}).get("b3_b1"),
        })

    df = pd.DataFrame(records).dropna()

    metric_list = [
        "roscoe-sa", "roscoe-ss", "roscoe-li", "roscoe-lc",
        "total_roscoe_mean", "judge_score", "mean_coherence",
        "mean_RECEval", "ourmetric_coherence_score",
        "ourmetric_quality_score", "ourmetric_evidence_score",
        "ourmetric_b1b2", "ourmetric_b2b3", "ourmetric_b3b1",
        "ourmetric_total_score"
    ]

    # Compute Somers' D and p-values
    rows = []
    for col in metric_list:
        d = somersd(df["perturbation_score"], df[col])
        rows.append({
            "Metric": col,
            "SomersD": d.statistic,
            "pValue": d.pvalue
        })

    results_df = pd.DataFrame(rows)

    print(results_df.to_string(index=False))

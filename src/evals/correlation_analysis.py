import json
import glob
import argparse
import pandas as pd
from scipy.stats import somersd


# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------
parser = argparse.ArgumentParser(
    description="Compute Somers' D correlation between perturbation score and evaluation metrics."
)

parser.add_argument("--dir", required=True, help="Directory containing JSON files")
parser.add_argument("--pattern", default="filtered_*.json", help="File pattern to match")
parser.add_argument("--save", default=None, help="Optional path to save correlation results (CSV)")

args = parser.parse_args()


# ---------------------------------------------------
# Metrics to Evaluate
# ---------------------------------------------------
METRIC_LIST = [
    "roscoe-sa",
    "roscoe-ss",
    "roscoe-li",
    "roscoe-lc",
    "total_roscoe_mean",
    "judge_score",
    "mean_coherence",
    "mean_RECEval",
    "ourmetric_coherence_score",
    "ourmetric_quality_score",
    "ourmetric_evidence_score",
    "ourmetric_b1b2",
    "ourmetric_b2b3",
    "ourmetric_b3b1",
    "ourmetric_total_score",
]


# ---------------------------------------------------
# Process Files
# ---------------------------------------------------
file_paths = glob.glob(f"{args.dir.rstrip('/')}/{args.pattern}")

if not file_paths:
    print("No files matched.")
    exit()

all_results = []

for file_path in file_paths:
    print(f"\nProcessing File: {file_path}\n")

    with open(file_path, "r") as f:
        data = json.load(f)

    records = []
    for entry in data:

        roscoe_means = [
            entry.get("mean_ROSCOE_SA"),
            entry.get("mean_ROSCOE_SS"),
            entry.get("mean_ROSCOE_LI"),
            entry.get("mean_ROSCOE_LC"),
        ]
        roscoe_means = [x for x in roscoe_means if x is not None]
        total_roscoe_mean = (
            sum(roscoe_means) / len(roscoe_means) if roscoe_means else None
        )

        records.append({
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

    df = pd.DataFrame(records)

    rows = []

    for metric in METRIC_LIST:
        if metric not in df.columns:
            continue

        sub_df = df[["perturbation_score", metric]].dropna()
        if sub_df.empty:
            continue

        d = somersd(sub_df["perturbation_score"], sub_df[metric])

        rows.append({
            "File": file_path.split("/")[-1],
            "Metric": metric,
            "SomersD": d.statistic,
            "pValue": d.pvalue,
            "N": len(sub_df),
        })

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))

    all_results.extend(rows)


# ---------------------------------------------------
# Optional Save
# ---------------------------------------------------
if args.save:
    pd.DataFrame(all_results).to_csv(args.save, index=False)
    print(f"\nSaved correlation results to: {args.save}")
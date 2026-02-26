import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon


# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(
    description="Wilcoxon signed-rank test across shot settings."
)

parser.add_argument("--shot1", required=True, help="Path to 1-shot JSON file")
parser.add_argument("--shot2", required=True, help="Path to 2-shot JSON file")
parser.add_argument("--shot4", required=True, help="Path to 4-shot JSON file")
parser.add_argument("--output", default="Shot_Difference_Wilcoxon_Results.csv")

args = parser.parse_args()


# -----------------------------
# Metric Extraction
# -----------------------------
def extract_metrics(entry):
    return {
        "ROSCOE-SA": entry.get("mean_ROSCOE_SA"),
        "ROSCOE-SS": entry.get("mean_ROSCOE_SS"),
        "ROSCOE-LI": entry.get("mean_ROSCOE_LI"),
        "ROSCOE-LC": entry.get("mean_ROSCOE_LC"),
        "ROSCOE_MEAN": np.nanmean([
            entry.get("mean_ROSCOE_SA"),
            entry.get("mean_ROSCOE_SS"),
            entry.get("mean_ROSCOE_LI"),
            entry.get("mean_ROSCOE_LC"),
        ]),
        "KOTONYA_AND_TONI": entry.get("coherence_scores", {}).get("mean_coherence"),
        "LLM_AS_A_JUDGE": entry.get("judge_score"),
        "RECEval": entry.get("mean_RECEval"),

        # Our Metric (OM)
        "OM_COHERENCE (B1)": entry.get("ourmetric", {}).get("coherence_score"),
        "OM_QUALITY (B2)": entry.get("ourmetric", {}).get("quality_score"),
        "OM_EVIDENCE (B3)": entry.get("ourmetric", {}).get("evidence_score"),
        "OM_B1_B2": entry.get("ourmetric", {}).get("b1_b2"),
        "OM_B2_B3": entry.get("ourmetric", {}).get("b2_b3"),
        "OM_B1_B3": entry.get("ourmetric", {}).get("b3_b1"),
        "OM_MEAN": entry.get("ourmetric", {}).get("total_score"),
    }


# -----------------------------
# Load Files
# -----------------------------
def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


data_1 = load_json(args.shot1)
data_2 = load_json(args.shot2)
data_4 = load_json(args.shot4)

print("✅ Loaded shot-wise files")


# -----------------------------
# Build DataFrame
# -----------------------------
def build_df(entries):
    rows = []
    for e in entries:
        row = {
            "claim_id": e.get("claim_id"),
            "dataset": e.get("dataset"),
            "model": e.get("model", e.get("claim_id")),
        }
        row.update(extract_metrics(e))
        rows.append(row)
    return pd.DataFrame(rows)


df_1 = build_df(data_1)
df_2 = build_df(data_2)
df_4 = build_df(data_4)

key_cols = ["claim_id", "dataset", "model"]

df_1 = df_1.set_index(key_cols)
df_2 = df_2.set_index(key_cols)
df_4 = df_4.set_index(key_cols)

common_index = (
    df_1.index
    .intersection(df_2.index)
    .intersection(df_4.index)
)

df_1 = df_1.loc[common_index]
df_2 = df_2.loc[common_index]
df_4 = df_4.loc[common_index]

print(f"✅ Aligned on {len(common_index)} common instances")


# -----------------------------
# Wilcoxon Analysis
# -----------------------------
def wilcoxon_analysis(a, b):
    diff = (b - a).dropna()

    if diff.nunique() <= 1:
        return np.nan, np.nan, "no-change", np.nan

    stat, p = wilcoxon(diff)
    median_diff = np.median(diff)

    n = len(diff)
    rbc = 1 - (2 * stat) / (n * (n + 1))

    direction = (
        "increase" if median_diff > 0 else
        "decrease" if median_diff < 0 else
        "no-change"
    )

    return median_diff, p, direction, rbc


results = []

metrics = df_1.columns

comparisons = [
    ("2-shot vs 1-shot", df_1, df_2),
    ("4-shot vs 1-shot", df_1, df_4),
    ("4-shot vs 2-shot", df_2, df_4),
]

for name, dfa, dfb in comparisons:
    for metric in metrics:
        med, p, direction, effect = wilcoxon_analysis(dfa[metric], dfb[metric])

        results.append({
            "comparison": name,
            "metric": metric,
            "median_difference": med,
            "wilcoxon_p": p,
            "effect_size_rbc": effect,
            "direction": direction,
            "significant(p<0.05)": (p < 0.05) if not np.isnan(p) else False,
        })


results_df = pd.DataFrame(results)
results_df.to_csv(args.output, index=False)

print("Wilcoxon analysis complete")
print(f"Results saved to {args.output}")
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------
# File → Dataset column
# ---------------------
file_to_dataset = {
    # "TESTBASELINES.json":"test",
    "entbank_random_150_R4-10_E_BASELINES_new.json": "ENTAILMENT_BANK",
    "proof_random_150_R4-10_E_BASELINES_new.json": "PROOFWRITER",
    "math_random_150_R4-R10_BASELINES.json": "GSM",
    "strat_random_150_R3+_E_BASELINES_new.json": "STRATQA",
    "pubhealth_random_150_R4-10_BASELINES.json": "PUBMED",
}

files = list(file_to_dataset.keys())

# Metrics (rows in final CSV)
metric_rows = [
    "ROSCOE-SA",
    "ROSCOE-SS",
    "ROSCOE-LI",
    "ROSCOE-LC",
    "ROSCOE_MEAN",
    "LLM_AS_A_JUDGE",
    "KOTONYA_AND_TONI",
    "RECEval",
    "OM_COHERENCE (B1)",
    "OM_QUALITY (B2)",
    "OM_EVIDENCE (B3)",
    "OM_B1_B2",
    "OM_B2_B3",
    "OM_B1_B3",
    "OM_MEAN"
]

# A dict to collect column outputs
summary = {metric: {} for metric in metric_rows}

# Utility to safely get nested values
def get_float(entry, *keys):
    cur = entry
    try:
        for k in keys:
            cur = cur[k]
        return float(cur)
    except:
        return np.nan

for filename in files:
    dataset_name = file_to_dataset[filename]
    path = Path(filename)

    if not path.exists():
        print(f"⚠️ Missing file: {filename}")
        for m in metric_rows:
            summary[m][dataset_name] = np.nan
        continue

    with path.open("r") as f:
        data = json.load(f)

    # Collect lists
    sa = []
    ss = []
    li = []
    lc = []
    judge = []
    coherence = []
    receval = []

    om_coh = []
    om_quality = []
    om_evidence = []
    om_b1_b2 = []
    om_b2_b3 = []
    om_b3_b1 = []
    om_total = []

    for entry in data:
        sa.append(get_float(entry, "mean_ROSCOE_SA"))
        ss.append(get_float(entry, "mean_ROSCOE_SS"))
        li.append(get_float(entry, "mean_ROSCOE_LI"))
        lc.append(get_float(entry, "mean_ROSCOE_LC"))
        judge.append(get_float(entry, "judge_score"))
        coherence.append(get_float(entry, "coherence_scores", "mean_coherence"))
        receval.append(get_float(entry, "mean_RECEval"))

        om_coh.append(get_float(entry, "ourmetric", "coherence_score"))
        om_quality.append(get_float(entry, "ourmetric", "quality_score"))
        om_evidence.append(get_float(entry, "ourmetric", "evidence_score"))
        om_b1_b2.append(get_float(entry, "ourmetric", "b1_b2"))
        om_b2_b3.append(get_float(entry, "ourmetric", "b2_b3"))
        om_b3_b1.append(get_float(entry, "ourmetric", "b3_b1"))
        om_total.append(get_float(entry, "ourmetric", "total_score"))

    # Compute means
    mean_sa = np.nanmean(sa)
    mean_ss = np.nanmean(ss)
    mean_li = np.nanmean(li)
    mean_lc = np.nanmean(lc)

    roscoe_mean = np.nanmean([mean_sa, mean_ss, mean_li, mean_lc])

    summary["ROSCOE-SA"][dataset_name] = mean_sa
    summary["ROSCOE-SS"][dataset_name] = mean_ss
    summary["ROSCOE-LI"][dataset_name] = mean_li
    summary["ROSCOE-LC"][dataset_name] = mean_lc
    summary["ROSCOE_MEAN"][dataset_name] = roscoe_mean

    summary["LLM_AS_A_JUDGE"][dataset_name] = np.nanmean(judge)
    summary["KOTONYA_AND_TONI"][dataset_name] = np.nanmean(coherence)
    summary["RECEval"][dataset_name] = np.nanmean(receval)

    summary["OM_COHERENCE (B1)"][dataset_name] = np.nanmean(om_coh)
    summary["OM_QUALITY (B2)"][dataset_name] = np.nanmean(om_quality)
    summary["OM_EVIDENCE (B3)"][dataset_name] = np.nanmean(om_evidence)

    summary["OM_B1_B2"][dataset_name] = np.nanmean(om_b1_b2)
    summary["OM_B2_B3"][dataset_name] = np.nanmean(om_b2_b3)
    summary["OM_B1_B3"][dataset_name] = np.nanmean(om_b3_b1)

    summary["OM_MEAN"][dataset_name] = np.nanmean(om_total)

# -------- BUILD DATAFRAME IN DESIRED SHAPE -------- #

df = pd.DataFrame(summary).T  # rows = metrics, columns = datasets

# Round to 4 decimals
df = df.applymap(lambda x: round(x, 4) if isinstance(x, float) else x)

# Save CSV
df.to_csv("metric_summary_table.csv")

print("✅ Saved: metric_summary_table.csv")
print(df)
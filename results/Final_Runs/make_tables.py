import json
import pandas as pd
from pathlib import Path

# -------- CONFIG --------
input_files = [
    "RT_Original_1shot.json",
    "RT_Original_2shot.json",
    "RT_Original_4shot.json",
]

output_csv = "RT_Original_merged_metrics.csv"
# ------------------------

rows = []

for file in input_files:
    with open(file, "r") as f:
        data = json.load(f)

    for entry in data:
        # --- Base metadata ---
        claim_id = entry.get("claim_id")
        model = entry.get("model")
        shots = entry.get("shots")
        dataset = entry.get("dataset")

        # --- ROSCOE ---
        roscoe_sa = entry.get("mean_ROSCOE_SA")
        roscoe_ss = entry.get("mean_ROSCOE_SS")
        roscoe_li = entry.get("mean_ROSCOE_LI")
        roscoe_lc = entry.get("mean_ROSCOE_LC")

        roscoe_values = [
            v for v in [roscoe_sa, roscoe_ss, roscoe_li, roscoe_lc] if v is not None
        ]
        roscoe_mean = sum(roscoe_values) / len(roscoe_values) if roscoe_values else None

        # --- LLM-as-a-Judge ---
        llm_as_judge = entry.get("judge_score")

        # --- Kotonya & Toni (Coherence) ---
        kt_coherence = entry.get("coherence_scores", {}).get("mean_coherence")

        # --- RECEval ---
        receval = entry.get("mean_RECEval")

        # --- MarODE / Our Metric ---
        om = entry.get("ourmetric", {})

        om_coherence = om.get("coherence_score")
        om_quality = om.get("quality_score")
        om_evidence = om.get("evidence_score")

        om_b1_b2 = om.get("b1_b2")
        om_b2_b3 = om.get("b2_b3")
        om_b1_b3 = om.get("b3_b1")

        om_components = [
            v for v in [om_coherence, om_quality, om_evidence] if v is not None
        ]
        om_mean = sum(om_components) / len(om_components) if om_components else None

        # --- Assemble row ---
        rows.append({
            "claim_id": claim_id,
            "model": model,
            "shots": shots,
            "dataset": dataset,

            "ROSCOE-SA": roscoe_sa,
            "ROSCOE-SS": roscoe_ss,
            "ROSCOE-LI": roscoe_li,
            "ROSCOE-LC": roscoe_lc,
            "ROSCOE_MEAN": roscoe_mean,

            "LLM_AS_A_JUDGE": llm_as_judge,
            "KOTONYA_AND_TONI": kt_coherence,
            "RECEval": receval,

            "MarODE_Coherence": om_coherence,
            "MarODE_Quality": om_quality,
            "MarODE_Evidence": om_evidence,

            "OM_B1_B2": om_b1_b2,
            "OM_B2_B3": om_b2_b3,
            "OM_B1_B3": om_b1_b3,

            "MarODE_MEAN": om_mean
        })

# --- Create DataFrame ---
df = pd.DataFrame(rows)

# --- Save CSV ---
df.to_csv(output_csv, index=False)

print(f"Saved merged CSV to: {output_csv}")
print(f"Total rows: {len(df)}")


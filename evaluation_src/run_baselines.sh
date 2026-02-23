#!/bin/bash

# Check if input, output and gpu are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <input_file> <output_file> <gpu>"
    exit 1
fi

# Input, output and GPU paths from command line
INPUT="$1"
OUTPUT="$2"
GPU="$3"

# Step 1: ROSCOE-SA
python ROSCOE-SA_processor.py --input "$INPUT" --output "$OUTPUT" --evidence "True" --gpu "$GPU"
echo "ROSCOE-SA Processing completed."

# Step 6: RECEval
python RECEVAL.py --input "$OUTPUT" --output "$OUTPUT" --gpu "$GPU"
echo "RECEval Processing completed."

# Step 3: ROSCOE-LI
python ROSCOE-LI_processor.py --input "$OUTPUT" --output "$OUTPUT" --evidence "True" --gpu "$GPU"
echo "ROSCOE-LI Processing completed."

# Step 2: ROSCOE-SS
python ROSCOE-SS_processor.py --input "$OUTPUT" --output "$OUTPUT" --evidence "True" --gpu "$GPU"
echo "ROSCOE-SS Processing compˀleted."

# Step 4: ROSCOE-LC
python ROSCOE-LC_processor.py --input "$OUTPUT" --output "$OUTPUT" --evidence "True" --gpu "$GPU"
echo "ROSCOE-LC Processing completed."

# Step 5: Kottoyama-Toni Coherence
python Kottoyama_processor.py --input "$OUTPUT" --output "$OUTPUT" --evidence "True" --gpu "$GPU"
echo "Kottoyama Processing completed."

# Step 7: LLM As A Judge
python Prometheus_LLM_JUDGE.py --input "$OUTPUT" --output "$OUTPUT" --evidence "True" --gpu "$GPU"
echo "Prometheus Judge completed."

# Step 8: Our Metric
python OurMetric.py --input "$OUTPUT" --output "$OUTPUT" --evidence "True" --gpu "$GPU"
echo "OurMetric Processing completed."

echo "Processing completed. Output saved to $OUTPUT"

# Example:
# ./run_baselines.sh /home/ojas/scripts/datasets/Final_Runs/DeepLlama_8B_1shot_PTRB.json /home/ojas/scripts/datasets/Final_Runs/DeepLlama_8B_1shot_PTRB_BASELINES.json.json 0

# ./run_baselines.sh /home/ojas/scripts/datasets/Human_Eval/sample_exps/test.json /home/ojas/scripts/datasets/Human_Eval/sample_exps/test_baseline.json 3

# ./run_baselines.sh /home/ojas/scripts/datasets/Human_Eval/strat_random_150_R3+_E.json /home/ojas/scripts/datasets/Human_Eval/strat_random_150_R3+_E_BASELINES_new.json 0

# ./run_baselines.sh /home/ojas/scripts/datasets/Human_Eval/proof_random_150_R4-10_E.json /home/ojas/scripts/datasets/Human_Eval/proof_random_150_R4-10_E_BASELINES_new.json 1

# ./run_baselines.sh /home/ojas/scripts/datasets/Human_Eval/entbank_random_150_R4-10_E.json /home/ojas/scripts/datasets/Human_Eval/entbank_random_150_R4-10_E_BASELINES_new.json 0


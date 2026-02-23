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
python ROSCOE-SA_processor.py --input "$INPUT" --output "$OUTPUT" --evidence "False" --gpu "$GPU"
echo "ROSCOE-SA Processing completed."

# Step 6: RECEval
python RECEVAL.py --input "$OUTPUT" --output "$OUTPUT" --gpu "$GPU"
echo "RECEval Processing completed."

# Step 3: ROSCOE-LI
python ROSCOE-LI_processor.py --input "$OUTPUT" --output "$OUTPUT" --evidence "False" --gpu "$GPU"
echo "ROSCOE-LI Processing completed."

# Step 2: ROSCOE-SS
python ROSCOE-SS_processor.py --input "$OUTPUT" --output "$OUTPUT" --evidence "False" --gpu "$GPU"
echo "ROSCOE-SS Processing completed."

# Step 4: ROSCOE-LC
python ROSCOE-LC_processor.py --input "$OUTPUT" --output "$OUTPUT" --evidence "False" --gpu "$GPU"
echo "ROSCOE-LC Processing completed."

# Step 5: Kottoyama-Toni Coherence
python Kottoyama_processor.py --input "$OUTPUT" --output "$OUTPUT" --evidence "False" --gpu "$GPU"
echo "Kottoyama Processing completed."

# Step 7: LLM As A Judge
python Prometheus_LLM_JUDGE.py --input "$OUTPUT" --output "$OUTPUT" --evidence "False" --gpu "$GPU"
echo "Prometheus Judge completed."

# Step 8: Our Metric
python OurMetric.py --input "$OUTPUT" --output "$OUTPUT" --evidence "False" --gpu "$GPU"
echo "OurMetric Processing completed."

echo "Processing completed. Output saved to $OUTPUT"

# Example:
# ./run_baselines_noevidence.sh /home/ojas/scripts/datasets/Human_Eval/math_random_150_R4-R10.json /home/ojas/scripts/datasets/Human_Eval/math_random_150_R4-R10_BASELINES.json 2

# ./run_baselines_noevidence.sh /home/ojas/scripts/datasets/Human_Eval/pubhealth_random_150_R4-10.json /home/ojas/scripts/datasets/Human_Eval/pubhealth_random_150_R4-10_BASELINES.json 2

# python OurMetric.py --input /home/ojas/scripts/datasets/Human_Eval/pubhealth_random_150_R4-10.json --output /home/ojas/scripts/datasets/Human_Eval/pubhealth_random_150_R4-10_OURMETRIC.json --evidence "False" --gpu 2

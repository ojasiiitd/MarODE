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

# Step 8: Our Metric
python OurMetric.py --input "$INPUT" --output "$OUTPUT" --evidence "True" --gpu "$GPU"
echo "OurMetric Processing completed."

# Example:
# ./run_quick_baselines.sh /home/ojas/scripts/datasets/Human_Eval/math_random_150_R4-R10_BASELINES.json /home/ojas/scripts/datasets/Human_Eval/math_random_150_R4-R10_BASELINES_check.json 0

# ./run_quick_baselines.sh /home/ojas/scripts/datasets/Final_Runs/DeepLlama_8B_1shot_BASELINES.json /home/ojas/scripts/datasets/Final_Runs/DeepLlama_8B_1shot_BASELINES_newcheck.json 0



# runs


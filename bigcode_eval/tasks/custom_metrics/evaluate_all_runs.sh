#!/bin/bash

# ============================================
# Concurrent Evaluation Script
# ============================================
# This script runs concurrent evaluations of all JSON files
# in a specified directory using a Python script.
# ============================================

# ----------------------------
# Configuration Variables
# ----------------------------

# Directory containing the JSON files to evaluate
LOAD_DIR="/Users/mpietruszka/Downloads/wszystkie_outy_ale_ewaluuj_moim_kodem"

# Path to the Python script to execute
PYTHON_SCRIPT="/Users/mpietruszka/Repos/aj-bigcode/main.py"

# Fixed parameters for the Python command
MODEL="openai-completions"
TASKS="ds_f_eng"
ALLOW_CODE_EXECUTION="--allow_code_execution"
USE_AUTH_TOKEN="--use_auth_token"
SAVE_GENERATIONS="--save_generations"
PRECISION="--precision bf16"
DO_SAMPLE="--do_sample False"
TRUST_REMOTE_CODE="--trust_remote_code"
MODEL_ARGS="--model_args model=o1-preview-2024-09-12,tokenizer=Xenova/gpt-4"
MAX_NEW_TOKENS="--max_new_tokens 3000"
BATCH_SIZE="--batch_size 3"

# Prefix for the save_generations_path to ensure uniqueness
SAVE_GENERATIONS_PATH_PREFIX="o1-preview-2024-09-12-run"

# ----------------------------
# Function Definitions
# ----------------------------

# Function to display script usage
usage() {
    echo "Usage: $0"
    echo "Ensure that LOAD_DIR and PYTHON_SCRIPT variables are set correctly in the script."
    exit 1
}

# Check if LOAD_DIR exists
if [ ! -d "$LOAD_DIR" ]; then
    echo "Error: Directory '$LOAD_DIR' does not exist."
    exit 1
fi

# Check if PYTHON_SCRIPT exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' does not exist."
    exit 1
fi

# ----------------------------
# Main Execution
# ----------------------------

echo "Starting concurrent evaluations..."

# Iterate over each JSON file in the LOAD_DIR
for file in "$LOAD_DIR"/*.json; do
    # Check if the glob didn't match any files
    if [ ! -e "$file" ]; then
        echo "No JSON files found in '$LOAD_DIR'."
        exit 0
    fi

    # Extract the base filename without extension for unique save path
    base_name=$(basename "$file" .json)
    SAVE_GENERATIONS_PATH="${SAVE_GENERATIONS_PATH_PREFIX}-${base_name}"

    echo "Processing '$file'..."

    # Execute the Python script in the background
    python3.10 "$PYTHON_SCRIPT" \
        --model="$MODEL" \
        --tasks="$TASKS" \
        $ALLOW_CODE_EXECUTION \
        $USE_AUTH_TOKEN \
        $SAVE_GENERATIONS \
        $PRECISION \
        $DO_SAMPLE \
        $TRUST_REMOTE_CODE \
        $MODEL_ARGS \
        $MAX_NEW_TOKENS \
        --save_generations_path "$SAVE_GENERATIONS_PATH" \
        $BATCH_SIZE \
        --load_generations_path "$file" &

    # Optional: Limit the number of concurrent jobs
    # Uncomment the following lines to limit to, for example, 4 concurrent jobs
    MAX_JOBS=6
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
done

# Wait for all background processes to finish
wait

echo "All evaluations completed successfully."

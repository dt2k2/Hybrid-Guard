#!/usr/bin/env bash
set -e

# Detect Python interpreter
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -f ".venv/Scripts/python" ]; then
    PYTHON_BIN=".venv/Scripts/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python"
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    exec "$PYTHON_BIN" train_url_model.py --help
fi

DATA_PATH="Dataset.csv"
OUTPUT_DIR="artifacts/lr_final"

if [[ -n "$1" && "$1" != -* ]]; then
    DATA_PATH="$1"
    shift
fi

if [[ -n "$1" && "$1" != -* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

echo "====================================================="
echo " Training Logistic Regression (LR) Model"
echo " Dataset:    $DATA_PATH"
echo " Output dir: $OUTPUT_DIR"
echo "====================================================="

"$PYTHON_BIN" train_url_model.py \
    --data "$DATA_PATH" \
    --model lr \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "====================================================="
echo " LR Training Completed!"
echo " Artifacts saved in: $OUTPUT_DIR"
echo "====================================================="

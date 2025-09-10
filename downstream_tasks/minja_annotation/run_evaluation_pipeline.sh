#!/bin/bash

# Script to run evaluation pipeline for multiple checkpoints
# Usage: ./run_evaluation_pipeline.sh --config <config_path> --device <device> <checkpoint_path1> [checkpoint_path2] [checkpoint_path3] ...

set -e  # Exit on any error

# Initialize variables
CONFIG_PATH=""
DEVICE=""
CHECKPOINT_PATHS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option $1"
            echo "Usage: $0 --config <config_path> --device <device> <checkpoint_path1> [checkpoint_path2] [checkpoint_path3] ..."
            echo "Example: $0 --config configs/basic.yaml --device 0 /path/to/checkpoint1.safetensors /path/to/checkpoint2.safetensors"
            exit 1
            ;;
        *)
            CHECKPOINT_PATHS+=("$1")
            shift
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$CONFIG_PATH" ]; then
    echo "Error: --config is required"
    echo "Usage: $0 --config <config_path> --device <device> <checkpoint_path1> [checkpoint_path2] [checkpoint_path3] ..."
    exit 1
fi

if [ -z "$DEVICE" ]; then
    echo "Error: --device is required"
    echo "Usage: $0 --config <config_path> --device <device> <checkpoint_path1> [checkpoint_path2] [checkpoint_path3] ..."
    exit 1
fi

if [ ${#CHECKPOINT_PATHS[@]} -eq 0 ]; then
    echo "Error: At least one checkpoint path is required"
    echo "Usage: $0 --config <config_path> --device <device> <checkpoint_path1> [checkpoint_path2] [checkpoint_path3] ..."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file '$CONFIG_PATH' does not exist"
    exit 1
fi

echo "=========================================="
echo "Starting evaluation pipeline"
echo "Config: $CONFIG_PATH"
echo "Device: $DEVICE"
echo "Checkpoints: ${CHECKPOINT_PATHS[*]}"
echo "=========================================="

# Process each checkpoint
for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing checkpoint: $CHECKPOINT_PATH"
    echo "=========================================="
    
    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "Error: Checkpoint file '$CHECKPOINT_PATH' does not exist"
        echo "Skipping this checkpoint..."
        continue
    fi
    
    # Step 1: Run evaluate_on_chromosome.py
    echo "Step 1: Running evaluate_on_chromosome.py..."
    GENALM_HOME=$(realpath ../../) CUDA_VISIBLE_DEVICES="$DEVICE" python evaluate_on_chromosome.py \
        --config "$CONFIG_PATH" \
        --model_cpt "$CHECKPOINT_PATH"
    
    if [ $? -ne 0 ]; then
        echo "Error: evaluate_on_chromosome.py failed for checkpoint $CHECKPOINT_PATH"
        echo "Skipping to next checkpoint..."
        continue
    fi
    
    # Determine the output directory where bigWig files were created
    # Based on the evaluate_on_chromosome.py logic:
    # output_dir = os.path.join(os.path.dirname(model_cpt), "eval", fasta_basename, chromosome)
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
    CHECKPOINT_BASENAME=$(basename "$CHECKPOINT_PATH" .safetensors)
    
    # We need to find the eval directory - it should be in the checkpoint directory
    EVAL_DIR="$CHECKPOINT_DIR/eval/NC_060944.1"
    
    if [ ! -d "$EVAL_DIR" ]; then
        echo "Error: Expected eval directory '$EVAL_DIR' not found"
        echo "Skipping to next checkpoint..."
        continue
    fi
    
    # Step 2: Run preds2metric.py with different thresholds
    echo "Step 2: Running preds2metric.py..."
    
    # Run with threshold 0.1
    echo "  Running with threshold 0.1..."
    python preds2metric.py \
        --bigwig_path "$EVAL_DIR" \
        --threshold 0.1 \
        --max_k 250
    
    if [ $? -ne 0 ]; then
        echo "Error: preds2metric.py failed for checkpoint $CHECKPOINT_PATH with threshold 0.1"
    else
        echo "  ✓ Completed with threshold 0.1"
    fi
    
    # Run with threshold 0.5
    echo "  Running with threshold 0.5..."
    python preds2metric.py \
        --bigwig_path "$EVAL_DIR" \
        --threshold 0.5 \
        --max_k 250
    
    if [ $? -ne 0 ]; then
        echo "Error: preds2metric.py failed for checkpoint $CHECKPOINT_PATH with threshold 0.5"
    else
        echo "  ✓ Completed with threshold 0.5"
    fi
    
    echo "✓ Completed processing checkpoint: $CHECKPOINT_PATH"
done

echo ""
echo "=========================================="
echo "Evaluation pipeline completed!"
echo "=========================================="

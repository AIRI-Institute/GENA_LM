#!/usr/bin/env bash
set -euo pipefail

PRETRAINED_CONFIG_NAME="answerdotai/ModernBERT-base"
MODEL_PATH="./checkpoint-106000/model.safetensors"
# DATA_DIR="/disk/10tb/home/chepurova/chepurova/mammals_data_contig_separated/"
DATA_DIR="data/mammals_data_contig_separated/"
MAX_LENGTH=512
BATCH_SIZE=8
CUDA_VISIBLE_DEVICES="0"
INFERENCE_RESULT_DIR="mammals_inference_runs/"
EXPERIMENT_NAME="mammals_chrY_ratio_0.25_modern_gena"
N_PER_SAMPLE=60_000

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --pretrained-config-name     Config of the used model
  --model-path PATH       Path to model checkpoint (default: $MODEL_PATH)
  --data-dir PATH         Path to mammals data directory (default: $DATA_DIR)
  --max_length            Size of the context window
  --batch-size N          Inference batch size (default: $BATCH_SIZE)
  --cuda-visible-devices  CUDA device id(s) to use, e.g. 0 or 1,2 (default: $CUDA_VISIBLE_DEVICES)
  --experiment-dir NAME   Experiment subfolder inside $INFERENCE_RESULT_DIR
                          (default: $EXPERIMENT_NAME)
  --n-per-sample N          Number of predictions per sample (default: $N_PER_SAMPLE)
  -h, --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained-config-name)
            PRETRAINED_CONFIG_NAME="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --cuda-visible-devices)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --experiment-dir)
            EXPERIMENT_NAME="${2%/}"
            EXPERIMENT_NAME="${EXPERIMENT_NAME##*/}"
            shift 2
            ;;
        --n-per-sample)
            N_PER_SAMPLE="$2"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

INFERENCE_RESULT_DIR="${INFERENCE_RESULT_DIR%/}/"

export CUDA_VISIBLE_DEVICES

mkdir -p "$INFERENCE_RESULT_DIR"
mkdir -p "${INFERENCE_RESULT_DIR}${EXPERIMENT_NAME}"

OUTPUT_FILE_PREFIX="${INFERENCE_RESULT_DIR}${EXPERIMENT_NAME}/"
METRICS_OUTPUT_FILE_PREFIX="${INFERENCE_RESULT_DIR}${EXPERIMENT_NAME}/"

echo "Pretrained config:       $PRETRAINED_CONFIG_NAME"
echo "Model path:              $MODEL_PATH"
echo "Data dir:                $DATA_DIR"
echo "Batch size:              $BATCH_SIZE"
echo "Max length:              $MAX_LENGTH"
echo "CUDA visible devices:    $CUDA_VISIBLE_DEVICES"
echo "Inference result dir:    $INFERENCE_RESULT_DIR"
echo "Experiment name:         $EXPERIMENT_NAME"
echo "Output prefix:           $OUTPUT_FILE_PREFIX"
echo "Number of predictions per sample: $N_PER_SAMPLE"  
# Piliocolobus tephrosceles
# Bos javanicus

TRAIN_SPECIES=(
    "Apodemus sylvaticus" "Arvicanthis niloticus" "Bos javanicus" "Bos taurus"
    "Budorcas taxicolor" "Canis lupus familiaris" "Delphinus delphis" "Globicephala melas" "Homo sapiens"
    "Lutra lutra" "Macaca mulatta" "Macaca thibetana thibetana" "Meles meles" "Mesoplodon densirostris" "Mus musculus"
    "Mustela erminea" "Mustela lutreola" "Neomonachus schauinslandi" "Ovis aries" "Pan paniscus" "Papio anubis"
    "Piliocolobus tephrosceles" "Pongo abelii" "Rattus norvegicus" "Rattus rattus" "Tursiops truncatus" "Zalophus californianus"
)

VALID_SPECIES=(
    "Balaenoptera musculus" "Callithrix jacchus" "Cervus canadensis" "Chionomys nivalis"
    "Dama dama" "Eubalaena glacialis" "Jaculus jaculus" "Lynx canadensis" "Meriones unguiculatus" "Neofelis nebulosa"
)

TEST_SPECIES=(
    "Camelus dromedarius" "Choloepus didactylus" "Cynocephalus volans" "Elephas maximus indicus"
    "Equus asinus" "Lemur catta" "Lepus europaeus" "Loxodonta africana" "Manis pentadactyla"
    "Monodelphis domestica" "Myotis daubentonii" "Nycticebus coucang" "Ochotona princeps" "Ornithorhynchus anatinus"
    "Phyllostomus discolor" "Sarcophilus harrisii" "Sciurus carolinensis" "Suncus etruscus" "Sus scrofa" "Tachyglossus aculeatus"
)

for species in "${TRAIN_SPECIES[@]}"; do
    echo "Evaluating $species on train set"
    python inference_mammals.py \
        --pretrained_config_name "$PRETRAINED_CONFIG_NAME" \
        --model_path "$MODEL_PATH" \
        --max_length "$MAX_LENGTH" \
        --data_dir "$DATA_DIR" \
        --split train \
        --force_species "$species" \
        --batch_size "$BATCH_SIZE" \
        --n_per_sample "$N_PER_SAMPLE" \
        --output_file_prefix "$OUTPUT_FILE_PREFIX" \
        --save_probs \
        --metrics_output_file_prefix "$METRICS_OUTPUT_FILE_PREFIX"
done

for species in "${VALID_SPECIES[@]}"; do
    echo "Evaluating $species on valid set"
    python inference_mammals.py \
        --pretrained_config_name "$PRETRAINED_CONFIG_NAME" \
        --model_path "$MODEL_PATH" \
        --max_length "$MAX_LENGTH" \
        --data_dir "$DATA_DIR" \
        --split valid \
        --force_species "$species" \
        --batch_size "$BATCH_SIZE" \
        --n_per_sample "$N_PER_SAMPLE" \
        --output_file_prefix "$OUTPUT_FILE_PREFIX" \
        --save_probs \
        --metrics_output_file_prefix "$METRICS_OUTPUT_FILE_PREFIX"
done

for species in "${TEST_SPECIES[@]}"; do
    echo "Evaluating $species on test set"
    python inference_mammals.py \
        --pretrained_config_name "$PRETRAINED_CONFIG_NAME" \
        --model_path "$MODEL_PATH" \
        --max_length "$MAX_LENGTH" \
        --data_dir "$DATA_DIR" \
        --split test \
        --force_species "$species" \
        --batch_size "$BATCH_SIZE" \
        --n_per_sample "$N_PER_SAMPLE" \
        --output_file_prefix "$OUTPUT_FILE_PREFIX" \
        --save_probs \
        --metrics_output_file_prefix "$METRICS_OUTPUT_FILE_PREFIX"
done

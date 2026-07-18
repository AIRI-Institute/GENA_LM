MODEL_PATH=checkpoint-33000/model.safetensors
DATA_DIR=/disk/10tb/home/chepurova/chepurova/mammals_data_contig_separated/
OUTPUT_FILE_PREFIX=mammals_inference_runs/mammals_chrY_ratio_0.25/inference_results
BATCH_SIZE=8
N_PER_SAMPLE=60_000
METRICS_OUTPUT_FILE_PREFIX=mammals_inference_runs/mammals_chrY_ratio_0.25/mammals

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
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --split train \
    --force_species "$species" \
    --batch_size $BATCH_SIZE \
    --n_per_sample $N_PER_SAMPLE \
    --output_file_prefix $OUTPUT_FILE_PREFIX \
    --save_probs \
    --metrics_output_file_prefix $METRICS_OUTPUT_FILE_PREFIX
done

for species in "${VALID_SPECIES[@]}"; do
    echo "Evaluating $species on valid set"
    python inference_mammals.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --split valid \
    --force_species "$species" \
    --batch_size $BATCH_SIZE \
    --n_per_sample $N_PER_SAMPLE \
    --output_file_prefix $OUTPUT_FILE_PREFIX \
    --save_probs \
    --metrics_output_file_prefix $METRICS_OUTPUT_FILE_PREFIX
done

for species in "${TEST_SPECIES[@]}"; do
    echo "Evaluating $species on test set"
    python inference_mammals.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --split test \
    --force_species "$species" \
    --batch_size $BATCH_SIZE \
    --n_per_sample $N_PER_SAMPLE \
    --output_file_prefix $OUTPUT_FILE_PREFIX \
    --save_probs \
    --metrics_output_file_prefix $METRICS_OUTPUT_FILE_PREFIX
done
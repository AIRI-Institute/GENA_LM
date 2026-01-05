## Pipelime:

1. Run model inferemce:

```bash

conda activate NT # (or bert24 for ModernGENA/GENA)
CUDA_VISIBLE_DEVICES=1; python compute_entropy.py --config configs/modgena-base-ep30-ba90700.ini --limit_bp 3000000

# run all moderngena cpts (see config files for large and xlarge scripts):
for cpt in data/mgena_cpts/base/*.pt; do n=$(basename $cpt | cut -d "-" -f1-2); echo $n; CUDA_VISIBLE_DEVICES=1; python compute_entropy.py --config configs/modgena-generic.ini --name modgena-base-$n cpt_path $cpt; done

# run evo (for cadueceus predictions were aquiered not corretly)

python3 process_h5_predictions.py --h5_path data/decoder-models/minja_human_chr21_only_hg38_evo2_mlm_embeddings.h5 --genome_path ../expression_prediction/datasets/data/genomes/hg38/hg38.fa --output_prefix data/decoder-models/evo2_chr21 --token_map evo2

# python3 process_h5_predictions.py --h5_path data/decoder-models/minja_human_chr21_only_hg38_caduceus_ps_mlm_embeddings.h5 --genome_path ../expression_prediction/datasets/data/genomes/hg38/hg38.fa --output_prefix data/decoder-models/caduceus_ps_chr21 --token_map caduceus

# python3 process_h5_predictions.py --h5_path data/decoder-models/minja_human_chr21_only_hg38_caduceus_ph_mlm_embeddings.h5 --genome_path ../expression_prediction/datasets/data/genomes/hg38/hg38.fa --output_prefix data/decoder-models/caduceus_ph_chr21 --token_map caduceus


```

2. Create accessible regions file (intersection of all predictions)

```bash

python create_accessible_intervals.py data/modgena-base-ep30-ba90700_chr21_is_correct.bedgraph data/gena-lm_chr21_is_correct.bedgraph data/accessible_regions.bed

```

3. Create annotatons

```bash

fetch_annotations.sh

```

3. Compute stats

```bash

python score_model.py --annotation_beds data/annotations/exons.bed data/annotations/introns.bed data/annotations/nestedRepeats.bed data/annotations/promoters.bed data/annotations/simpleRepeats.bed --prediction_bedgraphs data/gena-lm_chr21_is_correct.bedgraph data/modgena-base-ep30-ba90700_chr21_is_correct.bedgraph --accessible_regions data/accessible_regions.bed --output data/gena_and_moderngena.csv --n_shuffles 4
``` 

## Gathering data from all machines on aws

Copy data TO aws

``` bash
cd ~/DNALM/GENA_LM/downstream_tasks/entropy/data
aws s3 cp . s3://genalm/entropy/inference_results/ --profile airi --endpoint-url https://s3.cloud.ru --acl authenticated-read --recursive

```

Copy data FROM aws

```bash
aws s3 ls s3://genalm/entropy/inference_results/ --profile airi --endpoint-url https://s3.cloud.ru
```

Copy decode (Caduces and Evo) predictions

```
mkdir -p data/decoder-models/
cd data/decoder-models/
aws s3 cp s3://genalm/annogena/evo2/minja_human_chr21_only_hg38_evo2_mlm_embeddings.h5 . --profile airi --endpoint-url https://s3.cloud.ru
aws s3 cp s3://genalm/annogena/evo2/minja_human_chr21_only_hg38_caduceus_ph_mlm_embeddings.h5 . --profile airi --endpoint-url https://s3.cloud.ru
aws s3 cp s3://genalm/annogena/evo2/minja_human_chr21_only_hg38_caduceus_ps_mlm_embeddings.h5 . --profile airi --endpoint-url https://s3.cloud.ru
```

## Train-test splits for different models

I use chromosome 21, since most of models (_but not all_) have it in test split

- NT v1 and v2: Namely, sequences from chromosomes 20 and 21 were used for the test and the remainder were used for training the different models

- NT v3: used the same approach as for the Nucleotide Transformer model [21], following the standard BERT masking strategy [32]: for each sequence, 15% of tokens were selected for potential modification; of this subset, 80% were replaced with a [MASK] token, 10% were substituted with a random token from the vocabulary, and the final 10% remained unchanged. For each batch, the loss function was computed as the sum of the cross-entropy losses between the predicted probabilities over tokens and the ground-truth tokens at each selected position.

- SegmentNT: Namely, chromosomes 20 and 21 are used for test, chromosome 22 is used for validation

- GENA: we hold out human chromosomes 7 (CP068271.2) and 10 (CP068268.2) for testing; In alignment with the BERT pre-training methodology, 15% of the tokens were randomly selected for prediction. Among these, 80% were replaced with MASK tokens, 10% were swapped with random tokens and the remaining 10% were retained unchanged. 

- ModernGENA: 10% of the data is randomly selected for validation for all species except human. For human, chromosomes 8, 20, and 21 are used for validation, as described in the annotation.

- HyenaDNA: Uses Basenji split: curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > data/hg38/human-sequences.bed

- CADUCEUS: same split as Hyena (Basenji): "Data downloading instructions are copied from HyenaDNA repo".  For bi-directional models, we use the masking recipe presented in Devlin et al. (2018). Namely, we 'mask' 15% of tokens. Of the 'masked' tokens, 80% are replaced with a special [MASK] token, 10% are replaced with a random token from the vocabulary, and 10% are left unchanged.
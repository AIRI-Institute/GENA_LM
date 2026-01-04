## Pipelime:

1. Run model inferemce:

```bash

conda activate NT # (or bert24 for ModernGENA/GENA)
CUDA_VISIBLE_DEVICES=1; python compute_entropy.py --config configs/modgena-base-ep30-ba90700.ini --limit_bp 3000000

# run all moderngena cpts:
for cpt in data/mgena_cpts/base/*.pt; do n=$(basename $cpt | cut -d "-" -f1-2); echo $n; CUDA_VISIBLE_DEVICES=1; python compute_entropy.py --config configs/modgena-generic.ini --name modgena-base-$n cpt_path $cpt; done

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

## Train-test splits for different models

I use chromosome 21, since most of models (_but not all_) have it in test split

- NT v1 and v2: Namely, sequences from chromosomes 20 and 21 were used for the test and the remainder were used for training the different models

- NT v3: used the same approach as for the Nucleotide Transformer model [21], following the standard BERT masking strategy [32]: for each sequence, 15% of tokens were selected for potential modification; of this subset, 80% were replaced with a [MASK] token, 10% were substituted with a random token from the vocabulary, and the final 10% remained unchanged. For each batch, the loss function was computed as the sum of the cross-entropy losses between the predicted probabilities over tokens and the ground-truth tokens at each selected position.

- SegmentNT: Namely, chromosomes 20 and 21 are used for test, chromosome 22 is used for validation

- GENA: we hold out human chromosomes 7 (CP068271.2) and 10 (CP068268.2) for testing; In alignment with the BERT pre-training methodology, 15% of the tokens were randomly selected for prediction. Among these, 80% were replaced with MASK tokens, 10% were swapped with random tokens and the remaining 10% were retained unchanged. 

- ModernGENA: 10% of the data is randomly selected for validation for all species except human. For human, chromosomes 8, 20, and 21 are used for validation, as described in the annotation.


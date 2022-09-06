# Pathogenic mutations

Datasets:
- S1A_germline_variant_dataset
- S1B_somatic_variant_dataset
- S1C_eQTL_variant_dataset
- S1D_GWAS_variant_dataset

Zero-shot classification with GENA-LM: notebooks/gena_lm_masked_lm_example.ipynb
Hypothesis: Language model should assign higher probability to mutations that are not pathogenic.

data sample is preprocessed to form:
```
[CLS] left_context [MASK] right_context [SEP] [PAD]
```

## Fine-tuning on pathogenic mutations datasets
The task is considered as two label sequence-lvl classification:
1 - mutation is pathogenic, 0 - mutation is not pathogenic. 

data sample is preprocessed to form:
```
[CLS] left_context alternative_nucleotide right_context [SEP] [PAD]
```

We run 5-fold cross-validation to measure models performance on mutations datasets.
Example of fine-tuning with averaging metrics between folds could be found in `finetune_pathogenic.sh` script.
Paths in the script should be modified according to your setup.

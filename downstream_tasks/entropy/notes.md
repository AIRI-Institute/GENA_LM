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

- SegmentNT: Namely, chromosomes 20 and 21 are used for test, chromosome 22 is used for validation

- GENA: we hold out human chromosomes 7 (CP068271.2) and 10 (CP068268.2) for testing; In alignment with the BERT pre-training methodology, 15% of the tokens were randomly selected for prediction. Among these, 80% were replaced with MASK tokens, 10% were swapped with random tokens and the remaining 10% were retained unchanged. 

- ModernGENA: 10% of the data is randomly selected for validation for all species except human. For human, chromosomes 8, 20, and 21 are used for validation, as described in the annotation.


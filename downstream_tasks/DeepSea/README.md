# DeepSea Chromatin-Profile

Original DeepSea dataset can be downloaded [here](http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz)

Sequence length for original dataset: 1000bp ~ 192 bpe tokens

## Targets
- Transcription factors: [125:125 + 690]
- DNase I-hypersensitive sites: [:125]
- Histone marks: [125 + 690:125 + 690 + 104]
https://github.com/jimmyyhwu/deepsea/blob/4f9fdcfdaa1fec2a93d15047aa1d18f84292220d/compute_aucs.py#L46


## Data preprocessing
The script below converts files from .mat format to .csv.gz with `sequence` and target columns.
```python
python prepare_dataset.py --train_path ./train.mat--valid_path ./valid.mat --test_path ./test.mat
```
as result `train.csv.gz`, `valid.csv.gz`, and `test.csv.gz` should be generated.

### Extend context length (optional)
Original DeepSea dataset includes 1-kb sequences. To add additional context, use 
[extend_dataset.py](extend_dataset.py):

```bash 
python extend_dataset.py --path /path/to/folder/with/1kb/csv/files --fasta /path/to/hg19.genome.fasta --seqlen required_output_seq_length
```

note that extend_dataset.py requires [bwa](https://github.com/lh3/bwa) to be installed and located in PATH

## Finetuning
We follow the BigBird paper:
> We prefix and append each example with [CLS] and [SEP] token respectively. The output corresponding to the [CLS] token from BigBird transformer encoder is fed to a linear layer with 919 heads. Thus we jointly predict the 919 independent binary classification problems. We fine-tune the pretrained BigBird from App. F.1 using hyper-parameters described in Tab. 21. As the data is highly imbalanced data (way more negative examples than positive examples), we upweighted loss function for positive examples by factor of 8.

Modify paths and hyperparameters in the example script for finetuning and run:
```bash
CUDA_VISIBLE_DEVICES=0,1 NP=2 ./finetune_deepsea.sh
```
using GPUs that are available for you.
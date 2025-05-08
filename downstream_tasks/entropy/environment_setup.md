# NT
```
conda create --name NT python=3.9
conda activate NT
cd ~/DNALM
# git clone https://github.com/instadeepai/nucleotide-transformer
python3 -m pip install --upgrade git+https://github.com/huggingface/transformers.git
python3 -m pip install pysam tqdm matplotlib seaborn
python3 -m pip install numpy scipy pandas
python3 -m pip install torch
```

Run:

```
conda activate NT
python compute_entropy.py --genome_path ../../notebooks/data/hg38.gapless.fa --chrm chr21 --batch_size 8 --model nucleotide-transformer-v2-100m --limit_bp 10000 --batch_size 2
```
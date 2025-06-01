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
python3 compute_entropy.py --genome_path ../../notebooks/data/hg38.gapless.fa --chrm chr21 --batch_size 8 --model nucleotide-transformer-v2-100m --limit_bp 10000 --batch_size 32
```

# ModernGENA

Follow https://github.com/minjaf/ModernBERT , then:

```
conda activate bert24
python3 -m pip install pysam, scipy
cd downstr

python3 compute_entropy.py --genome_path ../../notebooks/data/hg38.gapless.fa --chrm chr21 --model ModernGENA_t2t_test --limit_bp 10000 --batch_size 2

python3 compute_entropy.py --genome_path ../../notebooks/data/hg38.gapless.fa --chrm chr21 --model ModernGENA_prom_multi --limit_bp 30000 --batch_size 8

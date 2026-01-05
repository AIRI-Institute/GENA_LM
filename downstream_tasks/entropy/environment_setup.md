# NT
```
conda create --name NT python=3.10 -y
conda activate NT
cd ~/DNALM
# git clone https://github.com/instadeepai/nucleotide-transformer
python3 -m pip install --upgrade git+https://github.com/huggingface/transformers.git
python3 -m pip install pysam tqdm matplotlib seaborn
python3 -m pip install numpy scipy pandas
python3 -m pip install torch
pip install -U "transformers==4.57.3"
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
python3 -m pip install pysam scipy pybedtools seaborn pybedtools
conda install bioconda::bedtools -y
```

# Caduceus

```
wget https://raw.githubusercontent.com/kuleshov-group/caduceus/refs/heads/main/caduceus_env.yml
conda env create -f caduceus_env.yml -y
conda activate caduceus_env
rm caduceus_env.yml # optional: cleanup env setup file
```
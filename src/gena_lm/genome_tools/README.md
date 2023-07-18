A single genome reference file can be processed with `create_corpus.py`.

## Multi-species corpus

To download and process multiple genome datasets, there are following steps:

1. Download fasta files:
```shell
cut -f2 ensemble_genomes_metadata.tsv | xargs -L 1 wget -P fasta --no-clobber
```

2. Download and extract processed agp information into `contigs/` folder

3. Run:
```shell
bash create_multigenome_corpus.sh           \
    /path/to/full_ensembl_genomes_metadata  \
    /path/to/directory/with/fasta/          \
    /path/to/directory/with/contigs/        \
    /path/to/output/folder
```

## Variation corpus

1. Download [1000 Genomes + HGDP whole genome callset from gnomAD](https://gnomad.broadinstitute.org/downloads#v3-hgdp-1kg):
```shell
    GOO=https://storage.googleapis.com/gcp-public-data--gnomad/
    AWS=https://gnomad-public-us-east-1.s3.amazonaws.com/
    AZU=https://datasetgnomad.blob.core.windows.net/dataset/
	# choose whichever cloud
	LINK=$GOO
	LINK=$LINK/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.hgdp_tgp.

	mkdir 1KG+HGDP
    for CHR in chr{1..22} chr{X,Y} ; do
		wget -P 1KG+HGDP -c $LINK.$CHR.vcf.{bgz,bgz.tbi}
    done
```

2. Download hg38 reference:
```shell
	mkdir hg38
	wget -P hg38 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/analysisSet/hg38.analysisSet.fa.gz
	gunzip hg38.analysisSet.fa.gz
	samtools faidx hg38.analysisSet.fa
```

3. Apply variants:
```shell
    # select some samples from across the world
    SAMPLES=$(cut -f1 | 1KG+HGDP.selected_samples.tsv | tr , \n)

    mkdir variation_corpus
    for CHR in chr{1..22} chr{X,Y} ; do 
    	python create_allele_corpus.py                                  \
            --reference hg38/hg38.analysisSet.fa                        \
            --vcf 1KG+HGDP/gnomad.genomes.v3.1.2.hgdp_tgp.$CHR.vcf.bgz  \
            --samples ${SAMPLES%?}                                      \
            --output-dir variation_corpus
    done
```

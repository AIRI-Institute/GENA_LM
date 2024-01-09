set -e

# download Progressive Cactus alignment of 298 drosophilid species
# preprint: https://www.biorxiv.org/content/10.1101/2023.10.02.560517v1
# dataset source: https://doi.org/10.5061/dryad.x0k6djhrd

wget https://datadryad.org/stash/downloads/file_stream/2752661 -O hals/drosophila.hal.00
wget https://datadryad.org/stash/downloads/file_stream/2752662 -O hals/drosophila.hal.01
wget https://datadryad.org/stash/downloads/file_stream/2752663 -O hals/drosophila.hal.02
wget https://datadryad.org/stash/downloads/file_stream/2752664 -O hals/drosophila.hal.03
wget https://datadryad.org/stash/downloads/file_stream/2752665 -O hals/drosophila.hal.04

cat hals/drosophila.hal.* > hals/drosophila.hal
rm hals/drosophila.hal.0*

# get hal tools
docker pull quay.io/comparative-genomics-toolkit/cactus:v2.7.0

# validate hals - note that it may take several hours
# docker run -v $(pwd):/data --rm  quay.io/comparative-genomics-toolkit/cactus:v2.7.0 halValidate hals/drosophila.hal

# get genome names
genomes=$(docker run -v $(pwd):/data --rm quay.io/comparative-genomics-toolkit/cactus:v2.7.0 halStats hals/drosophila.hal --genomes); echo $genomes | tr ' ' '\n' | grep -v "Anc" > fly.genomes.txt

# generate test holdouts for each species
# this is done by liftovering of the manually selected D_MELANOGASTER test holdout

echo "Genome test_size_before_merge test_after_merge n_test_intervals file_name" > fly_holdouts/stats.txt; \ 
line="D_MELANOGASTER"; \
before=$(awk 'BEGIN{s=0}; {s+=$3-$2}; END{print s}' fly_holdouts/$line.testholdout.bed); \
after=$before; \
n_intervals=$(wc -l fly_holdouts/$line.testholdout.bed); \
echo "$line $before $after $n_intervals" >> fly_holdouts/stats.txt; \
d=10000; cat fly.genomes.txt | while read -r line; \
    do \
        if [[ "$line" == "D_MELANOGASTER" ]]; \
        then \
            continue; \
        fi; \
        echo $(date -u) $line; \
        docker run -v $(pwd):/data --rm \
            quay.io/comparative-genomics-toolkit/cactus:v2.7.0 \
            halLiftover hals/drosophila.hal \
            D_MELANOGASTER fly_holdouts/D_MELANOGASTER.testholdout.bed \
            $line fly_holdouts/$line.testholdout.bed; \
        bedtools sort -i fly_holdouts/$line.testholdout.bed > fly_holdouts/$line.testholdout.sorted.bed; \
        bedtools merge -d $d -i fly_holdouts/$line.testholdout.sorted.bed > \
            fly_holdouts/$line.testholdout.merge${d}.bed; \
        before=$(awk 'BEGIN{s=0}; {s+=$3-$2}; END{print s}' fly_holdouts/$line.testholdout.bed); \
        after=$(awk 'BEGIN{s=0}; {s+=$3-$2}; END{print s}' fly_holdouts/$line.testholdout.merge${d}.bed); \
        n_intervals=$(wc -l fly_holdouts/$line.testholdout.merge${d}.bed); \
        echo "$line $before $after $n_intervals" >> fly_holdouts/stats.txt; \
        rm fly_holdouts/$line.testholdout.bed fly_holdouts/$line.testholdout.sorted.bed; \
    done;

# get genomic sequences for selected species
mkdir fasta

cat fly_holdouts/valid_species_stats.tsv | while read -r line; \
do \
    genome=$(echo $line | cut -d ' ' -f1); \
    if [[ $genome == 'Genome' ]]; then continue; fi; \
    echo $(date -u) $genome; \
    docker run -v $(pwd):/data --rm \
        quay.io/comparative-genomics-toolkit/cactus:v2.7.0 \
        hal2fasta hals/drosophila.hal $genome --outFaPath fasta/$genome.fa; \
done;

# get train and test fasta files
mkdir fly_traintest_splits

d=10000; cat fly_holdouts/valid_species_stats.tsv | while read -r line; \
do \
    genome=$(echo $line | cut -d ' ' -f1); \
    if [[ $genome == "Genome" ]]; then continue; fi; \
    echo $(date -u) $genome; \
    if [[ $genome == "D_MELANOGASTER" ]]; \
    then \
        suffix=""; \
    else \
        suffix="merge${d}."; \
    fi; \
    samtools faidx fasta/$genome.fa; \
    sort -k1,1 fasta/$genome.fa.fai > fasta/$genome.fa.sorted.fai; \
    sort -k1,1 -k2,2n fly_holdouts/$genome.testholdout.${suffix}bed > \
        fly_holdouts/$genome.testholdout.sorted.${suffix}bed; \
    bedtools complement \
        -i fly_holdouts/$genome.testholdout.sorted.${suffix}bed \
        -g fasta/$genome.fa.sorted.fai > \
        fly_traintest_splits/$genome.train.${suffix}bed; \
    awk -v genome="$genome" \
        '{print $1 "\t" $2 "\t" $3 "\t" genome "_" $1 "_" $2 "_" $3 "_test"}' \
        fly_holdouts/$genome.testholdout.${suffix}bed > \
        fly_traintest_splits/$genome.traintestsplit.${suffix}bed ;\
    awk -v genome="$genome" \
        '{print $1 "\t" $2 "\t" $3 "\t" genome "_" $1 "_" $2 "_" $3 "_train"}' \
        fly_traintest_splits/$genome.train.${suffix}bed >> \
        fly_traintest_splits/$genome.traintestsplit.${suffix}bed ;\
done;


mkdir fly_fasta_for_corpus_generation

d=10000; cat fly_holdouts/valid_species_stats.tsv | while read -r line; \
do \
    genome=$(echo $line | cut -d ' ' -f1); \
    if [[ $genome == "Genome" ]]; then continue; fi; \
    echo $(date -u) $genome; \
    if [[ $genome == "D_MELANOGASTER" ]]; \
    then \
        suffix=""; \
    else \
        suffix="merge${d}."; \
    fi; \
    bedtools getfasta \
        -fi fasta/$genome.fa \
        -bed fly_traintest_splits/$genome.traintestsplit.${suffix}bed \
        -nameOnly \
        > fly_fasta_for_corpus_generation/$genome.splits.fa
done;

# concatenate into final multifasta file
cat fly_fasta_for_corpus_generation/*.fa | gzip > fly_fasta_for_corpus_generation.fa.gz

# cleanup
rm -r fly_fasta_for_corpus_generation
# rm -r hals
# rm -r fasta
# rm -r fly_traintest_splits
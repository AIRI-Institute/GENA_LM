export sp="yeast"
export reference="sacCer3"
export d=10000
export hal=${sp}_hals/${sp}.full.hal
export fasta=${sp}_fasta
export splits=${sp}_traintest_splits
export fasta4corpus=${sp}_fasta_for_corpus_generation

#download data from https://doi.org/10.1038/s41588-023-01459-y

mkdir $fasta

grep "GCA" ENA_PRJEB59413_assmebly_links.tsv | \
while read -r line
do
    acc=$(echo $line | cut -d " " -f1)
    wget "https://www.ebi.ac.uk/ena/browser/api/fasta/${acc}?download=true&gzip=true" -O $fasta/$acc.gz
done

# download reference assembly
wget https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/sacCer3.fa.gz -O $fasta/sacCer3.fa.gz

# create cactus input file
echo "sacCer3 ${sp}_fasta/sacCer3.fa.gz" > cactus_minigraph_samples.tsv

grep "GCA" ENA_PRJEB59413_assmebly_links.tsv | \
while read -r line
do
    acc=$(echo $line | cut -d " " -f1)
    n=$(echo $acc | tr '.' '_')
    echo "$n ${sp}_fasta/$acc.gz" >> cactus_minigraph_samples.tsv
done

# get cactus and hal tools
docker pull quay.io/comparative-genomics-toolkit/cactus:v2.7.0

# create alignment using cactus (takes ~2h)
docker run -v $(pwd):/data --rm quay.io/comparative-genomics-toolkit/cactus:v2.7.0 \
    cactus-pangenome ./tmp cactus_minigraph_samples.tsv \
        --outDir ${sp}_hals/ --outName ${sp} --reference sacCer3

# validate hal - note that it may take several hours
run -v $(pwd):/data --rm  quay.io/comparative-genomics-toolkit/cactus:v2.7.0 halValidate $hal

# get genome names
genomes=$(docker run -v $(pwd):/data \
        --rm quay.io/comparative-genomics-toolkit/cactus:v2.7.0 \
        halStats $hal --genomes); echo $genomes | tr ' ' '\n' | grep -v "Anc" | grep -v "_MINIGRAPH_" > $sp.genomes.txt
echo "Found "$(wc -l $sp.genomes.txt)" genomes in file $sp.genomes.txt"


# generate test holdouts for each species
# this is done by liftovering of the manually selected reference test holdout set

echo "Genome test_size_before_merge test_after_merge n_test_intervals file_name" > ${sp}_holdouts/stats.txt; \
line="$reference"; \
before=$(awk 'BEGIN{s=0}; {s+=$3-$2}; END{print s}' ${sp}_holdouts/$line.testholdout.bed); \
after=$before; \
n_intervals=$(wc -l ${sp}_holdouts/$line.testholdout.bed); \
echo "$line $before $after $n_intervals" >> ${sp}_holdouts/stats.txt; \
cat ${sp}.genomes.txt | while read -r line; \
    do \
        if [[ "$line" == "$reference" ]]; \
        then \
            continue; \
        fi; \
        echo $(date -u) $line; \
        docker run -v $(pwd):/data --rm \
            quay.io/comparative-genomics-toolkit/cactus:v2.7.0 \
            halLiftover $hal \
            $reference ${sp}_holdouts/$reference.testholdout.bed \
            $line ${sp}_holdouts/$line.testholdout.bed; \
        bedtools sort -i ${sp}_holdouts/$line.testholdout.bed > ${sp}_holdouts/$line.testholdout.sorted.bed; \
        bedtools merge -d $d -i ${sp}_holdouts/$line.testholdout.sorted.bed > \
            ${sp}_holdouts/$line.testholdout.merge${d}.bed; \
        before=$(awk 'BEGIN{s=0}; {s+=$3-$2}; END{print s}' ${sp}_holdouts/$line.testholdout.bed); \
        after=$(awk 'BEGIN{s=0}; {s+=$3-$2}; END{print s}' ${sp}_holdouts/$line.testholdout.merge${d}.bed); \
        n_intervals=$(wc -l ${sp}_holdouts/$line.testholdout.merge${d}.bed); \
        echo "$line $before $after $n_intervals" >> ${sp}_holdouts/stats.txt; \
        rm ${sp}_holdouts/$line.testholdout.bed ${sp}_holdouts/$line.testholdout.sorted.bed; \
    done;

# get genomic sequences for selected species

# cat ${sp}_holdouts/valid_species_stats.tsv | while read -r line; \
# do \
#     genome=$(echo $line | cut -d ' ' -f1); \
#     if [[ $genome == 'Genome' ]]; then continue; fi; \
#     echo $(date -u) $genome; \
#     docker run -v $(pwd):/data --rm \
#         quay.io/comparative-genomics-toolkit/cactus:v2.7.0 \
#         hal2fasta $hal $genome --outFaPath $fasta/$genome.fa; \
# done;

gunzip $fasta/*
for i in $fasta/GCA*; do mv $i ${i}.fa; done

# get train and test fasta files
mkdir $splits

cat ${sp}_holdouts/valid_species_stats.tsv | while read -r line; \
do \
    genome=$(echo $line | cut -d ' ' -f1); \
    if [[ $genome == "Genome" ]]; then continue; fi; \
    echo $(date -u) $genome; \
    if [[ $genome == "$reference" ]]; \
    then \
        suffix=""; \
    else \
        suffix="merge${d}."; \
    fi; \
    samtools faidx $fasta/$genome.fa; \
    sort -k1,1 $fasta/$genome.fa.fai > $fasta/$genome.fa.sorted.fai; \
    sort -k1,1 -k2,2n ${sp}_holdouts/$genome.testholdout.${suffix}bed > \
        ${sp}_holdouts/$genome.testholdout.sorted.${suffix}bed; \
    bedtools complement \
        -i ${sp}_holdouts/$genome.testholdout.sorted.${suffix}bed \
        -g $fasta/$genome.fa.sorted.fai > \
        $splits/$genome.train.${suffix}bed; \
    awk -v genome="$genome" \
        '{print $1 "\t" $2 "\t" $3 "\t" genome "_" $1 "_" $2 "_" $3 "_test"}' \
        ${sp}_holdouts/$genome.testholdout.${suffix}bed > \
        $splits/$genome.traintestsplit.${suffix}bed ;\
    awk -v genome="$genome" \
        '{print $1 "\t" $2 "\t" $3 "\t" genome "_" $1 "_" $2 "_" $3 "_train"}' \
        $splits/$genome.train.${suffix}bed >> \
        $splits/$genome.traintestsplit.${suffix}bed ;\
done;

mkdir $fasta4corpus
cat ${sp}_holdouts/valid_species_stats.tsv | while read -r line; \
do \
    genome=$(echo $line | cut -d ' ' -f1); \
    if [[ $genome == "Genome" ]]; then continue; fi; \
    echo $(date -u) $genome; \
    if [[ $genome == "$reference" ]]; \
    then \
        suffix=""; \
    else \
        suffix="merge${d}."; \
    fi; \
    bedtools getfasta \
        -fi $fasta/$genome.fa \
        -bed $splits/$genome.traintestsplit.${suffix}bed \
        -nameOnly \
        > $fasta4corpus/$genome.splits.fa
done;

# concatenate into final multifasta file
cat $fasta4corpus/*.fa | gzip > ${sp}_fasta_for_corpus_generation.fa.gz

# cleanup
rm -r $fasta4corpus
# # rm -r ${sp}_hals
# # rm -r $fasta
# # rm -r $splits
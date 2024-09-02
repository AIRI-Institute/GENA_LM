export sp="athaliana"
export reference="GCF_000001735_4"
export d=10000
export hal=${sp}_hals/${sp}.full.hal
export fasta=${sp}_fasta
export splits=${sp}_traintest_splits
export fasta4corpus=${sp}_fasta_for_corpus_generation

#download data from https://doi.org/10.1038/s41467-023-42029-4

mkdir $fasta

wget 'https://figshare.com/ndownloader/files/41661786' -O $fasta/41661786.tar.gz
tar -xvf $fasta/41661786.tar.gz -C $fasta/
mv $fasta/ath_genome_and_annotation/genomes/* $fasta/
rm -r $fasta/ath_genome_and_annotation/
for i in $fasta/*.fasta; \
do \
    filename=$(basename "$i"); \
    filename="${filename%.*}"; \
    filename=$(echo $filename | sed 's/\./_/g'); \
    mv $i $fasta/${filename}.fa; \
done;
rm $fasta/41661786.tar.gz

# download reference assembly
datasets download genome accession GCF_000001735.4 --include genome --assembly-level chromosome --filename $fasta/$reference.zip
unzip $fasta/$reference.zip -d $fasta
mv $fasta/ncbi_dataset/data/GCF_000001735.4/GCF_000001735.4_TAIR10.1_genomic.fna  \
   $fasta/GCF_000001735_4.fa
rm  $fasta/$reference.zip
rm $fasta/README.md
rm -r $fasta/ncbi_dataset/

# create cactus input file

rm cactus_minigraph_samples.tsv
for i in $fasta/*.fa; \
do \
    filename=$(basename "$i"); \
    filename="${filename%.*}"; \
    sample=$(echo $filename | sed 's/\./_/g'); \
    echo "$sample $i" >> cactus_minigraph_samples.tsv; \
done;

# get cactus and hal tools
docker pull quay.io/comparative-genomics-toolkit/cactus:v2.7.0

# create alignment using cactus (takes ~12h for arabidopsis)
docker run -v $(pwd):/data --rm quay.io/comparative-genomics-toolkit/cactus:v2.7.0 \
    cactus-pangenome ./tmp cactus_minigraph_samples.tsv \
        --outDir ${sp}_hals/ --outName ${sp} --reference $reference

# # validate hal - note that it may take several hours
# run -v $(pwd):/data --rm  quay.io/comparative-genomics-toolkit/cactus:v2.7.0 halValidate $hal

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
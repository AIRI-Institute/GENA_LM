#!/bin/bash

GENA_INSTALL_DIR=${HOME}
INTERVALS=${GENA_INSTALL_DIR}/intervals

#SETUP GENA
cd ${GENA_INSTALL_DIR}

git clone https://github.com/AIRI-Institute/GENA_LM.git
cd GENA_LM
git checkout expression_frozen_embeddings
git submodule init
git submodule update


#DOWNLOAD DATA

cd ${GENA_INSTALL_DIR}/GENA_LM/downstream_tasks/expression_prediction/datasets/

#rna seq data (tmp)
cd data
mkdir -p multi_species_rna_geo_artem
aws s3 sync "s3://genalm/expr/datasets/multi_species_rna_geo_artem/" "./multi_species_rna_geo_artem"   --endpoint-url https://s3.cloud.ru   --exclude "*.bam"   --exclude "*/supplement/*"   --profile airi
cd ..

#genomes (10 total)

for genome in \
    GCF_000001405.40 \
    GCF_000001635.27 \
    GCF_036323735.1 \
    GCF_000003025.6 \
    GCF_002263795.3 \
    GCF_011100685.1 \
    GCF_049350105.2 \
    GCF_964237555.1 \
    GCF_041296265.1 \
    GCF_016772045.2; do
    aws s3 cp s3://genalm/expr/genome/${genome} ./genomes/${genome}   --endpoint-url https://s3.cloud.ru   --profile airi   --recursive --no-progress --only-show-errors


#file mappings (donwload into datasets/data dir)

aws s3 sync s3://genalm/expr/file_mappings_10_genomes/ ./ --endpoint-url https://s3.cloud.ru   --profile airi   --recursive --no-progress --only-show-errors


#description embeddings pickle files

aws s3 sync s3://genalm/expr/file_mappings_10_genomes/ ./ --endpoint-url https://s3.cloud.ru   --profile airi   --recursive --no-progress --only-show-errors


#intervals (9 total)

mkdir ${INTERVALS}

for interval in \
        blastp_borzoi \
        blastp_enformer \
        blastp_random \
        cactus_borzoi \
        cactus_enformer \
        cactus_random \
        random \
        random_borzoi \
        random_enformer; do

    aws s3 cp s3://genalm/phylogena/expr/${interval}/${interval}.tar.gz ${INTERVALS} --endpoint-url https://s3.cloud.ru   --profile airi  --no-progress --only-show-errors



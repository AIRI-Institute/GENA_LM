#!/bin/bash

# pip install huggingface_hub
# hf login (once)

# check hf login
echo $(hf auth whoami)

LOCAL_FILE_PATH=./src/gena_lm/modeling_bert.py
HF_REPO_FILE_PATH=modeling_bert.py


MSG="update model imports to be compatible with transformers 4.56.0"
REPOS=(
  "gena-lm-bert-base-t2t"
  "gena-lm-bert-large-t2t"
  "gena-lm-bert-base-lastln-t2t"
  "gena-lm-bigbird-base-sparse-t2t"
  "gena-lm-bert-base-t2t-multi"
  "gena-lm-bert-base-fly"
  "gena-lm-bert-base-yeast"
  "gena-lm-bert-base-athaliana"
)

ORG_NAME=AIRI-Institute
for REPO_NAME in "${REPOS[@]}"; do
    echo "Uploading to $REPO_NAME..."
    hf upload "$ORG_NAME/$REPO_NAME" "$LOCAL_FILE_PATH" "$HF_REPO_FILE_PATH" --repo-type model --commit-message "$MSG"
done

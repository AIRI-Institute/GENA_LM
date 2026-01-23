# Setup environment
```bash
conda env create -f environment.yaml -n expression_flash
conda activate expression_flash
pip install -r ../../../requirements.txt

SOFT_DIR=$HOME/soft
mkdir -p $SOFT_DIR
cd $SOFT_DIR
git clone --branch feat/trainer_with_accelerate https://github.com/yurakuratov/t5-experiments.git
cd t5-experiments
sed -i 's/requirements-lm-tools.txt/requirements-lm-tools-accel.txt/' setup.py
python -m pip install -e .
pip install "transformers==4.55.2" "tokenizers==0.21.4"
pip install "flash_attn==2.6.3" --no-build-isolation
pip install ipykernel
python -m ipykernel install --user --name expression_flash --display-name "Python (expression_flash)"
pip install hydra-core --upgrade
```

# Download models and data

Note: this will download model from aws; if you don't have access to aws, ask for gdrive folders with the model and input files

```bash
model_dir=$HOME/DNALM/GENA_LM/models/
mkdir -p $model_dir/full_model/
aws s3 cp s3://genalm/expr/runs/aspeedok/final/model_expression/20260105-202619/model_40000 $model_dir/full_model/ --recursive --profile airi --endpoint-url https://s3.cloud.ru

mkdir -p $model_dir/decoder/
aws s3 cp s3://genalm/runs/moderngena-expression/decoders/moderngena-expression-decoder-L3H1024I1024h8dp0.1/ $model_dir/decoder/ --recursive --profile airi --endpoint-url https://s3.cloud.ru

mkdir -p $model_dir/modernbert_large/
aws s3 cp s3://genalm/runs/moderngena-large-pretrain-promoters_multi_v2_all_checkpoints/ep36-ba108400-hf $model_dir/modernbert_large/ --recursive --profile airi --endpoint-url https://s3.cloud.ru

# included in github repository, keep for reference
# data_dir=$HOME/DNALM/GENA_LM/downstream_tasks/expression_prediction/inference_example/data/
# mkdir -p $data_dir
# aws s3 cp s3://genalm/expr/datasets/minja/metadata/ENCFF578UUD.json $data_dir/ --profile airi --endpoint-url https://s3.cloud.ru
# aws s3 cp s3://genalm/expr/datasets/minja/metadata/ENCFF588KDY.json $data_dir/ --profile airi --endpoint-url https://s3.cloud.ru
```

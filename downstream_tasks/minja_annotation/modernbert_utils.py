import os
import torch
from omegaconf import DictConfig, OmegaConf as om

# Add ModernBERT to path
import sys
# get ModernBERT environment variable
modernbert_env = os.environ.get("MODERNBERT_HOME")
if modernbert_env:
	sys.path.append(os.path.abspath(modernbert_env))
else:
	raise ValueError("MODERNBERT_HOME is not set; you need to set environment variable MODERNBERT_HOME to the path of the ModernBERT repository")

from src import flex_bert as flex_bert_module
from src import hf_bert as hf_bert_module
from src import mosaic_bert as mosaic_bert_module
from src.bert_layers.model import FlexBertModel
import src.bert_layers.configuration_bert as configuration_bert_module
from composer.utils.checkpoint import _ensure_valid_checkpoint

def build_model(cfg: DictConfig):
	if cfg.name == "hf_bert":
		return hf_bert_module.create_hf_bert_mlm(
			pretrained_model_name=cfg.pretrained_model_name,
			use_pretrained=cfg.get("use_pretrained", None),
			model_config=cfg.get("model_config", None),
			tokenizer_name=cfg.get("tokenizer_name", None),
			gradient_checkpointing=cfg.get("gradient_checkpointing", None),
		)
	elif cfg.name == "mosaic_bert":
		return mosaic_bert_module.create_mosaic_bert_mlm(
			pretrained_model_name=cfg.pretrained_model_name,
			pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
			model_config=cfg.get("model_config", None),
			tokenizer_name=cfg.get("tokenizer_name", None),
			gradient_checkpointing=cfg.get("gradient_checkpointing", None),
		)
	elif cfg.name == "flex_bert":
		return flex_bert_module.create_flex_bert_mlm(
			pretrained_model_name=cfg.pretrained_model_name,
			pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
			model_config=cfg.get("model_config", None),
			tokenizer_name=cfg.get("tokenizer_name", None),
			gradient_checkpointing=cfg.get("gradient_checkpointing", None),
			recompute_metric_loss=cfg.get("recompute_metric_loss", False),
			disable_train_metrics=cfg.get("disable_train_metrics", False),
		)
	else:
		raise ValueError(f"Not sure how to build model with name={cfg.name}")

def load_modernbert_model (model_path, logger):
	# Load config
	cfg_path = os.path.join(model_path, "cfg.yaml")
	yaml_cfg = om.load(cfg_path)

	model = build_model(yaml_cfg.model)
	
	# Load checkpoint
	# checkpoint_filepath = os.path.join(model_path, "ep11-ba68300-rank0.pt")
	checkpoint_filepath = os.path.join(model_path, "latest-rank0.pt")	
	logger.info(f"Loading checkpoint from {checkpoint_filepath}")
	state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu")
	state_dict = state.get("state", {})
	model_state = state_dict.get("model", {})
	assert len(model_state) > 0, "Model state is empty, please check the checkpoint and checkpoint path"
	model.load_state_dict(model_state)
	
	return model

def load_flexbert_model(model_path, logger,
						expected_missing = {},
						expected_unexpected = {
							"model.head.dense.weight",
							"model.head.norm.weight",
							"model.decoder.weight",
							"model.decoder.bias",
						}):

	cfg_path = os.path.join(model_path, "cfg.yaml")
	yaml_cfg = om.load(cfg_path)
	model_config = yaml_cfg["model"]["model_config"]
	pretrained_model_name = "bert-base-uncased"
	if isinstance(model_config, DictConfig):
			model_config = om.to_container(model_config, resolve=True)
	config = configuration_bert_module.FlexBertConfig.from_pretrained(pretrained_model_name, **model_config)
	model = FlexBertModel(config=config)

	checkpoint_filepath = os.path.join(model_path, "latest-rank0.pt")	
	logger.info(f"Loading checkpoint from {checkpoint_filepath}")
	state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu")
	state_dict = state.get("state", {})
	model_state = state_dict.get("model", {})
	assert len(model_state) > 0, "Model state is empty, please check the checkpoint and checkpoint path"
	model_state = {k.replace("model.bert.", ""): v for k, v in model_state.items()}
	test = model.load_state_dict(model_state, strict=False)

	# ensure that the missing and unexpected keys are as expected
	assert len(test.missing_keys) == 0 or \
			(set(test.missing_keys) == expected_missing and len(test.missing_keys) == len(expected_missing)), \
			f"missing_keys mismatch: {test.missing_keys}"
	assert len(test.unexpected_keys) == 0 or \
			(set(test.unexpected_keys) == expected_unexpected and len(test.unexpected_keys) == len(expected_unexpected)), \
		f"unexpected_keys mismatch: {test.unexpected_keys}"

	return model

# test
if __name__ == "__main__":
	import logging
	logger = logging.getLogger(__name__)
	model_path = "/mnt/nfs_dna/minja/DNALM/ModernBERT/runs/moderngena-base-pretrain-promoters_multi_v2_resume_ep30-ba90700/"
	model = load_model_and_tokenizer(model_path, logger)
	print(model)
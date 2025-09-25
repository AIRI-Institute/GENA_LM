import os
import torch
from omegaconf import DictConfig, OmegaConf as om
import numpy as np

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
from datetime import timedelta, datetime, timezone, date, time
from torch.torch_version import TorchVersion
import contextlib

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
	state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu", 
						weights_only=True)
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
	safe_ctx = getattr(torch.serialization, "safe_globals", None)
	SAFE_GLOBALS = [timedelta, datetime, timezone, date, time, TorchVersion]
	ctx = safe_ctx(SAFE_GLOBALS) if safe_ctx else contextlib.nullcontext()
	with ctx:
		state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu",
							weights_only=True)
	state_dict = state.get("state", {})
	model_state = state_dict.get("model", {})
	assert len(model_state) > 0, "Model state is empty, please check the checkpoint and checkpoint path"
	model_state = {k.replace("model.bert.", ""): v for k, v in model_state.items()}
	
	############# DEBUGGING CODE ##############
	# DEBUG: get weights for the first layer of the model
	# TODO: remove this debugging code at some point, leaving only one line:
	# test = model.load_state_dict(model_state, strict=False)
	# Print all layer names and the shape of their weights to help identify layers
	for layer_name, debug_param in model.named_parameters():
		if layer_name.find("tok_embeddings") != -1:
			continue
		if not layer_name in model_state:
			continue
		break
	debug_param = np.array(debug_param.detach().cpu().numpy(), copy=True)
	# logger.info(f"Layer name: {layer_name}, Weight shape: {debug_param.shape} saved for debugging")
	# logger.info(f"Is it the same as in cpt? {np.allclose(debug_param, model_state[layer_name])}")
	# logger.info(f"Debug param: {debug_param[:5]}")
	# logger.info(f"Model state: {model_state[layer_name][:5]}")
	
	test = model.load_state_dict(model_state, strict=False)

	# assert that params are changed
	for _, debug_param_updated in model.named_parameters():
		if _ == layer_name:	break
	debug_param_updated = np.array(debug_param_updated.detach().cpu().numpy(), copy=True)
	# logger.info(f"Debug param updated: {debug_param_updated[:5]}")

	# logger.info(f"Layer name: {_}, Weight shape: {debug_param_updated.shape} saved for debugging")
	# logger.info(f"Is it the same as before loading? {np.allclose(debug_param_updated, debug_param)}")
	# logger.info(f"Is it the same as in cpt? {np.allclose(debug_param_updated, model_state[_])}")

	assert not np.allclose(debug_param_updated, debug_param), f"Params are not changed, {layer_name}\n{debug_param}\n{debug_param_updated}"
	assert debug_param_updated.shape == debug_param.shape, "Shape mismatch"
	############# END DEBUGGING CODE ##############
	
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
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
import importlib
from dataclasses import dataclass
from typing import Optional
from transformers.modeling_outputs import TokenClassifierOutput

import os, logging

@dataclass
class AnnotationModelOutput(TokenClassifierOutput):
	loss: Optional[torch.FloatTensor] = None
	loss_TSS: Optional[torch.FloatTensor] = None
	loss_polya: Optional[torch.FloatTensor] = None
	logits: Optional[torch.FloatTensor] = None
	predicts: Optional[torch.FloatTensor] = None

class Linear_Classifier_with_dropout(nn.Module):
	def __init__(self, num_labels, num_layers, classifier_dropout):
		super().__init__()
		assert num_layers >= 1
		assert num_labels >= 1
		self.num_labels = num_labels
		self.num_layers = num_layers
		self.classifier_dropout = classifier_dropout
	
	def set_params(self, config):
		self.layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.num_layers - 1)])
		self.layers.append(nn.Linear(config.hidden_size, self.num_labels))
		self.dropout = nn.Dropout(self.classifier_dropout)

	def forward(self, x):	
		for layer in self.layers:
			x = self.dropout(x)
			x = layer(x)
		return x

class WeightedBCEWithLogitsLoss(nn.Module):
	def __init__(self, w_nonprimary_exists, w_intergenic):
		super().__init__()
		self.w_nonprimary_exists = w_nonprimary_exists
		self.w_intergenic = w_intergenic
		self.LogSig = nn.LogSigmoid()
	
	def forward(self, predicts, targets):
		# So basically loss = y*log(s(x)) + (1-y)*log(1-s(x))
		# where I am not sure y==0, i.e. I have non-primary transcript targets, I downweight second term:
		# w_nonprimary_exists*(1-y)*log(1-s(x))
		# same for intergenic regions:
		# w_intergenic*(1-y)*log(1-s(x))
		# we use total w as min (w_nonprimary_exists, w_intergenic)

		losses = {}
		class_index = 0
		for class_name in ["tss", "polya"]:
			losses[class_name] = torch.tensor(0.0, device=predicts.device)
			for strand in ["+", "-"]:
				# uncertain_target = 1 --> self.w_nonprimary_exists
				# uncertain_target = 0 --> 1
				# uncertain_target = 0.5 --> self.w_nonprimary_exists + (1-self.w_nonprimary_exists)/2
				# W = self.w_nonprimary_exists + (1-self.w_nonprimary_exists)*(1-uncertain_target)

				w_nonprimary_exists = self.w_nonprimary_exists + \
										(1-targets[f"uncertain_{class_name}_{strand}"])*(1-self.w_nonprimary_exists)
				if class_name == "polya":
					# intragenic are 1-intergenic so we don't need to invert it here
					w_intergenic = self.w_intergenic + (targets[f"intragenic_regions_{strand}"])*(1-self.w_intergenic)
					W = torch.minimum(w_intergenic, w_nonprimary_exists)
				else:
					W = w_nonprimary_exists
				
				# batch_size x num_tokens x num_labels
				X = predicts[:, :, class_index]
				Y = targets[f"primary_{class_name}_{strand}"]
				MASK = Y != -100
				loss = -(Y*self.LogSig(X) + W*(1-Y)*self.LogSig(1-X))
				if MASK.sum() != 0.0:
					loss = (loss*MASK).sum()/MASK.sum()
				else:
					loss = torch.tensor(0.0, device=predicts.device)
				assert loss >= 0, f"loss is {loss}"
				losses[class_name] += loss.sum()
				# if losses[class_name] == 0:
				# 	print (f"losses[{class_name}] is 0")
				# 	print ("Y", Y)
				# 	print ("X", X)
				# 	print ("W", W)
				# 	print ("loss", loss)
				# 	print ("MASK", MASK)
				# 	print ("loss.sum()", loss.sum())
				# 	print ("MASK.sum()", MASK.sum())
				# 	print ("--------------------------")
				# 	raise ValueError("losses[{class_name}] is 0")
				class_index += 1

		losses["total"] = (losses["tss"] + losses["polya"])/2
		return losses

class AnnotationModel(torch.nn.Module):
	def __init__(
		self,
		output_dir = None, # currently unused, for compatibility with older models
		config = None,
		pretrained_cpt = None,
		activation = None,
		classifier = None,
		loss_fct = None,
		logger = None,
	):		
		super().__init__()

		if logger is None:
			self.logger = logging.getLogger(__name__)
		else:
			self.logger = logger
			
		if pretrained_cpt is not None:
			if os.path.exists(pretrained_cpt):
				raise NotImplementedError("TODO: check this draft of the code below before using it")
				# assert config is not None
				# self.bert = BertModel(config, add_pooling_layer=False)
				# checkpoint = torch.load(pretrained_cpt, map_location="cpu")
				# state_dict = checkpoint["model_state_dict"]
				# self.logger.info(f"Loading checkpoint from local model {pretrained_cpt}")
				# missing_k, unexpected_k = self.bert.load_state_dict(state_dict, strict=False)
				# if len(missing_k) != 0:
				# 	self.logger.warning(f"{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.")
				# if len(unexpected_k) != 0:
				# 	self.logger.warning(f"{unexpected_k} were found in checkpoint, but model is not expecting them!")
			else:  # hf checkpoint
				model = AutoModel.from_pretrained(pretrained_cpt, trust_remote_code=True)
				module_name = model.__class__.__module__
				cls = getattr(importlib.import_module(module_name), 'BertModel')
				self.bert = cls.from_pretrained(pretrained_cpt, add_pooling_layer=False)
		else:
			self.logger.warning(f"Randomly initialized backbone.")
			self.bert = BertModel(config, add_pooling_layer=False)

		self.classifier = classifier
		self.classifier.set_params(self.bert.config)

		self.activation = activation
		self.loss_fct = loss_fct
	
	def forward(self, input_ids, attention_mask, targets, return_loss=True):
		outputs = self.bert(input_ids, attention_mask)
		# todo: check what is output here
		logits = self.classifier(outputs.last_hidden_state)
		predicts = self.activation(logits)
		if targets is not None:
			loss = self.loss_fct(predicts, targets)

		return AnnotationModelOutput(
			loss=loss['total'],
			loss_TSS=loss['tss'],
			loss_polya=loss['polya'],
			logits=logits,
			predicts=predicts,
		)
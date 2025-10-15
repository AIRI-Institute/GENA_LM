import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, AutoModelForCausalLM, ModernBertModel
import importlib
from dataclasses import dataclass
from typing import Optional, Any, Dict
from transformers.modeling_outputs import TokenClassifierOutput
import os, logging
import types
from collections import Counter


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
		modernbert_cpt = None,
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
			
		assert (pretrained_cpt is not None) != (modernbert_cpt is not None), "Either pretrained_cpt or modernbert_cpt must be provided, not both"
		
		if modernbert_cpt is not None:
			from modernbert_utils import load_flexbert_model
			self.logger.info(f"Loading ModernBERT model from {modernbert_cpt}")
			self.bert = load_flexbert_model(modernbert_cpt, logger=self.logger)
			self.is_modernbert_model = True
		else:
			self.is_modernbert_model = False
		
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
		# else:
		# 	self.logger.warning(f"Randomly initialized backbone.")
		# 	self.bert = BertModel(config, add_pooling_layer=False)

		self.classifier = classifier
		self.classifier.set_params(self.bert.config)

		self.activation = activation
		self.loss_fct = loss_fct
	
	def forward(self, input_ids, attention_mask, targets, return_loss=True):
		if self.is_modernbert_model:
			outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
			assert len(outputs.size()) == 3 # Batch_size x Seq_len x Hidden_size
			assert outputs.size()[0] == input_ids.size()[0] # batch size dimension
			assert outputs.size()[1] == input_ids.size()[1] # sequence length dimension
			logits = self.classifier(outputs)
		else:
			outputs = self.bert(input_ids, attention_mask)
			outputs = outputs.last_hidden_state

		logits = self.classifier(outputs)
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



# MODERNBERT_HOME="/home/jovyan/DNALM/ModernBERT"




class ARMT_AnnotationModel(nn.Module):
    """
    Wrap a BERT/ModernBERT TokenClassification backbone with ARMT
    (AssociativeMemoryCell + AssociativeRecurrentWrapper).

    - HF BERT path: AutoModel -> BertForTokenClassification with Identity head.
    - ModernBERT path: ModernBertModel loaded with SDPA.

    Downstream `classifier` must implement `set_params(config)` and accept a
    3D tensor of shape (batch, seq, hidden_size).
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        config: Optional[Any] = None,
        pretrained_cpt: Optional[str] = None,
        modernbert_cpt: Optional[str] = None,
        activation: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
        loss_fct: Optional[nn.Module] = None,
        logger: Optional[logging.Logger] = None,
        armt_repo_id: str = "irodkin/armt-neox-tiny",
        base_model_name: Optional[str] = None,
        num_mem_tokens: int = 16,
        d_mem: int = 32,
        segment_size: int = 128,
        segment_alignment: str = "left",
        sliding_window: bool = False,
        layers_attr: str = "bert.encoder.layer",
        wrap_pos: bool = False,
        correction: bool = True,
        n_heads: int = 1,
        use_denom: bool = True,
        gating: bool = False,
        freeze_mem: bool = False,
        act_on: bool = False,
        max_hop: int = 4,
        act_type: str = "associative",
        constant_depth: bool = False,
        act_format: str = "linear",
        noisy_halting: bool = False,
        attend_to_previous_input: bool = False,
        use_sink: bool = False,
        time_penalty: float = 0.0,
        sdpa_use_attn_mask: Optional[bool] = None,
    ):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.activation = activation
        self.loss_fct = loss_fct

        if (base_model_name is None) and (modernbert_cpt is None):
            raise ValueError("Provide either `base_model_name` (HF repo id) or `modernbert_cpt` (ModernBERT path).")
        if (base_model_name is not None) and (modernbert_cpt is not None):
            raise ValueError("Provide exactly one of `base_model_name` or `modernbert_cpt` (not both).")

        loaded = AutoModelForCausalLM.from_pretrained(armt_repo_id, trust_remote_code=True)
        armt_mod = importlib.import_module(loaded.__class__.__module__)
        AssociativeMemoryCell = getattr(armt_mod, "AssociativeMemoryCell")
        AssociativeRecurrentWrapper = getattr(armt_mod, "AssociativeRecurrentWrapper")

        if base_model_name is not None:
            auto_backbone = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
            gena_mod = importlib.import_module(auto_backbone.__class__.__module__)
            BertForTokenClassification = getattr(gena_mod, "BertForTokenClassification")

            base_tc = BertForTokenClassification(auto_backbone.config)
            missing, unexpected = base_tc.load_state_dict(auto_backbone.state_dict(), strict=False)
            if any(k for k in missing if not k.startswith("classifier.")):
                self.logger.warning(f"[ARMT_AnnotationModel] Missing non-classifier keys: {missing}")
            if unexpected:
                self.logger.warning(f"[ARMT_AnnotationModel] Unexpected keys during load: {unexpected}")

            if not hasattr(base_tc, "classifier"):
                raise RuntimeError("BertForTokenClassification must have `.classifier`.")
            base_tc.classifier = nn.Identity()

            base_tc._orig_forward = base_tc.forward
            def _forward_ignoring_cache(self_, *args, **kwargs):
                kwargs.pop("use_cache", None)
                kwargs.pop("past_key_values", None)
                return self_._orig_forward(*args, **kwargs)
            base_tc.forward = types.MethodType(_forward_ignoring_cache, base_tc)

            self._encoder_cfg = auto_backbone.config
            bert_encoder = base_tc.bert

        else:
            self.logger.info(f"Loading ModernBERT model from {modernbert_cpt}")
            base_tc = ModernBertModel.from_pretrained(
                modernbert_cpt,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            base_tc.config.deterministic_flash_attn = True
            base_tc.config.use_sdpa_attn_mask = True
            if sdpa_use_attn_mask is not None and hasattr(base_tc.config, "use_sdpa_attn_mask"):
                base_tc.config.use_sdpa_attn_mask = bool(sdpa_use_attn_mask)

            _orig_modern_forward = base_tc.forward
            def _forward_modern_return_logits(self_, *args, **kwargs):
                kwargs.pop("use_cache", None)
                kwargs.pop("past_key_values", None)
                out = _orig_modern_forward(*args, **kwargs)
                if hasattr(out, "last_hidden_state"):
                    seq = out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0:
                    seq = out[0]
                else:
                    raise RuntimeError("ModernBERT forward did not return last_hidden_state.")
                class _Out: pass
                o = _Out()
                o.logits = seq
                o.hidden_states = getattr(out, "hidden_states", None)
                return o
            base_tc.forward = types.MethodType(_forward_modern_return_logits, base_tc)

            if hasattr(base_tc, "config"):
                self._encoder_cfg = base_tc.config
            else:
                if hasattr(base_tc, "get_input_embeddings"):
                    emb_layer = base_tc.get_input_embeddings()
                else:
                    emb_layer = base_tc.embeddings.tok_embeddings
                h = emb_layer.weight.shape[1]
                class _Cfg: pass
                self._encoder_cfg = _Cfg(); self._encoder_cfg.hidden_size = h

            bert_encoder = base_tc

        memory_cell = AssociativeMemoryCell(
            base_model=base_tc,
            num_mem_tokens=num_mem_tokens,
            d_mem=d_mem,
            layers_attr=layers_attr,
            wrap_pos=wrap_pos,
            correction=correction,
            n_heads=n_heads,
            use_denom=use_denom,
            gating=gating,
            freeze_mem=freeze_mem,
            act_on=act_on,
            max_hop=max_hop,
            act_type=act_type,
            constant_depth=constant_depth,
            act_format=act_format,
            noisy_halting=noisy_halting,
            attend_to_previous_input=attend_to_previous_input,
            use_sink=use_sink,
        )

        recurrent = AssociativeRecurrentWrapper(
            memory_cell,
            segment_size=segment_size,
            segment_alignment=segment_alignment,
            sliding_window=sliding_window,
            attend_to_previous_input=attend_to_previous_input,
            act_on=act_on,
            time_penalty=time_penalty,
        )
        self.armt = recurrent

        self.classifier = classifier
        if not hasattr(self.classifier, "set_params"):
            raise RuntimeError("`classifier` must implement `set_params(config)`.")
        self.classifier.set_params(self._encoder_cfg)

    def forward(self, input_ids, attention_mask, targets=None, return_loss: bool = True):
        out = self.armt(input_ids=input_ids, attention_mask=attention_mask)

        assert hasattr(out, "logits") and out.logits is not None, "ARMT output must provide `logits`"
        hidden = out.logits
        assert len(hidden.size()) == 3, f"Expected 3D hidden states, got {hidden.size()}"
        assert hidden.size(0) == input_ids.size(0), f"Batch mismatch: {hidden.size(0)} != {input_ids.size(0)}"
        assert hidden.size(1) == input_ids.size(1), f"Seq len mismatch: {hidden.size(1)} != {input_ids.size(1)}"
        assert hidden.size(2) == getattr(self._encoder_cfg, "hidden_size"), \
            f"Hidden size mismatch: {hidden.size(2)} != {getattr(self._encoder_cfg, 'hidden_size')}"

        logits = self.classifier(hidden)
        predicts = self.activation(logits)

        losses = {
            "total": torch.tensor(0.0, device=logits.device),
            "tss":   torch.tensor(0.0, device=logits.device),
            "polya": torch.tensor(0.0, device=logits.device),
        }
        if (targets is not None) and return_loss:
            losses = self.loss_fct(predicts, targets)

        return AnnotationModelOutput(
            loss=losses["total"],
            loss_TSS=losses["tss"],
            loss_polya=losses["polya"],
            logits=logits,
            predicts=predicts,
        )



















# class ARMT_AnnotationModel(nn.Module):
# 	"""
# 	Wrap a GENA-LM BERT TokenClassification model with ARMT (AssociativeMemoryCell + AssociativeRecurrentWrapper).
# 	The base encoder is pulled from HF as a pretrained AutoModel, then packed into BertForTokenClassification
# 	with its classifier replaced by nn.Identity() so the model returns logits equal to last_hidden_state.
# 	"""

# 	def __init__(
# 		self,
# 		output_dir: Optional[str] = None,
# 		config: Optional[Any] = None,
# 		pretrained_cpt: Optional[str] = None,		# unused; require base_model_name explicitly
# 		modernbert_cpt: Optional[str] = None,		# unused; require base_model_name explicitly
# 		activation: Optional[nn.Module] = None,
# 		classifier: Optional[nn.Module] = None,
# 		loss_fct: Optional[nn.Module] = None,
# 		logger: Optional[logging.Logger] = None,

# 		# ARMT source (to import its classes)
# 		armt_repo_id: str = "irodkin/armt-neox-tiny",

# 		# REQUIRED backbone on HF (e.g., "AIRI-Institute/gena-lm-bert-base-t2t")
# 		base_model_name: Optional[str] = None,

# 		# ARMT configuration (complete set wired through)
# 		num_mem_tokens: int = 16,
# 		d_mem: int = 32,
# 		segment_size: int = 128,
# 		segment_alignment: str = "left",
# 		sliding_window: bool = False,
# 		layers_attr: str = "bert.encoder.layer",
# 		wrap_pos: bool = False,
# 		correction: bool = True,
# 		n_heads: int = 1,
# 		use_denom: bool = True,
# 		gating: bool = False,
# 		freeze_mem: bool = False,
# 		act_on: bool = False,
# 		max_hop: int = 4,
# 		act_type: str = "associative",
# 		constant_depth: bool = False,
# 		act_format: str = "linear",
# 		noisy_halting: bool = False,
# 		attend_to_previous_input: bool = False,
# 		use_sink: bool = False,
# 		time_penalty: float = 0.0,
# 	):
# 		super().__init__()
# 		self.logger = logger or logging.getLogger(__name__)
# 		self.activation = activation
# 		self.loss_fct = loss_fct

# 		if base_model_name is None:
# 			raise ValueError("Provide `base_model_name` (HF repo id).")

# 		# Import ARMT classes from the remote module
# 		loaded = AutoModelForCausalLM.from_pretrained(armt_repo_id, trust_remote_code=True)
# 		armt_mod = importlib.import_module(loaded.__class__.__module__)
# 		AssociativeMemoryCell = getattr(armt_mod, "AssociativeMemoryCell")
# 		AssociativeRecurrentWrapper = getattr(armt_mod, "AssociativeRecurrentWrapper")

# 		# Pull pretrained encoder and assemble a TokenClassification model with Identity head
# 		auto_backbone = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
# 		gena_mod = importlib.import_module(auto_backbone.__class__.__module__)
# 		BertForTokenClassification = getattr(gena_mod, "BertForTokenClassification")

# 		base_tc = BertForTokenClassification(auto_backbone.config)
# 		# Load encoder weights as-is (state dict names are already compatible in the repo)
# 		missing, unexpected = base_tc.load_state_dict(auto_backbone.state_dict(), strict=False)
# 		if any(k for k in missing if not k.startswith("classifier.")):
# 			self.logger.warning(f"[ARMT_AnnotationModel] Missing non-classifier keys: {missing}")
# 		if unexpected:
# 			self.logger.warning(f"[ARMT_AnnotationModel] Unexpected keys during load: {unexpected}")

# 		if not hasattr(base_tc, "classifier"):
# 			raise RuntimeError("BertForTokenClassification must have `.classifier`.")
# 		base_tc.classifier = nn.Identity()

# 		# Accept and ignore cache-related kwargs ARMT may pass
# 		base_tc._orig_forward = base_tc.forward
# 		def _forward_ignoring_cache(self_, *args, use_cache=None, past_key_values=None, **kwargs):
# 			kwargs.pop("use_cache", None)
# 			kwargs.pop("past_key_values", None)
# 			return self_._orig_forward(*args, **kwargs)
# 		base_tc.forward = types.MethodType(_forward_ignoring_cache, base_tc)

# 		self._encoder_cfg = auto_backbone.config
# 		bert_encoder = base_tc.bert

# 		def collect_param_summaries(model: nn.Module) -> tuple[float, int, Dict[str, float]]:
# 			"""
# 			Return (total_sum, param_count, per_param_sum_map) for all parameters in `model`.
# 			Per-parameter values are simple scalar sums of the underlying tensors.
# 			"""
# 			total = 0.0
# 			count = 0
# 			vals: Dict[str, float] = {}
# 			for name, p in model.named_parameters():
# 				if p is None or p.data is None:
# 					continue
# 				s = p.data.float().sum().item()
# 				vals[name] = s
# 				total += s
# 				count += 1
# 			return total, count, vals

# 		def quantize_sum(value: float, decimals: int = 6) -> float:
# 			"""Round a float to `decimals` places to absorb tiny numeric noise when matching by value."""
# 			return float(f"{value:.{decimals}f}")

# 		# First pass: capture encoder totals and value multiset
# 		pre_total_sum, pre_count, pre_map = collect_param_summaries(bert_encoder)
# 		self.logger.info(f"[ARMT_AnnotationModel] Encoder pre-ARMT total sum: {pre_total_sum:.6f}")
# 		first_pass_value_multiset = Counter(quantize_sum(v) for v in pre_map.values())

# 		# Build ARMT stack on the full TC model so outputs carry `logits`
# 		memory_cell = AssociativeMemoryCell(
# 			base_model=base_tc,
# 			num_mem_tokens=num_mem_tokens,
# 			d_mem=d_mem,
# 			layers_attr=layers_attr,
# 			wrap_pos=wrap_pos,
# 			correction=correction,
# 			n_heads=n_heads,
# 			use_denom=use_denom,
# 			gating=gating,
# 			freeze_mem=freeze_mem,
# 			act_on=act_on,
# 			max_hop=max_hop,
# 			act_type=act_type,
# 			constant_depth=constant_depth,
# 			act_format=act_format,
# 			noisy_halting=noisy_halting,
# 			attend_to_previous_input=attend_to_previous_input,
# 			use_sink=use_sink
# 		)

# 		recurrent = AssociativeRecurrentWrapper(
# 			memory_cell,
# 			segment_size=segment_size,
# 			segment_alignment=segment_alignment,
# 			sliding_window=sliding_window,
# 			attend_to_previous_input=attend_to_previous_input,
# 			act_on=act_on,
# 			time_penalty=time_penalty
# 		)
# 		self.armt = recurrent

# 		# Second pass: totals and value-based multiset matching (name-agnostic)
# 		enc_after = self.armt.memory_cell.model.bert
# 		post_total_sum, post_count, post_map = collect_param_summaries(enc_after)
# 		self.logger.info(f"[ARMT_AnnotationModel] Encoder post-ARMT total sum: {post_total_sum:.6f}")

# 		remaining = first_pass_value_multiset.copy()
# 		matches = 0
# 		for s in post_map.values():
# 			key = quantize_sum(s)
# 			if remaining.get(key, 0) > 0:
# 				remaining[key] -= 1
# 				if remaining[key] == 0:
# 					del remaining[key]
# 				matches += 1

# 		assert matches == pre_count, \
# 			f"Value-based match count mismatch: matched {matches} vs first-pass param count {pre_count}"

# 		# Size the downstream classifier head from the encoder config
# 		self.classifier = classifier
# 		self.classifier.set_params(self._encoder_cfg)

# 	def forward(self, input_ids, attention_mask, targets=None, return_loss: bool = True):
# 		out = self.armt(input_ids=input_ids, attention_mask=attention_mask)

# 		assert hasattr(out, "logits") and out.logits is not None, "ARMT output must provide `logits`"
# 		hidden = out.logits
# 		assert len(hidden.size()) == 3, f"Expected 3D hidden states, got {hidden.size()}"
# 		assert hidden.size(0) == input_ids.size(0), f"Batch mismatch: {hidden.size(0)} != {input_ids.size(0)}"
# 		assert hidden.size(1) == input_ids.size(1), f"Seq len mismatch: {hidden.size(1)} != {input_ids.size(1)}"
# 		assert hidden.size(2) == getattr(self._encoder_cfg, "hidden_size"), \
# 			f"Hidden size mismatch: {hidden.size(2)} != {getattr(self._encoder_cfg, 'hidden_size')}"

# 		logits = self.classifier(hidden)
# 		predicts = self.activation(logits)

# 		losses = {
# 			"total": torch.tensor(0.0, device=logits.device),
# 			"tss":   torch.tensor(0.0, device=logits.device),
# 			"polya": torch.tensor(0.0, device=logits.device),
# 		}
# 		if (targets is not None) and return_loss:
# 			losses = self.loss_fct(predicts, targets)

# 		return AnnotationModelOutput(
# 			loss=losses["total"],
# 			loss_TSS=losses["tss"],
# 			loss_polya=losses["polya"],
# 			logits=logits,
# 			predicts=predicts,
# 		)

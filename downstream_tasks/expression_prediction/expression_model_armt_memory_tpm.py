"""ARMT variant with auxiliary TPM supervision on decoder memory tokens."""

import torch

from downstream_tasks.expression_prediction.expression_model_armt_center import (
    ARMTExpressionModelOutput,
    ExpressionCountsARMT,
    cls_deviation_from_mean_loss,
    cls_multinomial_loss,
    pack_segment,
    segment_indices,
)


class ExpressionCountsARMTMemoryTPM(ExpressionCountsARMT):
    """Keep the CLS readout and train each memory token with the same TPM target."""

    def forward(self, input_ids=None, attention_mask=None, labels_mask=None, labels=None,
                return_dict=None, desc_input_ids=None, desc_attention_mask=None,
                dataset_flag=None, tss_token_index=None):
        if dataset_flag is None or tss_token_index is None:
            raise ValueError("dataset_flag and tss_token_index are required")
        batch, n_keys = dataset_flag.shape
        rows = batch * n_keys
        input_ids = input_ids.reshape(rows, -1)
        attention_mask = attention_mask.reshape(rows, -1)
        tss_token_index = tss_token_index.reshape(rows).to(input_ids.device)
        labels = labels.reshape(rows, -1, 1) if labels is not None else None
        labels_mask = labels_mask.reshape(rows, -1, 1).bool() if labels_mask is not None else None
        if labels is not None and labels_mask is None:
            raise ValueError("labels_mask is required when labels are supplied")
        if self.decoder.is_gradient_checkpointing:
            raise RuntimeError("Decoder checkpointing cannot replay mutable ARMT memory; use DNA checkpointing only")
        grid = torch.arange(rows, device=input_ids.device).reshape(batch, n_keys)
        flag = dataset_flag[:, :1].bool().expand(batch, n_keys)
        dna_source = torch.where(flag, grid[:, :1].expand_as(grid), grid).reshape(-1)
        dna_unique, dna_map = torch.unique(dna_source, sorted=True, return_inverse=True)
        desc_source = torch.where(flag, grid, grid[:, :1].expand_as(grid)).reshape(-1)
        desc_unique, desc_map = torch.unique(desc_source, sorted=True, return_inverse=True)
        desc_ids = desc_input_ids.reshape(rows, -1)[desc_unique]
        desc_mask = desc_attention_mask.reshape(rows, -1)[desc_unique]
        desc_hidden = self.desc_model(input_ids=desc_ids, attention_mask=desc_mask, return_dict=True).last_hidden_state
        positions = torch.arange(desc_mask.shape[1], device=desc_mask.device)[None]
        last = positions.masked_fill(~desc_mask.bool(), -1).max(-1).values.clamp_min(0)
        desc = self.desc_ln(self.desc_proj(desc_hidden[torch.arange(len(last), device=last.device), last]))[desc_map]
        length = input_ids.shape[1]
        output_logits = desc.new_zeros(rows, length, 1)
        processed = torch.zeros_like(attention_mask, dtype=torch.bool)
        final_expression = desc.new_zeros(rows, 1)
        cls_sum = desc.new_zeros((), dtype=torch.float32)
        signal_sum = cls_sum.clone()
        cls_count = cls_sum.clone()
        signal_count = cls_sum.clone()
        memory_tpm_sum = cls_sum.clone()
        deviation_sum, multinomial_sum = cls_sum.clone(), cls_sum.clone()
        regularizer_count = 0
        diagnostics = {}
        cls_id, sep_id = self.config.cls_token_id, self.config.sep_token_id
        if cls_id is None or sep_id is None:
            raise ValueError("DNA config must specify CLS and SEP token IDs")
        self.reset_memory()
        try:
            for step, offset in enumerate(self.offsets, 1):
                indices, valid = segment_indices(tss_token_index, offset, self.payload_size, attention_mask)
                active = valid.any(-1)
                local_ids, local_mask, indices, valid = pack_segment(
                    input_ids, indices, valid, cls_id, sep_id, self.config.pad_token_id)
                dna = self.bert(input_ids=local_ids[dna_unique], attention_mask=local_mask[dna_unique],
                                return_dict=True).last_hidden_state[dna_map]
                combined = self.dna_ln(dna) + desc[:, None].to(dna.dtype)
                position = self.segment_position.weight[offset + self.max_segments // 2]
                combined = combined + position.to(combined.dtype)
                memory = self.armt_memory.to(combined.dtype)[None].expand(rows, -1, -1)
                augmented = torch.cat([combined, memory], dim=1)
                augmented_mask = torch.cat([local_mask, torch.ones(rows, self.num_mem_tokens,
                                            device=valid.device, dtype=torch.bool)], dim=1)
                for layer in self.decoder.layers:
                    layer.active_rows = active
                decoded = self.decoder(inputs_embeds=augmented, attention_mask=augmented_mask,
                                       return_dict=True).last_hidden_state
                predictions = self.activation(self.classifier(decoded[:, :self.segment_size]))
                final_expression = torch.where(active[:, None], predictions[:, 0], final_expression)
                signal_prediction = predictions[:, 1:-1] * valid[:, :, None]
                output_logits = output_logits.scatter_add(1, indices[:, :, None], signal_prediction)
                visited = torch.zeros_like(attention_mask, dtype=torch.long).scatter_add_(1, indices, valid.long())
                processed |= visited.bool()
                if labels is not None:
                    expr_valid = labels_mask[:, 0, 0] & active
                    error = self.loss_fct(predictions[:, 0].float(), labels[:, 0].float()).reshape(rows)
                    step_sum = error.masked_fill(~expr_valid, 0).sum()
                    count = expr_valid.sum()
                    cls_sum, cls_count = cls_sum + step_sum, cls_count + count
                    memory_predictions = self.activation(self.classifier(decoded[:, self.segment_size:]))
                    tpm_target = labels[:, :1].float().expand_as(memory_predictions)
                    memory_errors = self.loss_fct(memory_predictions.float(), tpm_target)
                    memory_step_sum = memory_errors.mean(dim=(1, 2)).masked_fill(~expr_valid, 0).sum()
                    memory_tpm_sum = memory_tpm_sum + memory_step_sum
                    diagnostics[f"armt_memory_tpm_loss_step_{step}"] = (
                        memory_step_sum / count.clamp_min(1)).detach()
                    signal_labels = labels.gather(1, indices[:, :, None])
                    signal_valid = labels_mask.gather(1, indices[:, :, None]) & valid[:, :, None]
                    errors = self.loss_fct(predictions[:, 1:-1].float(), signal_labels.float())
                    step_signal = errors.masked_fill(~signal_valid, 0).sum()
                    count_signal = signal_valid.sum()
                    signal_sum, signal_count = signal_sum + step_signal, signal_count + count_signal
                    diagnostics[f"armt_expr_loss_step_{step}"] = (step_sum / count.clamp_min(1)).detach()
                    diagnostics[f"armt_signal_loss_step_{step}"] = (step_signal / count_signal.clamp_min(1)).detach()
                    if expr_valid.any():
                        step_labels = labels[:, :1]
                        step_mask = expr_valid[:, None, None]
                        if self.use_deviation_loss:
                            deviation_sum = deviation_sum + cls_deviation_from_mean_loss(
                                predictions[:, :1], step_labels, step_mask, n_keys, dataset_flag[:, 0].bool())
                        if self.use_multinomial_loss:
                            multinomial_sum = multinomial_sum + cls_multinomial_loss(
                                predictions[:, :1], step_labels, step_mask, n_keys,
                                group_mask=dataset_flag[:, 0].bool())
                        regularizer_count += 1
            output_logits = torch.cat([final_expression[:, None], output_logits[:, 1:]], dim=1)
            output_mask = None
            cls_loss = other_loss = loss = deviation_loss = multinomial_loss = None
            if labels is not None:
                output_mask = labels_mask & processed[:, :, None]
                output_mask[:, 0] = labels_mask[:, 0] & processed.any(-1)[:, None]
                cls_loss = cls_sum / cls_count.clamp_min(1)
                other_loss = signal_sum / signal_count.clamp_min(1)
                memory_tpm_loss = memory_tpm_sum / cls_count.clamp_min(1)
                diagnostics["armt_memory_tpm_loss"] = memory_tpm_loss.detach()
                loss = cls_loss + memory_tpm_loss + self.weight * other_loss
                if self.use_deviation_loss:
                    deviation_loss = deviation_sum / max(1, regularizer_count)
                    loss = loss + self.weight_deviation_loss * deviation_loss
                if self.use_multinomial_loss:
                    multinomial_loss = multinomial_sum / max(1, regularizer_count)
                    loss = loss + self.weight_multinomial_loss * multinomial_loss
            output = ARMTExpressionModelOutput(
                loss=loss, logits=output_logits, labels_reshaped=labels,
                labels_mask_reshaped=output_mask, cls_loss=cls_loss, other_loss=other_loss,
                deviation_loss=deviation_loss, multinomial_loss=multinomial_loss,
                armt_metrics=diagnostics or None)
            return output if return_dict is not False else (loss, output_logits)
        finally:
            self.reset_memory()

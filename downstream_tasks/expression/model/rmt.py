import torch
import torch.nn.functional as F

import math
import random

from gena_lm.modeling_rmt import RMTEncoderForSequenceClassification


class RMTEncoderExpression(RMTEncoderForSequenceClassification):
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__(base_model, **rmt_kwargs)
        self.rmt_config["sum_loss"] = True
        self.mixed_length_ratio = self.rmt_config.get("mixed_length_ratio", 0.0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        desc_vectors=None,
        tpm=None,
        dataset_mean=None,
        dataset_deviation=None,
        return_dict=None,
        **kwargs,
    ):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "labels_mask": labels_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "desc_vectors": desc_vectors,
        }

        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)

        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(
            input_ids, labels, labels_mask, tpm
        )

        losses = []
        loss_components = ["cls_loss", "other_loss", "mean_loss", "diviation_loss"]
        loss_components_metrics = {k: [] for k in loss_components}
        logits = []
        labels_reshaped_l = []
        labels_mask_reshaped_l = []
        logits_masks = []
        labels_segm = []

        max_n_segments = len(segmented)
        n_skipped_segments = 0
        # Если используется "mixed_length_ratio" во время обучения — рандомно уменьшаем кол-во сегментов
        if (
            self.mixed_length_ratio > 0.0
            and self.model.training
            and random.random() > self.mixed_length_ratio
        ):
            max_n_segments = (
                random.randint(2, max_n_segments)
                if max_n_segments > 1
                else max_n_segments
            )

        for seg_num, (
            segment_input_ids,
            segment_labels,
            segment_labels_mask,
        ) in enumerate(zip(segmented, segmented_labels, segmented_labels_mask)):
            if (self.rmt_config["bptt_depth"] > -1) and (
                (len(segmented) - seg_num) > self.rmt_config["bptt_depth"]
            ):
                memory = memory.detach()

            # Проверяем, не нужно ли нам «пропускать» сегменты в рамках mixed_length_ratio
            if (seg_num + 1 - n_skipped_segments > max_n_segments) and (
                self.mixed_length_ratio > 0.0
            ):
                break

            seg_kwargs = dict(**kwargs)
            seg_kwargs["output_hidden_states"] = True

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                n_skipped_segments += 1
                continue

            input_ids_ = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask_ = self.get_attention_mask(input_ids_)
            token_type_ids_ = self.get_token_type_ids(input_ids_)

            inputs_embeds_ = self.model.embeddings(input_ids_)
            inputs_embeds_[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs["input_ids"] = None
            seg_kwargs["inputs_embeds"] = inputs_embeds_
            seg_kwargs["attention_mask"] = attention_mask_
            seg_kwargs["token_type_ids"] = token_type_ids_

            if desc_vectors is not None:
                seg_kwargs["desc_vectors"] = desc_vectors[non_empty_mask]

            if dataset_mean is not None:
                seg_kwargs["dataset_mean"] = dataset_mean[non_empty_mask]
            if dataset_deviation is not None:
                seg_kwargs["dataset_deviation"] = dataset_deviation[non_empty_mask]

            if labels is not None:
                seg_kwargs["labels"] = torch.stack(
                    [el for el, m in zip(segment_labels, non_empty_mask) if m]
                )
            if labels_mask is not None:
                seg_kwargs["labels_mask"] = torch.stack(
                    [el for el, m in zip(segment_labels_mask, non_empty_mask) if m]
                )

            out = self.model(**seg_kwargs)

            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            # handle loss
            out_loss = out.get("loss", None)
            if out_loss is not None:
                losses.append(out_loss)

            # handle loss components
            for loss_component in loss_components:
                component_value = out.get(loss_component, None)
                if component_value is not None:
                    loss_components_metrics[loss_component].append(component_value)

            # handle other outputs
            # logits.append(out['logits'].detach())
            # labels_reshaped = out.get('labels_reshaped', None)
            # labels_mask_reshaped = out.get('labels_mask_reshaped', None)
            # labels_reshaped_l.append(labels_reshaped.detach())
            # labels_mask_reshaped_l.append(labels_mask_reshaped.detach())
            # labels_segm += [seg_kwargs['labels']]

            logits.append(out["logits"])
            labels_reshaped = out.get("labels_reshaped", None)
            labels_mask_reshaped = out.get("labels_mask_reshaped", None)
            labels_reshaped_l.append(labels_reshaped)
            labels_mask_reshaped_l.append(labels_mask_reshaped)
            labels_segm += [seg_kwargs["labels"]]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs["labels_mask"])

        # drop unnecessary hiddens to save memory
        if not kwargs.get("output_hidden_states"):
            for key in out.keys():
                if "hidden_state" in key:
                    out[key] = None

        for i, loss_tensor in enumerate(losses):
            out[f"loss_{i}"] = loss_tensor.mean()

        if out_loss is not None:
            out["loss"] = torch.stack(losses).mean()

        # aggregate loss components from all segments
        # TODO: may we want to log loss components per each segment?
        for loss_component in loss_components:
            if (
                loss_components_metrics[loss_component] is not None
                and len(loss_components_metrics[loss_component]) > 0
            ):
                out[f"loss_{loss_component}"] = torch.stack(
                    loss_components_metrics[loss_component]
                ).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        # for i in range(len(logits)):
        #     logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
        #     labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
        #     if len(logits_masks) > 0:
        #         logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out["logits"] = logits
        out["logits_segm"] = logits
        out["labels_segm"] = labels_segm
        out["labels_reshaped"] = labels_reshaped_l
        out["labels_mask_reshaped"] = labels_mask_reshaped_l
        if len(logits_masks) > 0:
            out["rmt_logits_masks"] = logits_masks
            out["rmt_logits_masks_segm"] = logits_masks

        return out

    def pad_and_segment(self, input_ids, labels=None, labels_mask=None, tpm=None):
        segmented_batch = []
        segmented_batch_labels = []
        segmented_batch_labels_mask = []

        if labels is None:
            labels = [None] * input_ids.shape[0]
        batch_labels = labels

        if labels_mask is None:
            labels_mask = [None] * input_ids.shape[0]
        batch_labels_mask = labels_mask

        for batch_index, (seq, seq_labels, seq_labels_mask) in enumerate(
            zip(input_ids, batch_labels, batch_labels_mask)
        ):
            content_tokens_mask = (
                (seq != self.pad_token_id)
                & (seq != self.cls_token.item())
                & (seq != self.sep_token.item())
            )
            seq = seq[content_tokens_mask]
            seq = seq[: self.segment_size * self.rmt_config["max_n_segments"]]

            if seq_labels is not None:
                seq_labels = seq_labels[content_tokens_mask]
                seq_labels = seq_labels[
                    : self.segment_size * self.rmt_config["max_n_segments"]
                ]
            if seq_labels_mask is not None:
                seq_labels_mask = seq_labels_mask[content_tokens_mask]
                seq_labels_mask = seq_labels_mask[
                    : self.segment_size * self.rmt_config["max_n_segments"]
                ]

            n_seg = math.ceil(len(seq) / self.segment_size)
            input_segments = torch.chunk(seq, n_seg)
            labels_segments = (
                torch.chunk(seq_labels, n_seg) if seq_labels is not None else None
            )
            labels_mask_segments = (
                torch.chunk(seq_labels_mask, n_seg)
                if seq_labels_mask is not None
                else None
            )

            segs_padded = []
            segs_labels_padded = []
            segs_mask_padded = []

            current_tpm = tpm[batch_index] if tpm is not None else None

            for i, seg in enumerate(input_segments):
                segs_padded.append(
                    self.pad_add_special_tokens(
                        seg,
                        self.rmt_config["input_size"],
                        add_to="inputs",
                        tpm=current_tpm,
                    )
                )
                if labels_segments is not None:
                    seg_labels = labels_segments[i]
                    segs_labels_padded.append(
                        self.pad_add_special_tokens(
                            seg_labels,
                            self.rmt_config["input_size"],
                            add_to="labels",
                            tpm=current_tpm,
                        )
                    )
                if labels_mask_segments is not None:
                    seg_labels_mask = labels_mask_segments[i]
                    segs_mask_padded.append(
                        self.pad_add_special_tokens(
                            seg_labels_mask,
                            self.rmt_config["input_size"],
                            add_to="labels_mask",
                            tpm=current_tpm,
                        )
                    )

            segmented_batch.append(segs_padded)
            if labels_segments is not None:
                segmented_batch_labels.append(segs_labels_padded)
            else:
                segmented_batch_labels.append([])
            if labels_mask_segments is not None:
                segmented_batch_labels_mask.append(segs_mask_padded)
            else:
                segmented_batch_labels_mask.append([])

        segmented_batch = [
            [s[::-1][i] if len(s) > i else None for s in segmented_batch]
            for i in range(self.rmt_config["max_n_segments"])
        ][::-1]

        if any(len(x) > 0 for x in segmented_batch_labels):
            segmented_batch_labels = [
                [s[::-1][i] if len(s) > i else None for s in segmented_batch_labels]
                for i in range(self.rmt_config["max_n_segments"])
            ][::-1]

        if any(len(x) > 0 for x in segmented_batch_labels_mask):
            segmented_batch_labels_mask = [
                [
                    s[::-1][i] if len(s) > i else None
                    for s in segmented_batch_labels_mask
                ]
                for i in range(self.rmt_config["max_n_segments"])
            ][::-1]

        return segmented_batch, segmented_batch_labels, segmented_batch_labels_mask

    def pad_add_special_tokens(self, tensor, segment_size, add_to="inputs", tpm=None):
        """
        Если add_to == 'labels'/'labels_mask', то для CLS-токена используется значение из tpm.
        """
        device = tensor.device
        input_elements = []

        if add_to == "inputs":
            input_elements += [
                self.cls_token,
                self.mem_token_ids,
                self.sep_token,
                tensor,
                self.sep_token,
            ]
            final_tensor = torch.cat(input_elements)
            pad_size = segment_size - final_tensor.shape[0]
            if pad_size > 0:
                final_tensor = F.pad(
                    final_tensor, (0, pad_size), value=self.pad_token_id
                )
            return final_tensor

        elif add_to == "labels":
            # shape [seq_len, n]
            seq_len = tensor.shape[0]
            feat_dim = tensor.shape[1]

            masked_labels = torch.zeros((1, feat_dim), device=device)

            #    if tpm is not None and not torch.isnan(tpm).all():
            if not torch.isnan(tpm).all():
                tpm_not_nan = torch.nan_to_num(tpm, nan=0.0)
                cls_label = tpm_not_nan.unsqueeze(0)
            else:
                cls_label = masked_labels

            input_elements = [
                cls_label,
                masked_labels.repeat(self.num_mem_tokens, 1),
                masked_labels,
                tensor,
                masked_labels,
            ]
            final_tensor = torch.cat(input_elements, dim=0)
            pad_size = segment_size - final_tensor.shape[0]
            if pad_size > 0:
                final_tensor = F.pad(final_tensor, (0, 0, 0, pad_size), value=0)
            return final_tensor

        elif add_to == "labels_mask":
            # shape [seq_len, n]
            seq_len, feat_dim = tensor.shape
            mask_value = torch.zeros((1, feat_dim), device=device)

            #            if tpm is not None and not torch.isnan(tpm).all():
            if not torch.isnan(tpm).all():
                cls_mask_value = (~torch.isnan(tpm)).float().unsqueeze(0)
            else:
                cls_mask_value = mask_value

            input_elements = [
                cls_mask_value,
                mask_value.repeat(self.num_mem_tokens, 1),
                mask_value,
                tensor,
                mask_value,
            ]
            final_tensor = torch.cat(input_elements, dim=0)
            pad_size = segment_size - final_tensor.shape[0]
            if pad_size > 0:
                final_tensor = F.pad(final_tensor, (0, 0, 0, pad_size), value=0)
            return final_tensor

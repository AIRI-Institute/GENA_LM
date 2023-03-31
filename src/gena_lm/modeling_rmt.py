import math
import torch
import torch.nn.functional as F


class RMTEncoderForSequenceClassification(torch.nn.Module):
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__()
        self.model = base_model
        self.set_params(**rmt_kwargs)

    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(num_mem_tokens)

        self.segment_size = rmt_config['input_size'] - num_mem_tokens - 3

    def set_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.model.embeddings(mem_token_ids)
        return memory

    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.register_buffer('cls_token', torch.tensor([tokenizer.cls_token_id]))
        self.register_buffer('sep_token', torch.tensor([tokenizer.sep_token_id]))

    def extend_word_embeddings(self, num_mem_tokens):
        vocab_size = self.model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)
        self.model.embeddings = self.model.base_model.embeddings.word_embeddings

        mem_start_ind = 1
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # todo: replace copy-pasted args with @functools.wraps(self.model.forward) decorator
        # need to change Trainer's usage of inspect.getfullargspec to inspect.signature to support @wraps
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented = self.pad_and_segment(input_ids)

        losses = []
        for seg_num, segment_input_ids in enumerate(segmented):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]

            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids

            out = self.model(**seg_kwargs)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            losses.append(out['loss'])

        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

        for i, l in enumerate(losses):
            out[f'loss_{i}'] = l.mean()

        if self.rmt_config['sum_loss']:
            out['loss'] = torch.stack(losses).mean(dim=0)

        mem_token_ids = self.mem_token_ids
        memory_tokens = self.model.embeddings(mem_token_ids)

        return out

    def pad_and_segment(self, input_ids):
        segmented_batch = []
        for seq in input_ids:
            seq = seq[(seq != self.pad_token_id) & (seq != self.cls_token.item()) & (seq != self.sep_token.item())]
            seq = seq[:self.segment_size * self.rmt_config['max_n_segments']]

            n_seg = math.ceil(len(seq) / self.segment_size)
            input_segments = torch.chunk(seq, n_seg)
            input_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size']) for t in input_segments]

            segmented_batch.append(input_segments)

        # batch of segments -> segmented batch
        # + align segments to right border
        # so that the last segment is always non-empty
        segmented_batch = [[s[::-1][i] if len(s) > i else None for s in segmented_batch]
                           for i in range(self.rmt_config['max_n_segments'])][::-1]
        return segmented_batch

    def pad_add_special_tokens(self, tensor, segment_size):
        input_elements = []
        input_elements += [self.cls_token, self.mem_token_ids, self.sep_token, tensor, self.sep_token]
        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size))
        return tensor

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)


class RMTEncoderForTokenClassification(RMTEncoderForSequenceClassification):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__(base_model, **rmt_kwargs)
        self.rmt_config['sum_loss'] = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        # todo: currently output from RMT model is not the same like from backbone model with 1 segment
        # because of inserted memory tokens and operations with cls/sep/pad in pad_and_segment
        # need to impl such that output from forward is like output from backbone model:
        # input -> segmented_inp -> segmented_logits -> output
        #                               | -> loss         | -> metrics
        #                           segmented_labels <- labels

        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'labels_mask': labels_mask, 'pos_weight': pos_weight,
                  'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  }

        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)

            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids
            if labels is not None:
                seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
            if labels_mask is not None:
                seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
            if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                pos_weight = pos_weight[0, 0, :][None, None, :]
                segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.model(**seg_kwargs)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            losses.append(out['loss'])
            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])

        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

        for i, l in enumerate(losses):
            out[f'loss_{i}'] = l.mean()

        # aggregate losses from all segments
        out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        mem_token_ids = self.mem_token_ids

        return out

    def pad_and_segment(self, input_ids, labels=None, labels_mask=None):
        segmented_batch = []
        segmented_batch_labels = []
        segmented_batch_labels_mask = []

        if labels is None:
            labels = [None] * input_ids.shape[0]
        batch_labels = labels

        if labels_mask is None:
            labels_mask = [None] * input_ids.shape[0]
        batch_labels_mask = labels_mask

        for seq, labels, labels_mask in zip(input_ids, batch_labels, batch_labels_mask):
            content_tokens_mask = (seq != self.pad_token_id) & (seq != self.cls_token.item()) & (seq != self.sep_token.item())
            seq = seq[content_tokens_mask]
            seq = seq[:self.segment_size * self.rmt_config['max_n_segments']]
            if labels is not None:
                labels = labels[content_tokens_mask]
                labels = labels[:self.segment_size * self.rmt_config['max_n_segments']]
            if labels_mask is not None:
                labels_mask = labels_mask[content_tokens_mask]
                labels_mask = labels_mask[:self.segment_size * self.rmt_config['max_n_segments']]

            n_seg = math.ceil(len(seq) / self.segment_size)
            input_segments = torch.chunk(seq, n_seg)
            input_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size']) for t in input_segments]
            segmented_batch.append(input_segments)

            if labels is not None:
                labels_segments = torch.chunk(labels, n_seg)
                labels_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size'], add_to='labels') for t in labels_segments]
                segmented_batch_labels.append(labels_segments)

            if labels_mask is not None:
                labels_mask_segments = torch.chunk(labels_mask, n_seg)
                labels_mask_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size'], add_to='labels_mask') for t in labels_mask_segments]
                segmented_batch_labels_mask.append(labels_mask_segments)

        # batch of segments -> segmented batch
        # + align segments to right border
        # so that the last segment is always non-empty
        segmented_batch = [[s[::-1][i] if len(s) > i else None for s in segmented_batch]
                           for i in range(self.rmt_config['max_n_segments'])][::-1]

        if len(segmented_batch_labels) > 0:
            segmented_batch_labels = [[s[::-1][i] if len(s) > i else None for s in segmented_batch_labels]
                                      for i in range(self.rmt_config['max_n_segments'])][::-1]

        if len(segmented_batch_labels_mask) > 0:
            segmented_batch_labels_mask = [[s[::-1][i] if len(s) > i else None for s in segmented_batch_labels_mask]
                                           for i in range(self.rmt_config['max_n_segments'])][::-1]

        return segmented_batch, segmented_batch_labels, segmented_batch_labels_mask

    def pad_add_special_tokens(self, tensor, segment_size, add_to='inputs'):
        input_elements = []
        if add_to == 'inputs':
            input_elements += [self.cls_token, self.mem_token_ids, self.sep_token, tensor, self.sep_token]
        elif add_to == 'labels':
            masked_labels = torch.zeros((1, tensor.shape[-1]), device=tensor.device)
            input_elements += [masked_labels, masked_labels.repeat(self.num_mem_tokens, 1), masked_labels, tensor, masked_labels]
        elif add_to == 'labels_mask':
            mask_value = torch.zeros((1), device=tensor.device)
            input_elements += [mask_value, mask_value.repeat(self.num_mem_tokens), mask_value, tensor, mask_value]

        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            if add_to == 'inputs':
                tensor = F.pad(tensor, (0, pad_size), value=self.pad_token_id)
            elif add_to == 'labels':
                # todo: labels pad value should be specified, if not multilable classification it could be just -100
                tensor = F.pad(tensor, (0, 0, 0, pad_size), value=0)
            elif add_to == 'labels_mask':
                tensor = F.pad(tensor, (0, pad_size), value=0)
        return tensor

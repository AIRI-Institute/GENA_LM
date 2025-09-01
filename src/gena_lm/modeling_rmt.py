import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchcrf import CRF
import numpy as np
from tqdm import tqdm
import copy
import time
from typing import Tuple


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
        hidden_states = []
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
            if seg_kwargs['labels'] is not None:
                seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]

            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids

            out = self.model(**seg_kwargs)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            losses.append(out.get('loss', torch.tensor(0.0)))
            if seg_kwargs.get('output_hidden_states'):
                hidden_states += [out['hidden_states']]

        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None
        else:
            for i in range(len(hidden_states)):
                out[f'hidden_states_{i}'] = hidden_states[i]

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
        # print(segmented)

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
            memory[non_empty_mask] = out['hidden_states'][-1][:, self.memory_position]

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
    







class DownSample1D(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=2):
        super().__init__()
        layers = [nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)]
        layers += [
            nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
            for _ in range(num_layers - 1)
        ]
        self.conv_layers = nn.ModuleList(layers)
        self.activation_fn = nn.SiLU()
        # Use ceil_mode=True to handle arbitrary sequence lengths
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.activation_fn(conv_layer(x))
        hidden = x  # Save for skip connection
        x = self.avg_pool(hidden)
        return x, hidden

class UpSample1D(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=2):
        super().__init__()
        # Use ConvTranspose1d for upsampling
        self.up = nn.ConvTranspose1d(input_channels, output_channels, kernel_size=2, stride=2)
        layers = [nn.Conv1d(output_channels * 2, output_channels, kernel_size=3, padding=1)]
        layers += [
            nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
            for _ in range(num_layers - 1)
        ]
        self.conv_layers = nn.ModuleList(layers)
        self.activation_fn = nn.SiLU()

    def forward(self, x, skip_connection):
        x = self.up(x)
        # Adjust size if necessary to match the skip connection
        diff = skip_connection.size(2) - x.size(2)
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :skip_connection.size(2)]
        # Concatenate skip connection
        x = torch.cat([skip_connection, x], dim=1)
        for conv_layer in self.conv_layers:
            x = self.activation_fn(conv_layer(x))
        return x

class FinalConv1D(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=2):
        super().__init__()
        layers = [nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)]
        layers += [
            nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
            for _ in range(num_layers - 1)
        ]
        self.conv_layers = nn.ModuleList(layers)
        self.activation_fn = nn.SiLU()

    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i < len(self.conv_layers) - 1:
                x = self.activation_fn(x)
        return x

class UNET1DSegmentationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, output_channels_list=None, num_conv_layers_per_block=2):
        super().__init__()
        if output_channels_list is None:
            output_channels_list = [64, 128, 256]  # Default values
        else:
            output_channels_list = list(output_channels_list)  # Ensure it's a list

        self.num_pooling_layers = len(output_channels_list)

        # Downsampling blocks
        downsample_input_channels_list = [embed_dim] + output_channels_list[:-1]
        self.downsample_blocks = nn.ModuleList([
            DownSample1D(in_ch, out_ch, num_conv_layers_per_block)
            for in_ch, out_ch in zip(downsample_input_channels_list, output_channels_list)
        ])

        # Upsampling blocks
        reversed_output_channels_list = output_channels_list[::-1]
        upsample_input_channels_list = [output_channels_list[-1]] + reversed_output_channels_list[:-1]
        upsample_output_channels_list = reversed_output_channels_list
        self.upsample_blocks = nn.ModuleList([
            UpSample1D(in_ch, out_ch, num_conv_layers_per_block)
            for in_ch, out_ch in zip(upsample_input_channels_list, upsample_output_channels_list)
        ])

        self.final_block = FinalConv1D(output_channels_list[0], num_classes, num_conv_layers_per_block)

    def forward(self, x):
        hiddens = []
        # Downsampling path
        for downsample_block in self.downsample_blocks:
            x, hidden = downsample_block(x)
            hiddens.append(hidden)

        # Upsampling path
        for i, upsample_block in enumerate(self.upsample_blocks):
            skip_connection = hiddens[-(i + 1)]
            x = upsample_block(x, skip_connection)

        x = self.final_block(x)
        return x    








# class DownSample1D(nn.Module):
#     """
#     1D-UNET down-sampling block.
#     """

#     def __init__(self,
#                  input_channels: int,
#                  output_channels: int,
#                  num_layers: int = 2):
#         super().__init__()

#         self.first_layer = [nn.Conv1d(
#             in_channels=input_channels,
#             out_channels=output_channels,
#             kernel_size=3,
#             stride=1,
#             dilation=1,
#             padding="same",
#         )]

#         self.next_layers = [
#             nn.Conv1d(
#                 in_channels=output_channels,
#                 out_channels=output_channels,
#                 kernel_size=3,
#                 stride=1,
#                 dilation=1,
#                 padding="same",
#             )
#             for _ in range(num_layers - 1)
#         ]
#         self.conv_layers = nn.ModuleList(self.first_layer + self.next_layers)

#         self.avg_pool = nn.AvgPool1d(
#             kernel_size=2,
#             stride=2,
#             padding=0,
#             ceil_mode=True,
#         )
#         self.activation_fn = nn.SiLU()

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         for conv_layer in self.conv_layers:
#             x = self.activation_fn(conv_layer(x))

#         hidden = x
#         x = self.avg_pool(hidden)
#         return x, hidden


# class UpSample1D(nn.Module):
#     """
#     1D-UNET up-sampling block.
#     """

#     def __init__(self,
#                  input_channels: int,
#                  output_channels: int,
#                  num_layers: int = 2):
#         super().__init__()

#         self._first_layer = [nn.ConvTranspose1d(
#             in_channels=input_channels,
#             out_channels=output_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )]

#         self._next_layers = [
#             nn.ConvTranspose1d(
#                 in_channels=output_channels,
#                 out_channels=output_channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             )
#             for _ in range(num_layers - 1)
#         ]
#         self.conv_layers = nn.ModuleList(self._first_layer + self._next_layers)
#         self._activation_fn = nn.SiLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for conv_layer in self.conv_layers:
#             x = self._activation_fn(conv_layer(x))

#         x = F.interpolate(x, size=2 * x.shape[2], mode="nearest")
#         return x


# class FinalConv1D(nn.Module):
#     """
#     Final output block of the 1D-UNET.
#     """
#     def __init__(self,
#                  input_channels: int,
#                  output_channels: int,
#                  num_layers: int = 2):
#         super().__init__()

#         self._first_layer = [nn.Conv1d(
#             in_channels=input_channels,
#             out_channels=output_channels,
#             kernel_size=3,
#             stride=1,
#             dilation=1,
#             padding="same",
#         )]

#         self._next_layers = [
#             nn.Conv1d(
#                 in_channels=output_channels,
#                 out_channels=output_channels,
#                 kernel_size=3,
#                 stride=1,
#                 dilation=1,
#                 padding="same",
#             )
#             for _ in range(num_layers - 1)
#         ]
#         self.conv_layers = nn.ModuleList(self._first_layer + self._next_layers)
#         self._activation_fn = nn.SiLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for i, conv_layer in enumerate(self.conv_layers):
#             x = conv_layer(x)
#             if i < len(self.conv_layers) - 1:
#                 x = self._activation_fn(x)
#         return x


# class UNET1DSegmentationHead(nn.Module):
#     """
#     1D-UNET head that now supports **any** input length with <1-element padding per level.
#     """

#     def __init__(self,
#                  embed_dim: int,
#                  num_classes: int,
#                  output_channels_list: Tuple[int, ...] = (64, 128, 256),
#                  num_conv_layers_per_block: int = 2):
#         super().__init__()

#         self._num_pooling_layers = len(output_channels_list)

#         downsample_input_channels_list = (embed_dim,) + output_channels_list[:-1]
#         self._downsample_blocks = nn.ModuleList([
#             DownSample1D(
#                 input_channels=i_ch,
#                 output_channels=o_ch,
#                 num_layers=num_conv_layers_per_block,
#             )
#             for i_ch, o_ch in zip(downsample_input_channels_list,
#                                   output_channels_list)
#         ])

#         rev = tuple(reversed(output_channels_list))
#         upsample_input_channels_list = (output_channels_list[-1],) + rev
#         self._upsample_blocks = nn.ModuleList([
#             UpSample1D(
#                 input_channels=i_ch,
#                 output_channels=o_ch,
#                 num_layers=num_conv_layers_per_block,
#             )
#             for i_ch, o_ch in zip(upsample_input_channels_list, rev)
#         ])

#         self.final_block = FinalConv1D(
#             input_channels=output_channels_list[0],
#             output_channels=num_classes * 2,
#             num_layers=num_conv_layers_per_block,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         original_len = x.shape[2]

#         hiddens = []
#         for downsample_block in self._downsample_blocks:
#             x, hidden = downsample_block(x)
#             hiddens.append(hidden)

#         for upsample_block, hidden in zip(self._upsample_blocks, reversed(hiddens)):
#             x = upsample_block(x)

#             assert hidden.shape[2] <= x.shape[2] + 1, (f"Skip connection too long: hidden={hidden.shape[2]}, "f"up={x.shape[2]} (difference > 1)")

#             if x.shape[2] > hidden.shape[2]:
#                 x = x[:, :, :hidden.shape[2]]
#             elif x.shape[2] < hidden.shape[2]:       
#                 hidden = hidden[:, :, :x.shape[2]]

#             x = x + hidden

#         x = self.final_block(x)

#         assert x.shape[2] == original_len
#         if x.shape[2] > original_len:
#             x = x[:, :, :original_len]

#         return x















class RMTEncoderForLetterLevelTokenClassificationBidirectionalUNET(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, forward_model, backward_model, **rmt_kwargs):
        super().__init__() 
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.nucleotide_embedding = nn.Embedding(100, 1024)
        self.unet = UNET1DSegmentationHead(
                            embed_dim=1024*3,
                            num_classes=1024*3,
                            output_channels_list=[256, 512, 1024],  # Example channel sizes as a list
                            num_conv_layers_per_block=2
                        )
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(1024*3, 5)
        
        self.set_params(**rmt_kwargs)
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(num_mem_tokens)

        self.segment_size = rmt_config['input_size'] - num_mem_tokens - 3

    def set_forward_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.forward_model.embeddings(mem_token_ids)
        return memory

    def set_backward_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.backward_model.embeddings(mem_token_ids)
        return memory

    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.register_buffer('cls_token', torch.tensor([tokenizer.cls_token_id]))
        self.register_buffer('sep_token', torch.tensor([tokenizer.sep_token_id]))

    def extend_word_embeddings(self, num_mem_tokens):
        vocab_size = self.forward_model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.forward_model.resize_token_embeddings(extended_vocab_size)
        self.forward_model.embeddings = self.forward_model.base_model.embeddings.word_embeddings
        mem_start_ind = 1
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

        vocab_size = self.backward_model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.backward_model.resize_token_embeddings(extended_vocab_size)
        self.backward_model.embeddings = self.backward_model.base_model.embeddings.word_embeddings
        mem_start_ind = 1
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

    def forward(self, input_ids_forward=None, attention_mask_forward=None, token_type_ids_forward=None, input_ids_backward=None, 
                attention_mask_backward=None, token_type_ids_backward=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask_forward=None, labels_mask_backward=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater_forward=None, embedding_repeater_backward=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
        # todo: currently output from RMT model is not the same like from backbone model with 1 segment
        # because of inserted memory tokens and operations with cls/sep/pad in pad_and_segment
        # need to impl such that output from forward is like output from backbone model:
        # input -> segmented_inp -> segmented_logits -> output
        #                               | -> loss         | -> metrics
        #                           segmented_labels <- labels


        bidirectional_out = dict()
        
        kwargs = {'input_ids': input_ids_forward, 'attention_mask': attention_mask_forward, 'token_type_ids': token_type_ids_forward,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'labels_mask': labels_mask_forward, 'pos_weight': pos_weight,
                  'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  }
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids_forward.shape

        memory = self.set_forward_memory()
        memory = memory.repeat(input_ids_forward.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids_forward, labels, labels_mask_forward)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :] # comment it
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True #???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)

            inputs_embeds = self.forward_model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids
            
            if labels is not None:
                seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
            if labels_mask_forward is not None:
                seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
            if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                pos_weight = pos_weight[0, 0, :][None, None, :]
                segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.forward_model(**seg_kwargs)
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask_forward is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)

        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids

        bidirectional_out['forward_logits'] = out['logits']
        bidirectional_out['forward_rmt_logits_masks'] = out['rmt_logits_masks']



        kwargs = {'input_ids': input_ids_backward, 'attention_mask': attention_mask_backward, 'token_type_ids': token_type_ids_backward,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'labels_mask': labels_mask_backward, 'pos_weight': pos_weight,
                  'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  }
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids_backward.shape

        memory = self.set_backward_memory()
        memory = memory.repeat(input_ids_backward.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids_backward, labels, labels_mask_backward)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :]
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True #???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)

            inputs_embeds = self.backward_model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids
            if labels is not None:
                seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
            if labels_mask_backward is not None:
                seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
            if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                pos_weight = pos_weight[0, 0, :][None, None, :]
                segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.backward_model(**seg_kwargs)
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask_backward is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)

        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids

        
        bidirectional_out['backward_logits'] = out['logits']
        bidirectional_out['backward_rmt_logits_masks'] = out['rmt_logits_masks']

        
        
        if embedding_repeater_forward is not None and embedding_repeater_backward is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits_forward = bidirectional_out['forward_logits'][b, bidirectional_out['forward_rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                curr_logits_backward = bidirectional_out['backward_logits'][b, bidirectional_out['backward_rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater_forward = embedding_repeater_forward[b][lllm]
                curr_repeater_backward = embedding_repeater_backward[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits_forward[:, curr_repeater_forward, :], torch.flip(curr_logits_backward[:, curr_repeater_backward, :], dims=[1])), dim=-1)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits_forward[:, curr_repeater_forward, :] + torch.flip(curr_logits_backward[:, curr_repeater_backward, :], dims=[1])
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                
                collected_repeated_logits = repeated_curr_logits_with_letter_embeddings.transpose(1, 2)
                
                collected_repeated_logits = self.activation_fn(self.unet(collected_repeated_logits))
                
                collected_repeated_logits = collected_repeated_logits.transpose(1, 2)
                
                collected_repeated_logits = self.fc(collected_repeated_logits)
                
                loss = None
                if letter_level_labels is not None:
                    
                    if self.rmt_config['use_crf']:
                        # print(torch.transpose(collected_repeated_logits.float(), 0, 1).shape, torch.argmax(torch.transpose(curr_letter_level_labels.long(), 0, 1), axis=-1).shape)
                        crf_loss = -self.crf_model(collected_repeated_logits.float(), curr_letter_level_labels) / collected_repeated_logits.shape[1]
                        
                        loss_fct = CrossEntropyLoss()
                        # print(collected_repeated_logits.shape, curr_letter_level_labels.shape)
                        loss = loss_fct(collected_repeated_logits.float().squeeze(), curr_letter_level_labels.squeeze()) + crf_loss
                        # print(f'CRF loss: {crf_loss}')
                        # print(f'CRF loss smoothed: {-crf_loss / collected_repeated_logits.shape[1]}')
                        crf_decoding = torch.tensor(self.crf_model.decode(collected_repeated_logits.float()))
                        # print(crf_decoding.shape)
                    else:
                        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight) # torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, needed_for_loss_mask, :]
                        # print(pos_weight.shape, collected_repeated_logits.shape, curr_letter_level_labels.shape, len(needed_for_loss_mask))
                        # print(collected_repeated_logits.float()[:, needed_for_loss_mask, :].shape, curr_letter_level_labels.float()[:, needed_for_loss_mask, :].shape)
                        # loss = loss_fct(collected_repeated_logits.float()[:, needed_for_loss_mask, :], curr_letter_level_labels.float()[:, needed_for_loss_mask, :])
                        loss = loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        # print(loss)

                        
                        
                batched_losses.append(loss) # loss

                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)












class RMTEncoderForLetterLevelTokenClassificationBidirectionalImprovedUNET(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, forward_model, backward_model, sub_model, **rmt_kwargs):
        super().__init__() 
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.sub_model = sub_model
        self.unet = UNET1DSegmentationHead(
                            embed_dim=1024,
                            num_classes=1024,
                            output_channels_list=[256, 512, 1024],  # Example channel sizes as a list
                            num_conv_layers_per_block=2
                        )
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(1024, 5)
        
        self.set_params(**rmt_kwargs)
        
        self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(num_mem_tokens)

        self.segment_size = rmt_config['input_size'] - num_mem_tokens - 3

    def set_forward_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.forward_model.embeddings(mem_token_ids)
        return memory

    def set_backward_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.backward_model.embeddings(mem_token_ids)
        return memory

    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.register_buffer('cls_token', torch.tensor([tokenizer.cls_token_id]))
        self.register_buffer('sep_token', torch.tensor([tokenizer.sep_token_id]))

    def extend_word_embeddings(self, num_mem_tokens):
        vocab_size = self.forward_model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.forward_model.resize_token_embeddings(extended_vocab_size)
        self.forward_model.embeddings = self.forward_model.base_model.embeddings.word_embeddings
        mem_start_ind = 1
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

        vocab_size = self.backward_model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.backward_model.resize_token_embeddings(extended_vocab_size)
        self.backward_model.embeddings = self.backward_model.base_model.embeddings.word_embeddings
        mem_start_ind = 1
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

    def forward(self, input_ids_forward=None, attention_mask_forward=None, token_type_ids_forward=None, input_ids_backward=None, 
                attention_mask_backward=None, token_type_ids_backward=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask_forward=None, labels_mask_backward=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater_forward=None, embedding_repeater_backward=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
        # todo: currently output from RMT model is not the same like from backbone model with 1 segment
        # because of inserted memory tokens and operations with cls/sep/pad in pad_and_segment
        # need to impl such that output from forward is like output from backbone model:
        # input -> segmented_inp -> segmented_logits -> output
        #                               | -> loss         | -> metrics
        #                           segmented_labels <- labels


        bidirectional_out = dict()
        
        kwargs = {'input_ids': input_ids_forward, 'attention_mask': attention_mask_forward, 'token_type_ids': token_type_ids_forward,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'labels_mask': labels_mask_forward, 'pos_weight': pos_weight,
                  'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  }
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids_forward.shape

        memory = self.set_forward_memory()
        memory = memory.repeat(input_ids_forward.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids_forward, labels, labels_mask_forward)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :] # comment it
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True #???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)

            inputs_embeds = self.forward_model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids
            
            if labels is not None:
                seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
            if labels_mask_forward is not None:
                seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
            if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                pos_weight = pos_weight[0, 0, :][None, None, :]
                segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.forward_model(**seg_kwargs)
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask_forward is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)

        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids

        bidirectional_out['forward_logits'] = out['logits']
        bidirectional_out['forward_rmt_logits_masks'] = out['rmt_logits_masks']



        kwargs = {'input_ids': input_ids_backward, 'attention_mask': attention_mask_backward, 'token_type_ids': token_type_ids_backward,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'labels_mask': labels_mask_backward, 'pos_weight': pos_weight,
                  'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  }
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids_backward.shape

        memory = self.set_backward_memory()
        memory = memory.repeat(input_ids_backward.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids_backward, labels, labels_mask_backward)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :]
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True #???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)

            inputs_embeds = self.backward_model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids
            if labels is not None:
                seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
            if labels_mask_backward is not None:
                seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
            if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                pos_weight = pos_weight[0, 0, :][None, None, :]
                segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.backward_model(**seg_kwargs)
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask_backward is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)

        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids

        
        bidirectional_out['backward_logits'] = out['logits']
        bidirectional_out['backward_rmt_logits_masks'] = out['rmt_logits_masks']

        
        
        if embedding_repeater_forward is not None and embedding_repeater_backward is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits_forward = bidirectional_out['forward_logits'][b, bidirectional_out['forward_rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                curr_logits_backward = bidirectional_out['backward_logits'][b, bidirectional_out['backward_rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater_forward = embedding_repeater_forward[b][lllm]
                curr_repeater_backward = embedding_repeater_backward[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                # curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                # repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits_forward[:, curr_repeater_forward, :], torch.flip(curr_logits_backward[:, curr_repeater_backward, :], dims=[1])), dim=-1)
                repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits_forward[:, curr_repeater_forward, :] + torch.flip(curr_logits_backward[:, curr_repeater_backward, :], dims=[1])
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                
                repeater_kwargs = dict()
                repeater_kwargs['input_ids'] = None
                repeater_kwargs['inputs_embeds'] = repeated_curr_logits_with_letter_embeddings
                # repeater_kwargs['rmt_embeds'] = curr_logits[:, curr_repeater, :]
                # print('BATCH SIZE', bs)
                # print(letter_level_attention_mask)
                repeater_kwargs['token_type_ids'] = letter_level_token_types_ids[b, lllm].unsqueeze(0)
                repeater_kwargs['attention_mask'] = letter_level_attention_mask[b, lllm].unsqueeze(0)
                
                
                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['sub_model_input_size'])
                
                sub_model_segment_mask = np.zeros(num_letter_level_segments)
                sub_model_segment_mask[:self.rmt_config['num_trainable_sub_model_segments']] = 1
                # sub_model_segment_mask[-1] = 1
                
                np.random.shuffle(sub_model_segment_mask)
                # sub_model_segment_mask = list(sub_model_segment_mask.astype(bool))
                # sub_model_segment_mask[0] = False
                # print(sub_model_segment_mask)
                
                # needed_seg_counter = 0
                needed_for_loss_mask = []
                custom_pos_weights = []
                all_edge_loss_mask = []
                
                repeated_logits = []
                for i in range(num_letter_level_segments):
                    seg_repeater_kwargs = dict()
                    seg_repeater_kwargs['input_ids'] = None
                    seg_repeater_kwargs['inputs_embeds'] = repeater_kwargs['inputs_embeds'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :]
                    # seg_repeater_kwargs['rmt_embeds'] = repeater_kwargs['rmt_embeds'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :]
                    seg_repeater_kwargs['attention_mask'] = repeater_kwargs['attention_mask'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size']].long()
                    seg_repeater_kwargs['token_type_ids'] = repeater_kwargs['token_type_ids'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size']].long()
                    
                    # print('curr_letter_level_labels.shape', curr_letter_level_labels.shape)
                    seg_labels = curr_letter_level_labels[:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :].detach().cpu().numpy()
                    
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['attention_mask'].shape[1]
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['token_type_ids'].shape[1]
                    
                    # custom_pos_weight = np.ones(seg_labels.shape) * 10 # np.zeros(seg_labels.shape)
                    # custom_pos_weight[:, :, 1] = 80.
                    # custom_pos_weight[:, :, 4] = 80.
                    # edge_loss_mask = np.array([False] * seg_labels.shape[1])
                    # for lp in range(custom_pos_weight.shape[1]-1):
                    #     if np.all(seg_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(seg_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(seg_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(seg_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                    #         seq_range = np.clip(lp+4, 0, seg_labels.shape[1]) - np.clip(lp-4, 0, None)
                    #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = np.array([pos_weight[0, 0, :].detach().cpu().numpy().tolist()] * seq_range) ** 7.0
                    #         edge_loss_mask[np.clip(lp-4, 0, None):lp+4] = True
                    # print('google', seg_repeater_kwargs['inputs_embeds'].shape)
                    
                    # print('4444444444444444444444444444444')
                    # print(self.model.device, self.sub_model.device)
                    if sub_model_segment_mask[i]:
                    # if [0, 0, 1, 0, 0, 0] in seg_labels and [0, 1, 0, 0, 1, 0] in seg_labels[0, :, :] and needed_seg_counter < self.rmt_config['num_trainable_sub_model_segments']:
                        # print('THIS IS WORKING SECTION!!!!!!!!!!!!!!!!!')
                        out_sub_model = self.sub_model(**seg_repeater_kwargs)
                        out_sub_model['logits'] = out_sub_model['logits'].transpose(1, 2)
                        out_sub_model['logits'] = self.activation_fn(self.unet(out_sub_model['logits']))
                        out_sub_model['logits'] = out_sub_model['logits'].transpose(1, 2)
                        out_sub_model['logits'] = self.fc(out_sub_model['logits'])
                        needed_for_loss_mask += [True] * seg_repeater_kwargs['inputs_embeds'].shape[1]
                        # needed_seg_counter += 1
                        assert out_sub_model['logits'].shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    else:
                        # print('----------------no grad----------------------')
                        with torch.no_grad():
                            out_sub_model = self.sub_model(**seg_repeater_kwargs)
                            out_sub_model['logits'] = out_sub_model['logits'].transpose(1, 2)
                            out_sub_model['logits'] = self.activation_fn(self.unet(out_sub_model['logits']))
                            out_sub_model['logits'] = out_sub_model['logits'].transpose(1, 2)
                            out_sub_model['logits'] = self.fc(out_sub_model['logits'])
                            needed_for_loss_mask += [False] * seg_repeater_kwargs['inputs_embeds'].shape[1]
                            assert out_sub_model['logits'].shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    # print('55555555555555555555555555555555555555')
                    repeated_logits.append(out_sub_model['logits'])
                    # custom_pos_weights.append(custom_pos_weight)
                    # all_edge_loss_mask.append(edge_loss_mask)
                    
                    # if i == (num_letter_level_segments-1):
                    #     print(out_sub_model)
                    
                collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                # collected_custom_pos_weights = np.concatenate(custom_pos_weights, axis=1)
                # all_edge_loss_mask = np.concatenate(all_edge_loss_mask, axis=0)
                # print('NEW POS WEIGHT SHAPE', collected_custom_pos_weights.shape)
                
                loss = None
                if letter_level_labels is not None:
                    
                    if self.rmt_config['use_crf']:
                        # print(torch.transpose(collected_repeated_logits.float(), 0, 1).shape, torch.argmax(torch.transpose(curr_letter_level_labels.long(), 0, 1), axis=-1).shape)
                        crf_loss = -self.crf_model(collected_repeated_logits.float(), curr_letter_level_labels) / collected_repeated_logits.shape[1]
                        
                        loss_fct = CrossEntropyLoss()
                        # print(collected_repeated_logits.shape, curr_letter_level_labels.shape)
                        loss = loss_fct(collected_repeated_logits.float().squeeze(), curr_letter_level_labels.squeeze()) + crf_loss
                        # print(f'CRF loss: {crf_loss}')
                        # print(f'CRF loss smoothed: {-crf_loss / collected_repeated_logits.shape[1]}')
                        crf_decoding = torch.tensor(self.crf_model.decode(collected_repeated_logits.float()))
                        # print(crf_decoding.shape)
                    else:
                        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight) # torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, needed_for_loss_mask, :]
                        # print(pos_weight.shape, collected_repeated_logits.shape, curr_letter_level_labels.shape, len(needed_for_loss_mask))
                        # print(collected_repeated_logits.float()[:, needed_for_loss_mask, :].shape, curr_letter_level_labels.float()[:, needed_for_loss_mask, :].shape)
                        loss = loss_fct(collected_repeated_logits.float()[:, needed_for_loss_mask, :], curr_letter_level_labels.float()[:, needed_for_loss_mask, :])
                        # loss = loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        # print(loss)

                        
                        
                batched_losses.append(loss) # loss

                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)










class RMTEncoderForLetterLevelTokenClassificationBidirectionalImproved(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, forward_model, backward_model, sub_model, **rmt_kwargs):
        super().__init__() 
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.sub_model = sub_model
        # self.nucleotide_embedding = nn.Embedding(100, 1024)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(num_mem_tokens)

        self.segment_size = rmt_config['input_size'] - num_mem_tokens - 3

    def set_forward_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.forward_model.embeddings(mem_token_ids)
        return memory

    def set_backward_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.backward_model.embeddings(mem_token_ids)
        return memory

    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.register_buffer('cls_token', torch.tensor([tokenizer.cls_token_id]))
        self.register_buffer('sep_token', torch.tensor([tokenizer.sep_token_id]))

    def extend_word_embeddings(self, num_mem_tokens):
        vocab_size = self.forward_model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.forward_model.resize_token_embeddings(extended_vocab_size)
        self.forward_model.embeddings = self.forward_model.base_model.embeddings.word_embeddings
        mem_start_ind = 1
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

        vocab_size = self.backward_model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.backward_model.resize_token_embeddings(extended_vocab_size)
        self.backward_model.embeddings = self.backward_model.base_model.embeddings.word_embeddings
        mem_start_ind = 1
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

    def forward(self, input_ids_forward=None, attention_mask_forward=None, token_type_ids_forward=None, input_ids_backward=None, 
                attention_mask_backward=None, token_type_ids_backward=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask_forward=None, labels_mask_backward=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater_forward=None, embedding_repeater_backward=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
        # todo: currently output from RMT model is not the same like from backbone model with 1 segment
        # because of inserted memory tokens and operations with cls/sep/pad in pad_and_segment
        # need to impl such that output from forward is like output from backbone model:
        # input -> segmented_inp -> segmented_logits -> output
        #                               | -> loss         | -> metrics
        #                           segmented_labels <- labels


        # bidirectional_out = dict()
        
        # kwargs = {'input_ids': input_ids_forward, 'attention_mask': attention_mask_forward, 'token_type_ids': token_type_ids_forward,
        #           'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
        #           'labels': labels, 'labels_mask': labels_mask_forward, 'pos_weight': pos_weight,
        #           'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
        #           'return_dict': return_dict,
        #           }
        # # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids_forward.shape

        # memory = self.set_forward_memory()
        # memory = memory.repeat(input_ids_forward.shape[0], 1, 1)
        # segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids_forward, labels, labels_mask_forward)

        # losses = []
        # logits = []
        # logits_masks = []
        # labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :] # comment it
        # for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
        #                                                                                        segmented_labels,
        #                                                                                        segmented_labels_mask)):
        #     if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
        #         memory = memory.detach()

        #     seg_kwargs = dict(**kwargs)
        #     seg_kwargs['output_hidden_states'] = True #???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

        #     non_empty_mask = [s is not None for s in segment_input_ids]
        #     if sum(non_empty_mask) == 0:
        #         continue
        #     input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        #     attention_mask = self.get_attention_mask(input_ids)
        #     token_type_ids = self.get_token_type_ids(input_ids)

        #     inputs_embeds = self.forward_model.embeddings(input_ids)
        #     inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

        #     seg_kwargs['input_ids'] = None
        #     seg_kwargs['inputs_embeds'] = inputs_embeds
        #     seg_kwargs['attention_mask'] = attention_mask
        #     seg_kwargs['token_type_ids'] = token_type_ids
            
        #     if labels is not None:
        #         seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
        #     if labels_mask_forward is not None:
        #         seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
        #     if pos_weight is not None:
        #         # all values in the second dimension of pos_weight should be the same
        #         pos_weight = pos_weight[0, 0, :][None, None, :]
        #         segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
        #         seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

        #     out = self.forward_model(**seg_kwargs)
        #     # print(out)
        #     memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

        #     logits.append(out['logits'].detach())
        #     labels_segm += [seg_kwargs['labels']]

        #     if labels_mask_forward is not None:
        #         logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # # drop unnecessary hiddens to save memory
        # if not kwargs.get('output_hidden_states'):
        #     for key in out.keys():
        #         if 'hidden_state' in key:
        #             out[key] = None

        # for i in range(len(logits)):
        #     logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
        #     labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
        #     if len(logits_masks) > 0:
        #         logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        # out['logits'] = torch.cat(logits, dim=1)

        # out['logits_segm'] = [logits]
        # out['labels_segm'] = [labels_segm]
        # if len(logits_masks) > 0:
        #     out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
        #     out['rmt_logits_masks_segm'] = [logits_masks]

        # # print(out['logits'])
        # mem_token_ids = self.mem_token_ids

        # bidirectional_out['forward_logits'] = out['logits']
        # bidirectional_out['forward_rmt_logits_masks'] = out['rmt_logits_masks']



        # kwargs = {'input_ids': input_ids_backward, 'attention_mask': attention_mask_backward, 'token_type_ids': token_type_ids_backward,
        #           'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
        #           'labels': labels, 'labels_mask': labels_mask_backward, 'pos_weight': pos_weight,
        #           'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
        #           'return_dict': return_dict,
        #           }
        # # print('POSPOSPOSPOSPOS', pos_weight.shape)
        # bs, seq_len = input_ids_backward.shape

        # memory = self.set_backward_memory()
        # memory = memory.repeat(input_ids_backward.shape[0], 1, 1)
        # segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids_backward, labels, labels_mask_backward)

        # losses = []
        # logits = []
        # logits_masks = []
        # labels_segm = []
        # # pos_weight = pos_weight[0, 0, :][None, None, :]
        # for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
        #                                                                                        segmented_labels,
        #                                                                                        segmented_labels_mask)):
        #     if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
        #         memory = memory.detach()

        #     seg_kwargs = dict(**kwargs)
        #     seg_kwargs['output_hidden_states'] = True #???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

        #     non_empty_mask = [s is not None for s in segment_input_ids]
        #     if sum(non_empty_mask) == 0:
        #         continue
        #     input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        #     attention_mask = self.get_attention_mask(input_ids)
        #     token_type_ids = self.get_token_type_ids(input_ids)

        #     inputs_embeds = self.backward_model.embeddings(input_ids)
        #     inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

        #     seg_kwargs['input_ids'] = None
        #     seg_kwargs['inputs_embeds'] = inputs_embeds
        #     seg_kwargs['attention_mask'] = attention_mask
        #     seg_kwargs['token_type_ids'] = token_type_ids
        #     if labels is not None:
        #         seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
        #     if labels_mask_backward is not None:
        #         seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
        #     if pos_weight is not None:
        #         # all values in the second dimension of pos_weight should be the same
        #         pos_weight = pos_weight[0, 0, :][None, None, :]
        #         segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
        #         seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

        #     out = self.backward_model(**seg_kwargs)
        #     # print(out)
        #     memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

        #     logits.append(out['logits'].detach())
        #     labels_segm += [seg_kwargs['labels']]

        #     if labels_mask_backward is not None:
        #         logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # # drop unnecessary hiddens to save memory
        # if not kwargs.get('output_hidden_states'):
        #     for key in out.keys():
        #         if 'hidden_state' in key:
        #             out[key] = None

        # for i in range(len(logits)):
        #     logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
        #     labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
        #     if len(logits_masks) > 0:
        #         logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        # out['logits'] = torch.cat(logits, dim=1)

        # out['logits_segm'] = [logits]
        # out['labels_segm'] = [labels_segm]
        # if len(logits_masks) > 0:
        #     out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
        #     out['rmt_logits_masks_segm'] = [logits_masks]

        # # print(out['logits'])
        # mem_token_ids = self.mem_token_ids

        
        # bidirectional_out['backward_logits'] = out['logits']
        # bidirectional_out['backward_rmt_logits_masks'] = out['rmt_logits_masks']

        
        
        if True: #embedding_repeater_forward is not None and embedding_repeater_backward is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                # curr_logits_forward = bidirectional_out['forward_logits'][b, bidirectional_out['forward_rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # curr_logits_backward = bidirectional_out['backward_logits'][b, bidirectional_out['backward_rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater_forward = embedding_repeater_forward[b][lllm]
                curr_repeater_backward = embedding_repeater_backward[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                # curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                # repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits_forward[:, curr_repeater_forward, :], torch.flip(curr_logits_backward[:, curr_repeater_backward, :], dims=[1])), dim=-1)
                repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                
                repeater_kwargs = dict()
                repeater_kwargs['input_ids'] = None
                repeater_kwargs['inputs_embeds'] = repeated_curr_logits_with_letter_embeddings
                # repeater_kwargs['rmt_embeds'] = curr_logits[:, curr_repeater, :]
                # print('BATCH SIZE', bs)
                # print(letter_level_attention_mask)
                repeater_kwargs['token_type_ids'] = letter_level_token_types_ids[b, lllm].unsqueeze(0)
                repeater_kwargs['attention_mask'] = letter_level_attention_mask[b, lllm].unsqueeze(0)
                
                
                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['sub_model_input_size'])
                
                sub_model_segment_mask = np.zeros(num_letter_level_segments)
                sub_model_segment_mask[:self.rmt_config['num_trainable_sub_model_segments']] = 1
                # sub_model_segment_mask[-1] = 1
                
                np.random.shuffle(sub_model_segment_mask)
                # sub_model_segment_mask = list(sub_model_segment_mask.astype(bool))
                # sub_model_segment_mask[0] = False
                # print(sub_model_segment_mask)
                
                # needed_seg_counter = 0
                needed_for_loss_mask = []
                custom_pos_weights = []
                all_edge_loss_mask = []
                
                repeated_logits = []
                for i in range(num_letter_level_segments):
                    seg_repeater_kwargs = dict()
                    seg_repeater_kwargs['input_ids'] = None
                    seg_repeater_kwargs['inputs_embeds'] = repeater_kwargs['inputs_embeds'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :]
                    # seg_repeater_kwargs['rmt_embeds'] = repeater_kwargs['rmt_embeds'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :]
                    seg_repeater_kwargs['attention_mask'] = repeater_kwargs['attention_mask'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size']].long()
                    seg_repeater_kwargs['token_type_ids'] = repeater_kwargs['token_type_ids'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size']].long()
                    
                    # print('curr_letter_level_labels.shape', curr_letter_level_labels.shape)
                    seg_labels = curr_letter_level_labels[:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :].detach().cpu().numpy()
                    
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['attention_mask'].shape[1]
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['token_type_ids'].shape[1]
                    
                    # custom_pos_weight = np.ones(seg_labels.shape) * 10 # np.zeros(seg_labels.shape)
                    # custom_pos_weight[:, :, 1] = 80.
                    # custom_pos_weight[:, :, 4] = 80.
                    # edge_loss_mask = np.array([False] * seg_labels.shape[1])
                    # for lp in range(custom_pos_weight.shape[1]-1):
                    #     if np.all(seg_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(seg_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(seg_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(seg_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                    #         seq_range = np.clip(lp+4, 0, seg_labels.shape[1]) - np.clip(lp-4, 0, None)
                    #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = np.array([pos_weight[0, 0, :].detach().cpu().numpy().tolist()] * seq_range) ** 7.0
                    #         edge_loss_mask[np.clip(lp-4, 0, None):lp+4] = True
                    # print('google', seg_repeater_kwargs['inputs_embeds'].shape)
                    
                    # print('4444444444444444444444444444444')
                    # print(self.model.device, self.sub_model.device)
                    if sub_model_segment_mask[i]:
                    # if [0, 0, 1, 0, 0, 0] in seg_labels and [0, 1, 0, 0, 1, 0] in seg_labels[0, :, :] and needed_seg_counter < self.rmt_config['num_trainable_sub_model_segments']:
                        # print('THIS IS WORKING SECTION!!!!!!!!!!!!!!!!!')
                        out_sub_model = self.sub_model(**seg_repeater_kwargs)
                        needed_for_loss_mask += [True] * seg_repeater_kwargs['inputs_embeds'].shape[1]
                        # needed_seg_counter += 1
                        assert out_sub_model['logits'].shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    else:
                        # print('----------------no grad----------------------')
                        with torch.no_grad():
                            out_sub_model = self.sub_model(**seg_repeater_kwargs)
                            needed_for_loss_mask += [False] * seg_repeater_kwargs['inputs_embeds'].shape[1]
                            assert out_sub_model['logits'].shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    # print('55555555555555555555555555555555555555')
                    repeated_logits.append(out_sub_model['logits'])
                    # custom_pos_weights.append(custom_pos_weight)
                    # all_edge_loss_mask.append(edge_loss_mask)
                    
                    # if i == (num_letter_level_segments-1):
                    #     print(out_sub_model)
                    
                collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                # collected_custom_pos_weights = np.concatenate(custom_pos_weights, axis=1)
                # all_edge_loss_mask = np.concatenate(all_edge_loss_mask, axis=0)
                # print('NEW POS WEIGHT SHAPE', collected_custom_pos_weights.shape)
                
                loss = None
                if letter_level_labels is not None:
                    
                    if self.rmt_config['use_crf']:
                        # print(torch.transpose(collected_repeated_logits.float(), 0, 1).shape, torch.argmax(torch.transpose(curr_letter_level_labels.long(), 0, 1), axis=-1).shape)
                        crf_loss = -self.crf_model(collected_repeated_logits.float(), curr_letter_level_labels) / collected_repeated_logits.shape[1]
                        
                        loss_fct = CrossEntropyLoss()
                        # print(collected_repeated_logits.shape, curr_letter_level_labels.shape)
                        loss = loss_fct(collected_repeated_logits.float().squeeze(), curr_letter_level_labels.squeeze()) + crf_loss
                        # print(f'CRF loss: {crf_loss}')
                        # print(f'CRF loss smoothed: {-crf_loss / collected_repeated_logits.shape[1]}')
                        crf_decoding = torch.tensor(self.crf_model.decode(collected_repeated_logits.float()))
                        # print(crf_decoding.shape)
                    else:
                        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight) # torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, needed_for_loss_mask, :]
                        # print(pos_weight.shape, collected_repeated_logits.shape, curr_letter_level_labels.shape, len(needed_for_loss_mask))
                        # print(collected_repeated_logits.float()[:, needed_for_loss_mask, :].shape, curr_letter_level_labels.float()[:, needed_for_loss_mask, :].shape)
                        loss = loss_fct(collected_repeated_logits.float()[:, needed_for_loss_mask, :], curr_letter_level_labels.float()[:, needed_for_loss_mask, :])
                        # loss = loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        # print(loss)

                        
                        
                batched_losses.append(loss) # loss

                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)













class RMTEncoderForLetterLevelTokenClassificationBidirectional(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, forward_model, backward_model, sub_model, **rmt_kwargs):
        super().__init__() 
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.sub_model = sub_model
        self.nucleotide_embedding = nn.Embedding(100, 1024)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(num_mem_tokens)

        self.segment_size = rmt_config['input_size'] - num_mem_tokens - 3

    def set_forward_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.forward_model.embeddings(mem_token_ids)
        return memory

    def set_backward_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.backward_model.embeddings(mem_token_ids)
        return memory

    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.register_buffer('cls_token', torch.tensor([tokenizer.cls_token_id]))
        self.register_buffer('sep_token', torch.tensor([tokenizer.sep_token_id]))

    def extend_word_embeddings(self, num_mem_tokens):
        vocab_size = self.forward_model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.forward_model.resize_token_embeddings(extended_vocab_size)
        self.forward_model.embeddings = self.forward_model.base_model.embeddings.word_embeddings
        mem_start_ind = 1
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

        vocab_size = self.backward_model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.backward_model.resize_token_embeddings(extended_vocab_size)
        self.backward_model.embeddings = self.backward_model.base_model.embeddings.word_embeddings
        mem_start_ind = 1
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

    def forward(self, input_ids_forward=None, attention_mask_forward=None, token_type_ids_forward=None, input_ids_backward=None, 
                attention_mask_backward=None, token_type_ids_backward=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask_forward=None, labels_mask_backward=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater_forward=None, embedding_repeater_backward=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
        # todo: currently output from RMT model is not the same like from backbone model with 1 segment
        # because of inserted memory tokens and operations with cls/sep/pad in pad_and_segment
        # need to impl such that output from forward is like output from backbone model:
        # input -> segmented_inp -> segmented_logits -> output
        #                               | -> loss         | -> metrics
        #                           segmented_labels <- labels


        bidirectional_out = dict()
        
        kwargs = {'input_ids': input_ids_forward, 'attention_mask': attention_mask_forward, 'token_type_ids': token_type_ids_forward,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'labels_mask': labels_mask_forward, 'pos_weight': pos_weight,
                  'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  }
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids_forward.shape

        memory = self.set_forward_memory()
        memory = memory.repeat(input_ids_forward.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids_forward, labels, labels_mask_forward)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :]
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True #???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)

            inputs_embeds = self.forward_model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids
            
            if labels is not None:
                seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
            if labels_mask_forward is not None:
                seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
            if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                pos_weight = pos_weight[0, 0, :][None, None, :]
                segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.forward_model(**seg_kwargs)
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask_forward is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)

        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids

        bidirectional_out['forward_logits'] = out['logits']
        bidirectional_out['forward_rmt_logits_masks'] = out['rmt_logits_masks']



        kwargs = {'input_ids': input_ids_backward, 'attention_mask': attention_mask_backward, 'token_type_ids': token_type_ids_backward,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'labels_mask': labels_mask_backward, 'pos_weight': pos_weight,
                  'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  }
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids_backward.shape

        memory = self.set_backward_memory()
        memory = memory.repeat(input_ids_backward.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids_backward, labels, labels_mask_backward)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :]
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True #???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)

            inputs_embeds = self.backward_model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids
            if labels is not None:
                seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
            if labels_mask_backward is not None:
                seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
            if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                pos_weight = pos_weight[0, 0, :][None, None, :]
                segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.backward_model(**seg_kwargs)
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask_backward is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)

        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids

        
        bidirectional_out['backward_logits'] = out['logits']
        bidirectional_out['backward_rmt_logits_masks'] = out['rmt_logits_masks']

        
        
        if embedding_repeater_forward is not None and embedding_repeater_backward is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits_forward = bidirectional_out['forward_logits'][b, bidirectional_out['forward_rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                curr_logits_backward = bidirectional_out['backward_logits'][b, bidirectional_out['backward_rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater_forward = embedding_repeater_forward[b][lllm]
                curr_repeater_backward = embedding_repeater_backward[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits_forward[:, curr_repeater_forward, :], torch.flip(curr_logits_backward[:, curr_repeater_backward, :], dims=[1])), dim=-1)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                
                repeater_kwargs = dict()
                repeater_kwargs['input_ids'] = None
                repeater_kwargs['inputs_embeds'] = repeated_curr_logits_with_letter_embeddings
                # repeater_kwargs['rmt_embeds'] = curr_logits[:, curr_repeater, :]
                # print('BATCH SIZE', bs)
                # print(letter_level_attention_mask)
                repeater_kwargs['token_type_ids'] = letter_level_token_types_ids[b, lllm].unsqueeze(0)
                repeater_kwargs['attention_mask'] = letter_level_attention_mask[b, lllm].unsqueeze(0)
                
                
                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['sub_model_input_size'])
                
                sub_model_segment_mask = np.zeros(num_letter_level_segments)
                sub_model_segment_mask[:self.rmt_config['num_trainable_sub_model_segments']] = 1
                # sub_model_segment_mask[-1] = 1
                
                np.random.shuffle(sub_model_segment_mask)
                # sub_model_segment_mask = list(sub_model_segment_mask.astype(bool))
                # sub_model_segment_mask[0] = False
                # print(sub_model_segment_mask)
                
                # needed_seg_counter = 0
                needed_for_loss_mask = []
                custom_pos_weights = []
                all_edge_loss_mask = []
                
                repeated_logits = []
                for i in range(num_letter_level_segments):
                    seg_repeater_kwargs = dict()
                    seg_repeater_kwargs['input_ids'] = None
                    seg_repeater_kwargs['inputs_embeds'] = repeater_kwargs['inputs_embeds'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :]
                    # seg_repeater_kwargs['rmt_embeds'] = repeater_kwargs['rmt_embeds'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :]
                    seg_repeater_kwargs['attention_mask'] = repeater_kwargs['attention_mask'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size']].long()
                    seg_repeater_kwargs['token_type_ids'] = repeater_kwargs['token_type_ids'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size']].long()
                    
                    # print('curr_letter_level_labels.shape', curr_letter_level_labels.shape)
                    seg_labels = curr_letter_level_labels[:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :].detach().cpu().numpy()
                    
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['attention_mask'].shape[1]
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['token_type_ids'].shape[1]
                    
                    custom_pos_weight = np.ones(seg_labels.shape) * 10 # np.zeros(seg_labels.shape)
                    # custom_pos_weight[:, :, 1] = 80.
                    # custom_pos_weight[:, :, 4] = 80.
                    # edge_loss_mask = np.array([False] * seg_labels.shape[1])
                    for lp in range(custom_pos_weight.shape[1]-1):
                        if np.all(seg_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(seg_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(seg_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(seg_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                            seq_range = np.clip(lp+4, 0, seg_labels.shape[1]) - np.clip(lp-4, 0, None)
                            custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = np.array([pos_weight[0, 0, :].detach().cpu().numpy().tolist()] * seq_range) ** 7.0
                    #         edge_loss_mask[np.clip(lp-4, 0, None):lp+4] = True
                    # print('google', seg_repeater_kwargs['inputs_embeds'].shape)
                    
                    # print('4444444444444444444444444444444')
                    # print(self.model.device, self.sub_model.device)
                    if sub_model_segment_mask[i]:
                    # if [0, 0, 1, 0, 0, 0] in seg_labels and [0, 1, 0, 0, 1, 0] in seg_labels[0, :, :] and needed_seg_counter < self.rmt_config['num_trainable_sub_model_segments']:
                        # print('THIS IS WORKING SECTION!!!!!!!!!!!!!!!!!')
                        out_sub_model = self.sub_model(**seg_repeater_kwargs)
                        needed_for_loss_mask += [True] * seg_repeater_kwargs['inputs_embeds'].shape[1]
                        # needed_seg_counter += 1
                        assert out_sub_model['logits'].shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    else:
                        # print('----------------no grad----------------------')
                        with torch.no_grad():
                            out_sub_model = self.sub_model(**seg_repeater_kwargs)
                            needed_for_loss_mask += [False] * seg_repeater_kwargs['inputs_embeds'].shape[1]
                            assert out_sub_model['logits'].shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    # print('55555555555555555555555555555555555555')
                    repeated_logits.append(out_sub_model['logits'])
                    custom_pos_weights.append(custom_pos_weight)
                    # all_edge_loss_mask.append(edge_loss_mask)
                    
                    # if i == (num_letter_level_segments-1):
                    #     print(out_sub_model)
                    
                collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                collected_custom_pos_weights = np.concatenate(custom_pos_weights, axis=1)
                # all_edge_loss_mask = np.concatenate(all_edge_loss_mask, axis=0)
                # print('NEW POS WEIGHT SHAPE', collected_custom_pos_weights.shape)
                
                loss = None
                if letter_level_labels is not None:
                    
                    if self.rmt_config['use_crf']:
                        # print(torch.transpose(collected_repeated_logits.float(), 0, 1).shape, torch.argmax(torch.transpose(curr_letter_level_labels.long(), 0, 1), axis=-1).shape)
                        crf_loss = -self.crf_model(collected_repeated_logits.float(), curr_letter_level_labels) / collected_repeated_logits.shape[1]
                        
                        loss_fct = CrossEntropyLoss()
                        # print(collected_repeated_logits.shape, curr_letter_level_labels.shape)
                        loss = loss_fct(collected_repeated_logits.float().squeeze(), curr_letter_level_labels.squeeze()) + crf_loss
                        # print(f'CRF loss: {crf_loss}')
                        # print(f'CRF loss smoothed: {-crf_loss / collected_repeated_logits.shape[1]}')
                        crf_decoding = torch.tensor(self.crf_model.decode(collected_repeated_logits.float()))
                        # print(crf_decoding.shape)
                    else:
                        loss_fct = BCEWithLogitsLoss(pos_weight=torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, needed_for_loss_mask, :]) # torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, needed_for_loss_mask, :]
                        # print(pos_weight.shape, collected_repeated_logits.shape, curr_letter_level_labels.shape, len(needed_for_loss_mask))
                        # print(collected_repeated_logits.float()[:, needed_for_loss_mask, :].shape, curr_letter_level_labels.float()[:, needed_for_loss_mask, :].shape)
                        loss = loss_fct(collected_repeated_logits.float()[:, needed_for_loss_mask, :], curr_letter_level_labels.float()[:, needed_for_loss_mask, :])
                        # loss = loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        # print(loss)

                        
                        
                batched_losses.append(loss) # loss

                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)






class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky





    
class RMTEncoderForLetterLevelTokenClassification(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, sub_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        self.sub_model = sub_model
        # self.nucleotide_embedding = nn.Embedding(100, 1024)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                # curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                # repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                
                repeater_kwargs = dict()
                repeater_kwargs['input_ids'] = None
                repeater_kwargs['inputs_embeds'] = repeated_curr_logits_with_letter_embeddings
                # repeater_kwargs['rmt_embeds'] = curr_logits[:, curr_repeater, :]
                # print('BATCH SIZE', bs)
                # print(letter_level_attention_mask)
                repeater_kwargs['token_type_ids'] = letter_level_token_types_ids[b, lllm].unsqueeze(0)
                repeater_kwargs['attention_mask'] = letter_level_attention_mask[b, lllm].unsqueeze(0)

                # print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                
                
                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['sub_model_input_size'])
                
                sub_model_segment_mask = np.zeros(num_letter_level_segments)
                sub_model_segment_mask[:self.rmt_config['num_trainable_sub_model_segments']] = 1
                # sub_model_segment_mask[-1] = 1
                
                np.random.shuffle(sub_model_segment_mask)
                # sub_model_segment_mask = list(sub_model_segment_mask.astype(bool))
                # sub_model_segment_mask[0] = False
                # print(sub_model_segment_mask)
                
                # needed_seg_counter = 0
                needed_for_loss_mask = []
                custom_pos_weights = []
                all_edge_loss_mask = []
                
                repeated_logits = []
                for i in range(num_letter_level_segments):
                    seg_repeater_kwargs = dict()
                    seg_repeater_kwargs['input_ids'] = None
                    seg_repeater_kwargs['inputs_embeds'] = repeater_kwargs['inputs_embeds'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :]
                    # seg_repeater_kwargs['rmt_embeds'] = repeater_kwargs['rmt_embeds'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :]
                    seg_repeater_kwargs['attention_mask'] = repeater_kwargs['attention_mask'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size']].long()
                    seg_repeater_kwargs['token_type_ids'] = repeater_kwargs['token_type_ids'][:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size']].long()
                    
                    # print('curr_letter_level_labels.shape', curr_letter_level_labels.shape)
                    seg_labels = curr_letter_level_labels[:, i*self.rmt_config['sub_model_input_size']:(i+1)*self.rmt_config['sub_model_input_size'], :].detach().cpu().numpy()
                    
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['attention_mask'].shape[1]
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['token_type_ids'].shape[1]
                    
                    # custom_pos_weight = np.ones(seg_labels.shape) / 2 # np.zeros(seg_labels.shape)
                    # custom_pos_weight[:, :, 1] = 80.
                    # custom_pos_weight[:, :, 4] = 80.
                    # edge_loss_mask = np.array([False] * seg_labels.shape[1])
                    # for lp in range(custom_pos_weight.shape[1]-1):
                    #     if np.all(seg_labels[0, lp, :] == np.array([0, 0, 1, 0, 0, 0])) and np.all(seg_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1, 0])) or np.all(seg_labels[0, lp, :] == np.array([0, 1, 0, 0, 1, 0])) and np.all(seg_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0, 0])):
                    #         seq_range = np.clip(lp+4, 0, seg_labels.shape[1]) - np.clip(lp-4, 0, None)
                    #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = np.array([pos_weight[0, 0, :].detach().cpu().numpy().tolist()] * seq_range) ** 2.0
                    #         edge_loss_mask[np.clip(lp-4, 0, None):lp+4] = True
                    # print('google', seg_repeater_kwargs['inputs_embeds'].shape)
                    
                    # print('4444444444444444444444444444444')
                    # print(self.model.device, self.sub_model.device)
                    if sub_model_segment_mask[i]:
                    # if [0, 0, 1, 0, 0, 0] in seg_labels and [0, 1, 0, 0, 1, 0] in seg_labels[0, :, :] and needed_seg_counter < self.rmt_config['num_trainable_sub_model_segments']:
                        # print('THIS IS WORKING SECTION!!!!!!!!!!!!!!!!!')
                        out_sub_model = self.sub_model(**seg_repeater_kwargs)
                        needed_for_loss_mask += [True] * seg_repeater_kwargs['inputs_embeds'].shape[1]
                        # needed_seg_counter += 1
                        assert out_sub_model['logits'].shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    else:
                        # print('----------------no grad----------------------')
                        with torch.no_grad():
                            out_sub_model = self.sub_model(**seg_repeater_kwargs)
                            needed_for_loss_mask += [False] * seg_repeater_kwargs['inputs_embeds'].shape[1]
                            assert out_sub_model['logits'].shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    # print('55555555555555555555555555555555555555')
                    repeated_logits.append(out_sub_model['logits'])
                    # custom_pos_weights.append(custom_pos_weight)
                    # all_edge_loss_mask.append(edge_loss_mask)
                    
                    # if i == (num_letter_level_segments-1):
                    #     print(out_sub_model)
                    
                collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                # collected_custom_pos_weights = np.concatenate(custom_pos_weights, axis=1)
                # all_edge_loss_mask = np.concatenate(all_edge_loss_mask, axis=0)
                # print('NEW POS WEIGHT SHAPE', collected_custom_pos_weights.shape)
                
                loss = None
                if letter_level_labels is not None:
                    
                    if self.rmt_config['use_crf']:
                        # print(torch.transpose(collected_repeated_logits.float(), 0, 1).shape, torch.argmax(torch.transpose(curr_letter_level_labels.long(), 0, 1), axis=-1).shape)
                        crf_loss = -self.crf_model(collected_repeated_logits.float(), curr_letter_level_labels) / collected_repeated_logits.shape[1]
                        
                        loss_fct = CrossEntropyLoss()
                        # print(collected_repeated_logits.shape, curr_letter_level_labels.shape)
                        loss = loss_fct(collected_repeated_logits.float().squeeze(), curr_letter_level_labels.squeeze()) + crf_loss
                        # print(f'CRF loss: {crf_loss}')
                        # print(f'CRF loss smoothed: {-crf_loss / collected_repeated_logits.shape[1]}')
                        crf_decoding = torch.tensor(self.crf_model.decode(collected_repeated_logits.float()))
                        # print(crf_decoding.shape)
                    else:

                        tversky_loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
                        loss = 0
                        for i in range(collected_repeated_logits.shape[-1]):
                            loss += tversky_loss_fn(torch.sigmoid(collected_repeated_logits.float()[:, needed_for_loss_mask, i]), curr_letter_level_labels.float()[:, needed_for_loss_mask, i])
                        
                        # loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight) # torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, needed_for_loss_mask, :]
                        # # print(pos_weight.shape, collected_repeated_logits.shape, curr_letter_level_labels.shape, len(needed_for_loss_mask))
                        # # print(collected_repeated_logits.float()[:, needed_for_loss_mask, :].shape, curr_letter_level_labels.float()[:, needed_for_loss_mask, :].shape)
                        # loss = loss_fct(collected_repeated_logits.float()[:, needed_for_loss_mask, :], curr_letter_level_labels.float()[:, needed_for_loss_mask, :])
                        # # loss = loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        # # print(loss)
                        # loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight) # torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, all_edge_loss_mask, :]
                        # edge_loss = loss_fct(collected_repeated_logits.float()[:, all_edge_loss_mask, :], curr_letter_level_labels.float()[:, all_edge_loss_mask, :])
                        # if torch.isnan(edge_loss):
                        #     edge_loss = torch.tensor(0.0).to(f"cuda:{edge_loss.get_device()}")
                        # # print('LOOOOOOOOOOOOOOOSSSSSSS', np.sum(all_edge_loss_mask), collected_repeated_logits.float()[:, all_edge_loss_mask, :].shape, curr_letter_level_labels.float()[:, all_edge_loss_mask, :].shape)
                        
                        # not_all_edge_loss_mask = ~all_edge_loss_mask
                        # loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight) # torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, not_all_edge_loss_mask, :]
                        # not_edge_loss = loss_fct(collected_repeated_logits.float()[:, not_all_edge_loss_mask, :], curr_letter_level_labels.float()[:, not_all_edge_loss_mask, :])
                        # if torch.isnan(not_edge_loss):
                        #     not_edge_loss = torch.tensor(0.0).to(f"cuda:{not_edge_loss.get_device()}")
                        
                        
                batched_losses.append(loss) # loss
                # edge_losses.append(edge_loss)
                # no_edge_losses.append(not_edge_loss)
                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            # final_model_output['edge_loss'] = torch.stack(edge_losses).mean()
            # final_model_output['not_edge_loss'] = torch.stack(no_edge_losses).mean()
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)

    
    
    
    
    
    
    
    
    
class RMTEncoderForCLSLetterLevelTokenClassification(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, sub_model, intermediate_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        self.sub_model = sub_model
        self.intermediate_model = intermediate_model
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, batched_crf_predictions = [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding #+ curr_logits[:, curr_repeater, :]
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                
                repeater_kwargs = dict()
                repeater_kwargs['input_ids'] = None
                repeater_kwargs['inputs_embeds'] = repeated_curr_logits_with_letter_embeddings
                repeater_kwargs['rmt_embeds'] = curr_logits[:, curr_repeater, :]
                # print('BATCH SIZE', bs)
                # print(letter_level_attention_mask)
                repeater_kwargs['token_type_ids'] = letter_level_token_types_ids[b, lllm].unsqueeze(0)
                repeater_kwargs['attention_mask'] = letter_level_attention_mask[b, lllm].unsqueeze(0)
                
                
                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / (self.rmt_config['sub_model_input_size']-1))
                
                # sub_model_segment_mask = np.zeros(num_letter_level_segments)
                # sub_model_segment_mask[:self.rmt_config['num_trainable_sub_model_segments']] = 1
                # sub_model_segment_mask[-1] = 1
                
                # np.random.shuffle(sub_model_segment_mask)
                # sub_model_segment_mask = list(sub_model_segment_mask.astype(bool))
                # sub_model_segment_mask[0] = False
                # print(sub_model_segment_mask)
                
                needed_seg_counter = 0
                needed_for_loss_mask = []
                
                repeated_logits = []
                for i in range(num_letter_level_segments):
                    seg_repeater_kwargs = dict()
                    seg_repeater_kwargs['input_ids'] = None
                    seg_repeater_kwargs['inputs_embeds'] = repeater_kwargs['rmt_embeds'][:, i*(self.rmt_config['sub_model_input_size']-1):(i+1)*(self.rmt_config['sub_model_input_size']-1), :] # repeater_kwargs['inputs_embeds'][:, i*(self.rmt_config['sub_model_input_size']-1):(i+1)*(self.rmt_config['sub_model_input_size']-1), :]
                    # seg_repeater_kwargs['rmt_embeds'] = repeater_kwargs['rmt_embeds'][:, i*(self.rmt_config['sub_model_input_size']-1):(i+1)*(self.rmt_config['sub_model_input_size']-1), :]
                    seg_repeater_kwargs['attention_mask'] = repeater_kwargs['attention_mask'][:, i*(self.rmt_config['sub_model_input_size']-1):(i+1)*(self.rmt_config['sub_model_input_size']-1)].long()
                    seg_repeater_kwargs['token_type_ids'] = repeater_kwargs['token_type_ids'][:, i*(self.rmt_config['sub_model_input_size']-1):(i+1)*(self.rmt_config['sub_model_input_size']-1)].long()
                    
                    # print('curr_letter_level_labels.shape', curr_letter_level_labels.shape)
                    seg_labels = curr_letter_level_labels[:, i*(self.rmt_config['sub_model_input_size']-1):(i+1)*(self.rmt_config['sub_model_input_size']-1), :].detach().cpu().numpy()
                    
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['attention_mask'].shape[1]
                    assert seg_repeater_kwargs['inputs_embeds'].shape[1] == seg_repeater_kwargs['token_type_ids'].shape[1]
                    
                    
                    # print('google', seg_repeater_kwargs['inputs_embeds'].shape)
                    
                    # print('4444444444444444444444444444444')
                    # print(self.model.device, self.sub_model.device)
                    # if sub_model_segment_mask[i]:
                    if [0, 0, 1, 0, 0, 0] in seg_labels and [0, 1, 0, 0, 1, 0] in seg_labels[0, :, :] and needed_seg_counter < self.rmt_config['num_trainable_sub_model_segments']:
                        # print('THIS IS WORKING SECTION!!!!!!!!!!!!!!!!!')
                        # np.random.random() > 0.5
                        intermediate_model_cls_embedding = self.intermediate_model.base_model.embeddings.word_embeddings(torch.tensor([1]).long().unsqueeze(0).to(f"cuda:{seg_repeater_kwargs['inputs_embeds'].get_device()}"))
                        seg_repeater_kwargs['inputs_embeds'] = torch.cat((intermediate_model_cls_embedding, seg_repeater_kwargs['inputs_embeds']), dim=1)
                        seg_repeater_kwargs['attention_mask'] = torch.cat((torch.ones(1, 1).long().to(f"cuda:{seg_repeater_kwargs['inputs_embeds'].get_device()}"), seg_repeater_kwargs['attention_mask']), dim=1)
                        seg_repeater_kwargs['token_type_ids'] = torch.cat((torch.zeros(1, 1).long().to(f"cuda:{seg_repeater_kwargs['inputs_embeds'].get_device()}"), seg_repeater_kwargs['token_type_ids']), dim=1)
                        out_intermediate_model = self.intermediate_model(**seg_repeater_kwargs)
                        
                        seg_repeater_kwargs['inputs_embeds'] = torch.cat((out_intermediate_model.logits[:, 0, :].unsqueeze(1), repeater_kwargs['inputs_embeds'][:, i*(self.rmt_config['sub_model_input_size']-1):(i+1)*(self.rmt_config['sub_model_input_size']-1), :]), dim=1)
                        out_sub_model = self.sub_model(**seg_repeater_kwargs)
                        needed_for_loss_mask += [True] * (seg_repeater_kwargs['inputs_embeds'].shape[1]-1)
                        needed_seg_counter += 1
                        assert out_sub_model.logits.shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    else:
                        # print('----------------no grad----------------------')
                        with torch.no_grad():
                            intermediate_model_cls_embedding = self.intermediate_model.base_model.embeddings.word_embeddings(torch.tensor([1]).long().unsqueeze(0).to(f"cuda:{seg_repeater_kwargs['inputs_embeds'].get_device()}"))
                            seg_repeater_kwargs['inputs_embeds'] = torch.cat((intermediate_model_cls_embedding, seg_repeater_kwargs['inputs_embeds']), dim=1)
                            seg_repeater_kwargs['attention_mask'] = torch.cat((torch.ones(1, 1).long().to(f"cuda:{seg_repeater_kwargs['inputs_embeds'].get_device()}"), seg_repeater_kwargs['attention_mask']), dim=1)
                            seg_repeater_kwargs['token_type_ids'] = torch.cat((torch.ones(1, 1).long().to(f"cuda:{seg_repeater_kwargs['inputs_embeds'].get_device()}"), seg_repeater_kwargs['token_type_ids']), dim=1)
                            out_intermediate_model = self.intermediate_model(**seg_repeater_kwargs)

                            seg_repeater_kwargs['inputs_embeds'] = torch.cat((out_intermediate_model.logits[:, 0, :].unsqueeze(1), repeater_kwargs['inputs_embeds'][:, i*(self.rmt_config['sub_model_input_size']-1):(i+1)*(self.rmt_config['sub_model_input_size']-1), :]), dim=1)
                            out_sub_model = self.sub_model(**seg_repeater_kwargs)
                            needed_for_loss_mask += [False] * (seg_repeater_kwargs['inputs_embeds'].shape[1]-1)
                            assert out_sub_model.logits.shape[1] == seg_repeater_kwargs['inputs_embeds'].shape[1]
                    # print('55555555555555555555555555555555555555')
                    repeated_logits.append(out_sub_model.logits[:, 1:, :])
                    
                    # if i == (num_letter_level_segments-1):
                    #     print(out_sub_model)
                    
                collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                
                loss = None
                if letter_level_labels is not None:
                    
                    if self.rmt_config['use_crf']:
                        # print(torch.transpose(collected_repeated_logits.float(), 0, 1).shape, torch.argmax(torch.transpose(curr_letter_level_labels.long(), 0, 1), axis=-1).shape)
                        crf_loss = -self.crf_model(collected_repeated_logits.float(), curr_letter_level_labels) / collected_repeated_logits.shape[1]
                        
                        loss_fct = CrossEntropyLoss()
                        # print(collected_repeated_logits.shape, curr_letter_level_labels.shape)
                        loss = loss_fct(collected_repeated_logits.float().squeeze(), curr_letter_level_labels.squeeze()) + crf_loss
                        # print(f'CRF loss: {crf_loss}')
                        # print(f'CRF loss smoothed: {-crf_loss / collected_repeated_logits.shape[1]}')
                        crf_decoding = torch.tensor(self.crf_model.decode(collected_repeated_logits.float()))
                        # print(crf_decoding.shape)
                    else:
                        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)
                        # print(pos_weight.shape, collected_repeated_logits.shape, curr_letter_level_labels.shape, len(needed_for_loss_mask))
                        # print(collected_repeated_logits.float()[:, needed_for_loss_mask, :].shape, curr_letter_level_labels.float()[:, needed_for_loss_mask, :].shape)
                        loss = loss_fct(collected_repeated_logits.float()[:, needed_for_loss_mask, :], curr_letter_level_labels.float()[:, needed_for_loss_mask, :])
                        # loss = loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        # print(loss)
                        
                batched_losses.append(loss)
                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)

    


    
    

    
    
    
    
    
    
    
    
class RMTEncoderForLetterLevelTokenClassificationUNET(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        self.sub_model = UNET1DSegmentationHead(
                            embed_dim=2048,
                            num_classes=2048,
                            output_channels_list=[256, 512, 1024],  # Example channel sizes as a list
                            num_conv_layers_per_block=2
                        )
        self.nucleotide_embedding = nn.Embedding(1000, 1024)
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(2048, 5)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # print(repeated_curr_logits_with_letter_embeddings)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False
                collected_repeated_logits = repeated_curr_logits_with_letter_embeddings.transpose(1, 2)
                
                collected_repeated_logits = self.activation_fn(self.sub_model(collected_repeated_logits))
                
                collected_repeated_logits = collected_repeated_logits.transpose(1, 2)
                
                collected_repeated_logits = self.fc(collected_repeated_logits)
                
                
                
                loss = None
                if letter_level_labels is not None:
                    
                    if self.rmt_config['use_crf']:
                        # print(torch.transpose(collected_repeated_logits.float(), 0, 1).shape, torch.argmax(torch.transpose(curr_letter_level_labels.long(), 0, 1), axis=-1).shape)
                        crf_loss = -self.crf_model(collected_repeated_logits.float(), curr_letter_level_labels) / collected_repeated_logits.shape[1]
                        
                        loss_fct = CrossEntropyLoss()
                        # print(collected_repeated_logits.shape, curr_letter_level_labels.shape)
                        loss = loss_fct(collected_repeated_logits.float().squeeze(), curr_letter_level_labels.squeeze()) + crf_loss
                        # print(f'CRF loss: {crf_loss}')
                        # print(f'CRF loss smoothed: {-crf_loss / collected_repeated_logits.shape[1]}')
                        crf_decoding = torch.tensor(self.crf_model.decode(collected_repeated_logits.float()))
                        # print(crf_decoding.shape)
                    else:
                        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight) # torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, needed_for_loss_mask, :]
                        # print(pos_weight.shape, collected_repeated_logits.shape, curr_letter_level_labels.shape, len(needed_for_loss_mask))
                        # print(collected_repeated_logits.float()[:, needed_for_loss_mask, :].shape, curr_letter_level_labels.float()[:, needed_for_loss_mask, :].shape)
                        loss = loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        # loss = loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        # print(loss)
                        
                        
                batched_losses.append(loss) # loss

                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)








class RMTEncoderForLetterLevelTokenClassificationUNETsegmented(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        self.sub_model = UNET1DSegmentationHead(
                            embed_dim=2048,
                            num_classes=2048,
                            output_channels_list=[256, 512, 1024],  # Example channel sizes as a list
                            num_conv_layers_per_block=2
                        )
        self.nucleotide_embedding = nn.Embedding(1000, 1024)
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(2048, 5)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # print(repeated_curr_logits_with_letter_embeddings)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                # custom_pos_weight = np.ones(curr_letter_level_labels.shape)
                # for lp in range(custom_pos_weight.shape[1]-1):
                #     if np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = 100.0

                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['unet_sub_model_input_size'])

                repeated_logits = []
                for i in range(num_letter_level_segments):
                    curr_unet_inputs_embeds = repeated_curr_logits_with_letter_embeddings[:, i*self.rmt_config['unet_sub_model_input_size']:(i+1)*self.rmt_config['unet_sub_model_input_size'], :]
                    curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)
                    curr_unet_inputs_embeds = self.activation_fn(self.sub_model(curr_unet_inputs_embeds))
                    curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)
                    curr_unet_inputs_embeds = self.fc(curr_unet_inputs_embeds)

                    repeated_logits.append(curr_unet_inputs_embeds)

                collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                
                
                
                loss = None
                if letter_level_labels is not None:
                    
                    if self.rmt_config['use_crf']:
                        # print(torch.transpose(collected_repeated_logits.float(), 0, 1).shape, torch.argmax(torch.transpose(curr_letter_level_labels.long(), 0, 1), axis=-1).shape)
                        crf_loss = -self.crf_model(collected_repeated_logits.float(), curr_letter_level_labels) / collected_repeated_logits.shape[1]
                        
                        loss_fct = CrossEntropyLoss()
                        # print(collected_repeated_logits.shape, curr_letter_level_labels.shape)
                        loss = loss_fct(collected_repeated_logits.float().squeeze(), curr_letter_level_labels.squeeze()) + crf_loss
                        # print(f'CRF loss: {crf_loss}')
                        # print(f'CRF loss smoothed: {-crf_loss / collected_repeated_logits.shape[1]}')
                        crf_decoding = torch.tensor(self.crf_model.decode(collected_repeated_logits.float()))
                        # print(crf_decoding.shape)
                    else:
                        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight) # torch.tensor(collected_custom_pos_weights).to(f"cuda:{pos_weight.get_device()}")[:, needed_for_loss_mask, :]
                        # print(pos_weight.shape, collected_repeated_logits.shape, curr_letter_level_labels.shape, len(needed_for_loss_mask))
                        # print(collected_repeated_logits.float()[:, needed_for_loss_mask, :].shape, curr_letter_level_labels.float()[:, needed_for_loss_mask, :].shape)

                        # loss = 0
                        # num_loss_segments = math.ceil(collected_repeated_logits.shape[1] / 2048)
                        # for i in range(num_loss_segments):
                        #     loss += loss_fct(collected_repeated_logits.float()[:, i*2048:(i+1)*2048, :], curr_letter_level_labels.float()[:, i*2048:(i+1)*2048, :])
                
                        loss = loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        # print(loss)
                        # tversky_loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
                        # loss = 0
                        # for i in range(collected_repeated_logits.shape[-1]):
                        #     loss += tversky_loss_fn(torch.sigmoid(collected_repeated_logits.float()[:, :, i]), curr_letter_level_labels.float()[:, :, i])
                        
                        
                batched_losses.append(loss) # loss

                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)









class RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeater(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        self.sub_model = UNET1DSegmentationHead(
                            embed_dim=1536,
                            num_classes=1536,
                            output_channels_list=[192, 384, 768],  # Example channel sizes as a list
                            num_conv_layers_per_block=2
                        )
        self.nucleotide_embedding = nn.Embedding(1000, 768)
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(1536, 9)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'])
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # print(repeated_curr_logits_with_letter_embeddings)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                # custom_pos_weight = np.ones(curr_letter_level_labels.shape)
                # for lp in range(custom_pos_weight.shape[1]-1):
                #     if np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = 100.0

                loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)

                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['unet_sub_model_input_size'])

                loss = 0

                cycles = 1
                for c in range(cycles):

                    repeated_logits = []
                    repeated_embeddings = []
                    for i in range(num_letter_level_segments):
                        curr_unet_inputs_embeds = repeated_curr_logits_with_letter_embeddings[:, i*self.rmt_config['unet_sub_model_input_size']:(i+1)*self.rmt_config['unet_sub_model_input_size'], :]
                        curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)
                        curr_unet_inputs_embeds = self.activation_fn(self.sub_model(curr_unet_inputs_embeds))
                        curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)

                        repeated_embeddings.append(curr_unet_inputs_embeds)
                        
                        curr_unet_inputs_embeds = self.fc(curr_unet_inputs_embeds)
                        repeated_logits.append(curr_unet_inputs_embeds)
    
                    collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                    collected_repeated_embeddings = torch.cat(repeated_embeddings, dim=1)

                    loss += loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                    
                    repeated_curr_logits_with_letter_embeddings += collected_repeated_embeddings
                        
                batched_losses.append(loss / cycles) # loss

                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)







class RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeaterLargeCycles3(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        self.sub_model = UNET1DSegmentationHead(
                            embed_dim=2048,
                            num_classes=2048,
                            output_channels_list=[256, 512, 1024],  # Example channel sizes as a list
                            num_conv_layers_per_block=2
                        )
        self.nucleotide_embedding = nn.Embedding(1000, 1024)
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(2048, 5)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'])
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # print(repeated_curr_logits_with_letter_embeddings)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                # custom_pos_weight = np.ones(curr_letter_level_labels.shape)
                # for lp in range(custom_pos_weight.shape[1]-1):
                #     if np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = 100.0

                loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)

                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['unet_sub_model_input_size'])

                loss = 0

                cycles = 3
                for c in range(cycles):

                    repeated_logits = []
                    repeated_embeddings = []
                    for i in range(num_letter_level_segments):
                        curr_unet_inputs_embeds = repeated_curr_logits_with_letter_embeddings[:, i*self.rmt_config['unet_sub_model_input_size']:(i+1)*self.rmt_config['unet_sub_model_input_size'], :]
                        curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)
                        curr_unet_inputs_embeds = self.activation_fn(self.sub_model(curr_unet_inputs_embeds))
                        curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)

                        repeated_embeddings.append(curr_unet_inputs_embeds)
                        
                        curr_unet_inputs_embeds = self.fc(curr_unet_inputs_embeds)
                        repeated_logits.append(curr_unet_inputs_embeds)
    
                    collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                    collected_repeated_embeddings = torch.cat(repeated_embeddings, dim=1)

                    loss += loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                    
                    repeated_curr_logits_with_letter_embeddings += collected_repeated_embeddings
                        
                batched_losses.append(loss / cycles) # loss

                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            final_model_output['hidden_state'] = curr_logits[:, curr_repeater, :]
            final_model_output['embedding_repeater'] = curr_repeater.squeeze()
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)










class RMTEncoderForLetterLevelTokenClassificationLinearSegmentedRepeater(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        self.nucleotide_embedding = nn.Embedding(1000, 768)
        # self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(1536, 5)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
        if self.rmt_config['use_crf']:
            self.num_crf_classes = self.rmt_config['num_crf_classes']
            self.crf_model = CRF(self.num_crf_classes, batch_first=True)
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'])
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # print(repeated_curr_logits_with_letter_embeddings)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                # custom_pos_weight = np.ones(curr_letter_level_labels.shape)
                # for lp in range(custom_pos_weight.shape[1]-1):
                #     if np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = 100.0

                loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)

                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['unet_sub_model_input_size'])

                loss = 0

                cycles = 1
                for c in range(cycles):

                    repeated_logits = []
                    repeated_embeddings = []
                    for i in range(num_letter_level_segments):
                        curr_unet_inputs_embeds = repeated_curr_logits_with_letter_embeddings[:, i*self.rmt_config['unet_sub_model_input_size']:(i+1)*self.rmt_config['unet_sub_model_input_size'], :]

                        repeated_embeddings.append(curr_unet_inputs_embeds)
                        
                        curr_unet_inputs_embeds = self.fc(curr_unet_inputs_embeds)
                        repeated_logits.append(curr_unet_inputs_embeds)
    
                    collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                    collected_repeated_embeddings = torch.cat(repeated_embeddings, dim=1)

                    loss += loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                    
                    repeated_curr_logits_with_letter_embeddings += collected_repeated_embeddings
                        
                batched_losses.append(loss / cycles) # loss

                if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                    collected_repeated_logits = F.pad(collected_repeated_logits, (0, 0, 0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]))
                    if self.rmt_config['use_crf']:
                        crf_decoding = F.pad(crf_decoding, (0, letter_level_tokens.shape[1] - crf_decoding.shape[1]))
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                if self.rmt_config['use_crf']:
                    batched_crf_predictions.append(crf_decoding)
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        if self.rmt_config['use_crf']:
            final_model_output = TokenClassifierOutput(
                loss=torch.stack(batched_losses).mean(),
                logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            )
            final_model_output['decode'] = torch.cat(batched_crf_predictions, dim=0)
            return final_model_output
        else:
            # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            final_model_output = dict() # TokenClassifierOutput(
            #     loss=torch.stack(batched_losses).mean(),
            #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
            # )
            
            final_model_output['loss'] = torch.stack(batched_losses).mean()
            final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)
            
            return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)




















class CRFLayer(nn.Module):
    def __init__(self, num_tags: int, bos_tag_id: int):
        super(CRFLayer, self).__init__()
        self.num_tags = num_tags  
        self.bos_tag_id = bos_tag_id  

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))  

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor = None, initial_state: torch.Tensor = None):
       
        batch_size, seq_len, num_labels = emissions.size()
        assert num_labels == self.num_tags - 1, "Emissions should have 24 labels (excluding BOS)."

        bos_col = torch.full((batch_size, seq_len, 1), fill_value=-1e4, device=emissions.device)  

        emissions25 = torch.cat([emissions, bos_col], dim=2)  

        if initial_state is None:

            initial_state = emissions25.new_zeros((batch_size, self.num_tags))

        alpha = initial_state  

        for t in range(seq_len):
            emit_t = emissions25[:, t, :]  

            prev_alpha = alpha.unsqueeze(2)              
            trans = self.transitions.unsqueeze(0)        

            score = prev_alpha + trans  

            alpha = torch.logsumexp(score, dim=2)  

            alpha = alpha + emit_t  

        log_Z = torch.logsumexp(alpha, dim=1)  

        if tags is None:

            return log_Z

        gold_score = emissions25.new_zeros((batch_size,))  

        first_tags = tags[:, 0]  

        init_prev_tag = initial_state.argmax(dim=1)  

        trans_score = self.transitions[first_tags, init_prev_tag]   
        emit_score = emissions25[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze(1)  
        gold_score += trans_score + emit_score

        for t in range(1, seq_len):
            curr_tags = tags[:, t]       
            prev_tags = tags[:, t-1]     

            trans_score = self.transitions[curr_tags, prev_tags]   

            emit_score = emissions25[:, t].gather(1, curr_tags.unsqueeze(1)).squeeze(1)  
            gold_score += trans_score + emit_score

        nll = log_Z - gold_score  

        return nll.mean()

    def decode(self, emissions: torch.Tensor, initial_state: torch.Tensor):
       
        batch_size, seq_len, _ = emissions.size()

        bos_col = torch.full((batch_size, seq_len, 1), fill_value=-1e4, device=emissions.device)
        emissions25 = torch.cat([emissions, bos_col], dim=2)

        viterbi_vars = initial_state.clone()  

        backpointers = []

        for t in range(seq_len):
            emit_t = emissions25[:, t, :]  
            next_vars = []    
            bptr_t = []       

            scores = viterbi_vars.unsqueeze(1) + self.transitions.unsqueeze(0)  

            max_scores, best_prev_tags = scores.max(dim=2)  

            max_scores = max_scores + emit_t  
            backpointers.append(best_prev_tags)  
            viterbi_vars = max_scores  

        best_paths = []

        end_best_scores, end_best_tags = viterbi_vars.max(dim=1)  

        for b in range(batch_size):
            best_tag = int(end_best_tags[b].item())
            seq_tags = [best_tag]

            for ptr_t in reversed(backpointers):
                best_tag = int(ptr_t[b, best_tag].item())
                seq_tags.append(best_tag)
            seq_tags.reverse()

            seq_tags = seq_tags[1:]
            best_paths.append(seq_tags)
        return best_paths

# class CADUSEUS_for_token_classification_CRF(torch.nn.Module):
#     def __init__(self, caduseus_model):
#         super().__init__() 
#         self.caduseus_model = caduseus_model
#         self.fc = nn.Linear(256, 24)

#     def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
#         batch_size, total_seq_len = input_ids.shape[:2]

#         total_loss = 0.0
#         all_preds = [[] for _ in range(batch_size)]
#         num_chunks_total = 0

#         if attention_mask is not None:
#             actual_lengths = attention_mask.sum(dim=1).tolist()  
#         else:
#             actual_lengths = [total_seq_len] * batch_size

#         for b in range(batch_size):
#             seq_len = actual_lengths[b]

#             seq_input_ids = input_ids[b, :seq_len].unsqueeze(0)  
#             seq_mask = attention_mask[b, :seq_len].unsqueeze(0) if attention_mask is not None else None
#             seq_labels = labels[b, :seq_len].unsqueeze(0) if labels is not None else None

#             prev_tag = None  

#             for start in range(0, seq_len, self.max_chunk_len):
#                 end = min(start + self.max_chunk_len, seq_len)
#                 chunk_len = end - start

#                 chunk_input_ids = seq_input_ids[:, start:end]
#                 chunk_mask = seq_mask[:, start:end] if seq_mask is not None else None
#                 chunk_labels = seq_labels[:, start:end] if seq_labels is not None else None

#                 outputs = self.base_model(chunk_input_ids, attention_mask=chunk_mask, **kwargs)
#                 sequence_output = outputs[0]  

#                 chunk_logits = self.classifier(sequence_output)  

#                 if start == 0:

#                     init_dist = chunk_logits.new_zeros((1, 25))  

#                 else:

#                     init_dist = chunk_logits.new_full((1, 25), -1e4)
#                     if labels is not None:

#                         prev_tag_idx = int(prev_tag)  
#                     else:

#                         prev_tag_idx = int(prev_tag)
#                     init_dist[0, prev_tag_idx] = 0.0  

#                 if chunk_labels is not None:
#                     nll = self.crf(chunk_logits, tags=chunk_labels, initial_state=init_dist)
#                     total_loss += nll.item()  
#                     num_chunks_total += 1

#                 best_path = self.crf.decode(chunk_logits, initial_state=init_dist)[0]  

#                 all_preds[b].extend(best_path)

#                 prev_tag = best_path[-1]

#         avg_loss = None
#         if labels is not None:
#             avg_loss = total_loss / num_chunks_total

#         max_len = max(len(seq) for seq in all_preds)
#         output_logits = torch.zeros((batch_size, max_len, 24), device=input_ids.device)

#         for b in range(batch_size):
#             pred_seq = all_preds[b]
#             seq_length = len(pred_seq)
#             for t, tag in enumerate(pred_seq):
#                 if t < seq_length:

#                     output_logits[b, t, tag] = 10.0  

#         return TokenClassifierOutput(loss=avg_loss, logits=output_logits)







class CADUSEUS_for_token_classification_CRF(torch.nn.Module):
    def __init__(self, caduseus_model):
        super().__init__() 
        self.caduseus_model = caduseus_model
        self.fc = nn.Linear(256, 24)
        self.crf = CRF(24, batch_first=True)

    def forward(self, input_ids=None, letter_level_labels=None, letter_level_labels_mask=None):
        hidden_states = self.caduseus_model(input_ids).last_hidden_state
        bs = hidden_states.shape[0]
        batched_collected_preds, batched_losses = [], []
        for b in range(bs):  # aggregate in one batch
            curr_logits = hidden_states[b, letter_level_labels_mask[b], :].unsqueeze(0)
            curr_logits = self.fc(curr_logits)
            # Mask for CRF: shape (1, seq_len)
            # curr_mask = torch.ones(curr_logits.shape[:2], dtype=torch.bool, device=curr_logits.device)
            if letter_level_labels is not None:
                curr_labels = letter_level_labels[b, letter_level_labels_mask[b]].unsqueeze(0)
                print('AAAAAAAAAAA')
                loss = -self.crf(curr_logits.float(), curr_labels.long(), reduction='mean')
                print('BBBBBBBBBBB')
                batched_losses.append(loss)
            # Decoding
            print('CCCCCCCCCC')
            pred_labels = torch.tensor(self.crf.decode(curr_logits.float()), device=curr_logits.device)
            print('DDDDDDDDDD')
            # print('AAAAAAAAA', pred_labels.shape)
            # assert False
            pred_labels = F.pad(pred_labels, (letter_level_labels.shape[1] - pred_labels.shape[1] - 1, 1), value=-100)
            # print('BBBBBBBBB', pred_labels.shape)
            batched_collected_preds.append(pred_labels)
        final_model_output = dict()
        if letter_level_labels is not None and batched_losses:
            final_model_output['loss'] = torch.stack(batched_losses).mean()
        # Pad predictions to seq_len, fill with -100 for padding
        # maxlen = letter_level_labels.shape[1] if letter_level_labels is not None else max([p.size(0) for p in batched_collected_preds])
        # preds_padded = [F.pad(p, (0, maxlen - p.size(0)), value=-100) for p in batched_collected_preds]
        final_model_output['logits'] = torch.cat(batched_collected_preds, dim=0)
        # print(final_model_output['logits'].shape)
        return final_model_output





from torch_struct import LinearChainCRF
# import torch
# import torch.nn.functional as F
# from torch import nn

# class CADUSEUS_for_token_classification_CRF_fast(torch.nn.Module):
#     """
#     • takes `letter_level_labels_mask` to drop padding / system tokens
#     • expects `letter_level_labels` to be class indices in [0, 23]
#     • returns:
#         - loss (mean NLL) if labels are given
#         - logits = decoded label indices padded with –100
#     """

#     def __init__(self, caduseus_model):
#         super().__init__()
#         self.caduseus_model = caduseus_model           # any HF encoder
#         self.num_labels     = 24                      # K
#         self.fc             = nn.Linear(256, self.num_labels)
#         # transition scores  (to , from)
#         self.transitions    = nn.Parameter(torch.randn(self.num_labels,
#                                                        self.num_labels))

#     # ---------- helper ----------------------------------------------------
#     def _pairwise_potentials(self, emis: torch.Tensor) -> torch.Tensor:
#         """
#         emis  : (B , T , K)  – per-token scores from the linear head
#         return: (B , T-1 , K , K)  – log-potentials for LinearChainCRF
#         pot[b,t,i,j] = trans[i,j] + emis[b,t,i] + emis[b,t+1,j]
#         """
#         B, T, K = emis.size()
#         trans   = self.transitions                      # (K , K)
#         # expand dims so broadcast adds correctly
#         trans   = trans.unsqueeze(0).unsqueeze(0)       # (1 , 1 , K , K)
#         emis_i  = emis[:, :-1].unsqueeze(3)             # (B , T-1 , K , 1)
#         emis_j  = emis[:, 1: ].unsqueeze(2)             # (B , T-1 , 1 , K)
#         return trans + emis_i + emis_j                  # (B , T-1 , K , K)

#     # ---------- forward ---------------------------------------------------
#     def forward(
#         self,
#         input_ids               = None,    # (bs , seq_len)
#         letter_level_labels     = None,    # (bs , seq_len)  or  None
#         letter_level_labels_mask = None    # bool mask
#     ):
#         hidden = self.caduseus_model(input_ids).last_hidden_state  # (bs , seq_len , 256)
#         bs, full_len, _ = hidden.size()

#         losses, pred_rows = [], []

#         for b in range(bs):

#             # ---- select only “real” tokens for this sample ----------------
#             real_mask = letter_level_labels_mask[b]                 # (seq_len,)
#             h        = hidden[b, real_mask]                         # (T , 256)
#             if h.numel() == 0:                                      # empty after mask?
#                 # pad entirely with –100 and continue
#                 pred_rows.append(torch.full((1, full_len),
#                                   -100, device=hidden.device, dtype=torch.long))
#                 continue

#             h        = h.unsqueeze(0)                               # (1 , T , 256)
#             emis     = self.fc(h)                                   # (1 , T , K)
#             T        = emis.size(1)

#             # ---- build linear-chain potentials ---------------------------
#             if T == 1:
#                 # Torch-Struct needs length≥2; fall back to greedy
#                 best = emis.argmax(-1)                              # (1 , 1)
#                 pot  = emis.new_empty(1, 0, self.num_labels, self.num_labels)
#                 dist = LinearChainCRF(pot)                          # dummy CRF
#             else:
#                 pot  = self._pairwise_potentials(emis)              # (1 , T-1 , K , K)
#                 dist = LinearChainCRF(pot)

#             # ---- loss -----------------------------------------------------
#             if letter_level_labels is not None:
#                 gold = letter_level_labels[b, real_mask]            # (T,)
#                 # sanity-check: no label outside range
#                 if (gold.min() < 0) or (gold.max() >= self.num_labels):
#                     raise ValueError(
#                         f"Gold labels out of range 0-{self.num_labels-1}: {gold}")
#                 nll  = -(dist.log_prob(gold.unsqueeze(0))           # (1,)
#                          if T > 1 else
#                          emis.squeeze(0).log_softmax(-1)[torch.arange(T), gold].sum().neg())
#                 losses.append(nll)

#             # ---- decoding -------------------------------------------------
#             if T == 1:
#                 path = best.squeeze(0)                              # (1,)
#             else:
#                 path = dist.argmax.squeeze(0)                       # (T,)

#             # ---- pad back to `full_len` with –100 ------------------------
#             padded = torch.full((full_len,), -100,
#                                 dtype=torch.long, device=hidden.device)
#             padded[real_mask] = path
#             pred_rows.append(padded.unsqueeze(0))                   # (1 , full_len)

#         # ------------ aggregate batch --------------------------------------
#         output = {}
#         if losses:
#             output["loss"] = torch.stack(losses).mean()
#         output["logits"] = torch.cat(pred_rows, dim=0)              # (bs , seq_len)
#         return output



class CADUSEUS_for_token_classification_CRF_fast(torch.nn.Module):
    def __init__(self, caduseus_model):
        super().__init__() 
        self.caduseus_model = caduseus_model
        self.fcn = nn.Linear(512, 2)
        # self.A = nn.Parameter(torch.load('/home/jovyan/dnalm/downstream_tasks/annotation/transition_matrix_neurips_rebuttal_human.pt'), requires_grad=False)
        self.A = nn.Parameter(torch.tensor([[-7.0442418e-03, -4.9590716e+00], [-8.2220173e+00, -1.3983005e-03]]), requires_grad=False)
   
    def forward(self, input_ids=None, letter_level_labels=None, letter_level_labels_mask=None):

        hidden_states = self.caduseus_model(input_ids).last_hidden_state

        bs = hidden_states.shape[0]
        
        batched_collected_logits, batched_losses = [], []
        for b in range(bs): # aggregate in one batch
            repeater_kwargs = dict()
            
            curr_logits = hidden_states[b, letter_level_labels_mask[b], :].unsqueeze(0)
            curr_logits = self.fcn(curr_logits)

            # loss_fct = BCEWithLogitsLoss()

            if letter_level_labels is not None:

                device = curr_logits.device

                tags = letter_level_labels[b, letter_level_labels_mask[b]].unsqueeze(0).long()
                e = curr_logits

                B, T, K = e.shape

                # print(B, T, K)

                big_neg = -1e4
                first   = e.new_full((B, 1, K, K), big_neg)
                first[:, 0, :, 0] = e[:, 0]
        
                # edges 1…T-1
                rest = self.A.view(1, 1, K, K) + e[:, 1:].unsqueeze(3)  # (B,T-1,K,K)
                pot  = torch.cat([first, rest], 1)                      # (B,T,K,K)
        
                dist  = LinearChainCRF(pot)
                logZ  = dist.partition                                  # (B,)
        
                parts = torch.zeros_like(pot)
        
                # edge 0
                parts[torch.arange(B), 0, tags[:, 0], 0] = 1.
        
                # edges 1…T-1  (row = next , col = prev)
                time_idx   = torch.arange(1, T, device=device)          # (T-1,)
                batch_idx  = torch.arange(B, device=device).unsqueeze(1)# (B,1)
                parts[batch_idx, time_idx, tags[:, 1:], tags[:, :-1]] = 1.
        
                gold_score = (pot * parts).sum((1, 2, 3))               # (B,)
                nll = (logZ - gold_score).mean() #################################################################################### FIX
        
                edges = dist.argmax[0]                 # (T,K,K)
                rows  = edges.argmax(1)                # (T,K) row idx of 1 for each col
                pred  = torch.empty(T, dtype=torch.long, device=device)
                pred[0] = rows[0, 0]                   
                for t in range(1, T):                 
                    pred[t] = rows[t, pred[t-1]]
                        
                batched_losses.append(nll)

            collected_repeated_logits = F.pad(pred.unsqueeze(0), (letter_level_labels.shape[1] - pred.shape[0] - 1, 1), value=-100)

            batched_collected_logits.append(collected_repeated_logits)
        
        final_model_output = dict()

        if letter_level_labels is not None:
            final_model_output['loss'] = torch.stack(batched_losses).mean()
        final_model_output['logits'] = torch.cat(batched_collected_logits, dim=0)

        # print(final_model_output['loss'], final_model_output['logits'].shape)
        
        return final_model_output




class CADUSEUS_for_token_classification(torch.nn.Module):
    def __init__(self, caduseus_model):
        super().__init__() 
        self.caduseus_model = caduseus_model
        self.fc = nn.Linear(512, 9)
   
    def forward(self, input_ids=None, letter_level_labels=None, letter_level_labels_mask=None):

        hidden_states = self.caduseus_model(input_ids).last_hidden_state

        bs = hidden_states.shape[0]
        
        batched_collected_logits, batched_losses = [], []
        for b in range(bs): # aggregate in one batch
            repeater_kwargs = dict()
            
            curr_logits = hidden_states[b, letter_level_labels_mask[b], :].unsqueeze(0)
            curr_logits = self.fc(curr_logits)

            loss_fct = BCEWithLogitsLoss() # weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0], device=curr_logits.device)

            if letter_level_labels is not None:

                # print(curr_logits, letter_level_labels[b, letter_level_labels_mask[b], :].unsqueeze(0))
                loss = loss_fct(curr_logits, letter_level_labels[b, letter_level_labels_mask[b], :].unsqueeze(0))
                        
                batched_losses.append(loss)

            collected_repeated_logits = F.pad(curr_logits, (0, 0, letter_level_labels.shape[1] - curr_logits.shape[1] - 1, 1))

            batched_collected_logits.append(collected_repeated_logits)
        
        final_model_output = dict()

        if letter_level_labels is not None:
            final_model_output['loss'] = torch.stack(batched_losses).mean()
        final_model_output['logits'] = torch.cat(batched_collected_logits, dim=0)
        
        return final_model_output














class RMTEncoderForLetterLevelTokenClassificationUNETsegmentedRepeaterCRFfast(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        self.sub_model = UNET1DSegmentationHead(
                            embed_dim=1536,
                            num_classes=1536,
                            output_channels_list=[192, 384, 768],  # Example channel sizes as a list
                            num_conv_layers_per_block=2
                        )
        self.nucleotide_embedding = nn.Embedding(1000, 768)
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(1536, 24)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True

        self.A   = nn.Parameter(torch.randn(24, 24))
        
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'])
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # print(repeated_curr_logits_with_letter_embeddings)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                # custom_pos_weight = np.ones(curr_letter_level_labels.shape)
                # for lp in range(custom_pos_weight.shape[1]-1):
                #     if np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = 100.0

                # loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)

                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['unet_sub_model_input_size'])

                loss = 0

                cycles = 1
                for c in range(cycles):

                    repeated_logits = []
                    repeated_embeddings = []
                    for i in range(num_letter_level_segments):
                        curr_unet_inputs_embeds = repeated_curr_logits_with_letter_embeddings[:, i*self.rmt_config['unet_sub_model_input_size']:(i+1)*self.rmt_config['unet_sub_model_input_size'], :]
                        curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)
                        curr_unet_inputs_embeds = self.activation_fn(self.sub_model(curr_unet_inputs_embeds))
                        curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)

                        repeated_embeddings.append(curr_unet_inputs_embeds)
                        
                        curr_unet_inputs_embeds = self.fc(curr_unet_inputs_embeds)
                        repeated_logits.append(curr_unet_inputs_embeds)
    
                    collected_repeated_logits = torch.cat(repeated_logits, dim=1)
                    collected_repeated_embeddings = torch.cat(repeated_embeddings, dim=1)

                    device = collected_repeated_logits.device

                    tags = curr_letter_level_labels
                    e = collected_repeated_logits

                    B, T, K = e.shape

                    # print(B, T, K)
    
                    big_neg = -1e4
                    first   = e.new_full((B, 1, K, K), big_neg)
                    first[:, 0, :, 0] = e[:, 0]
            
                    # edges 1…T-1
                    rest = self.A.view(1, 1, K, K) + e[:, 1:].unsqueeze(3)  # (B,T-1,K,K)
                    pot  = torch.cat([first, rest], 1)                      # (B,T,K,K)
            
                    dist  = LinearChainCRF(pot)
                    logZ  = dist.partition                                  # (B,)
            
                    parts = torch.zeros_like(pot)
            
                    # edge 0
                    parts[torch.arange(B), 0, tags[:, 0], 0] = 1.
            
                    # edges 1…T-1  (row = next , col = prev)
                    time_idx   = torch.arange(1, T, device=device)          # (T-1,)
                    batch_idx  = torch.arange(B, device=device).unsqueeze(1)# (B,1)
                    parts[batch_idx, time_idx, tags[:, 1:], tags[:, :-1]] = 1.
            
                    gold_score = (pot * parts).sum((1, 2, 3))               # (B,)
                    nll = (logZ - gold_score).mean() #################################################################################### FIX
            
                    edges = dist.argmax[0]                 # (T,K,K)
                    rows  = edges.argmax(1)                # (T,K) row idx of 1 for each col
                    pred  = torch.empty(T, dtype=torch.long, device=device)
                    pred[0] = rows[0, 0]                   
                    for t in range(1, T):                 
                        pred[t] = rows[t, pred[t-1]]

                    # loss += loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                    
                    repeated_curr_logits_with_letter_embeddings += collected_repeated_embeddings
                        
                batched_losses.append(nll / cycles) # loss

                # if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                collected_repeated_logits = F.pad(pred.unsqueeze(0), (0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]), value=-100)
                    
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        final_model_output = dict() # TokenClassifierOutput(
        #     loss=torch.stack(batched_losses).mean(),
        #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
        # )
        
        final_model_output['loss'] = torch.stack(batched_losses).mean()
        final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)

        # print(collected_repeated_logits)
        
        return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)


















class RMTEncoderForLetterLevelTokenClassificationLinearSegmentedRepeaterCRFfast(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        
        self.nucleotide_embedding = nn.Embedding(1000, 768)
       
        self.fc = nn.Linear(1536, 24)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True

        self.A   = nn.Parameter(torch.randn(24, 24))
        
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'])
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # print(repeated_curr_logits_with_letter_embeddings)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                # custom_pos_weight = np.ones(curr_letter_level_labels.shape)
                # for lp in range(custom_pos_weight.shape[1]-1):
                #     if np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = 100.0

                # loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)

                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['unet_sub_model_input_size'])

                loss = 0

                cycles = 1
                for c in range(cycles):

                    repeated_logits = []
                    for i in range(num_letter_level_segments):
                        curr_unet_inputs_embeds = repeated_curr_logits_with_letter_embeddings[:, i*self.rmt_config['unet_sub_model_input_size']:(i+1)*self.rmt_config['unet_sub_model_input_size'], :]
                        
                        curr_unet_inputs_embeds = self.fc(curr_unet_inputs_embeds)
                        repeated_logits.append(curr_unet_inputs_embeds)
    
                    collected_repeated_logits = torch.cat(repeated_logits, dim=1)

                    device = collected_repeated_logits.device

                    tags = curr_letter_level_labels
                    e = collected_repeated_logits

                    B, T, K = e.shape

                    # print(B, T, K)
    
                    big_neg = -1e4
                    first   = e.new_full((B, 1, K, K), big_neg)
                    first[:, 0, :, 0] = e[:, 0]
            
                    # edges 1…T-1
                    rest = self.A.view(1, 1, K, K) + e[:, 1:].unsqueeze(3)  # (B,T-1,K,K)
                    pot  = torch.cat([first, rest], 1)                      # (B,T,K,K)
            
                    dist  = LinearChainCRF(pot)
                    logZ  = dist.partition                                  # (B,)
            
                    parts = torch.zeros_like(pot)
            
                    # edge 0
                    parts[torch.arange(B), 0, tags[:, 0], 0] = 1.
            
                    # edges 1…T-1  (row = next , col = prev)
                    time_idx   = torch.arange(1, T, device=device)          # (T-1,)
                    batch_idx  = torch.arange(B, device=device).unsqueeze(1)# (B,1)
                    parts[batch_idx, time_idx, tags[:, 1:], tags[:, :-1]] = 1.
            
                    gold_score = (pot * parts).sum((1, 2, 3))               # (B,)
                    nll = (logZ - gold_score).mean() #################################################################################### FIX
            
                    edges = dist.argmax[0]                 # (T,K,K)
                    rows  = edges.argmax(1)                # (T,K) row idx of 1 for each col
                    pred  = torch.empty(T, dtype=torch.long, device=device)
                    pred[0] = rows[0, 0]                   
                    for t in range(1, T):                 
                        pred[t] = rows[t, pred[t-1]]

                    # loss += loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        
                batched_losses.append(nll / cycles) # loss

                # if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                collected_repeated_logits = F.pad(pred.unsqueeze(0), (0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]), value=-100)
                    
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        final_model_output = dict() # TokenClassifierOutput(
        #     loss=torch.stack(batched_losses).mean(),
        #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
        # )
        
        final_model_output['loss'] = torch.stack(batched_losses).mean()
        final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)

        # print(collected_repeated_logits)
        
        return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)














class RMTEncoderForLetterLevelTokenClassificationLinearSegmentedRepeaterCRFfastFixedTransition(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        
        self.nucleotide_embedding = nn.Embedding(1000, 768)
       
        self.fc = nn.Linear(1536, 24)
        # self.middle_norm = nn.LayerNorm(1024)
        # self.middle_dropout = nn.Dropout(p=0.992)
        
        self.set_params(**rmt_kwargs)
        
        # self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True

        self.A = nn.Parameter(torch.load('/home/jovyan/dnalm/downstream_tasks/annotation/transition_matrix_v2.pt'), requires_grad=True)
        
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        pos_weight = pos_weight[0, 0, :][None, None, :]
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
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'])
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None:
            batched_collected_repeated_logits, batched_losses, edge_losses, no_edge_losses, batched_crf_predictions = [], [], [], [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                curr_repeater = embedding_repeater[b][lllm]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                curr_letter_level_embedding = self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                # repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding + curr_logits[:, curr_repeater, :] # combine this with post merging?
                repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # # print('222222222222222222222')
                # repeated_attention_mask = torch.ones((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # # print('3333333333333333333333333333')
                # repeated_token_types_ids = torch.zeros((1, repeated_curr_logits_with_letter_embeddings.shape[1])).to(curr_logits.device)
                # print(repeated_curr_logits_with_letter_embeddings)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                # custom_pos_weight = np.ones(curr_letter_level_labels.shape)
                # for lp in range(custom_pos_weight.shape[1]-1):
                #     if np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 0, 1, 0, 0])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 1, 0, 0, 1])) or np.all(curr_letter_level_labels[0, lp, :] == np.array([0, 1, 0, 0, 1])) and np.all(curr_letter_level_labels[0, lp+1, :] == np.array([0, 0, 1, 0, 0])):
                #         custom_pos_weight[0, np.clip(lp-4, 0, None):lp+4, :] = 100.0

                # loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)

                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['unet_sub_model_input_size'])

                loss = 0

                cycles = 1
                for c in range(cycles):

                    repeated_logits = []
                    for i in range(num_letter_level_segments):
                        curr_unet_inputs_embeds = repeated_curr_logits_with_letter_embeddings[:, i*self.rmt_config['unet_sub_model_input_size']:(i+1)*self.rmt_config['unet_sub_model_input_size'], :]
                        
                        curr_unet_inputs_embeds = self.fc(curr_unet_inputs_embeds)
                        repeated_logits.append(curr_unet_inputs_embeds)
    
                    collected_repeated_logits = torch.cat(repeated_logits, dim=1)

                    device = collected_repeated_logits.device

                    tags = curr_letter_level_labels
                    e = collected_repeated_logits

                    B, T, K = e.shape

                    # print(B, T, K)
    
                    big_neg = -1e4
                    first   = e.new_full((B, 1, K, K), big_neg)
                    first[:, 0, :, 0] = e[:, 0]
            
                    # edges 1…T-1
                    rest = self.A.view(1, 1, K, K) + e[:, 1:].unsqueeze(3)  # (B,T-1,K,K)
                    pot  = torch.cat([first, rest], 1)                      # (B,T,K,K)
            
                    dist  = LinearChainCRF(pot)
                    logZ  = dist.partition                                  # (B,)
            
                    parts = torch.zeros_like(pot)
            
                    # edge 0
                    parts[torch.arange(B), 0, tags[:, 0], 0] = 1.
            
                    # edges 1…T-1  (row = next , col = prev)
                    time_idx   = torch.arange(1, T, device=device)          # (T-1,)
                    batch_idx  = torch.arange(B, device=device).unsqueeze(1)# (B,1)
                    parts[batch_idx, time_idx, tags[:, 1:], tags[:, :-1]] = 1.
            
                    gold_score = (pot * parts).sum((1, 2, 3))               # (B,)
                    nll = (logZ - gold_score).mean() #################################################################################### FIX
            
                    edges = dist.argmax[0]                 # (T,K,K)
                    rows  = edges.argmax(1)                # (T,K) row idx of 1 for each col
                    pred  = torch.empty(T, dtype=torch.long, device=device)
                    pred[0] = rows[0, 0]                   
                    for t in range(1, T):                 
                        pred[t] = rows[t, pred[t-1]]

                    # loss += loss_fct(collected_repeated_logits.float(), curr_letter_level_labels.float())
                        
                batched_losses.append(nll / cycles) # loss

                # if collected_repeated_logits.shape[1] != letter_level_tokens.shape[1]:
                collected_repeated_logits = F.pad(pred.unsqueeze(0), (0, letter_level_tokens.shape[1] - collected_repeated_logits.shape[1]), value=-100)
                    
                # print(crf_decoding.shape, collected_repeated_logits.shape)
                batched_collected_repeated_logits.append(collected_repeated_logits)
                
        else:
            raise Exception('No embedding_repeater!')
            
        # print(torch.cat(batched_collected_repeated_logits, dim=0)) 
          
        # print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        final_model_output = dict() # TokenClassifierOutput(
        #     loss=torch.stack(batched_losses).mean(),
        #     logits=torch.cat(batched_collected_repeated_logits, dim=0) # CHANGE!
        # )
        
        final_model_output['loss'] = torch.stack(batched_losses).mean()
        final_model_output['logits'] = torch.cat(batched_collected_repeated_logits, dim=0)

        # print(collected_repeated_logits)
        
        return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)















class CADUSEUS_for_token_classification_UNET(torch.nn.Module):
    def __init__(self, caduseus_model):
        super().__init__() 
        self.caduseus_model = caduseus_model
        self.sub_model = UNET1DSegmentationHead(
                            embed_dim=256,
                            num_classes=5,
                            output_channels_list=(32, 64, 128),
                            num_conv_layers_per_block=2
                        )
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(5*2, 5)
   
    def forward(self, input_ids=None, letter_level_labels=None, letter_level_labels_mask=None):

        hidden_states = self.caduseus_model(input_ids).last_hidden_state

        bs = hidden_states.shape[0]
        
        batched_collected_logits, batched_losses = [], []
        for b in range(bs): # aggregate in one batch
            repeater_kwargs = dict()

            if letter_level_labels_mask is not None:
                curr_logits = hidden_states[b, letter_level_labels_mask[b], :].unsqueeze(0)
            else:
                curr_logits = hidden_states[b, :, :].unsqueeze(0)

            segment_size = 8192

            num_letter_level_segments = math.ceil(curr_logits.shape[1] / segment_size)

            repeated_logits = []
            for i in range(num_letter_level_segments):
                curr_unet_inputs_embeds = curr_logits[:, i*segment_size:(i+1)*segment_size, :]
                curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)
                curr_unet_inputs_embeds = self.activation_fn(self.sub_model(curr_unet_inputs_embeds))
                curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)

                repeated_logits.append(curr_unet_inputs_embeds)

            curr_logits = torch.cat(repeated_logits, dim=1)
            
            curr_logits = self.fc(curr_logits)

            if letter_level_labels is not None:

                loss_fct = BCEWithLogitsLoss()

                # print(curr_logits, letter_level_labels[b, letter_level_labels_mask[b], :].unsqueeze(0))
                loss = loss_fct(curr_logits, letter_level_labels[b, letter_level_labels_mask[b], :].unsqueeze(0))
                        
                batched_losses.append(loss)

            if letter_level_labels is not None:
                collected_repeated_logits = F.pad(curr_logits, (0, 0, 0, letter_level_labels.shape[1] - curr_logits.shape[1]))
            else:
                collected_repeated_logits = curr_logits

            batched_collected_logits.append(collected_repeated_logits)
        
        final_model_output = dict()

        if letter_level_labels is not None:
            final_model_output['loss'] = torch.stack(batched_losses).mean()
        final_model_output['logits'] = torch.cat(batched_collected_logits, dim=0)
        
        return final_model_output

  









class RMTEncoderDecoderUNET(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, decoder_model, **rmt_kwargs):
        super().__init__()
    
        # Encoder (base_model) remains as before
        self.model = base_model
        
        # New: decoder in BERT-decoder mode (or any other decoder)
        self.decoder = decoder_model

        self.sub_model = UNET1DSegmentationHead(
                            embed_dim=1024,
                            num_classes=1024,
                            output_channels_list=[256, 512, 1024],  # Example channel sizes as a list
                            num_conv_layers_per_block=2
                        )
        self.activation_fn = nn.SiLU()

        self.nucleotide_embedding = nn.Embedding(100, 1024)
        
        # New: label embedding, vocab size=26, hidden size=1024
        self.label_embedding = nn.Embedding(26, 1024)
        
        self.set_params(**rmt_kwargs)
        
        self.rmt_config['sum_loss'] = True
        self.rmt_config['unet_sub_model_input_size'] = 8192
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        # print(input_ids, input_ids.shape)

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :]
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            # print(f'ITERATION {seg_num, segment_input_ids}')

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
            # if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                # pos_weight = pos_weight[0, 0, :][None, None, :]
                # segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                # seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.model(**seg_kwargs)
            # assert False
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        # assert False
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None: # CHECK HOW LABELS MASK WORKS IN RMT WITH BATCH SIZE > 1!!!!!!!!!!!!!!!!!!!!!!!
            batched_collected_repeated_logits, batched_losses = [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                lllm[0] = True
                # print(letter_level_labels, lllm, letter_level_labels.shape, lllm.shape)
                # assert False
                # print(letter_level_labels.shape)
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                # print(curr_letter_level_labels)
                assert 24 in curr_letter_level_labels
                lllm[0] = False
                # print(curr_letter_level_labels, curr_letter_level_labels.shape)
                # assert False
                curr_repeater = embedding_repeater[b][lllm[1:]]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                # print(letter_level_tokens)
                curr_letter_level_embedding = F.pad(self.nucleotide_embedding(letter_level_tokens[b, lllm[1:]].unsqueeze(0)), (0, 0, 1, 0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # assert False
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)
                # print('input_ids', torch.sum(input_ids[b] != 3))

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding[:, 1:, :] + curr_logits[:, curr_repeater, :] # combine this with post merging?
                # repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                num_letter_level_segments = math.ceil(repeated_curr_logits_with_letter_embeddings.shape[1] / self.rmt_config['unet_sub_model_input_size'])

                repeated_logits = []
                for i in range(num_letter_level_segments):
                    curr_unet_inputs_embeds = repeated_curr_logits_with_letter_embeddings[:, i*self.rmt_config['unet_sub_model_input_size']:(i+1)*self.rmt_config['unet_sub_model_input_size'], :]
                    curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)
                    curr_unet_inputs_embeds = self.activation_fn(self.sub_model(curr_unet_inputs_embeds))
                    curr_unet_inputs_embeds = curr_unet_inputs_embeds.transpose(1, 2)

                    repeated_logits.append(curr_unet_inputs_embeds)

                repeated_curr_logits_with_letter_embeddings = torch.cat(repeated_logits, dim=1)

                
                decoder_chunk_size = self.rmt_config["decoder_chunk_size"]  # e.g. 512
                look_back_size = self.rmt_config["decoder_look_back_size"]  # e.g. 512
                assert decoder_chunk_size > look_back_size
                enc_len = repeated_curr_logits_with_letter_embeddings.shape[1]
                w = torch.tensor([11531128.0, 11549727.0, 11531095.0, 1.0, 488589751.0, 265281019.0, 463184462.0, 65733.0, 43130.0, 91413.0, 65774.0, 43172.0, 91380.0, 18329.0, 18394.0, 27016189.0, 27147388.0, 324472532.0, 245528623.0, 38243.0, 38121.0, 23780.0, 23839.0, 552140823.0, 1.0, 1.0], device=repeated_curr_logits_with_letter_embeddings.device)
                loss_fct = nn.CrossEntropyLoss(weight=torch.sum(w) / w, ignore_index=-100)  # typical for seq2seq
                # loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # typical for seq2seq
                
                
                # We'll chunk the encoder embeddings and the decoder input tokens in steps of decoder_chunk_size
                # num_chunks = math.ceil(enc_len / (decoder_chunk_size - look_back_size))
                
                all_decoder_logits = []
                all_losses = []

                chunk_idx = 0
                
                # for chunk_idx in range(num_chunks):
                while True:
                    # print(enc_len, chunk_idx)
                    start_enc = chunk_idx * (decoder_chunk_size - look_back_size)
                    end_enc = min(start_enc + decoder_chunk_size, enc_len)

                    assert end_enc - start_enc <= decoder_chunk_size
                
                    start_dec = chunk_idx * (decoder_chunk_size - look_back_size)
                    end_dec = min(start_dec + decoder_chunk_size, enc_len + 1)

                    assert end_dec - start_dec <= decoder_chunk_size

                    # print(start_enc, end_enc, repeated_curr_logits_with_letter_embeddings.shape, curr_letter_level_labels.shape)
                    
                    # Slice the encoder output for this chunk
                    chunk_encoder_embs = repeated_curr_logits_with_letter_embeddings[:, start_enc:end_enc, :]  # shape: (1, chunk_size_enc, 1024)
                
                    # Slice the decoder tokens for this chunk
                    chunk_decoder_input_ids = curr_letter_level_labels[:, start_dec:end_dec]  # shape: (1, chunk_size_dec)
                    # print(curr_letter_level_labels[:, start_dec:end_dec].shape, curr_letter_level_labels[:, start_dec:end_dec])
                    if chunk_idx == 0:
                        # print('chunk_decoder_input_ids', chunk_decoder_input_ids)
                        assert chunk_decoder_input_ids[0, 0].item() == torch.tensor(24, device=chunk_encoder_embs.device)
                
                    # Embed the decoder input tokens
                    # print(curr_letter_level_labels)
                    # assert False
                    chunk_decoder_input_embeds = self.label_embedding(chunk_decoder_input_ids)  # (1, chunk_size_dec, 1024)
                    # chunk_decoder_input_embeds += curr_letter_level_embedding[:, start_dec:end_dec, :]
                
                    # Prepare attention masks
                    # (Assuming full attention on the chunked encoder part, and no future masking for chunked decoder in teacher forcing)
                    encoder_attention_mask = torch.ones(chunk_encoder_embs.size()[:-1], device=chunk_encoder_embs.device)
                    decoder_attention_mask = (chunk_decoder_input_ids != 25).long()  # if 25 is your pad

                    # print(chunk_decoder_input_embeds.shape, chunk_encoder_embs.shape, encoder_attention_mask.shape, decoder_attention_mask.shape)
                    # print(chunk_decoder_input_embeds, chunk_encoder_embs, encoder_attention_mask, decoder_attention_mask)
                    # assert False
                
                    # Forward through the decoder (in cross-attn mode)
                    decoder_outputs = self.decoder(
                        inputs_embeds=chunk_decoder_input_embeds,
                        encoder_hidden_states=chunk_encoder_embs,
                        encoder_attention_mask=encoder_attention_mask,
                        attention_mask=decoder_attention_mask
                    )
                    # Suppose decoder_outputs.logits -> shape: (1, chunk_size_dec, vocab_size)
                    chunk_logits = decoder_outputs.logits

                    ############################################################################# should we consider recalculation on the edges between segemnts?
                
                    # Teacher forcing: shift by 1
                    # If chunk_size_dec=1 in the last chunk, skipping the loss is safe 
                    # because there's no "next token" to predict inside that chunk
                    if chunk_logits.size(1) > 1:
                        if chunk_idx == 0:
                            # we remove the last logit & first label
                            logits_for_loss = chunk_logits[:, :-1, :]  # shape: (1, chunk_size_dec-1, vocab_size)
                            labels_for_loss = chunk_decoder_input_ids[:, 1:]  # shape: (1, chunk_size_dec-1)
                            # compute cross-entropy
                            loss_chunk = loss_fct(
                                logits_for_loss.reshape(-1, logits_for_loss.size(-1)), 
                                labels_for_loss.reshape(-1)
                            )
                            all_losses.append(loss_chunk)
                        else:
                            assert chunk_logits.size(1) - look_back_size >= 0
                            if chunk_logits.size(1) - look_back_size > 1:
                                # we remove the last logit & first label
                                logits_for_loss = chunk_logits[:, look_back_size:-1, :]  # shape: (1, chunk_size_dec-1, vocab_size)
                                labels_for_loss = chunk_decoder_input_ids[:, look_back_size+1:]  # shape: (1, chunk_size_dec-1)
                                # compute cross-entropy
                                loss_chunk = loss_fct(
                                    logits_for_loss.reshape(-1, logits_for_loss.size(-1)), 
                                    labels_for_loss.reshape(-1)
                                )
                                all_losses.append(loss_chunk)
                            else:
                                loss_chunk = torch.tensor(0.0, device=chunk_encoder_embs.device)
                    else:
                        # no next-token to predict if chunk size is 1
                        loss_chunk = torch.tensor(0.0, device=chunk_encoder_embs.device)

                    # print('GOGOOGGOGOG')
                    if chunk_idx == 0:
                        all_decoder_logits.append(chunk_logits)
                    else:
                        all_decoder_logits.append(chunk_logits[:, look_back_size:, :])

                    if chunk_idx == 0 and chunk_logits.size(1) < decoder_chunk_size:
                        break
                    elif chunk_logits.size(1) == decoder_chunk_size and end_enc != enc_len:
                        chunk_idx += 1
                    else:
                        break

                
                # Combine all chunk-level losses
                if all_losses:
                    final_loss = torch.stack(all_losses).mean()
                else:
                    final_loss = torch.tensor(0.0, device=repeated_curr_logits_with_letter_embeddings.device)
                
                # Concatenate chunked decoder logits for reporting
                final_logits = torch.cat(all_decoder_logits, dim=1)  # shape (1, total_decoder_tokens, vocab_size)

                # print(final_logits.shape, curr_letter_level_labels.shape, letter_level_labels.shape)
                # print('OLOLOLOOLLLOLOLO')
                final_logits = F.pad(final_logits, pad=(0, 0, 1, letter_level_labels.shape[1] - final_logits.shape[1] - 1))
                # print(final_logits.shape, curr_letter_level_labels.shape, letter_level_labels.shape, final_logits)
                # assert False
                batched_collected_repeated_logits.append(final_logits)
                batched_losses.append(final_loss)
        
        final_model_output = {
            "loss": torch.stack(batched_losses).mean(),
            "logits": torch.cat(batched_collected_repeated_logits, dim=0)
        }
        return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)














class RMTEncoderDecoderFeatured(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, decoder_model, **rmt_kwargs):
        super().__init__()
    
        # Encoder (base_model) remains as before
        self.model = base_model
        
        # New: decoder in BERT-decoder mode (or any other decoder)
        self.decoder = decoder_model

        self.nucleotide_embedding = nn.Embedding(100, 1024)
        
        # New: label embedding, vocab size=26, hidden size=1024
        self.label_embedding = nn.Embedding(26, 1024)
        
        self.set_params(**rmt_kwargs)

        self.wpe = nn.Embedding(self.rmt_config['decoder_chunk_size'] * 2, 1024)
        # self.wpe = nn.Embedding(64_000, 1024)
        
        self.rmt_config['sum_loss'] = True
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        # print(input_ids, input_ids.shape)

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :]
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            # print(f'ITERATION {seg_num, segment_input_ids}')

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
            # if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                # pos_weight = pos_weight[0, 0, :][None, None, :]
                # segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                # seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.model(**seg_kwargs)
            # assert False
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        # assert False
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None: # CHECK HOW LABELS MASK WORKS IN RMT WITH BATCH SIZE > 1!!!!!!!!!!!!!!!!!!!!!!!
            batched_collected_repeated_logits, batched_losses = [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
                # print(curr_logits.dtype)
                # assert False
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                lllm[0] = True
                # print(letter_level_labels, lllm, letter_level_labels.shape, lllm.shape)
                # assert False
                # print(letter_level_labels.shape)
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                # print(curr_letter_level_labels)
                assert 24 in curr_letter_level_labels
                lllm[0] = False
                # print(curr_letter_level_labels, curr_letter_level_labels.shape)
                # assert False
                curr_repeater = embedding_repeater[b][lllm[1:]]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                # print(letter_level_tokens)
                curr_letter_level_embedding = F.pad(self.nucleotide_embedding(letter_level_tokens[b, lllm[1:]].unsqueeze(0)), (0, 0, 1, 0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # assert False
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)
                # print('input_ids', torch.sum(input_ids[b] != 3))

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding[:, 1:, :] + curr_logits[:, curr_repeater, :] # combine this with post merging?
                # repeated_curr_logits_with_letter_embeddings += self.wpe(torch.arange(0, repeated_curr_logits_with_letter_embeddings.shape[1], dtype=torch.long, device=repeated_curr_logits_with_letter_embeddings.device).unsqueeze(0))
                # repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                
                decoder_chunk_size = self.rmt_config["decoder_chunk_size"]  # e.g. 512
                look_back_size = self.rmt_config["decoder_look_back_size"]  # e.g. 512
                assert decoder_chunk_size > look_back_size
                enc_len = repeated_curr_logits_with_letter_embeddings.shape[1]
                # w = torch.tensor([11531128.0, 11549727.0, 11531095.0, 1.0, 488589751.0, 265281019.0, 463184462.0, 65733.0, 43130.0, 91413.0, 65774.0, 43172.0, 91380.0, 18329.0, 18394.0, 27016189.0, 27147388.0, 324472532.0, 245528623.0, 38243.0, 38121.0, 23780.0, 23839.0, 552140823.0, 1.0, 1.0], device=repeated_curr_logits_with_letter_embeddings.device)
                # loss_fct = nn.CrossEntropyLoss(weight=torch.sum(w) / w, ignore_index=-100)  # typical for seq2seq
                # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 10000.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0, 1.0, 10000.0, 10000.0], device=repeated_curr_logits_with_letter_embeddings.device), ignore_index=-100)  # typical for seq2seq
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # typical for seq2seq
                
                
                # We'll chunk the encoder embeddings and the decoder input tokens in steps of decoder_chunk_size
                # num_chunks = math.ceil(enc_len / (decoder_chunk_size - look_back_size))
                
                all_decoder_logits = []
                all_losses = []

                chunk_idx = 0
                
                # for chunk_idx in range(num_chunks):
                while True:
                    # print(enc_len, chunk_idx)
                    start_enc = chunk_idx * (decoder_chunk_size - look_back_size)
                    end_enc = min(start_enc + decoder_chunk_size * 2, enc_len)

                    assert end_enc - start_enc <= decoder_chunk_size * 2
                
                    start_dec = chunk_idx * (decoder_chunk_size - look_back_size)
                    end_dec = min(start_dec + decoder_chunk_size, enc_len + 1)

                    assert end_dec - start_dec <= decoder_chunk_size

                    # print(start_enc, end_enc, repeated_curr_logits_with_letter_embeddings.shape, curr_letter_level_labels.shape)
                    
                    # Slice the encoder output for this chunk
                    chunk_encoder_embs = repeated_curr_logits_with_letter_embeddings[:, start_enc:end_enc, :]  # shape: (1, chunk_size_enc, 1024)
                    # print(chunk_encoder_embs, chunk_encoder_embs.shape)
                    # assert False
                    # print(chunk_encoder_embs.dtype) ########################################################### during training with mixed precision, dtype is not bf16
                    # print(self.wpe(torch.arange(0, chunk_encoder_embs.shape[1], dtype=torch.long, device=chunk_encoder_embs.device).unsqueeze(0)).shape, chunk_encoder_embs.dtype)
                    # assert False
                    chunk_encoder_embs += self.wpe(torch.arange(0, chunk_encoder_embs.shape[1], dtype=torch.long, device=chunk_encoder_embs.device).unsqueeze(0))

                
                    # Slice the decoder tokens for this chunk
                    chunk_decoder_input_ids = curr_letter_level_labels[:, start_dec:end_dec]  # shape: (1, chunk_size_dec)
                    # print(curr_letter_level_labels[:, start_dec:end_dec].shape, curr_letter_level_labels[:, start_dec:end_dec])
                    if chunk_idx == 0:
                        # print('chunk_decoder_input_ids', chunk_decoder_input_ids)
                        assert chunk_decoder_input_ids[0, 0].item() == torch.tensor(24, device=chunk_encoder_embs.device)
                
                    # Embed the decoder input tokens
                    # print(curr_letter_level_labels)
                    # assert False
                    chunk_decoder_input_embeds = self.label_embedding(chunk_decoder_input_ids)  # (1, chunk_size_dec, 1024)
                    # chunk_decoder_input_embeds += curr_letter_level_embedding[:, start_dec:end_dec, :]
                    chunk_decoder_input_embeds += F.pad(repeated_curr_logits_with_letter_embeddings, (0, 0, 1, 0))[:, start_dec:end_dec, :]
                
                    # Prepare attention masks
                    # (Assuming full attention on the chunked encoder part, and no future masking for chunked decoder in teacher forcing)
                    encoder_attention_mask = torch.ones(chunk_encoder_embs.size()[:-1], device=chunk_encoder_embs.device)
                    decoder_attention_mask = (chunk_decoder_input_ids != 25).long()  # if 25 is your pad

                    # print(chunk_decoder_input_embeds.shape, chunk_encoder_embs.shape, encoder_attention_mask.shape, decoder_attention_mask.shape)
                    # print(chunk_decoder_input_embeds, chunk_encoder_embs, encoder_attention_mask, decoder_attention_mask)
                    # assert False
                
                    # Forward through the decoder (in cross-attn mode)
                    decoder_outputs = self.decoder(
                        inputs_embeds=chunk_decoder_input_embeds,
                        encoder_hidden_states=chunk_encoder_embs,
                        encoder_attention_mask=encoder_attention_mask,
                        attention_mask=decoder_attention_mask
                    )
                    # Suppose decoder_outputs.logits -> shape: (1, chunk_size_dec, vocab_size)
                    chunk_logits = decoder_outputs.logits

                    ############################################################################# should we consider recalculation on the edges between segemnts?
                
                    # Teacher forcing: shift by 1
                    # If chunk_size_dec=1 in the last chunk, skipping the loss is safe 
                    # because there's no "next token" to predict inside that chunk
                    if chunk_logits.size(1) > 1:
                        if chunk_idx == 0:
                            # we remove the last logit & first label
                            logits_for_loss = chunk_logits[:, :-1, :]  # shape: (1, chunk_size_dec-1, vocab_size)
                            labels_for_loss = chunk_decoder_input_ids[:, 1:]  # shape: (1, chunk_size_dec-1)
                            # compute cross-entropy
                            loss_chunk = loss_fct(
                                logits_for_loss.reshape(-1, logits_for_loss.size(-1)), 
                                labels_for_loss.reshape(-1)
                            )
                            all_losses.append(loss_chunk)
                        else:
                            assert chunk_logits.size(1) - look_back_size >= 0
                            if chunk_logits.size(1) - look_back_size > 1:
                                # we remove the last logit & first label
                                logits_for_loss = chunk_logits[:, look_back_size:-1, :]  # shape: (1, chunk_size_dec-1, vocab_size)
                                labels_for_loss = chunk_decoder_input_ids[:, look_back_size+1:]  # shape: (1, chunk_size_dec-1)
                                # compute cross-entropy
                                loss_chunk = loss_fct(
                                    logits_for_loss.reshape(-1, logits_for_loss.size(-1)), 
                                    labels_for_loss.reshape(-1)
                                )
                                all_losses.append(loss_chunk)
                            else:
                                loss_chunk = torch.tensor(0.0, device=chunk_encoder_embs.device)
                    else:
                        # no next-token to predict if chunk size is 1
                        loss_chunk = torch.tensor(0.0, device=chunk_encoder_embs.device)

                    # print('GOGOOGGOGOG')
                    if chunk_idx == 0:
                        all_decoder_logits.append(chunk_logits)
                    else:
                        all_decoder_logits.append(chunk_logits[:, look_back_size:, :])

                    if chunk_idx == 0 and chunk_logits.size(1) < decoder_chunk_size:
                        break
                    elif chunk_logits.size(1) == decoder_chunk_size and end_dec != (enc_len + 1): # end_enc != enc_len: ############### watch this (because of 2x encoder context) ################
                        chunk_idx += 1
                    else:
                        break

                
                # Combine all chunk-level losses
                if all_losses:
                    final_loss = torch.stack(all_losses).mean()
                else:
                    final_loss = torch.tensor(0.0, device=repeated_curr_logits_with_letter_embeddings.device)
                
                # Concatenate chunked decoder logits for reporting
                final_logits = torch.cat(all_decoder_logits, dim=1)  # shape (1, total_decoder_tokens, vocab_size)

                # print(final_logits.shape, curr_letter_level_labels.shape, letter_level_labels.shape)
                # print('OLOLOLOOLLLOLOLO')
                final_logits = F.pad(final_logits, pad=(0, 0, 1, letter_level_labels.shape[1] - final_logits.shape[1] - 1))
                # print(final_logits.shape, curr_letter_level_labels.shape, letter_level_labels.shape, final_logits)
                # assert False
                batched_collected_repeated_logits.append(final_logits)
                batched_losses.append(final_loss)
        
        final_model_output = {
            "loss": torch.stack(batched_losses).mean(),
            "logits": torch.cat(batched_collected_repeated_logits, dim=0)
        }
        return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)


















class RMTEncoderDecoder(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, decoder_model, **rmt_kwargs):
        super().__init__()
    
        # Encoder (base_model) remains as before
        self.model = base_model
        
        # New: decoder in BERT-decoder mode (or any other decoder)
        self.decoder = decoder_model

        self.nucleotide_embedding = nn.Embedding(100, 1024)
        
        # New: label embedding, vocab size=26, hidden size=1024
        self.label_embedding = nn.Embedding(26, 1024)
        
        self.set_params(**rmt_kwargs)

        # self.wpe = nn.Embedding(self.rmt_config['decoder_chunk_size'] * 2, 1024)
        # self.wpe = nn.Embedding(35_000, 1024)
        
        self.rmt_config['sum_loss'] = True
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
        # todo: currently output from RMT model is not the same like from backbone model with 1 segment
        # because of inserted memory tokens and operations with cls/sep/pad in pad_and_segment
        # need to impl such that output from forward is like output from backbone model:
        # input -> segmented_inp -> segmented_logits -> output
        #                               | -> loss         | -> metrics
        #                           segmented_labels <- labels
        # self.model.eval()
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'labels_mask': labels_mask, 'pos_weight': pos_weight,
                  'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  }
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        # print(input_ids, input_ids.shape)

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :]
        for seg_num, (segment_input_ids, segment_labels, segment_labels_mask) in enumerate(zip(segmented,
                                                                                               segmented_labels,
                                                                                               segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            # print(f'ITERATION {seg_num, segment_input_ids}')

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
            # if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                # pos_weight = pos_weight[0, 0, :][None, None, :]
                # segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                # seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

            out = self.model(**seg_kwargs)
            # assert False
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits']) # .detach()
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        # assert False
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'])
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None: # CHECK HOW LABELS MASK WORKS IN RMT WITH BATCH SIZE > 1!!!!!!!!!!!!!!!!!!!!!!!
            batched_collected_repeated_logits, batched_losses = [], []
            for b in range(bs): # aggregate in one batch
                repeater_kwargs = dict()
                
                # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
                # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
                curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)#.detach()
                # print('MODEL WEIGHTS', self.model.bert.encoder.layer[0].attention.self.query.weight)
                # print('TENSOR!!!', curr_logits[0, 10, :])
                # print(curr_logits.dtype)
                # assert False
                # print('CURR LOGITS SHAPE', curr_logits.shape)
                lllm = letter_level_labels_mask[b]
                lllm[0] = True
                # print(letter_level_labels, lllm, letter_level_labels.shape, lllm.shape)
                # assert False
                # print(letter_level_labels.shape)
                curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
                # print(curr_letter_level_labels)
                assert 24 in curr_letter_level_labels
                lllm[0] = False
                # print(curr_letter_level_labels, curr_letter_level_labels.shape)
                # assert False
                curr_repeater = embedding_repeater[b][lllm[1:]]
                # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
                # assert 0 == 1
                # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
                # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
                # print(letter_level_tokens)
                curr_letter_level_embedding = F.pad(self.nucleotide_embedding(letter_level_tokens[b, lllm[1:]].unsqueeze(0)), (0, 0, 1, 0))
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
                # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
                # assert False
                # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
                # print('888888888888888888888888888', curr_logits.shape)
                # print('input_ids', torch.sum(input_ids[b] != 3))

                # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
                repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding[:, 1:, :] + curr_logits[:, curr_repeater, :] # combine this with post merging?
                # repeated_curr_logits_with_letter_embeddings += self.wpe(torch.arange(0, repeated_curr_logits_with_letter_embeddings.shape[1], dtype=torch.long, device=repeated_curr_logits_with_letter_embeddings.device).unsqueeze(0))
                # repeated_curr_logits_with_letter_embeddings = torch.cat((curr_letter_level_embedding, curr_logits[:, curr_repeater, :]), dim=-1)
                # print(repeated_curr_logits_with_letter_embeddings.shape)
                # assert False

                
                decoder_chunk_size = self.rmt_config["decoder_chunk_size"]  # e.g. 512
                look_back_size = self.rmt_config["decoder_look_back_size"]  # e.g. 512
                assert decoder_chunk_size > look_back_size
                enc_len = repeated_curr_logits_with_letter_embeddings.shape[1]
                # w = torch.tensor([11531128.0, 11549727.0, 11531095.0, 1.0, 488589751.0, 265281019.0, 463184462.0, 65733.0, 43130.0, 91413.0, 65774.0, 43172.0, 91380.0, 18329.0, 18394.0, 27016189.0, 27147388.0, 324472532.0, 245528623.0, 38243.0, 38121.0, 23780.0, 23839.0, 552140823.0, 1.0, 1.0], device=repeated_curr_logits_with_letter_embeddings.device)
                # loss_fct = nn.CrossEntropyLoss(weight=torch.sum(w) / w, ignore_index=-100)  # typical for seq2seq
                # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 1.0, 20.0, 20.0], device=repeated_curr_logits_with_letter_embeddings.device), ignore_index=-100)  # typical for seq2seq
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # typical for seq2seq
                
                
                # We'll chunk the encoder embeddings and the decoder input tokens in steps of decoder_chunk_size
                # num_chunks = math.ceil(enc_len / (decoder_chunk_size - look_back_size))
                
                all_decoder_logits = []
                all_losses = []

                chunk_idx = 0
                
                # for chunk_idx in range(num_chunks):
                while True:
                    # print(torch.sum(repeated_curr_logits_with_letter_embeddings))
                    # print(enc_len, chunk_idx)
                    start_enc = chunk_idx * (decoder_chunk_size - look_back_size)
                    end_enc = min(start_enc + decoder_chunk_size * 2, enc_len)

                    assert end_enc - start_enc <= decoder_chunk_size * 2
                
                    start_dec = chunk_idx * (decoder_chunk_size - look_back_size)
                    end_dec = min(start_dec + decoder_chunk_size, enc_len + 1)

                    assert end_dec - start_dec <= decoder_chunk_size

                    # print(start_enc, end_enc, start_dec, end_dec, repeated_curr_logits_with_letter_embeddings.shape, curr_letter_level_labels.shape)
                    
                    # Slice the encoder output for this chunk
                    chunk_encoder_embs = repeated_curr_logits_with_letter_embeddings[:, start_enc:end_enc, :]  # shape: (1, chunk_size_enc, 1024)
                    # print(chunk_encoder_embs, chunk_encoder_embs.shape)
                    # assert False
                    # print(chunk_encoder_embs.dtype) ########################################################### during training with mixed precision, dtype is not bf16
                    # print(self.wpe(torch.arange(0, chunk_encoder_embs.shape[1], dtype=torch.long, device=chunk_encoder_embs.device).unsqueeze(0)).shape, chunk_encoder_embs.dtype)
                    # assert False
                    # chunk_encoder_embs += self.wpe(torch.arange(0, chunk_encoder_embs.shape[1], dtype=torch.long, device=chunk_encoder_embs.device).unsqueeze(0))

                
                    # Slice the decoder tokens for this chunk
                    chunk_decoder_input_ids = curr_letter_level_labels[:, start_dec:end_dec]  # shape: (1, chunk_size_dec)
                    # print(curr_letter_level_labels[:, start_dec:end_dec].shape, curr_letter_level_labels[:, start_dec:end_dec])
                    if chunk_idx == 0:
                        # print('chunk_decoder_input_ids', chunk_decoder_input_ids)
                        assert chunk_decoder_input_ids[0, 0].item() == torch.tensor(24, device=chunk_encoder_embs.device)
                
                    # Embed the decoder input tokens
                    # print(curr_letter_level_labels)
                    # assert False
                    chunk_decoder_input_embeds = self.label_embedding(chunk_decoder_input_ids)  # (1, chunk_size_dec, 1024)
                    # chunk_decoder_input_embeds += curr_letter_level_embedding[:, start_dec:end_dec, :]
                    chunk_decoder_input_embeds += F.pad(repeated_curr_logits_with_letter_embeddings, (0, 0, 1, 0))[:, start_dec:end_dec, :]
                
                    # Prepare attention masks
                    # (Assuming full attention on the chunked encoder part, and no future masking for chunked decoder in teacher forcing)
                    encoder_attention_mask = torch.ones(chunk_encoder_embs.size()[:-1], device=chunk_encoder_embs.device)
                    decoder_attention_mask = (chunk_decoder_input_ids != 25).long()  # if 25 is your pad

                    # print(chunk_decoder_input_embeds.shape, chunk_encoder_embs.shape, encoder_attention_mask.shape, decoder_attention_mask.shape)
                    # print(chunk_decoder_input_embeds, chunk_encoder_embs, encoder_attention_mask, decoder_attention_mask)
                    # assert False
                
                    # Forward through the decoder (in cross-attn mode)
                    decoder_outputs = self.decoder(
                        inputs_embeds=chunk_decoder_input_embeds,
                        encoder_hidden_states=chunk_encoder_embs,
                        encoder_attention_mask=encoder_attention_mask,
                        attention_mask=decoder_attention_mask
                    )
                    # Suppose decoder_outputs.logits -> shape: (1, chunk_size_dec, vocab_size)
                    chunk_logits = decoder_outputs.logits

                    ############################################################################# should we consider recalculation on the edges between segemnts?
                
                    # Teacher forcing: shift by 1
                    # If chunk_size_dec=1 in the last chunk, skipping the loss is safe 
                    # because there's no "next token" to predict inside that chunk
                    if chunk_logits.size(1) > 1:
                        if chunk_idx == 0:
                            # we remove the last logit & first label
                            logits_for_loss = chunk_logits[:, :-1, :]  # shape: (1, chunk_size_dec-1, vocab_size)
                            labels_for_loss = chunk_decoder_input_ids[:, 1:]  # shape: (1, chunk_size_dec-1)
                            # compute cross-entropy
                            loss_chunk = loss_fct(
                                logits_for_loss.reshape(-1, logits_for_loss.size(-1)), 
                                labels_for_loss.reshape(-1)
                            )
                            all_losses.append(loss_chunk)
                        else:
                            assert chunk_logits.size(1) - look_back_size >= 0
                            if chunk_logits.size(1) - look_back_size > 1:
                                # we remove the last logit & first label
                                logits_for_loss = chunk_logits[:, look_back_size:-1, :]  # shape: (1, chunk_size_dec-1, vocab_size)
                                labels_for_loss = chunk_decoder_input_ids[:, look_back_size+1:]  # shape: (1, chunk_size_dec-1)
                                # compute cross-entropy
                                loss_chunk = loss_fct(
                                    logits_for_loss.reshape(-1, logits_for_loss.size(-1)), 
                                    labels_for_loss.reshape(-1)
                                )
                                all_losses.append(loss_chunk)
                            else:
                                loss_chunk = torch.tensor(0.0, device=chunk_encoder_embs.device)
                    else:
                        # no next-token to predict if chunk size is 1
                        loss_chunk = torch.tensor(0.0, device=chunk_encoder_embs.device)

                    # print('GOGOOGGOGOG')
                    if chunk_idx == 0:
                        all_decoder_logits.append(chunk_logits)
                    else:
                        all_decoder_logits.append(chunk_logits[:, look_back_size:, :])

                    if chunk_idx == 0 and chunk_logits.size(1) < decoder_chunk_size:
                        break
                    elif chunk_logits.size(1) == decoder_chunk_size and end_dec != (enc_len + 1): # end_enc != enc_len: ############### watch this (because of 2x encoder context) ################
                        chunk_idx += 1
                    else:
                        break

                
                # Combine all chunk-level losses
                # print(all_losses, torch.stack(all_losses).max())
                if all_losses:
                    final_loss = torch.stack(all_losses).max() # .mean()
                else:
                    final_loss = torch.tensor(0.0, device=repeated_curr_logits_with_letter_embeddings.device)
                
                # Concatenate chunked decoder logits for reporting
                final_logits = torch.cat(all_decoder_logits, dim=1)  # shape (1, total_decoder_tokens, vocab_size)

                # print(final_logits.shape, curr_letter_level_labels.shape, letter_level_labels.shape)
                # print('OLOLOLOOLLLOLOLO')
                final_logits = F.pad(final_logits, pad=(0, 0, 1, letter_level_labels.shape[1] - final_logits.shape[1] - 1))
                # print(final_logits.shape, curr_letter_level_labels.shape, letter_level_labels.shape, final_logits)
                # assert False
                batched_collected_repeated_logits.append(final_logits)
                batched_losses.append(final_loss)
        
        final_model_output = {
            "loss": torch.stack(batched_losses).mean(),
            "logits": torch.cat(batched_collected_repeated_logits, dim=0)
        }
        return final_model_output

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)







class RMTEncoderDecoderValidation(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, decoder_model, **rmt_kwargs):
        super().__init__()
    
        # Encoder (base_model) remains as before
        self.model = base_model
        
        # New: decoder in BERT-decoder mode (or any other decoder)
        self.decoder = decoder_model

        self.nucleotide_embedding = nn.Embedding(100, 1024)
        
        # New: label embedding, vocab size=26, hidden size=1024
        self.label_embedding = nn.Embedding(26, 1024)
        
        self.set_params(**rmt_kwargs)

        # self.wpe = nn.Embedding(35_000, 1024)
        
        self.rmt_config['sum_loss'] = True
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        # print(input_ids, input_ids.shape)

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        # print(segmented)
        # print(segmented_labels)
        # print(segmented_labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :]
        for seg_num, (segment_input_ids, segment_labels_mask) in enumerate(zip(segmented, segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            # print(f'ITERATION {seg_num, segment_input_ids}')

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
            # if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                # pos_weight = pos_weight[0, 0, :][None, None, :]
                # segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                # seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)
            
            out = self.model(**seg_kwargs)
            # assert False
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        # assert False
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            # labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'].shape)
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        # out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            # out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None: # CHECK HOW LABELS MASK WORKS IN RMT WITH BATCH SIZE > 1!!!!!!!!!!!!!!!!!!!!!!!
            batched_collected_repeated_logits, batched_losses = [], []
            assert bs == 1
            b = 0
            repeater_kwargs = dict()
            
            # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
            # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
            curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
            print('CURR LOGITS SHAPE', curr_logits.shape)
            lllm = letter_level_labels_mask[b]
            # lllm[0] = True
            # print(letter_level_labels, lllm, letter_level_labels.shape, lllm.shape)
            # assert False
            # print(letter_level_labels.shape)
            # curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
            # # print(curr_letter_level_labels)
            # assert 24 in curr_letter_level_labels
            # lllm[0] = False
            # print(curr_letter_level_labels, curr_letter_level_labels.shape)
            # assert False
            curr_repeater = embedding_repeater[b][lllm]
            # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
            # assert 0 == 1
            # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
            # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
            # print(letter_level_tokens)
            curr_letter_level_embedding = F.pad(self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0)), (0, 0, 1, 0))
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
            # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
            # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
            # print('888888888888888888888888888', curr_logits.shape)
            # print('input_ids', torch.sum(input_ids[b] != 3))
    
            # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
            repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding[:, 1:, :] + curr_logits[:, curr_repeater, :] # combine this with post merging?
            # repeated_curr_logits_with_letter_embeddings += self.wpe(torch.arange(0, repeated_curr_logits_with_letter_embeddings.shape[1], dtype=torch.long, device=repeated_curr_logits_with_letter_embeddings.device).unsqueeze(0))
            # print(repeated_curr_logits_with_letter_embeddings.shape, repeated_curr_logits_with_letter_embeddings.dtype)
            # assert False
                
        return repeated_curr_logits_with_letter_embeddings, F.pad(repeated_curr_logits_with_letter_embeddings, (0, 0, 1, 0)) # curr_letter_level_embedding

    def generate_process(self, generated_tokens, encoder_embeddings, curr_letter_level_embedding, chunk_size, device, targets_without_bos_token=None):
        current_len = len(generated_tokens)

        if targets_without_bos_token is not None:
            generated_tokens_new = [24] + list(targets_without_bos_token[:current_len-1])
                
        # Decoder input: only last 'chunk_size' tokens
        start_dec = max(0, current_len - chunk_size)
        if targets_without_bos_token is not None:
            dec_input_ids = torch.tensor([generated_tokens_new[start_dec:]], device=device)
        else:
            dec_input_ids = torch.tensor([generated_tokens[start_dec:]], device=device)
        chunk_decoder_input_embeds = self.label_embedding(dec_input_ids)
        chunk_decoder_input_embeds += curr_letter_level_embedding[:, start_dec:current_len, :]
        
        # Similarly, chunk the encoder if desired:
        # e.g., "sliding window" on the encoder side
        start_enc = start_dec  # align them for simplicity
        # print(start_enc + chunk_size, encoder_embeddings.size(1))
        end_enc = min(start_enc + chunk_size * 2, encoder_embeddings.size(1))
        chunk_enc_embs = encoder_embeddings[:, start_enc:end_enc, :]
        # print(chunk_enc_embs.shape)
        # print('VALID ENC EMB', chunk_enc_embs)
        # assert False

        # print(chunk_enc_embs.shape, chunk_decoder_input_embeds.shape)
        # time.sleep(1)
        
        
        enc_attn_mask = torch.ones(chunk_enc_embs.shape[:-1], device=device)
        dec_attn_mask = torch.ones(dec_input_ids.shape, device=device).long()
        
        out = self.decoder(
            inputs_embeds=chunk_decoder_input_embeds,
            encoder_hidden_states=chunk_enc_embs,
            encoder_attention_mask=enc_attn_mask,
            attention_mask=dec_attn_mask
        )
        # out.logits shape: (1, seq_len, vocab_size)
        next_token_logits = out.logits[:, -1, :]  # last step

        return next_token_logits.flatten()

    def find_intervals(self, lst):
        if not lst:
            return []
        
        intervals = []
        start = 0
        current_value = lst[0]
        
        # Iterate from the second element to the end.
        for i in range(1, len(lst)):
            if lst[i] != current_value:
                # Append the tuple: (value, start index, current index)
                intervals.append((current_value, start, i))
                current_value = lst[i]
                start = i
                
        # Append the last run, which goes until the end of the list.
        intervals.append((current_value, start, len(lst)))
        return intervals
        

    def generate(self, model_input_data, generate_the_most_probable=True, threshold=0.2, temperature=1.0, targets_without_bos_token=None):
        """
        Generate tokens with a sliding window approach for the decoder.
        encoder_embeddings: shape (1, enc_len, hidden_size), presumably huge.
        max_gen_length: max tokens to generate (including <BOS>).
        """
        label_dict = {0: 'CDS-0', 1:'CDS-1', 2:'CDS-2', 3:'CDS-skip', 4:'intron-0', 5:'intron-1', 6:'intron-2', 7:'ASS-0', 8:'ASS-1', 9:'ASS-2', 10:'DSS-0', 11:'DSS-1', 12:'DSS-2', 13:'START', 14:'STOP', 15:'nc_exon_plus', 16:'nc_exon_minus', 17:'nc_intron_plus', 18:'nc_intron_minus', 19:'nc_ASS', 20:'nc_DSS', 21:'TSS' , 22:'PolyA', 23:'IR'}
        
        encoder_embeddings, curr_letter_level_embedding = self.forward(**model_input_data)
        max_gen_length = encoder_embeddings.shape[1]
        device = encoder_embeddings.device
        chunk_size = self.rmt_config["decoder_chunk_size"]
        bos_id = 24
    
        generated_tokens = [bos_id] # start with <BOS>

        if generate_the_most_probable:
        
            for _ in tqdm(range(max_gen_length)):

                next_token_logits = self.generate_process(generated_tokens, encoder_embeddings, curr_letter_level_embedding, chunk_size, device, targets_without_bos_token)
                
                # print(torch.round(next_token_logits.softmax(dim=-1), decimals=2))
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                # next_token_id = torch.multinomial((next_token_logits / temperature).softmax(dim=-1), num_samples=1)
                
                generated_tokens.append(next_token_id.item())
            
            return generated_tokens[1:]

        else:
            print('GENERATION START'.center(30, '='))
            
            all_genes_dict = dict()

            while True:

                if len(generated_tokens) % 1000 == 0:
                    print(f"We found {len(all_genes_dict)} genes, and {len(generated_tokens)} tokens were generated out of {max_gen_length + 1}")
                # print(f"\rWe found {len(all_genes_dict)} genes, and {len(generated_tokens)} tokens were generated out of {max_gen_length + 1}", end='', flush=True)
                
                next_token_logits = self.generate_process(generated_tokens, encoder_embeddings, curr_letter_level_embedding, chunk_size, device)
                # print(next_token_logits.shape)
                next_token_values_3, next_token_ids_3 = torch.topk(next_token_logits, 3)
                next_token_values_3 = next_token_values_3.softmax(dim=-1)
                # print(next_token_values_3, next_token_ids_3)

                important_tokens = []
                if 21 in next_token_ids_3:
                    # print(next_token_ids_3)
                    if next_token_values_3[torch.where(next_token_ids_3 == 21)[0].item()] > threshold:
                        important_tokens.append(21)
                if 22 in next_token_ids_3:
                    if next_token_values_3[torch.where(next_token_ids_3 == 22)[0].item()] > threshold:
                        important_tokens.append(22)
                if 23 in next_token_ids_3:
                    if next_token_values_3[torch.where(next_token_ids_3 == 23)[0].item()] > threshold:
                        important_tokens.append(23)

                if len(important_tokens) > 0:

                    # print(f"\rWe found {len(all_genes_dict)} genes, and {len(generated_tokens)} tokens were generated out of {max_gen_length + 1}, fork point {len(generated_tokens)}", end='      ', flush=True)

                    assert len(next_token_ids_3) <= 3

                    for next_token_id in important_tokens:

                        generated_tokens_3 = copy.deepcopy(generated_tokens)

                        generated_tokens_3.append(next_token_id)
        
                        if len(generated_tokens_3) >= max_gen_length + 1:
                            break
        
                        if next_token_id in [21, 22]:
                            if next_token_id == 21:
                                curr_next_token_id = 21
                                gene_end_token = 22
                            else:
                                curr_next_token_id = 22
                                gene_end_token = 21
                            gene_start_idx = len(generated_tokens_3) - 1
                            current_generated_tokens = copy.deepcopy(generated_tokens_3)
        
                            red_flag_intergenic = False
        
                            while curr_next_token_id != gene_end_token:
        
                                if len(current_generated_tokens) >= max_gen_length + 1:
                                    red_flag_intergenic = True
                                    break
        
                                next_token_logits = self.generate_process(current_generated_tokens, encoder_embeddings, curr_letter_level_embedding, chunk_size, device)
                                curr_next_token_id = torch.argmax(next_token_logits, dim=-1) # torch.multinomial((next_token_logits / temperature).softmax(dim=-1), num_samples=1)
                                if curr_next_token_id >= 23: # intergenic or system tokens
                                    # add abrupted gene counter
                                    red_flag_intergenic = True
                                    break

                                # print(f"\rWe found {len(all_genes_dict)} genes, and {len(generated_tokens)} tokens were generated out of {max_gen_length + 1}, fork generated {len(current_generated_tokens)} out of possible {max_gen_length + 1} tokens, now we're in {label_dict[curr_next_token_id.item()]}", end='', flush=True)
                                current_generated_tokens.append(curr_next_token_id.item())
        
                            if not red_flag_intergenic:
                                gene_only_sequence = current_generated_tokens[gene_start_idx:]
                                all_genes_dict[f'gene_{len(all_genes_dict)}'] = self.find_intervals(gene_only_sequence)
                

                    generated_tokens.append(23)
                    if len(generated_tokens) >= max_gen_length + 1:
                        break
                    
                            

                else:
                    generated_tokens.append(next_token_ids_3[0].item()) # the most probable one
                    if len(generated_tokens) >= max_gen_length + 1:
                        break
                        
            return all_genes_dict


    
    
                    
                
                


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

        # print(input_elements)
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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)















class RMTEncoderDecoderValidationFeatured(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, decoder_model, **rmt_kwargs):
        super().__init__()
    
        # Encoder (base_model) remains as before
        self.model = base_model
        
        # New: decoder in BERT-decoder mode (or any other decoder)
        self.decoder = decoder_model

        self.nucleotide_embedding = nn.Embedding(100, 1024)
        
        # New: label embedding, vocab size=26, hidden size=1024
        self.label_embedding = nn.Embedding(26, 1024)
        
        self.set_params(**rmt_kwargs)

        self.wpe = nn.Embedding(self.rmt_config['decoder_chunk_size'] * 2, 1024)
        
        self.rmt_config['sum_loss'] = True
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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
        # print('POSPOSPOSPOSPOS', pos_weight.shape)
        bs, seq_len = input_ids.shape

        # print(input_ids, input_ids.shape)

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented, segmented_labels, segmented_labels_mask = self.pad_and_segment(input_ids, labels, labels_mask)

        # print(segmented)
        # print(segmented_labels)
        # print(segmented_labels_mask)

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        # pos_weight = pos_weight[0, 0, :][None, None, :]
        for seg_num, (segment_input_ids, segment_labels_mask) in enumerate(zip(segmented, segmented_labels_mask)):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']):
                memory = memory.detach()

            # print(f'ITERATION {seg_num, segment_input_ids}')

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
            # if pos_weight is not None:
                # all values in the second dimension of pos_weight should be the same
                # pos_weight = pos_weight[0, 0, :][None, None, :]
                # segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
                # seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)
            
            out = self.model(**seg_kwargs)
            # assert False
            # print(out)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        # assert False
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            # labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        out['logits'] = torch.cat(logits, dim=1)
        # out['logits'] = self.middle_dropout(self.middle_norm(torch.cat(logits, dim=1)))
        # print(out['logits'].shape)
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        # out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            # out['rmt_logits_masks_segm'] = [logits_masks]

        # print(out['logits'])
        mem_token_ids = self.mem_token_ids
        
        if embedding_repeater is not None: # CHECK HOW LABELS MASK WORKS IN RMT WITH BATCH SIZE > 1!!!!!!!!!!!!!!!!!!!!!!!
            batched_collected_repeated_logits, batched_losses = [], []
            assert bs == 1
            b = 0
            repeater_kwargs = dict()
            
            # print('google', out['rmt_logits_masks'][b, :].bool()[:10])
            # print(out['logits'].shape, out['rmt_logits_masks'][b, :].bool().shape)
            curr_logits = out['logits'][b, out['rmt_logits_masks'][b, :].bool(), :].unsqueeze(0)
            print('CURR LOGITS SHAPE', curr_logits.shape)
            lllm = letter_level_labels_mask[b]
            # lllm[0] = True
            # print(letter_level_labels, lllm, letter_level_labels.shape, lllm.shape)
            # assert False
            # print(letter_level_labels.shape)
            # curr_letter_level_labels = letter_level_labels[b, lllm].unsqueeze(0)
            # # print(curr_letter_level_labels)
            # assert 24 in curr_letter_level_labels
            # lllm[0] = False
            # print(curr_letter_level_labels, curr_letter_level_labels.shape)
            # assert False
            curr_repeater = embedding_repeater[b][lllm]
            # print('LLT SHAPE', letter_level_tokens[b, lllm].shape)
            # assert 0 == 1
            # print(set(list(letter_level_tokens[b, lllm].unsqueeze(0).flatten().detach().cpu().numpy())))
            # curr_letter_level_embedding = self.sub_model.base_model.embeddings.word_embeddings(letter_level_tokens[b, lllm].unsqueeze(0))#.requires_grad_() # check
            # print(letter_level_tokens)
            curr_letter_level_embedding = F.pad(self.nucleotide_embedding(letter_level_tokens[b, lllm].unsqueeze(0)), (0, 0, 1, 0))
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', curr_letter_level_embedding)
            # print('1111111111111111111111111111', curr_letter_level_embedding.shape)
            # print('777777777777777777777777777', curr_repeater.shape, torch.max(curr_repeater))
            # print('888888888888888888888888888', curr_logits.shape)
            # print('input_ids', torch.sum(input_ids[b] != 3))
    
            # print('ALL SHAPES!!!!!!!!!!!!!!!', curr_logits[:, curr_repeater, :].shape, curr_letter_level_embedding.shape)
            repeated_curr_logits_with_letter_embeddings = curr_letter_level_embedding[:, 1:, :] + curr_logits[:, curr_repeater, :] # combine this with post merging?
            # repeated_curr_logits_with_letter_embeddings += self.wpe(torch.arange(0, repeated_curr_logits_with_letter_embeddings.shape[1], dtype=torch.long, device=repeated_curr_logits_with_letter_embeddings.device).unsqueeze(0))
            # print(repeated_curr_logits_with_letter_embeddings.shape, repeated_curr_logits_with_letter_embeddings.dtype)
            # assert False
                
        return repeated_curr_logits_with_letter_embeddings, F.pad(repeated_curr_logits_with_letter_embeddings, (0, 0, 1, 0)) # curr_letter_level_embedding

    def generate_process(self, generated_tokens, encoder_embeddings, curr_letter_level_embedding, chunk_size, device, targets_without_bos_token=None):
        current_len = len(generated_tokens)

        if targets_without_bos_token is not None:
            generated_tokens_new = [24] + list(targets_without_bos_token[:current_len-1])
                
        # Decoder input: only last 'chunk_size' tokens
        start_dec = max(0, current_len - chunk_size)
        if targets_without_bos_token is not None:
            dec_input_ids = torch.tensor([generated_tokens_new[start_dec:]], device=device)
        else:
            dec_input_ids = torch.tensor([generated_tokens[start_dec:]], device=device)
        chunk_decoder_input_embeds = self.label_embedding(dec_input_ids)
        chunk_decoder_input_embeds += curr_letter_level_embedding[:, start_dec:current_len, :]
        
        # Similarly, chunk the encoder if desired:
        # e.g., "sliding window" on the encoder side
        start_enc = start_dec  # align them for simplicity
        # print(start_enc + chunk_size, encoder_embeddings.size(1))
        end_enc = min(start_enc + chunk_size * 2, encoder_embeddings.size(1))
        chunk_enc_embs = encoder_embeddings[:, start_enc:end_enc, :].detach().clone()
        chunk_enc_embs += self.wpe(torch.arange(0, chunk_enc_embs.shape[1], dtype=torch.long, device=chunk_enc_embs.device).unsqueeze(0))
        # print(chunk_enc_embs.shape)
        # print('VALID ENC EMB', chunk_enc_embs)
        # assert False

        # print(chunk_enc_embs.shape, chunk_decoder_input_embeds.shape)
        # time.sleep(1)
        
        
        enc_attn_mask = torch.ones(chunk_enc_embs.shape[:-1], device=device)
        dec_attn_mask = torch.ones(dec_input_ids.shape, device=device).long()

        # print(chunk_decoder_input_embeds.shape, chunk_enc_embs.shape, start_enc, end_enc)
        # print(chunk_enc_embs[0, 0, :10])
        # if current_len == 3:
        #     assert False
        
        out = self.decoder(
            inputs_embeds=chunk_decoder_input_embeds,
            encoder_hidden_states=chunk_enc_embs,
            encoder_attention_mask=enc_attn_mask,
            attention_mask=dec_attn_mask
        )
        # out.logits shape: (1, seq_len, vocab_size)
        next_token_logits = out.logits[:, -1, :]  # last step

        return next_token_logits.flatten()

    def find_intervals(self, lst):
        if not lst:
            return []
        
        intervals = []
        start = 0
        current_value = lst[0]
        
        # Iterate from the second element to the end.
        for i in range(1, len(lst)):
            if lst[i] != current_value:
                # Append the tuple: (value, start index, current index)
                intervals.append((current_value, start, i))
                current_value = lst[i]
                start = i
                
        # Append the last run, which goes until the end of the list.
        intervals.append((current_value, start, len(lst)))
        return intervals
        

    def generate(self, model_input_data, generate_the_most_probable=True, threshold=0.2, temperature=1.0, targets_without_bos_token=None):
        """
        Generate tokens with a sliding window approach for the decoder.
        encoder_embeddings: shape (1, enc_len, hidden_size), presumably huge.
        max_gen_length: max tokens to generate (including <BOS>).
        """
        label_dict = {0: 'CDS-0', 1:'CDS-1', 2:'CDS-2', 3:'CDS-skip', 4:'intron-0', 5:'intron-1', 6:'intron-2', 7:'ASS-0', 8:'ASS-1', 9:'ASS-2', 10:'DSS-0', 11:'DSS-1', 12:'DSS-2', 13:'START', 14:'STOP', 15:'nc_exon_plus', 16:'nc_exon_minus', 17:'nc_intron_plus', 18:'nc_intron_minus', 19:'nc_ASS', 20:'nc_DSS', 21:'TSS' , 22:'PolyA', 23:'IR'}
        
        encoder_embeddings, curr_letter_level_embedding = self.forward(**model_input_data)
        max_gen_length = encoder_embeddings.shape[1]
        device = encoder_embeddings.device
        chunk_size = self.rmt_config["decoder_chunk_size"]
        bos_id = 24
    
        generated_tokens = [bos_id] # start with <BOS>
        generated_logits = []

        if generate_the_most_probable:
        
            for _ in tqdm(range(max_gen_length)):

                next_token_logits = self.generate_process(generated_tokens, encoder_embeddings, curr_letter_level_embedding, chunk_size, device, targets_without_bos_token)
                
                # print(torch.round(next_token_logits.softmax(dim=-1), decimals=2))
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                # next_token_id = torch.multinomial((next_token_logits / temperature).softmax(dim=-1), num_samples=1)
                
                generated_tokens.append(next_token_id.item())
                generated_logits.append(next_token_logits.softmax(dim=-1).detach().cpu().numpy())
            
            
            # return generated_tokens[1:]
            return np.array(generated_logits)[:, :24]
        

        else:
            print('GENERATION START'.center(30, '='))
            
            all_genes_dict = dict()

            while True:

                if len(generated_tokens) % 1000 == 0:
                    print(f"We found {len(all_genes_dict)} genes, and {len(generated_tokens)} tokens were generated out of {max_gen_length + 1}")
                # print(f"\rWe found {len(all_genes_dict)} genes, and {len(generated_tokens)} tokens were generated out of {max_gen_length + 1}", end='', flush=True)
                
                next_token_logits = self.generate_process(generated_tokens, encoder_embeddings, curr_letter_level_embedding, chunk_size, device)
                # print(next_token_logits.shape)
                next_token_values_3, next_token_ids_3 = torch.topk(next_token_logits, 3)
                next_token_values_3 = next_token_values_3.softmax(dim=-1)
                # print(next_token_values_3, next_token_ids_3)

                important_tokens = []
                if 21 in next_token_ids_3:
                    # print(next_token_ids_3)
                    if next_token_values_3[torch.where(next_token_ids_3 == 21)[0].item()] > threshold:
                        important_tokens.append(21)
                if 22 in next_token_ids_3:
                    if next_token_values_3[torch.where(next_token_ids_3 == 22)[0].item()] > threshold:
                        important_tokens.append(22)
                if 23 in next_token_ids_3:
                    if next_token_values_3[torch.where(next_token_ids_3 == 23)[0].item()] > threshold:
                        important_tokens.append(23)

                if len(important_tokens) > 0:

                    # print(f"\rWe found {len(all_genes_dict)} genes, and {len(generated_tokens)} tokens were generated out of {max_gen_length + 1}, fork point {len(generated_tokens)}", end='      ', flush=True)

                    assert len(next_token_ids_3) <= 3

                    for next_token_id in important_tokens:

                        generated_tokens_3 = copy.deepcopy(generated_tokens)

                        generated_tokens_3.append(next_token_id)
        
                        if len(generated_tokens_3) >= max_gen_length + 1:
                            break
        
                        if next_token_id in [21, 22]:
                            if next_token_id == 21:
                                curr_next_token_id = 21
                                gene_end_token = 22
                            else:
                                curr_next_token_id = 22
                                gene_end_token = 21
                            gene_start_idx = len(generated_tokens_3) - 1
                            current_generated_tokens = copy.deepcopy(generated_tokens_3)
        
                            red_flag_intergenic = False
        
                            while curr_next_token_id != gene_end_token:
        
                                if len(current_generated_tokens) >= max_gen_length + 1:
                                    red_flag_intergenic = True
                                    break
        
                                next_token_logits = self.generate_process(current_generated_tokens, encoder_embeddings, curr_letter_level_embedding, chunk_size, device)
                                curr_next_token_id = torch.argmax(next_token_logits, dim=-1) # torch.multinomial((next_token_logits / temperature).softmax(dim=-1), num_samples=1)
                                if curr_next_token_id >= 23: # intergenic or system tokens
                                    # add abrupted gene counter
                                    red_flag_intergenic = True
                                    break

                                # print(f"\rWe found {len(all_genes_dict)} genes, and {len(generated_tokens)} tokens were generated out of {max_gen_length + 1}, fork generated {len(current_generated_tokens)} out of possible {max_gen_length + 1} tokens, now we're in {label_dict[curr_next_token_id.item()]}", end='', flush=True)
                                current_generated_tokens.append(curr_next_token_id.item())
        
                            if not red_flag_intergenic:
                                gene_only_sequence = current_generated_tokens[gene_start_idx:]
                                all_genes_dict[f'gene_{len(all_genes_dict)}'] = self.find_intervals(gene_only_sequence)
                

                    generated_tokens.append(23)
                    if len(generated_tokens) >= max_gen_length + 1:
                        break
                    
                            

                else:
                    generated_tokens.append(next_token_ids_3[0].item()) # the most probable one
                    if len(generated_tokens) >= max_gen_length + 1:
                        break
                        
            return all_genes_dict  


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

        # print(input_elements)
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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)













class RMTEncoderForLetterLevelTokenClassificationUnetSegmentedEmbeddingOnly(torch.nn.Module):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__() 
        self.model = base_model
        self.sub_model = UNET1DSegmentationHead(
                            embed_dim=2048,
                            num_classes=2048,
                            output_channels_list=[256, 512, 1024],  # Example channel sizes as a list
                            num_conv_layers_per_block=2
                        )
        self.nucleotide_embedding = nn.Embedding(1000, 1024)
        self.activation_fn = nn.SiLU()
        self.fc = nn.Linear(2048, 5)        
        self.set_params(**rmt_kwargs)
        
        #self.sub_model.embeddings = self.sub_model.base_model.embeddings.word_embeddings
        
        self.rmt_config['sum_loss'] = True
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, embedding_repeater=None, letter_level_tokens=None, letter_level_labels=None,
                letter_level_labels_mask=None, letter_level_token_types_ids=None, letter_level_attention_mask=None):
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

        all_memory_embeddings = []

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
            # print(input_ids)
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
            # print(self.memory_position)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            all_memory_embeddings.append(out.hidden_states[-1][:, self.memory_position])

            logits.append(out['logits'].detach())
            labels_segm += [seg_kwargs['labels']]

            if labels_mask is not None:
                logits_masks.append(seg_kwargs['labels_mask'])
        
        
        
        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

#         for i, l in enumerate(losses):
#             out[f'loss_{i}'] = l.mean()

#         # aggregate losses from all segments
#         out['loss'] = torch.stack(losses).mean()

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

        
            
        return {'all_memory_embeddings': torch.stack(all_memory_embeddings, dim=0),
                'labels_segm': torch.stack(out['labels_segm'][0], dim=0),
                'rmt_logits_masks_segm': torch.stack(out['rmt_logits_masks_segm'][0], dim=0)}

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

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)


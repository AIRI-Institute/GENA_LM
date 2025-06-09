import json
import logging
import os
from pathlib import Path
from itertools import zip_longest
import psutil

from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from scipy import stats
import numpy as np

import accelerate
from accelerate import DistributedDataParallelKwargs

from lm_experiments_tools import TrainerAccelerate as Trainer
from lm_experiments_tools import TrainerAccelerateArgs as TrainerArgs

from lm_experiments_tools.utils import prepare_run, get_cls_by_name, get_optimizer

load_dotenv()

import sys
sys.path.insert(0, os.getcwd())

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')

from lm_experiments_tools import get_optimizer
from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration, get_git_diff
import lm_experiments_tools.optimizers as optimizers

from downstream_tasks.annotation.transcript_GenePredictionDataset_letter_level_bidirectional import AnnotationDataset

# import horovod.torch as hvd

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# hvd.init()

# torch.set_num_threads(4)
# torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--data_path', type=str, help='path to the training data')
parser.add_argument('--valid_data_path', type=str, help='path to the valid data')
parser.add_argument('--test_data_path', type=str, help='path to the test data (dataset_test_0.csv)')
parser.add_argument('--seed', type=int, default=42, help='random seed')

# data args
parser.add_argument('--input_seq_len', type=int, default=64, help='input sequnce length (default: 64).')
parser.add_argument('--letter_level_input_seq_len', type=int, default=64, help='input sequnce length (default: 64).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')
parser.add_argument('--targets_offset', type=int, default=5000, help='default: 5000')
parser.add_argument('--targets_len', type=int, default=5000, help='default: 5000')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')


# rmt args
parser.add_argument('--backbone_cls', type=str, default=None,
                    help='backbone class name to use for RMT')

parser.add_argument('--backbone_trainable', action='store_true', default=False,
                    help='make all model weights trainable, not only task-specific head.')
parser.add_argument('--forward_checkpoint', type=str,
                    help='forward checkpoint (default: None).')
parser.add_argument('--backward_checkpoint', type=str,
                    help='backward checkpoint (default: None).')

parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')

parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])

parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')

parser.add_argument('--full_checkpoint', type=str,
                    help='full checkpoint (default: None).')

if __name__ == '__main__':
    args = parser.parse_args()
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')
    logger.info(f'accelerator state: {accelerator.state}')

    prepare_run(args, logger, logger_fmt, accelerator=accelerator)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    # global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    # get train dataset
    logger.info(f'preparing training data from: {args.data_path}')
    data_path = Path(args.data_path).expanduser().absolute()
    train_dataset = AnnotationDataset(data_path, max_seq_len=args.input_seq_len, tokenizer=tokenizer, max_atcg_seq_len=args.letter_level_input_seq_len)
    logger.info(f'len(train_dataset): {len(train_dataset)}')
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=accelerator.process_index, num_replicas=accelerator.num_processes, shuffle=True,
                                       drop_last=False, seed=args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler, **kwargs)

    if args.valid_data_path:
        logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_dataset = AnnotationDataset(valid_data_path, max_seq_len=args.input_seq_len, tokenizer=tokenizer, max_atcg_seq_len=args.letter_level_input_seq_len, tmp_valid_len=1000)
        valid_sampler = DistributedSampler(valid_dataset, rank=accelerator.process_index, num_replicas=accelerator.num_processes, shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler, **kwargs)
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
        logger.info(f'len(valid_dataset): {len(valid_dataset)}')
    else:
        valid_dataloader = None
        logger.info('No validation data is used.')

    # define model
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    # labels: 0, 1, 2; multi-class multi-label classification
    # model_cfg.num_labels = 6
    # model_cfg.problem_type = 'multi_label_classification'
    model_cls = get_cls_by_name(args.backbone_cls)
    
    logger.info(f'Using forward and backward model class: {model_cls}')
    model_forward = model_cls(config=model_cfg)
    model_backward = model_cls(config=model_cfg)

    # Aydar # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        rmt_config = {
            'num_mem_tokens': args.num_mem_tokens,
            'max_n_segments': args.max_n_segments,
            # 'segment_ordering': args.segment_ordering,
            'input_size': args.input_size,
            'bptt_depth': args.bptt_depth,
            'sum_loss': True,
            'tokenizer': tokenizer,
            'use_crf': False
        }
        rmt_cls = get_cls_by_name(args.model_cls)
        logger.info(f'Wrapping in: {rmt_cls}')
        model = rmt_cls(model_forward, model_backward, **rmt_config)
        
        if args.forward_checkpoint is not None:
            logger.info(f'loading forward checkpoit from {args.forward_checkpoint}')
            checkpoint = torch.load(args.forward_checkpoint, map_location='cpu')
            checkpoint_new = dict()
            for k, v in checkpoint.items():
                if 'model' in k:
                    checkpoint_new[k[6:]] = v
                else:
                    checkpoint_new[k] = v
            # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', args.backbone_checkpoint)
            missing_k, unexpected_k = model.forward_model.load_state_dict(checkpoint_new, strict=False)
            if len(missing_k) != 0:
                logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized. Forward.')
            if len(unexpected_k) != 0:
                logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them! Forward.')

        if args.backward_checkpoint is not None:
            logger.info(f'loading backward checkpoit from {args.backward_checkpoint}')
            checkpoint = torch.load(args.backward_checkpoint, map_location='cpu')
            checkpoint_new = dict()
            for k, v in checkpoint.items():
                if 'model' in k:
                    checkpoint_new[k[6:]] = v
                else:
                    checkpoint_new[k] = v
            # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', args.backbone_checkpoint)
            missing_k, unexpected_k = model.backward_model.load_state_dict(checkpoint_new, strict=False)
            if len(missing_k) != 0:
                logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized. Backward.')
            if len(unexpected_k) != 0:
                logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them! Backward.')
                
                
        if args.full_checkpoint is not None:
            logger.info(f'loading full checkpoint from {args.full_checkpoint}')
            checkpoint = torch.load(args.full_checkpoint, map_location='cpu')
            

            missing_k, unexpected_k = model.load_state_dict(checkpoint, strict=False)
            if len(missing_k) != 0:
                logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized. Backbone.')
            if len(unexpected_k) != 0:
                logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them! Backbone.')

        if args.input_seq_len / model.segment_size > rmt_config['max_n_segments']:
            raise RuntimeError(f"Input sequence does not fully fit into selected number of segments: "
                               f"{args.input_seq_len} / {model.segment_size} > {rmt_config['max_n_segments']}")

    print('Total number of parameters:', sum(p.numel() for p in model.parameters()))
    
    if not args.backbone_trainable:
        print('FREEZING WEIGHTS')
        # for name, param in model.named_parameters():
        #     if 'classifier' not in name:
        #         logger.info(f'{name} is frozen')
        #         param.requires_grad = False
        for param in model.forward_model.parameters():
            param.requires_grad = False

        for param in model.backward_model.parameters():
            param.requires_grad = False

    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  scale_parameter=args.scale_parameter,
                                  relative_step=args.relative_step,
                                  warmup_init=args.warmup_init,
                                  weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # label counts in test set: [8378616.,    9842.,   10258.])
    # upweight class 1 and 2
    pos_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]) # torch.tensor([128.0, 11.8, 1.0, 33.6, 27.4, 11.7]) # torch.tensor([2.0, 0.05, 0.5])

    def batch_transform_fn(batch):
        # print('AAAAAAAAAAAAAAA', batch['input_ids'].shape)
        bs, seq_len = batch['input_ids_forward'].shape
        # mod_pos_weight = pos_weight.repeat(bs, seq_len, 1)
        # for i in range(mod_pos_weight.shape[1]-1):
        #     curr_pos_weight = mod_pos_weight[:, i, :]
        #     curr_labels = batch['letter_level_labels'][:, i, :]
        #     curr_labels_next = batch['letter_level_labels'][:, i+1, :]
        #     if -100 not in curr_labels:
        #         if torch.all(curr_labels == torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])) and torch.all(curr_labels_next == torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])) or torch.all(curr_labels_next == torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])) and torch.all(curr_labels == torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])):
        #             mod_pos_weight[:, i, :] = curr_pos_weight + 10000
        #         else:
        #             continue
        #     else:
        #         continue
        return {
            'input_ids_forward': batch['input_ids_forward'],
            'token_type_ids_forward': batch['token_type_ids_forward'],
            'attention_mask_forward': batch['attention_mask_forward'],
            'input_ids_backward': batch['input_ids_backward'],
            'token_type_ids_backward': batch['token_type_ids_backward'],
            'attention_mask_backward': batch['attention_mask_backward'],
            'labels': batch['labels_ohe'],
            'labels_mask_forward': batch['labels_mask_forward'],
            'labels_mask_backward': batch['labels_mask_backward'],
            'pos_weight': pos_weight.repeat(bs, seq_len, 1), ################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            'letter_level_tokens': batch['letter_level_tokens'], 
            'letter_level_labels': batch['letter_level_labels'],
            'letter_level_labels_mask': batch['letter_level_labels_mask'],
            'embedding_repeater_forward': batch['embedding_repeater_forward'],
            'embedding_repeater_backward': batch['embedding_repeater_backward'],
            'letter_level_attention_mask' : batch['letter_level_attention_mask'],
            'letter_level_token_types_ids': batch['letter_level_token_types_ids']
        }

    def keep_for_metrics_fn(batch, output): # metrics data
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data['predictions'] = output['logits'].detach().cpu()
        data['letter_level_labels_mask'] = batch['letter_level_labels_mask'].detach().cpu()
        data['letter_level_labels'] = batch['letter_level_labels'].detach().cpu()
        return data

    def find_segments_ones(array):
        ones_idx = np.where(array == 1)[0]
        if len(ones_idx) == 0:
            return []
    
        split_idx = np.where(np.diff(ones_idx) > 1)[0] + 1
    
        split_ones_idx = np.split(ones_idx, split_idx)
        segments = [(segment[0], segment[-1] + 1) for segment in split_ones_idx]
    
        return segments

    def exon_level(threshold, y_labels, p_labels, metrics):     
        """
        Update metrics with chosen threshold
        """
        y_labels_segments = find_segments_ones(np.where(y_labels >= threshold, 1, 0))    
        p_labels_segments = find_segments_ones(np.where(p_labels >= threshold, 1, 0))
    
    
        y_exons_set = set(sorted(y_labels_segments))
        p_exons_set = set(sorted(p_labels_segments))
    
        assert metrics[f'TP_{threshold}'] == 0
        assert metrics[f'FP_{threshold}'] == 0
        assert metrics[f'FN_{threshold}'] == 0
        
        metrics[f'TP_{threshold}'] += len(y_exons_set & p_exons_set)
        metrics[f'FP_{threshold}'] += len(p_exons_set - y_exons_set)
        metrics[f'FN_{threshold}'] += len(y_exons_set - p_exons_set)

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        print(data['predictions'].shape, data['letter_level_labels'].shape, data['letter_level_labels_mask'].shape) #?????????????????????????????????????????????????????????????????????????????????????????????????????????????
        all_pred = []
        all_y = []
        for i in range(data['predictions'].shape[0]):
            all_pred.append(data['predictions'][i][data['letter_level_labels_mask'][i], :])
            all_y.append(data['letter_level_labels'][i][data['letter_level_labels_mask'][i], :])
            
        all_pred = torch.sigmoid(torch.cat(all_pred, axis=0))
        all_y = torch.cat(all_y, axis=0)
        
        # print(all_pred.shape, all_y.shape)
        
        for label in [0, 1, 2, 3, 4]:
            y_label = all_y[:, label]
            p_label = all_pred[:, label]
            
            metrics[f'mcc_05_{label}'] = matthews_corrcoef(y_label.long(), (p_label > 0.5).long())
            metrics[f'pearson_{label}'] = stats.pearsonr(y_label, p_label)[0]
            
            # print(y_label, p_label)
            
            if not np.isnan(p_label).any():
                pr_auc = average_precision_score(y_label, p_label, pos_label=1)
            else:
                pr_auc = np.nan
            # to be compatible with sklearn 1.1+
            metrics[f'pr_auc_{label}'] = pr_auc if not np.isnan(pr_auc) else 0.0
            
        metrics['pr_auc_mean'] = (metrics['pr_auc_1'] + metrics['pr_auc_2'] + metrics['pr_auc_0'] + metrics['pr_auc_3'] + metrics['pr_auc_4']) / 5
        metrics['pr_auc_exon_cds'] = (metrics['pr_auc_1'] + metrics['pr_auc_4']) / 2

        exon_level_data = {}

        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            exon_level_data[f'TP_{threshold}'] = 0
            exon_level_data[f'FP_{threshold}'] = 0
            exon_level_data[f'FN_{threshold}'] = 0

        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            exon_level(threshold, all_y[:, 1], all_pred[:, 1], exon_level_data)

        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            if exon_level_data[f'TP_{threshold}'] == 0 and exon_level_data[f'FN_{threshold}'] == 0:
                recall = 0
            else:
                recall = exon_level_data[f'TP_{threshold}'] / (exon_level_data[f'TP_{threshold}'] + exon_level_data[f'FN_{threshold}'])
                
            if exon_level_data[f'TP_{threshold}'] == 0 and exon_level_data[f'FP_{threshold}'] == 0:
                precision = 0
            else:
                precision = exon_level_data[f'TP_{threshold}'] / (exon_level_data[f'TP_{threshold}'] + exon_level_data[f'FP_{threshold}'])    
        
            if precision == 0 and recall == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision) 

            metrics[f'f1_exon_level_{threshold}'] = f1
            metrics[f'precision_exon_level_{threshold}'] = precision
            metrics[f'recall_exon_level_{threshold}'] = recall

        metrics['max_f1_exon_level'] = np.max([metrics[f'f1_exon_level_{threshold}'] for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]])
        
            
#         y = data['labels'][data['labels_mask'] == 1.0]

#         # make list of segments (batches x segments -> segments x batches)
#         data['labels_segm'] = list(zip_longest(*data['labels_segm']))
#         # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$1')
#         data['predictions_segm'] = list(zip_longest(*data['predictions_segm']))
#         # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$2')
#         data['rmt_logits_masks_segm'] = list(zip_longest(*data['rmt_logits_masks_segm']))
#         # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$3')

#         # collecting labels and prediction for rmt from each segment
#         y_rmt, p_rmt = [], []
#         for i in range(len(data['labels_segm'])):
#             # print('LOLOLOLOLOLOLOLOLOLOLOLO')
#             # filter None segments (None comes from shorter examples split on less segments)
#             data['labels_segm'][i] = list(filter(lambda x: x is not None, data['labels_segm'][i]))
#             data['predictions_segm'][i] = list(filter(lambda x: x is not None, data['predictions_segm'][i]))
#             data['rmt_logits_masks_segm'][i] = list(filter(lambda x: x is not None, data['rmt_logits_masks_segm'][i]))
#             # and cat
#             data['labels_segm'][i] = torch.cat(data['labels_segm'][i])
#             data['predictions_segm'][i] = torch.cat(data['predictions_segm'][i])
#             data['rmt_logits_masks_segm'][i] = torch.cat(data['rmt_logits_masks_segm'][i])

#             y_segm, p_segm = data['labels_segm'][i], data['predictions_segm'][i]
#             y_segm = y_segm[data['rmt_logits_masks_segm'][i] == 1.0]
#             p_segm = torch.sigmoid(p_segm[data['rmt_logits_masks_segm'][i] == 1.0])
#             y_rmt += [y_segm]
#             p_rmt += [p_segm]

#         y_rmt = torch.cat(y_rmt)
#         p_rmt = torch.cat(p_rmt)
#         # compute pr-auc for each class independetly
#         if y_rmt.shape != p_rmt.shape:
#             raise RuntimeError(f'y_rmt.shape != p_rmt.shape: {y_rmt.shape} != {p_rmt.shape}')
#         for label in [0, 1, 2, 3, 4, 5]:
#             # print(y_rmt)
#             y_label = y_rmt[:, label]
#             p_label = p_rmt[:, label]
#             if y_label.sum() != y[:, label].sum():
#                 raise RuntimeError(f'y_label.sum() != y[:, label].sum(): {y_label.sum()} != {y[:, label].sum()}'
#                                    f'for label {label}, it means that original labels and rmt labels are not the same!')
#             if not np.isnan(p_label).any():
#                 pr_auc = average_precision_score(y_label, p_label, pos_label=1)
#             else:
#                 pr_auc = np.nan
#             # to be compatible with sklearn 1.1+
#             metrics[f'pr_auc_{label}'] = pr_auc if not np.isnan(pr_auc) else 0.0
        # metrics['pr_auc_mean'] = 0.8 # (metrics['pr_auc_1'] + metrics['pr_auc_2'] + metrics['pr_auc_0'] + metrics['pr_auc_3'] + metrics['pr_auc_4'] + metrics['pr_auc_5']) / 6
        return metrics

    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    
    model, optimizer = accelerator.prepare(model, optimizer)

    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, valid_dataloader=valid_dataloader,
                      train_sampler=train_sampler, batch_transform_fn=batch_transform_fn,
                      batch_metrics_fn=batch_metrics_fn, keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn)
    
    # train loop
    accelerator.wait_for_everyone()
    trainer.train()
    # make sure all workers are done
    # hvd.barrier()
    # run validation after training
    if args.save_best:
        best_model_path = str(Path(args.model_path) / 'model_best.pth')
        logger.info(f'Loading best saved model from {best_model_path}')
        trainer.load(best_model_path)

    if args.valid_data_path:
        logger.info('Runnning validation on valid data:')
        trainer.validate(valid_dataloader, write_tb=False)

    if args.test_data_path:
        # get test dataset
        logger.info(f'preparing test data from: {args.test_data_path}')
        test_data_path = Path(args.test_data_path).expanduser().absolute()
        test_dataset = AnnotationDataset(test_data_path, max_seq_len=args.input_seq_len, tokenizer=tokenizer, max_atcg_seq_len=args.letter_level_input_seq_len)
        test_sampler = DistributedSampler(test_dataset, rank=accelerator.process_index, num_replicas=accelerator.num_processes, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler, **kwargs)
        logger.info(f'len(test_dataset): {len(test_dataset)}')
        logger.info('Runnning validation on test data:')
        trainer.validate(test_dataloader, split='test', write_tb=True)

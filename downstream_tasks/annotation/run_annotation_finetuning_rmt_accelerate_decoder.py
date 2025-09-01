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

from downstream_tasks.annotation.transcript_GenePredictionDataset_decoder import AnnotationDataset

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
parser.add_argument('--decoder_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
#parser.add_argument('--sub_model_cfg', type=str, help='path to model configuration file (default: None)')

# rmt args
parser.add_argument('--backbone_cls', type=str, default=None,
                    help='backbone class name to use for RMT')
parser.add_argument('--decoder_cls', type=str, default=None,
                   help='backbone class name to use for RMT')
parser.add_argument('--backbone_trainable', action='store_true', default=False,
                    help='make all model weights trainable, not only task-specific head.')
parser.add_argument('--encoder_checkpoint', type=str,
                    help='pre-trained backbone checkpoint (default: None).')
#parser.add_argument('--submodel_checkpoint', type=str,
                    # help='pre-trained backbone checkpoint (default: None).')
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--decoder_input_size', type=int, default=None, help='maximal input size of the sub model model')
parser.add_argument('--decoder_look_back_size', type=int, default=None, help='look back size of the sub model model')

parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])

parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')

#parser.add_argument('--num_trainable_sub_model_segments', type=int, default=35, help='default: 35')
parser.add_argument('--full_checkpoint', type=str,
                    help='full checkpoint (default: None).')

# val_counter = 0

if __name__ == '__main__':
    args = parser.parse_args()
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
    train_dataset = AnnotationDataset(data_path, max_seq_len=args.input_seq_len, tokenizer=tokenizer, max_atcg_seq_len=args.letter_level_input_seq_len, shuffle_starts_in_intergenic=True)
    logger.info(f'len(train_dataset): {len(train_dataset)}')
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=accelerator.process_index, num_replicas=accelerator.num_processes, shuffle=True,
                                       drop_last=False, seed=args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler, **kwargs)

    if args.valid_data_path:
        logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_dataset = AnnotationDataset(valid_data_path, max_seq_len=args.input_seq_len, tokenizer=tokenizer, max_atcg_seq_len=args.letter_level_input_seq_len, tmp_valid_len=1000, shuffle_starts_in_intergenic=False) # tmp_valid_len=1000
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
    
    logger.info(f'Using encoder model class: {model_cls}')
    model = model_cls(config=model_cfg)

    # define decoder
    model_cfg = AutoConfig.from_pretrained(args.decoder_cfg)
    print(model_cfg.pad_token_id)
    # labels: 0, 1, 2; multi-class multi-label classification
    model_cls = get_cls_by_name(args.decoder_cls)
    
    logger.info(f'Using decoder model class: {model_cls}')
    decoder = model_cls(config=model_cfg)

    # Aydar # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        rmt_config = {
            'num_mem_tokens': args.num_mem_tokens,
            'max_n_segments': args.max_n_segments,
            # 'segment_ordering': args.segment_ordering,
            'input_size': args.input_size,
            'decoder_chunk_size': args.decoder_input_size,
            'decoder_look_back_size': args.decoder_look_back_size,
            'bptt_depth': args.bptt_depth,
            'sum_loss': True,
            'tokenizer': tokenizer
        }
        rmt_cls = get_cls_by_name(args.model_cls)
        logger.info(f'Wrapping in: {rmt_cls}')
        model = rmt_cls(model, decoder, **rmt_config)

        if args.encoder_checkpoint is not None:
            logger.info(f'loading encoder from {args.encoder_checkpoint}')
            checkpoint = torch.load(args.encoder_checkpoint, map_location='cpu')
            checkpoint_new = dict()
            for k, v in checkpoint.items():
                if 'model' in k:
                    checkpoint_new[k[6:]] = v
                else:
                    checkpoint_new[k] = v
    
            missing_k, unexpected_k = model.model.load_state_dict(checkpoint_new, strict=False)
            if len(missing_k) != 0:
                logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized. Encoder.')
            if len(unexpected_k) != 0:
                logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them! Encoder.')
                    
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

    if not args.backbone_trainable:
        print('FREEZING WEIGHTS')
        # for name, param in model.named_parameters():
        #     if 'classifier' not in name:
        #         logger.info(f'{name} is frozen')
        #         param.requires_grad = False
        for param in model.model.parameters():
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
    # pos_weight = torch.tensor([47.8, 47.8, 47.8, 10000.0, 1.1, 2.0, 1.1, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 10000.0, 10000.0, 20.0, 20.0, 20.0, 20.0, 8000.0, 8000.0, 10000.0, 10000.0, 1.0, 1.0]) # torch.tensor([128.0, 11.8, 1.0, 33.6, 27.4, 11.7]) # torch.tensor([128.0, 11.8, 1.0, 33.6, 27.4, 11.7]) # torch.tensor([2.0, 0.05, 0.5])

    def batch_transform_fn(batch):
        # print('AAAAAAAAAAAAAAA', batch['input_ids'].shape)
        bs, seq_len = batch['input_ids'].shape
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
            'input_ids': batch['input_ids'],
            'token_type_ids': batch['token_type_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels_ohe'],
            'labels_mask': batch['labels_mask'],
            # 'pos_weight': pos_weight, #.repeat(bs, seq_len, 1), ################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            'letter_level_tokens': batch['letter_level_tokens'], 
            'letter_level_labels': batch['letter_level_labels'],
            'letter_level_labels_mask': batch['letter_level_labels_mask'],
            'embedding_repeater': batch['embedding_repeater'],
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

    def exon_level(y_labels, p_labels, metrics):     
        """
        Update metrics with chosen threshold
        """
        y_labels_segments = find_segments_ones(y_labels)    
        p_labels_segments = find_segments_ones(p_labels)
    
    
        y_exons_set = set(sorted(y_labels_segments))
        p_exons_set = set(sorted(p_labels_segments))
    
        assert metrics[f'TP'] == 0
        assert metrics[f'FP'] == 0
        assert metrics[f'FN'] == 0
        
        metrics[f'TP'] += len(y_exons_set & p_exons_set)
        metrics[f'FP'] += len(p_exons_set - y_exons_set)
        metrics[f'FN'] += len(y_exons_set - p_exons_set)
    

    def metrics_fn(data):
        # global val_counter
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        # print(data['letter_level_labels'])
        print(data['predictions'].shape, data['letter_level_labels'].shape, data['letter_level_labels_mask'].shape) #?????????????????????????????????????????????????????????????????????????????????????????????????????????????
        all_pred = []
        all_y = []
        for i in range(data['predictions'].shape[0]):
            lm = data['letter_level_labels_mask'][i]
            tgt = data['letter_level_labels'][i][lm]
            prd = data['predictions'][i][lm, :]
            assert len(tgt) == len(prd)
            all_pred.append(prd)
            all_y.append(tgt)

        # print(torch.cat(all_pred, axis=0).shape)
        all_pred = torch.argmax(torch.softmax(torch.cat(all_pred, axis=0), dim=1), dim=-1)
        all_y = torch.cat(all_y, axis=0)

        assert 24 not in all_y

        assert len(all_y) == len(all_pred)

        # print('all_y', all_y)

        # print(all_pred.shape, all_y.shape)
        # assert False

        label_dict = {0: 'CDS-0', 1:'CDS-1', 2:'CDS-2', 3:'CDS-skip', 4:'intron-0', 5:'intron-1', 6:'intron-2', 7:'ASS-0', 8:'ASS-1', 9:'ASS-2', 10:'DSS-0', 11:'DSS-1', 12:'DSS-2', 13:'START', 14:'STOP', 15:'nc_exon_plus', 16:'nc_exon_minus', 17:'nc_intron_plus', 18:'nc_intron_minus', 19:'nc_ASS', 20:'nc_DSS', 21:'TSS' , 22:'PolyA', 23:'IR'}

        metrics['pr_auc_mean'] = 0
        metrics['f1_mean'] = 0
        # metrics['val_counter'] = val_counter
        # val_counter += 1
        for label in range(24):
            # print(y_rmt)
            y_label = (all_y == label).type(torch.int32)
            p_label = (all_pred == label).type(torch.int32)

            # print(y_label, p_label)
            # assert False
            
            if not np.isnan(p_label).any():
                pr_auc = average_precision_score(y_label, p_label, pos_label=1)
            else:
                pr_auc = np.nan
            # to be compatible with sklearn 1.1+
            metrics[f'pr_auc_{label_dict[label]}'] = pr_auc if not np.isnan(pr_auc) else 0.0

            metrics['pr_auc_mean'] += metrics[f'pr_auc_{label_dict[label]}']
        
            exon_level_data = {}
    
            exon_level_data[f'TP'] = 0
            exon_level_data[f'FP'] = 0
            exon_level_data[f'FN'] = 0
    
            exon_level(y_label, p_label, exon_level_data)
    
            if exon_level_data[f'TP'] == 0 and exon_level_data[f'FN'] == 0:
                recall = 0
            else:
                recall = exon_level_data[f'TP'] / (exon_level_data[f'TP'] + exon_level_data[f'FN'])
                
            if exon_level_data[f'TP'] == 0 and exon_level_data[f'FP'] == 0:
                precision = 0
            else:
                precision = exon_level_data[f'TP'] / (exon_level_data[f'TP'] + exon_level_data[f'FP'])    
        
            if precision == 0 and recall == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision) 
    
            metrics[f'f1_{label_dict[label]}_level'] = f1
            metrics['f1_mean'] += metrics[f'f1_{label_dict[label]}_level']
            metrics[f'precision_{label_dict[label]}_level'] = precision
            metrics[f'recall_{label_dict[label]}_level'] = recall

        metrics['pr_auc_mean'] /= len(metrics) / 4
        metrics['f1_mean'] /= len(metrics) / 4
        
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

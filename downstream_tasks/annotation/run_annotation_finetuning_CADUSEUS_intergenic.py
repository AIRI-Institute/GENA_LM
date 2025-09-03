import json
import logging
import os
from pathlib import Path
from itertools import zip_longest
import psutil
from safetensors.torch import load_file

from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModel
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from scipy import stats
import numpy as np

import accelerate
from accelerate import DistributedDataParallelKwargs

# from transformers import Trainer, TrainingArguments

from lm_experiments_tools import TrainerAccelerate as Trainer
from lm_experiments_tools import TrainerAccelerateArgs as TrainerArgs

from lm_experiments_tools.utils import prepare_run, get_cls_by_name, get_optimizer
from huggingface_hub import hf_hub_download

load_dotenv()
import sys
sys.path.insert(0, os.getcwd())

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')

from lm_experiments_tools import get_optimizer
from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration, get_git_diff
import lm_experiments_tools.optimizers as optimizers

from downstream_tasks.annotation.transcript_GenePredictionDataset_letter_level_CADUSEUS_intergenic import AnnotationDataset

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
parser.add_argument('--letter_level_input_seq_len', type=int, default=64, help='input sequnce length (default: 64).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')
parser.add_argument('--targets_offset', type=int, default=5000, help='default: 5000')
parser.add_argument('--targets_len', type=int, default=5000, help='default: 5000')

parser.add_argument('--checkpoint', type=str, default=None,
                    help='path to saved checkpoint')

parser.add_argument('--backbone_cls', type=str, default=None,
                    help='backbone class name to use')

parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')
parser.add_argument('--backbone_trainable', action='store_true', default=False,
                    help='make all model weights trainable, not only task-specific head.')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')

#parser.add_argument('--num_trainable_sub_model_segments', type=int, default=35, help='default: 35')
parser.add_argument('--full_checkpoint', type=str,
                    help='full checkpoint (default: None).')

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
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer_caduseus = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_cls = get_cls_by_name(args.backbone_cls)
    
    logger.info(f'Using model class: {model_cls}')
    config_caduseus = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config_caduseus.bidirectional_weight_tie = False
    model_caduseus = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config_caduseus)
    model = model_cls(model_caduseus)

    print('ALL WEIGHTS', sum(p.sum() for p in model_caduseus.parameters()))
    
    if args.checkpoint is not None:
        if config_caduseus.bidirectional_weight_tie:
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            state_dict_new = dict()
            for k, v in state_dict.items():
                if 'module' in k:
                    state_dict_new[k[7:]] = v
                else:
                    state_dict_new[k] = v
            state_dict = state_dict_new
        else:
            state_dict = load_file(args.checkpoint)
        # print('\n', state_dict.keys(), '\n')
        print('Loading weights from checkpoint')
        missing_k, unexpected_k = model.load_state_dict(state_dict, strict=False)
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized. CADUSEUS.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them! CADUSEUS.')
        

    # print(model)

    if not args.backbone_trainable:
        print('FREEZING WEIGHTS')
        for param in model.caduseus_model.parameters():
            param.requires_grad = False





    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    # global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    # get train dataset
    logger.info(f'preparing training data from: {args.data_path}')
    data_path = hf_hub_download(repo_id="shmelev/encode4_full_human", filename=args.data_path, repo_type="dataset") # Path(args.data_path).expanduser().absolute()
    train_dataset = AnnotationDataset(data_path, tokenizer=tokenizer, max_atcg_seq_len=args.letter_level_input_seq_len, tokenizer_caduseus=tokenizer_caduseus)
    logger.info(f'len(train_dataset): {len(train_dataset)}')
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=accelerator.process_index, num_replicas=accelerator.num_processes, shuffle=True,
                                       drop_last=False, seed=args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, persistent_workers=True, sampler=train_sampler, **kwargs)

    if args.valid_data_path:
        logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = hf_hub_download(repo_id="shmelev/encode4_full_human", filename=args.valid_data_path, repo_type="dataset") # Path(args.valid_data_path).expanduser().absolute()
        valid_dataset = AnnotationDataset(valid_data_path, tokenizer=tokenizer, max_atcg_seq_len=args.letter_level_input_seq_len, tokenizer_caduseus=tokenizer_caduseus, tmp_valid_len=1000)
        valid_sampler = DistributedSampler(valid_dataset, rank=accelerator.process_index, num_replicas=accelerator.num_processes, shuffle=True, drop_last=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler, **kwargs)
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
        logger.info(f'len(valid_dataset): {len(valid_dataset)}')
    else:
        valid_dataloader = None
        logger.info('No validation data is used.')

    
    

    

    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    print(args.lr)
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
    # pos_weight = torch.tensor([1.0]*5) # torch.tensor([128.0, 11.8, 1.0, 33.6, 27.4, 11.7]) # torch.tensor([128.0, 11.8, 1.0, 33.6, 27.4, 11.7]) # torch.tensor([2.0, 0.05, 0.5])

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
            'letter_level_labels': batch['letter_level_labels'],
            'letter_level_labels_mask': batch['letter_level_labels_mask'],
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
        y_labels_segments = find_segments_ones(np.where(y_labels >= 0.5, 1, 0))    
        p_labels_segments = find_segments_ones(np.where(p_labels >= 0.5, 1, 0))
    
    
        y_exons_set = set(sorted(y_labels_segments))
        p_exons_set = set(sorted(p_labels_segments))
    
        assert metrics[f'TP'] == 0
        assert metrics[f'FP'] == 0
        assert metrics[f'FN'] == 0
        
        metrics[f'TP'] += len(y_exons_set & p_exons_set)
        metrics[f'FP'] += len(p_exons_set - y_exons_set)
        metrics[f'FN'] += len(y_exons_set - p_exons_set)


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
        
        # label_dict = {0: 'CDS-0', 1:'CDS-1', 2:'CDS-2', 3:'CDS-skip', 4:'intron-0', 5:'intron-1', 6:'intron-2', 7:'ASS-0', 8:'ASS-1', 9:'ASS-2', 10:'DSS-0', 11:'DSS-1', 12:'DSS-2', 13:'START', 14:'STOP', 15:'nc_exon_plus', 16:'nc_exon_minus', 17:'nc_intron_plus', 18:'nc_intron_minus', 19:'nc_ASS', 20:'nc_DSS', 21:'TSS' , 22:'PolyA', 23:'IR'}

        label_dict = {0:'TSS_+', 1:'TSS_-', 2:'PolyA_+', 3:'PolyA_-'} #???


        metrics['pr_auc_mean'] = 0
        # metrics['f1_mean'] = 0
        # metrics['val_counter'] = val_counter
        # val_counter += 1
        for label in range(4):
            # print(y_rmt)
            y_label = all_y[:, label]
            p_label = all_pred[:, label]

            # print(y_label, p_label)
            # assert False
            
            if not np.isnan(p_label).any():
                pr_auc = average_precision_score(y_label, p_label, pos_label=1)
            else:
                pr_auc = np.nan
            # to be compatible with sklearn 1.1+
            metrics[f'pr_auc_{label_dict[label]}'] = pr_auc if not np.isnan(pr_auc) else 0.0

            metrics['pr_auc_mean'] += metrics[f'pr_auc_{label_dict[label]}']
        
            # exon_level_data = {}
    
            # exon_level_data[f'TP'] = 0
            # exon_level_data[f'FP'] = 0
            # exon_level_data[f'FN'] = 0
    
            # exon_level(y_label, p_label, exon_level_data)
    
            # if exon_level_data[f'TP'] == 0 and exon_level_data[f'FN'] == 0:
            #     recall = 0
            # else:
            #     recall = exon_level_data[f'TP'] / (exon_level_data[f'TP'] + exon_level_data[f'FN'])
                
            # if exon_level_data[f'TP'] == 0 and exon_level_data[f'FP'] == 0:
            #     precision = 0
            # else:
            #     precision = exon_level_data[f'TP'] / (exon_level_data[f'TP'] + exon_level_data[f'FP'])    
        
            # if precision == 0 and recall == 0:
            #     f1 = 0
            # else:
            #     f1 = 2 * recall * precision / (recall + precision) 
    
            # metrics[f'f1_{label_dict[label]}_level'] = f1
            # metrics['f1_mean'] += metrics[f'f1_{label_dict[label]}_level']
            # metrics[f'precision_{label_dict[label]}_level'] = precision
            # metrics[f'recall_{label_dict[label]}_level'] = recall

        metrics['pr_auc_mean'] /= 4
        # metrics['pr_auc_tss_polya'] = (metrics[f'pr_auc_TSS'] + metrics[f'pr_auc_PolyA']) / 2
        # metrics['f1_mean'] /= 24
        
        return metrics
    

    # def metrics_fn(data):
    #     # compute metrics based on stored labels, predictions, ...
    #     metrics = {}
    #     print(data['predictions'].shape, data['letter_level_labels'].shape, data['letter_level_labels_mask'].shape) #?????????????????????????????????????????????????????????????????????????????????????????????????????????????
    #     all_pred = []
    #     all_y = []
    #     for i in range(data['predictions'].shape[0]):
    #         all_pred.append(data['predictions'][i][data['letter_level_labels_mask'][i], :])
    #         all_y.append(data['letter_level_labels'][i][data['letter_level_labels_mask'][i], :])
            
    #     all_pred = torch.argmax(torch.sigmoid(torch.cat(all_pred, axis=0)), dim=-1)
    #     all_y = torch.argmax(torch.cat(all_y, axis=0), dim=-1)
        
    #     label_dict = {0: 'CDS-0', 1:'CDS-1', 2:'CDS-2', 3:'CDS-skip', 4:'intron-0', 5:'intron-1', 6:'intron-2', 7:'ASS-0', 8:'ASS-1', 9:'ASS-2', 10:'DSS-0', 11:'DSS-1', 12:'DSS-2', 13:'START', 14:'STOP', 15:'nc_exon_plus', 16:'nc_exon_minus', 17:'nc_intron_plus', 18:'nc_intron_minus', 19:'nc_ASS', 20:'nc_DSS', 21:'TSS' , 22:'PolyA', 23:'IR'}

    #     metrics['pr_auc_mean'] = 0
    #     metrics['f1_mean'] = 0
    #     # metrics['val_counter'] = val_counter
    #     # val_counter += 1
    #     for label in range(24):
    #         # print(y_rmt)
    #         y_label = (all_y == label).type(torch.int32)
    #         p_label = (all_pred == label).type(torch.int32)

    #         # print(y_label, p_label)
    #         # assert False
            
    #         # if not np.isnan(p_label).any():
    #         #     pr_auc = average_precision_score(y_label, p_label, pos_label=1)
    #         # else:
    #         #     pr_auc = np.nan
    #         # # to be compatible with sklearn 1.1+
    #         # metrics[f'pr_auc_{label_dict[label]}'] = pr_auc if not np.isnan(pr_auc) else 0.0

    #         # metrics['pr_auc_mean'] += metrics[f'pr_auc_{label_dict[label]}']
        
    #         exon_level_data = {}
    
    #         exon_level_data[f'TP'] = 0
    #         exon_level_data[f'FP'] = 0
    #         exon_level_data[f'FN'] = 0
    
    #         exon_level(y_label, p_label, exon_level_data)
    
    #         if exon_level_data[f'TP'] == 0 and exon_level_data[f'FN'] == 0:
    #             recall = 0
    #         else:
    #             recall = exon_level_data[f'TP'] / (exon_level_data[f'TP'] + exon_level_data[f'FN'])
                
    #         if exon_level_data[f'TP'] == 0 and exon_level_data[f'FP'] == 0:
    #             precision = 0
    #         else:
    #             precision = exon_level_data[f'TP'] / (exon_level_data[f'TP'] + exon_level_data[f'FP'])    
        
    #         if precision == 0 and recall == 0:
    #             f1 = 0
    #         else:
    #             f1 = 2 * recall * precision / (recall + precision) 
    
    #         metrics[f'f1_{label_dict[label]}_level'] = f1
    #         metrics['f1_mean'] += metrics[f'f1_{label_dict[label]}_level']
    #         metrics[f'precision_{label_dict[label]}_level'] = precision
    #         metrics[f'recall_{label_dict[label]}_level'] = recall

    #     metrics['pr_auc_mean'] /= 24
    #     # metrics['f1_mean'] /= 24
        
    #     return metrics

    # def find_segments_ones(array):
    #     ones_idx = np.where(array == 1)[0]
    #     if len(ones_idx) == 0:
    #         return []
    
    #     split_idx = np.where(np.diff(ones_idx) > 1)[0] + 1
    
    #     split_ones_idx = np.split(ones_idx, split_idx)
    #     segments = [(segment[0], segment[-1] + 1) for segment in split_ones_idx]
    
    #     return segments

    # def exon_level(y_labels, p_labels, metrics):     
    #     """
    #     Update metrics with chosen threshold
    #     """
    #     y_labels_segments = find_segments_ones(np.where(y_labels >= 0.5, 1, 0))    
    #     p_labels_segments = find_segments_ones(np.where(p_labels >= 0.5, 1, 0))
    
    
    #     y_exons_set = set(sorted(y_labels_segments))
    #     p_exons_set = set(sorted(p_labels_segments))
    
    #     assert metrics[f'TP'] == 0
    #     assert metrics[f'FP'] == 0
    #     assert metrics[f'FN'] == 0
        
    #     metrics[f'TP'] += len(y_exons_set & p_exons_set)
    #     metrics[f'FP'] += len(p_exons_set - y_exons_set)
    #     metrics[f'FN'] += len(y_exons_set - p_exons_set)
    

    # def metrics_fn(data):
    #     # compute metrics based on stored labels, predictions, ...
    #     metrics = {}
    #     # print(data['letter_level_labels'])
    #     print(data['predictions'].shape, data['letter_level_labels'].shape, data['letter_level_labels_mask'].shape) #?????????????????????????????????????????????????????????????????????????????????????????????????????????????
    #     all_pred = []
    #     all_y = []
    #     for i in range(data['predictions'].shape[0]):
    #         lm = data['letter_level_labels_mask'][i]
    #         tgt = data['letter_level_labels'][i][lm, :]
    #         prd = data['predictions'][i][lm, :]
    #         assert len(tgt) == len(prd)
    #         all_pred.append(prd)
    #         all_y.append(tgt)

    #     # print(torch.cat(all_pred, axis=0).shape)
    #     all_pred = torch.cat(all_pred, axis=0)
    #     all_y = torch.cat(all_y, axis=0)

    #     # print('all_y', all_y)

    #     print(all_pred.shape, all_y.shape)
    #     # assert False

    #     label_dict = {0: 'CDS-0', 1:'CDS-1', 2:'CDS-2', 3:'CDS-skip', 4:'intron-0', 5:'intron-1', 6:'intron-2', 7:'ASS-0', 8:'ASS-1', 9:'ASS-2', 10:'DSS-0', 11:'DSS-1', 12:'DSS-2', 13:'START', 14:'STOP', 15:'nc_exon_plus', 16:'nc_exon_minu', 17:'nc_intron_plus', 18:'nc_intron_minu', 19:'nc_ASS', 20:'nc_DSS', 21:'TSS' , 22:'PolyA', 23:'IR'}

    #     metrics['pr_auc_mean'] = 0
    #     for label in range(24):
    #         # print(y_rmt)
    #         y_label = all_y[:, label]
    #         p_label = all_pred[:, label]

    #         # print(y_label, p_label)
    #         # assert False
            
    #         if not np.isnan(p_label).any():
    #             pr_auc = average_precision_score(y_label, p_label, pos_label=1)
    #         else:
    #             pr_auc = np.nan
    #         # to be compatible with sklearn 1.1+
    #         metrics[f'pr_auc_{label_dict[label]}'] = pr_auc if not np.isnan(pr_auc) else 0.0

    #         metrics['pr_auc_mean'] += metrics[f'pr_auc_{label_dict[label]}']
        
    #         exon_level_data = {}
    
    #         exon_level_data[f'TP'] = 0
    #         exon_level_data[f'FP'] = 0
    #         exon_level_data[f'FN'] = 0
    
    #         exon_level(y_label, p_label, exon_level_data)
    
    #         if exon_level_data[f'TP'] == 0 and exon_level_data[f'FN'] == 0:
    #             recall = 0
    #         else:
    #             recall = exon_level_data[f'TP'] / (exon_level_data[f'TP'] + exon_level_data[f'FN'])
                
    #         if exon_level_data[f'TP'] == 0 and exon_level_data[f'FP'] == 0:
    #             precision = 0
    #         else:
    #             precision = exon_level_data[f'TP'] / (exon_level_data[f'TP'] + exon_level_data[f'FP'])    
        
    #         if precision == 0 and recall == 0:
    #             f1 = 0
    #         else:
    #             f1 = 2 * recall * precision / (recall + precision) 
    
    #         metrics[f'f1_{label_dict[label]}_level'] = f1
    #         metrics[f'precision_{label_dict[label]}_level'] = precision
    #         metrics[f'recall_{label_dict[label]}_level'] = recall

    #     metrics['pr_auc_mean'] /= len(metrics) / 4
        
    #     return metrics

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

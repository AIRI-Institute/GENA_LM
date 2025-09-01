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
from sklearn.metrics import average_precision_score, f1_score
import numpy as np

import accelerate
from accelerate import DistributedDataParallelKwargs

from lm_experiments_tools import TrainerAccelerate as Trainer
from lm_experiments_tools import TrainerAccelerateArgs as TrainerArgs

from lm_experiments_tools.utils import prepare_run, get_cls_by_name, get_optimizer

load_dotenv()

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')

from lm_experiments_tools import get_optimizer
from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration, get_git_diff
import lm_experiments_tools.optimizers as optimizers

from downstream_tasks.annotation.transcript_no_rmt_GenePredictionDataset import AnnotationDataset

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
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')

parser.add_argument('--initial_checkpoint', type=str,
                    help='pretrained checkpoint (default: None).')

parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')

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
    train_dataset = AnnotationDataset(data_path, max_seq_len=args.input_seq_len, tmp_valid_len=None)
    logger.info(f'len(train_dataset): {len(train_dataset)}')
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=accelerator.process_index, num_replicas=accelerator.num_processes, shuffle=True,
                                       drop_last=False, seed=args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler, **kwargs)

    if args.valid_data_path:
        logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_dataset = AnnotationDataset(valid_data_path, max_seq_len=args.input_seq_len, tmp_valid_len=None)
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
    model_cfg.num_labels = 6
    model_cfg.problem_type = 'multi_label_classification'
    model_cls = get_cls_by_name(args.model_cls)
    
    logger.info(f'Using model class: {model_cls}')
    model = model_cls(config=model_cfg)

    if args.initial_checkpoint is not None:
        logger.info(f'loading pre-trained checkpoint from {args.initial_checkpoint}')
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        if args.initial_checkpoint[-4:] == '.bin':
            missing_k, unexpected_k = model.load_state_dict(checkpoint, strict=False)
        elif args.initial_checkpoint[-4:] == '.pth':
            missing_k, unexpected_k = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise Exception('Wrong data type! Should be .bit or .pth')
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

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
    pos_weight = torch.tensor([128.0, 11.8, 1.0, 33.6, 27.4, 11.7]) ** 0.5 # torch.tensor([128.0, 11.8, 1.0, 33.6, 27.4, 11.7]) # torch.tensor([2.0, 0.05, 0.5])
    

    def batch_transform_fn(batch):
        bs, seq_len = batch['input_ids'].shape
        # mod_pos_weight = pos_weight.repeat(bs, seq_len, 1)
        # for i in range(mod_pos_weight.shape[1]):
        #     curr_pos_weight = mod_pos_weight[:, i, :]
        #     curr_labels = batch['labels_ohe'][:, i, :]
        #     if -100 not in curr_labels:
        #         if torch.all(curr_labels == torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])) or torch.all(curr_labels == torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])) or torch.all(curr_labels == torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])) or torch.all(curr_labels == torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])):
        #             mod_pos_weight[:, i, :] = curr_pos_weight + 500
        #         else:
        #             continue
        #     else:
        #         continue
        return {
            'input_ids': batch['input_ids'],
            'token_type_ids': batch['token_type_ids'].long(),
            'attention_mask': batch['attention_mask'].long(),
            'labels': batch['labels'],
            'labels_mask': batch['labels_mask'],
            'pos_weight': pos_weight.repeat(bs, seq_len, 1),
        }

    def keep_for_metrics_fn(batch, output): # metrics data
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data['labels'] = batch['labels']
        data['predictions'] = output['logits'].detach()
        data['labels_mask'] = batch['labels_mask']
        return data

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = data['labels'], torch.sigmoid(data['predictions'])
        y = y[data['labels_mask'] == 1.0]
        p = p[data['labels_mask'] == 1.0]
        for label in [0, 1, 2, 3, 4, 5]:
            # print(y_rmt)
            y_label = y[:, label]
            p_label = p[:, label]
            if y_label.sum() != y[:, label].sum():
                raise RuntimeError(f'y_label.sum() != y[:, label].sum(): {y_label.sum()} != {y[:, label].sum()}'
                                   f'for label {label}, it means that original labels and rmt labels are not the same!')
            if not np.isnan(p_label).any():
                pr_auc = average_precision_score(y_label, p_label, pos_label=1)
            else:
                pr_auc = np.nan
            # to be compatible with sklearn 1.1+
            metrics[f'pr_auc_{label}'] = pr_auc if not np.isnan(pr_auc) else 0.0
        metrics['pr_auc_mean'] = (metrics['pr_auc_1'] + metrics['pr_auc_2'] + metrics['pr_auc_0'] + metrics['pr_auc_3'] + metrics['pr_auc_4'] + metrics['pr_auc_5']) / 6
        metrics['pr_auc_exon_cds'] = (metrics['pr_auc_1'] + metrics['pr_auc_4']) / 2
        return metrics
    
    model, optimizer = accelerator.prepare(model, optimizer)

    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, valid_dataloader=valid_dataloader,
                      train_sampler=train_sampler, batch_transform_fn=batch_transform_fn,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn)
    
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
        test_dataset = AnnotationDataset(test_data_path, max_seq_len=args.input_seq_len, tmp_valid_len=None)
        test_sampler = DistributedSampler(test_dataset, rank=accelerator.process_index, num_replicas=accelerator.num_processes, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler, **kwargs)
        logger.info(f'len(test_dataset): {len(test_dataset)}')
        logger.info('Runnning validation on test data:')
        trainer.validate(test_dataloader, split='test', write_tb=True)

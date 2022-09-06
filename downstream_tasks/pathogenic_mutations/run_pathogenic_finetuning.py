import json
import logging
import os
from pathlib import Path
import gc

import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from lm_experiments_tools import Trainer, TrainerArgs, get_optimizer
from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration
import lm_experiments_tools.optimizers as optimizers

from downstream_tasks.pathogenic_mutations.PathogenicMutationsDataset import PathogenicMutationsDataset

import horovod.torch as hvd

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

hvd.init()

torch.set_num_threads(4)
torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--data_path', type=str, help='path the training data, could be a folder')
parser.add_argument('--n_splits', type=int, default=5, help='number of folds for cross-validation (default: 5)')
parser.add_argument('--seed', type=int, default=42, help='random seed')

# bert data args
parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--mid_token', type=str, default='Alt', help='token that would be put in the mid (default: Alt).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')

# tokenizer
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')


if __name__ == '__main__':
    args = parser.parse_args()
    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    if hvd.rank() == 0 and args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing training data from: {args.data_path}')
    data_path = Path(args.data_path).expanduser().absolute()
    data = pd.read_csv(args.data_path, sep='\t')
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    valid_metrics = []
    model_path = Path(args.model_path)
    for i, (train_idx, valid_idx) in tqdm(enumerate(kf.split(data, data['Label'])), total=args.n_splits,
                                          desc='Running KFold', disable=(hvd.rank() != 0)):
        # create model path and save configuration
        if args.model_path is not None:
            run_path = Path(model_path) / f'run_{i+1}'
            args.model_path = str(run_path)
            if hvd.rank() == 0:
                if not run_path.exists():
                    run_path.mkdir(parents=True)
                args_dict = collect_run_configuration(args)
                json.dump(args_dict, open(run_path/'config.json', 'w'), indent=4)

        train_dataset = PathogenicMutationsDataset(data.iloc[train_idx], tokenizer, mid_token=args.mid_token,
                                                   max_seq_len=args.input_seq_len)

        # shuffle train data each epoch (one loop over train_dataset)
        train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                           drop_last=False, seed=args.seed)

        train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler, **kwargs)

        valid_dataset = PathogenicMutationsDataset(data.iloc[valid_idx], tokenizer, mid_token=args.mid_token,
                                                   max_seq_len=args.input_seq_len)
        valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler, **kwargs)
        if args.valid_interval is None:
            args.valid_interval = args.log_interval

        # define model
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        # classification with two labels
        model_cfg.num_labels = 2
        model_cls = get_cls_by_name(args.model_cls)
        if hvd.rank() == 0:
            logger.info(f'Using model class: {model_cls}')
        model = model_cls(config=model_cfg)

        # define optimizer
        optimizer_cls = get_optimizer(args.optimizer)
        if optimizer_cls is None:
            raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

        if hvd.rank() == 0:
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

        def batch_transform_fn(batch):
            return {
                'input_ids': batch['input_ids'],
                'token_type_ids': batch['token_type_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['pathogenic']}

        def keep_for_metrics_fn(batch, output):
            # select data from batch and model output that would be used to compute metrics
            data = {}
            data['labels'] = batch['labels']
            data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
            return data

        def metrics_fn(data):
            # compute metrics based on stored labels, predictions, ...
            metrics = {}
            y, p = data['labels'], data['predictions']
            # accuracy
            metrics['accuracy'] = ((p == y).sum() / len(y)).item()
            # f1, precision, recall, mcc
            metrics['f1'] = f1_score(y, p)
            metrics['f1_macro'] = f1_score(y, p, average='macro')
            metrics['f1_weighted'] = f1_score(y, p, average='weighted')
            metrics['precision'] = precision_score(y, p)
            metrics['recall'] = recall_score(y, p)
            metrics['mcc'] = matthews_corrcoef(y, p)
            return metrics

        trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                          batch_transform_fn=batch_transform_fn, keep_for_metrics_fn=keep_for_metrics_fn,
                          metrics_fn=metrics_fn)

        # train loop
        trainer.train()
        # make sure all workers are done
        hvd.barrier()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best.pth')
            if hvd.rank() == 0:
                logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
            if hvd.rank() == 0:
                logger.info('Runnning validation on valid data:')
            valid_metrics += [trainer.validate(valid_dataloader, write_tb=False)]

        # release GPU memory for further runs
        del trainer
        gc.collect()

    if len(valid_metrics) != 0:
        if hvd.rank() == 0:
            for i, m in enumerate(valid_metrics):
                logger.info(f'fold_{i}: {m}')
            agg_metrics = {k: [m[k] for m in valid_metrics] for k in valid_metrics[0]}
            for mn in agg_metrics:
                logger.info(f'{mn}: {np.mean(agg_metrics[mn]):.5f}+-{np.std(agg_metrics[mn]):.5f}')

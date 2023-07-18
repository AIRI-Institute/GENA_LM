import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from lm_experiments_tools import Trainer, TrainerArgs, get_optimizer
from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration
from lm_experiments_tools.utils import get_git_diff
import lm_experiments_tools.optimizers as optimizers

from downstream_tasks.promoter_prediction.dataset import EPDnewPromoterDataset

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
parser.add_argument('--valid_data_path', type=str, help='path the valid data, could be a folder')
parser.add_argument('--test_data_path', type=str, help='path the test data, could be a folder')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--seed', type=int, default=42, help='random seed')

# bert data args
parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')
parser.add_argument('--pad_to_max_seq_len', action='store_true', default=False,
                    help='pad all examples to input_seq_len (default: False)')
parser.add_argument('--truncate', type=str, default='right',
                    help='truncate input sequence to input_seq_len from right|mid|left (default: right)')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')

# tokenizer
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')
parser.add_argument('--bpe_dropout', type=float, default=None, help='bpe dropout value, used during training only')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')

parser.add_argument('--body_lr_multiplier', type=float, default=1.0,
                    help='multiplier to lr to set learning rate for pre-trained body (default: 1.0)')


if __name__ == '__main__':
    args = parser.parse_args()
    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    if hvd.rank() == 0 and args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    # create model path and save configuration
    if hvd.rank() == 0 and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)
        open(model_path / 'git.diff', 'w').write(get_git_diff())

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.bpe_dropout is not None and args.bpe_dropout > 0.0:
        train_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if hasattr(train_tokenizer._tokenizer.model, 'dropout'):
            train_tokenizer._tokenizer.model.dropout = args.bpe_dropout
        elif hvd.rank() == 0:
            logger.warning('BPE dropout is not set as tokenizer does not support it.')
    else:
        train_tokenizer = tokenizer

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    if not args.pad_to_max_seq_len:
        pad_token_ids = {'input_ids': tokenizer.pad_token_id, 'token_type_ids': 0, 'attention_mask': 0}
        pad_to_divisible_by = 64

        # could be used to pad on-the-fly
        def collate_fn(batch):
            feature_keys = ['input_ids', 'token_type_ids', 'attention_mask']
            padded_batch = {k: [] for k in feature_keys}
            max_seq_len = max([len(el['input_ids']) for el in batch])
            max_seq_len += (
                (pad_to_divisible_by - max_seq_len % pad_to_divisible_by)
                if max_seq_len % pad_to_divisible_by != 0
                else 0
            )
            for k in feature_keys:
                for i, el in enumerate(batch):
                    padded_batch[k] += [
                        np.concatenate(
                            [
                                batch[i][k],
                                np.array([pad_token_ids[k]] * max(0, max_seq_len - len(el[k])), dtype=np.int64),
                            ]
                        )
                    ]
            for k in padded_batch:
                padded_batch[k] = torch.from_numpy(np.stack(padded_batch[k]))
            padded_batch['labels'] = torch.tensor([el['labels'] for el in batch])
            return padded_batch

        kwargs['collate_fn'] = collate_fn

    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing training data from: {args.data_path}')
    data_path = Path(args.data_path).expanduser().absolute()
    train_dataset = EPDnewPromoterDataset(data_path, train_tokenizer, x_field='sequence',
                                          label_field='promoter_presence', max_seq_len=args.input_seq_len,
                                          pad_to_max=args.pad_to_max_seq_len, truncate=args.truncate)

    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)

    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler, **kwargs)
    # get validation dataset
    if args.valid_data_path:
        if hvd.rank() == 0:
            logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_dataset = EPDnewPromoterDataset(valid_data_path, tokenizer, x_field='sequence',
                                              label_field='promoter_presence', max_seq_len=args.input_seq_len,
                                              pad_to_max=args.pad_to_max_seq_len, truncate=args.truncate)
        valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler, **kwargs)
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
    else:
        valid_dataloader = None
        if hvd.rank() == 0:
            logger.info('No validation data is used.')
    # get test dataset
    if args.test_data_path:
        if hvd.rank() == 0:
            logger.info(f'preparing test data from: {args.test_data_path}')
        test_data_path = Path(args.test_data_path).expanduser().absolute()
        test_dataset = EPDnewPromoterDataset(test_data_path, tokenizer, x_field='sequence',
                                             label_field='promoter_presence', max_seq_len=args.input_seq_len,
                                             pad_to_max=args.pad_to_max_seq_len, truncate=args.truncate)
        test_sampler = DistributedSampler(test_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler, **kwargs)

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
        if args.body_lr_multiplier != 1.0:
            raise RuntimeError('Adafactor optimizer and body_lr_multiplier != 1.0 is not supported')
    else:
        if args.body_lr_multiplier == 1.0:
            optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optimizer_cls(
                    [{'params': model.bert.parameters(), 'lr': args.lr * args.body_lr_multiplier},
                     {'params': model.classifier.parameters()}],
                    lr=args.lr, weight_decay=args.weight_decay)

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
        metrics['accuracy'] = (p == y).sum() / len(y)
        # f1, precision, recall, mcc
        metrics['f1'] = f1_score(y, p)
        metrics['f1_macro'] = f1_score(y, p, average='macro')
        metrics['precision'] = precision_score(y, p)
        metrics['recall'] = recall_score(y, p)
        metrics['mcc'] = matthews_corrcoef(y, p)
        return metrics

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn)

    if not args.validate_only:
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
        if args.valid_data_path:
            if hvd.rank() == 0:
                logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        if args.test_data_path:
            if hvd.rank() == 0:
                logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, split='test', write_tb=True)
    else:
        # run validation, do not write to tensorboard
        if hvd.rank() == 0:
            logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, write_tb=False)
        if args.valid_data_path:
            if hvd.rank() == 0:
                logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        if args.test_data_path:
            if hvd.rank() == 0:
                logger.info('Running validation on test data:')
            trainer.validate(test_dataloader, split='test', write_tb=False)

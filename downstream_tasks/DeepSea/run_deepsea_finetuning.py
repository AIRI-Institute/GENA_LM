import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from sklearn.metrics import roc_auc_score
import numpy as np

from lm_experiments_tools import Trainer, TrainerArgs, get_optimizer
from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration
import lm_experiments_tools.optimizers as optimizers

from downstream_tasks.DeepSea.DeepSeaDataset import DeepSeaDataset

import horovod.torch as hvd

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

hvd.init()

torch.set_num_threads(4)
torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--data_path', type=str, help='path to the training data')
parser.add_argument('--valid_data_path', type=str, help='path to the valid data')
parser.add_argument('--test_data_path', type=str, help='path to the test data (dataset_test_0.csv)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')

# data args
parser.add_argument('--input_seq_len', type=int, default=64, help='input sequnce length (default: 64).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')

parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')

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
        json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing training data from: {args.data_path}')
    data_path = Path(args.data_path).expanduser().absolute()
    train_dataset = DeepSeaDataset(data_path, tokenizer, max_seq_len=args.input_seq_len)
    if hvd.rank() == 0:
        logger.info(f'len(train_dataset): {len(train_dataset)}')
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler, **kwargs)

    if args.valid_data_path:
        if hvd.rank() == 0:
            logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_dataset = DeepSeaDataset(valid_data_path, tokenizer, max_seq_len=args.input_seq_len)
        valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler, **kwargs)
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
        if hvd.rank() == 0:
            logger.info(f'len(valid_dataset): {len(valid_dataset)}')
    else:
        valid_dataloader = None
        if hvd.rank() == 0:
            logger.info('No validation data is used.')

    # define model
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    model_cfg.num_labels = 919
    model_cfg.problem_type = 'multi_label_classification'
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

    # upweight positive examples like in BigBird paper (see App. F3)
    pos_weight = torch.tensor([8.0] * model_cfg.num_labels)

    def batch_transform_fn(batch):
        bs, _ = batch['input_ids'].shape
        return {
            'input_ids': batch['input_ids'],
            'token_type_ids': batch['token_type_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels'],
            'pos_weight': pos_weight.repeat(bs, 1),
        }

    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data['labels'] = batch['labels']
        data['predictions'] = output['logits'].detach()
        return data

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = data['labels'], torch.sigmoid(data['predictions'])
        # compute auc for each class independetly, https://github.com/jimmyyhwu/deepsea/blob/master/compute_aucs.py#L46
        aucs = np.zeros(model_cfg.num_labels, dtype=np.float32)
        for i in range(model_cfg.num_labels):
            try:
                aucs[i] = roc_auc_score(y[:, i], p[:, i])
            except ValueError:
                aucs[i] = 0.5
        metrics['TF_median_auc'] = np.median(aucs[125:125 + 690])
        metrics['DHS_median_auc'] = np.median(aucs[:125])
        metrics['HM_median_auc'] = np.median(aucs[125 + 690:125 + 690 + 104])
        metrics['mean_auc'] = (metrics['TF_median_auc'] + metrics['DHS_median_auc'] + metrics['HM_median_auc']) / 3.0
        return metrics

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader=valid_dataloader,
                      train_sampler=train_sampler, batch_transform_fn=batch_transform_fn,
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

    # if we validate after training -- we take the best ckpt for evaluation
    # if we run only validation -- validation ckpt should be specified with init_checkpoint.
    #   also, model_path could be set None to not save logs
    if args.valid_data_path:
        if hvd.rank() == 0:
            logger.info('Runnning validation on valid data:')
        trainer.validate(valid_dataloader, write_tb=False)

    if args.test_data_path:
        # get test dataset
        if hvd.rank() == 0:
            logger.info(f'preparing test data from: {args.test_data_path}')
        test_data_path = Path(args.test_data_path).expanduser().absolute()
        test_dataset = DeepSeaDataset(test_data_path, tokenizer, max_seq_len=args.input_seq_len)
        test_sampler = DistributedSampler(test_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler, **kwargs)
        if hvd.rank() == 0:
            logger.info(f'len(test_dataset): {len(test_dataset)}')
            logger.info('Runnning validation on test data:')
        trainer.validate(test_dataloader, split='test', write_tb=True)

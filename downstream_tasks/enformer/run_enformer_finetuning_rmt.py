import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
import numpy as np

from lm_experiments_tools import Trainer, TrainerArgs, get_optimizer
from lm_experiments_tools.utils import get_cls_by_name, prepare_run
import lm_experiments_tools.optimizers as optimizers

from downstream_tasks.enformer.enformer_dataset import EnformerDataset
from downstream_tasks.enformer.enformer_metrics import MeanPearsonCorrCoefPerChannel

import horovod.torch as hvd

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger()

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

# data args
parser.add_argument('--input_seq_len', type=int, default=64, help='input sequnce length (default: 64).')
parser.add_argument('--bins_per_sample', type=int, default=896, help='input sequnce length (default: 64).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')
parser.add_argument('--n_valid_samples', type=int, default=None, help='number of samples to use for validation')
parser.add_argument('--use_augs', type=int, default=0, help='train set augmentaions: reverse compl + random shift')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--add_head_dense', type=int, default=0,
                    help='add dense layer to classification head with specified size (default: 0)')

# rmt args
parser.add_argument('--backbone_cls', type=str, default=None,
                    help='backbone class name to use for RMT')
parser.add_argument('--backbone_trainable', action='store_true', default=False,
                    help='make all model weights trainable, not only task-specific head.')
parser.add_argument('--backbone_checkpoint', type=str,
                    help='pre-trained backbone checkpoint (default: None).')
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--max_bins_per_segment', type=int, default=24, help='maximal number of bins in single segment')
parser.add_argument('--mixed_length_ratio', type=float, default=0.0, help='used for mixed length curriculum. '
                    'r > 0.0 means that we will start to sample batches with lengths <= max_n_segments')

parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')


def main():
    args = parser.parse_args()

    prepare_run(args, logger, logger_fmt)

    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    if hvd.rank() == 0 and args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    pad_token_ids = {'input_ids': tokenizer.pad_token_id, 'token_type_ids': 0, 'attention_mask': 0,
                     'bins_mask': 0, 'labels': 0}
    pad_to_divisible_by = 2

    def collate_fn(batch):
        feature_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask']
        padded_batch = {k: [] for k in feature_keys}
        max_seq_len = max([len(el['input_ids']) for el in batch])
        max_seq_len += (
            (pad_to_divisible_by - max_seq_len % pad_to_divisible_by)
            if max_seq_len % pad_to_divisible_by != 0
            else 0
        )
        for k in feature_keys:
            for i, el in enumerate(batch):
                dtype = batch[i][k].dtype
                pad = np.array([pad_token_ids[k]] * max(0, max_seq_len - len(el[k])), dtype=dtype)
                padded_batch[k] += [np.concatenate([batch[i][k], pad])]

        max_labels_len = max([len(el['labels']) for el in batch])
        padded_batch['labels'] = []
        padded_batch['labels_mask'] = torch.ones((len(batch), max_labels_len), dtype=torch.bool)
        for i, el in enumerate(batch):
            el = el['labels']
            pad = np.ones((max(0, max_labels_len - len(el)), el.shape[-1])) * pad_token_ids['labels']
            padded_batch['labels'] += [np.concatenate([batch[i]['labels'], pad])]
            padded_batch['labels_mask'][i, len(el):] = 0

        for k in padded_batch:
            padded_batch[k] = torch.from_numpy(np.stack(padded_batch[k]))

        return padded_batch

    kwargs['collate_fn'] = collate_fn

    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing training data from: {args.data_path}')
    data_path = Path(args.data_path).expanduser().absolute()
    train_dataset = EnformerDataset(tokenizer, data_path, max_seq_len=args.input_seq_len,
                                    bins_per_sample=args.bins_per_sample, augment=bool(args.use_augs))
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
        valid_dataset = EnformerDataset(tokenizer, valid_data_path, max_seq_len=args.input_seq_len,
                                        bins_per_sample=args.bins_per_sample, n_samples=args.n_valid_samples)
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
    model_cfg.num_labels = EnformerDataset.TG_COUNT
    model_cfg.add_head_dense = args.add_head_dense
    model_cls = get_cls_by_name(args.backbone_cls)
    if hvd.rank() == 0:
        logger.info(f'Using backbone model class: {model_cls}')
    model = model_cls(config=model_cfg)

    if args.backbone_checkpoint is not None:
        if hvd.rank() == 0:
            logger.info(f'loading pre-trained backbone from {args.backbone_checkpoint}')
        checkpoint = torch.load(args.backbone_checkpoint, map_location='cpu')
        missing_k, unexpected_k = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if len(missing_k) != 0 and hvd.rank() == 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0 and hvd.rank() == 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

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
            'max_n_bins': args.bins_per_sample,
            'max_bins_per_segment': args.max_bins_per_segment,
            'mixed_length_ratio': args.mixed_length_ratio,
        }
        rmt_cls = get_cls_by_name(args.model_cls)
        if hvd.rank() == 0:
            logger.info(f'Wrapping in: {rmt_cls}')
        model = rmt_cls(model, **rmt_config)

        if args.input_seq_len / model.segment_size > rmt_config['max_n_segments']:
            raise RuntimeError(f"Input sequence does not fully fit into selected number of segments: "
                               f"{args.input_seq_len} / {model.segment_size} > {rmt_config['max_n_segments']}")

    if not args.backbone_trainable:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                if hvd.rank() == 0:
                    logger.info(f'{name} is frozen')
                param.requires_grad = False

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
            'labels': batch['labels'],
            'labels_mask': batch['labels_mask'],
            'bins_mask': batch['bins_mask'],
        }

    def pearson_corr_coef(x, y, dim=1, reduce_dims=(-1,)):
        x_centered = x - x.mean(dim=dim, keepdim=True)
        y_centered = y - y.mean(dim=dim, keepdim=True)
        return torch.nn.functional.cosine_similarity(x_centered, y_centered, dim=dim).mean(dim=reduce_dims)

    def keep_for_metrics_fn(batch, output):
        predictions_segm = [[el.detach().cpu() for el in s] for s in output['logits_segm']]
        rmt_logits_masks_segm = [s.detach().cpu() for s in output['rmt_logits_masks_segm']]
        labels_segm = [[el.detach().cpu() for el in s] for s in output['labels_segm']]
        rmt_labels_masks_segm = [[el.detach().cpu() for el in s] for s in output['rmt_labels_masks_segm']]
        # collecting labels and prediction for rmt from each segment
        y_rmt, p_rmt = [], []
        for i in range(len(labels_segm)):
            # and cat
            labels_segm[i] = torch.stack(labels_segm[i])
            predictions_segm[i] = torch.stack(predictions_segm[i])
            rmt_labels_masks_segm[i] = torch.stack(rmt_labels_masks_segm[i])

            y_segm, p_segm = labels_segm[i], predictions_segm[i]
            y_segm = y_segm[rmt_labels_masks_segm[i]]
            p_segm = torch.nn.functional.softplus(p_segm[rmt_logits_masks_segm[i]].float())
            assert y_segm.shape == p_segm.shape
            y_rmt += [y_segm]
            p_rmt += [p_segm]

        y_rmt = torch.cat(y_rmt)
        p_rmt = torch.cat(p_rmt)
        assert y_rmt.shape == p_rmt.shape
        # keep states for correlation metric
        preds = p_rmt.cpu().unsqueeze(1)
        target = y_rmt.cpu().unsqueeze(1)
        reduce_dims = (0, 1)
        data = {}
        data['_product'] = torch.sum(preds * target, dim=reduce_dims).unsqueeze(0)
        data['_true'] = torch.sum(target, dim=reduce_dims).unsqueeze(0)
        data['_true_squared'] = torch.sum(torch.square(target), dim=reduce_dims).unsqueeze(0)
        data['_pred'] = torch.sum(preds, dim=reduce_dims).unsqueeze(0)
        data['_pred_squared'] = torch.sum(torch.square(preds), dim=reduce_dims).unsqueeze(0)
        data['_count'] = torch.sum(torch.ones_like(target), dim=reduce_dims).unsqueeze(0)
        return data

    def batch_metrics_fn(batch, output):
        metrics = {'loss': output['loss'].detach()}

        predictions_segm = [[el.detach().cpu() for el in s] for s in output['logits_segm']]
        rmt_logits_masks_segm = [s.detach().cpu() for s in output['rmt_logits_masks_segm']]
        labels_segm = [[el.detach().cpu() for el in s] for s in output['labels_segm']]
        rmt_labels_masks_segm = [[el.detach().cpu() for el in s] for s in output['rmt_labels_masks_segm']]

        # collecting labels and prediction for rmt from each segment
        y_rmt, p_rmt = [], []
        for i in range(len(labels_segm)):
            # and cat
            labels_segm[i] = torch.stack(labels_segm[i])
            predictions_segm[i] = torch.stack(predictions_segm[i])
            rmt_labels_masks_segm[i] = torch.stack(rmt_labels_masks_segm[i])

            y_segm, p_segm = labels_segm[i], predictions_segm[i]
            y_segm = y_segm[rmt_labels_masks_segm[i]]
            p_segm = torch.nn.functional.softplus(p_segm[rmt_logits_masks_segm[i]].float())
            assert y_segm.shape == p_segm.shape
            y_rmt += [y_segm]
            p_rmt += [p_segm]

        y_rmt = torch.cat(y_rmt)
        p_rmt = torch.cat(p_rmt)
        assert y_rmt.shape == p_rmt.shape

        metrics['pearson_corr'] = pearson_corr_coef(p_rmt, y_rmt)

        corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=EnformerDataset.TG_COUNT)
        corr_coef(preds=p_rmt.cpu().unsqueeze(1), target=y_rmt.cpu().unsqueeze(1))
        corr_coef = corr_coef.compute()
        corr_coef = corr_coef.nansum() / (EnformerDataset.TG_COUNT - torch.isnan(corr_coef).sum())
        metrics['pearson_corr_enformer'] = corr_coef

        # loss_fct = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
        # loss_2 = loss_fct(p_rmt, y_rmt)
        # loss_1 = metrics['loss'].item()
        # logger.info(f'y_rmt.shape: {y_rmt.shape}')
        # logger.info(f'p_rmt.shape: {p_rmt.shape}')
        # for i in range(len(rmt_logits_masks_segm)):
        #     logger.info(f'rmt_logits_masks_segm[{i}].shape: {rmt_logits_masks_segm[i].shape}')
        #     logger.info(f'rmt_logits_masks_segm[{i}].sum(): {rmt_logits_masks_segm[i].sum()}')
        # if not np.isclose(loss_1, loss_2, atol=1e-03):
        #     logger.info(f'NOTGOOD: {loss_1:.6f} {loss_2:.6f}')
        # logger.info(f'loss1 vs loss2: {loss_1:.6f} {loss_2.sum().item() / (p_rmt.shape[0]*EnformerDataset.TG_COUNT):.6f}')
        # logger.info(f'loss2: {loss_2}')
        return metrics

    def metrics_fn(data):
        metrics = {}
        # aggregate correlation states
        data['_product'] = torch.sum(data['_product'], dim=0)
        data['_true'] = torch.sum(data['_true'], dim=0)
        data['_true_squared'] = torch.sum(data['_true_squared'], dim=0)
        data['_pred'] = torch.sum(data['_pred'], dim=0)
        data['_pred_squared'] = torch.sum(data['_pred_squared'], dim=0)
        data['_count'] = torch.sum(data['_count'], dim=0)
        # compute correlation
        true_mean = data['_true'] / data['_count']
        pred_mean = data['_pred'] / data['_count']

        covariance = (data['_product'] - true_mean * data['_pred'] - pred_mean * data['_true']
                      + data['_count'] * true_mean * pred_mean)

        true_var = data['_true_squared'] - data['_count'] * torch.square(true_mean)
        pred_var = data['_pred_squared'] - data['_count'] * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        corr_coef = covariance / tp_var
        corr_coef = corr_coef.nansum() / (EnformerDataset.TG_COUNT - torch.isnan(corr_coef).sum())
        metrics['pearson_corr_enformer_statefull'] = corr_coef.item()
        return metrics

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader=valid_dataloader,
                      train_sampler=train_sampler, batch_transform_fn=batch_transform_fn,
                      batch_metrics_fn=batch_metrics_fn, metrics_fn=metrics_fn, keep_for_metrics_fn=keep_for_metrics_fn)
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
        # get test dataset
        if hvd.rank() == 0:
            logger.info(f'preparing test data from: {args.test_data_path}')
        test_data_path = Path(args.test_data_path).expanduser().absolute()
        test_dataset = EnformerDataset(tokenizer, test_data_path, max_seq_len=args.input_seq_len,
                                       bins_per_sample=args.bins_per_sample)
        test_sampler = DistributedSampler(test_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler, **kwargs)
        if hvd.rank() == 0:
            logger.info(f'len(test_dataset): {len(test_dataset)}')
            logger.info('Runnning validation on test data:')
        trainer.validate(test_dataloader, split='test', write_tb=True)

    trainer.save_metrics(args.model_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)

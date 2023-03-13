import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from torchmetrics import Metric
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
import numpy as np
from tqdm import tqdm

from lm_experiments_tools.utils import get_cls_by_name

from downstream_tasks.enformer.enformer_dataset import EnformerDataset

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_num_threads(4)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to the training data')
parser.add_argument('--valid_data_path', type=str, help='path to the valid data')
parser.add_argument('--test_data_path', type=str, help='path to the test data (dataset_test_0.csv)')
parser.add_argument('--seed', type=int, default=42, help='random seed')

# data args
parser.add_argument('--input_seq_len', type=int, default=64, help='input sequnce length (default: 64).')
parser.add_argument('--bins_per_sample', type=int, default=896, help='input sequnce length (default: 64).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')
parser.add_argument('--n_samples', type=int, default=None, help='number of samples to use to compute metrics')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

parser.add_argument('--batch_size', type=int, default=64, help='evaluation batch size (default: 64).')

parser.add_argument('--experiment_cfg', type=str, help='path experiment config.json (to get model_cfg, model_cls)')
parser.add_argument('--init_checkpoint', type=str, help='path to checkpoint to evaluate')


# taken from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/metrics.py
class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    full_state_update:bool = False
    higher_is_better: Optional[bool] = True
    def __init__(self, n_channels:int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        super().__init__(dist_sync_on_step=dist_sync_on_step, full_state_update=False)
        self.reduce_dims=(0, 1)
        self.add_state("product", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (self.product
                    - true_mean * self.pred
                    - pred_mean * self.true
                    + self.count * true_mean * pred_mean)

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation


if __name__ == '__main__':
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

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
    train_dataloader = None
    valid_dataloader = None
    test_dataloader = None

    if args.data_path:
        logger.info(f'preparing training data from: {args.data_path}')
        data_path = Path(args.data_path).expanduser().absolute()
        train_dataset = EnformerDataset(tokenizer, data_path, max_seq_len=args.input_seq_len,
                                        bins_per_sample=args.bins_per_sample, n_samples=args.n_samples)
        logger.info(f'len(train_dataset): {len(train_dataset)}')
        # shuffle train data each epoch (one loop over train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,  **kwargs)

    if args.valid_data_path:
        logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_dataset = EnformerDataset(tokenizer, valid_data_path, max_seq_len=args.input_seq_len,
                                        bins_per_sample=args.bins_per_sample, n_samples=args.n_samples)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,  **kwargs)
        logger.info(f'len(valid_dataset): {len(valid_dataset)}')

    if args.test_data_path:
        # get test dataset
        logger.info(f'preparing test data from: {args.test_data_path}')
        test_data_path = Path(args.test_data_path).expanduser().absolute()
        test_dataset = EnformerDataset(tokenizer, test_data_path, max_seq_len=args.input_seq_len,
                                       bins_per_sample=args.bins_per_sample, n_samples=args.n_samples)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, **kwargs)

    # define model
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    model_cfg.num_labels = EnformerDataset.TG_COUNT
    model_cls = get_cls_by_name(args.model_cls)
    logger.info(f'Using model class: {model_cls}')
    model = model_cls(config=model_cfg)

    logger.info(f'loading weights from: {args.init_checkpoint}')
    checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
    missing_k, unexpected_k = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if len(missing_k) != 0:
        logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
    if len(unexpected_k) != 0:
        logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    if 'metrics' in checkpoint:
        logger.info(f'checkpoint metrics: {checkpoint["metrics"]}')

    def batch_transform_fn(batch):
        return {
            'input_ids': batch['input_ids'],
            'token_type_ids': batch['token_type_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels'],
            'labels_mask': batch['labels_mask'],
            'bins_mask': batch['bins_mask'],
        }

    model = model.cuda()
    model = model.eval()

    datasets = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
    for sn in datasets:
        dl = datasets[sn]
        if dl is not None:
            corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=EnformerDataset.TG_COUNT)
            with torch.no_grad():
                for batch in tqdm(dl, desc=sn):
                    batch = batch_transform_fn(batch)
                    batch = {k: batch[k].cuda() for k in batch}
                    output = model(**batch)
                    pred = torch.nn.functional.softplus(output['logits'].detach())
                    labels = batch['labels'][batch['labels_mask']]
                    corr_coef(preds=pred.cpu(), target=labels.cpu())
            logger.info(f'{sn} corr_coef: {corr_coef.compute().mean()}')

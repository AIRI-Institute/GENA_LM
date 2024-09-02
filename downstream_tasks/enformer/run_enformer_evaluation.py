import argparse
import logging
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
import numpy as np
from tqdm import tqdm
import h5py

from lm_experiments_tools.utils import get_cls_by_name

from downstream_tasks.enformer.enformer_dataset import EnformerDataset
from downstream_tasks.enformer.enformer_metrics import MeanPearsonCorrCoefPerChannel

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
parser.add_argument('--input_seq_len', type=int, help='input sequnce length.')
parser.add_argument('--bins_per_sample', type=int, help='input sequnce length (~896 per sample)')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')
parser.add_argument('--n_samples', type=int, default=None, help='number of samples to use to compute metrics')
parser.add_argument('--n_context_bins', type=int, default=0,
                    help='number of bins to use as context (do not predict on them) (default: 0)')
parser.add_argument('--context_mode', type=str, default=None,
                    help='context mode: left (default: None)')
parser.add_argument('--remove_context', type=bool, default=True)
# augmenation args
parser.add_argument('--augment', dest='augment', action='store_true', default=False, help='apply data augmentations')
parser.add_argument('--aug_shift_min', type=int, default=-3)
parser.add_argument('--aug_shift_max', type=int, default=3)
parser.add_argument('--aug_rc', type=str, default='rnd', help='keep | rc | rnd')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, help='model class name to use')
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

parser.add_argument('--batch_size', type=int, default=64, help='evaluation batch size (default: 64).')

parser.add_argument('--experiment_cfg', type=str, help='path experiment config.json (to get model_cfg, model_cls)')
parser.add_argument('--init_checkpoint', type=str, help='path to checkpoint to evaluate')

# rmt args
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--max_n_segments', type=int, default=None, help='maximal segment number')
parser.add_argument('--max_bins_per_segment', type=int, default=None, help='maximal number of bins in single segment')

# script args
# add save to metrics.json
parser.add_argument('--save_predictions', type=str, default=None, help='filename where to save predictions')
parser.add_argument('--save_metrics', dest='save_metrics', action='store_true')
parser.add_argument('--dont_save_metrics', dest='save_metrics', action='store_false')
parser.set_defaults(save_metrics=True)

if __name__ == '__main__':
    args = parser.parse_args()

    # get unset args from experiment config
    exp_cfg = {}
    if args.experiment_cfg is not None:
        exp_cfg = json.load(open(args.experiment_cfg, 'r'))
        if args.model_cls is None:
            args.model_cls = exp_cfg['model_cls']
            # args.model_cls = 'src.gena_lm.modeling_rmt_dbg:RMTEncoderForEnformer'
            # args.batch_size = 1
        if args.tokenizer is None:
            args.tokenizer = exp_cfg['tokenizer']
        if args.input_seq_len is None:
            args.input_seq_len = exp_cfg['input_seq_len']
        if args.bins_per_sample is None:
            args.bins_per_sample = exp_cfg['bins_per_sample']
        if 'RMT' in args.model_cls:
            if args.input_size is None:
                args.input_size = exp_cfg['input_size']
            if args.max_n_segments is None:
                args.max_n_segments = exp_cfg['max_n_segments']
            if args.max_bins_per_segment is None:
                args.max_bins_per_segment = exp_cfg['max_bins_per_segment']

    is_rmt = 'RMT' in args.model_cls
    logger.info(f'RMT: {is_rmt}')

    logger.info(f'model_cls: {args.model_cls}')
    logger.info(f'input_seq_len: {args.input_seq_len}')
    logger.info(f'bins_per_sample: {args.bins_per_sample}')
    if is_rmt:
        logger.info(f"num_mem_tokens: {exp_cfg['num_mem_tokens']}")
        logger.info(f'input_size: {args.input_size}')
        logger.info(f'max_bins_per_segment: {args.max_bins_per_segment}')
        logger.info(f'max_n_segments: {args.max_n_segments}')

    if args.context_mode is not None:
        args.remove_context = False

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
                                        bins_per_sample=args.bins_per_sample, n_samples=args.n_samples,
                                        remove_context=args.remove_context, context_mode=args.context_mode,
                                        n_context_bins=args.n_context_bins,
                                        augment=args.augment, aug_rc=args.aug_rc,
                                        aug_shift_min=args.aug_shift_min, aug_shift_max=args.aug_shift_max,
                                        )
        logger.info(f'len(train_dataset): {len(train_dataset)}')
        # shuffle train data each epoch (one loop over train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,  **kwargs)

    if args.valid_data_path:
        logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_dataset = EnformerDataset(tokenizer, valid_data_path, max_seq_len=args.input_seq_len,
                                        bins_per_sample=args.bins_per_sample, n_samples=args.n_samples,
                                        remove_context=args.remove_context, context_mode=args.context_mode,
                                        n_context_bins=args.n_context_bins,
                                        augment=args.augment, aug_rc=args.aug_rc,
                                        aug_shift_min=args.aug_shift_min, aug_shift_max=args.aug_shift_max,
                                        )
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,  **kwargs)
        logger.info(f'len(valid_dataset): {len(valid_dataset)}')

    if args.test_data_path:
        # get test dataset
        logger.info(f'preparing test data from: {args.test_data_path}')
        test_data_path = Path(args.test_data_path).expanduser().absolute()
        test_dataset = EnformerDataset(tokenizer, test_data_path, max_seq_len=args.input_seq_len,
                                       bins_per_sample=args.bins_per_sample, n_samples=args.n_samples,
                                       remove_context=args.remove_context, context_mode=args.context_mode,
                                       n_context_bins=args.n_context_bins,
                                       augment=args.augment, aug_rc=args.aug_rc,
                                       aug_shift_min=args.aug_shift_min, aug_shift_max=args.aug_shift_max,
                                       )
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, **kwargs)

    # define modelmetrics[f'{sn}'] = {'pearson_corr_enformer': corr_coef_mean}
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    model_cfg.num_labels = EnformerDataset.TG_COUNT
    if is_rmt:
        model_cls = get_cls_by_name(exp_cfg['backbone_cls'])
    else:
        model_cls = get_cls_by_name(args.model_cls)
    logger.info(f'Using model class: {model_cls}')
    model = model_cls(config=model_cfg)

    if is_rmt:
        rmt_config = {
            'num_mem_tokens': exp_cfg['num_mem_tokens'],
            'max_n_segments': args.max_n_segments,
            # 'segment_ordering': args.segment_ordering,
            'input_size': args.input_size,
            'bptt_depth': exp_cfg['bptt_depth'],
            'tokenizer': tokenizer,
            'max_n_bins': args.bins_per_sample,
            'max_bins_per_segment': args.max_bins_per_segment,
        }
        logger.info(f'rmt_config: {rmt_config}')
        rmt_cls = get_cls_by_name(args.model_cls)
        logger.info(f'Wrapping in: {rmt_cls}')
        model = rmt_cls(model, **rmt_config)

    logger.info(f'loading weights from: {args.init_checkpoint}')
    checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
    missing_k, unexpected_k = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if len(missing_k) != 0:
        logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
    if len(unexpected_k) != 0:
        logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    if 'metrics' in checkpoint:
        logger.info(f'checkpoint metrics: {checkpoint["metrics"]}')

    metrics_file = Path(args.init_checkpoint).parent / 'metrics.json'
    if metrics_file.is_file():
        metrics = json.load(metrics_file.open('r'))
    elif 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
    else:
        metrics = {}

    logger.info('loaded metrics (from metrics.json or ckpt):')
    for sn in metrics:
        for m in metrics[sn]:
            if m != 'pearson_corr_enformer_per_target':
                logger.info(f'{sn} {m}: {metrics[sn][m]}')

    def batch_transform_fn(batch):
        b = {'input_ids': batch['input_ids'],
             'labels': batch['labels'],
             'labels_mask': batch['labels_mask'],
             'bins_mask': batch['bins_mask']}
        if not is_rmt:
            b['token_type_ids'] = batch['token_type_ids']
            b['attention_mask'] = batch['attention_mask']
        return b

    model = model.cuda()
    model = model.eval()
    preds = []
    labelss = []

    datasets = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
    for sn in datasets:
        dl = datasets[sn]
        if dl is not None:
            sn_detailed = f'{sn}_full' if args.n_samples is None or args.n_samples < 0 else f'{sn}_{args.n_samples}'
            sn_detailed = f'{sn_detailed}_{args.input_seq_len}_{args.bins_per_sample}'
            if args.context_mode is not None:
                sn_detailed = f'{sn_detailed}_ctx_{args.context_mode}_{args.n_context_bins}'
            if is_rmt:
                sn_detailed = f'{sn_detailed}_{args.input_size}_{args.max_bins_per_segment}'

            if args.save_predictions is not None:
                preds_filename = f'{args.save_predictions}_{sn_detailed}'
                if args.augment:
                    preds_filename = f'{preds_filename}_augs_{args.augment}'
                    preds_filename = f'{preds_filename}_shift_{args.aug_shift_min}_{args.aug_shift_max}'
                    preds_filename = f'{preds_filename}_rc_{args.aug_rc}.h5py'
                else:
                    preds_filename = f'{preds_filename}.h5py'
                preds_filename = Path(args.init_checkpoint).parent / preds_filename

                preds_file = h5py.File(preds_filename, 'w')
                preds_file.create_dataset('labels', shape=(1, 1, 5313), maxshape=(None, 1, 5313), dtype=np.float32,
                                          compression='gzip', chunks=True)
                preds_file.create_dataset('pred', shape=(1, 1, 5313), maxshape=(None, 1, 5313), dtype=np.float32,
                                          compression='gzip', chunks=True)
                n_written = 0

            corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=EnformerDataset.TG_COUNT)
            with torch.no_grad():
                for batch in tqdm(dl, desc=sn):
                    batch = batch_transform_fn(batch)
                    batch = {k: batch[k].cuda() for k in batch}
                    output = model(**batch)
                    if not is_rmt:
                        pred = torch.nn.functional.softplus(output['logits'].detach())
                        labels = batch['labels'][batch['labels_mask']]
                    else:
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
                            if y_segm.shape[0] == 0:
                                # segment without targets
                                continue
                            y_rmt += [y_segm]
                            p_rmt += [p_segm]

                        y_rmt = torch.cat(y_rmt)
                        p_rmt = torch.cat(p_rmt)
                        assert y_rmt.shape == p_rmt.shape

                        pred = p_rmt
                        labels = y_rmt

                    pred = pred.cpu().unsqueeze(1)
                    labels = labels.cpu().unsqueeze(1)
                    corr_coef(preds=pred, target=labels)
                    if args.save_predictions is not None:
                        n_new = pred.shape[0]
                        n_free = preds_file['labels'].shape[0] - n_written
                        n_to_add = max(n_new - n_free, 0)
                        preds_file['labels'].resize(preds_file['labels'].shape[0] + n_to_add, axis=0)
                        preds_file['pred'].resize(preds_file['pred'].shape[0] + n_to_add, axis=0)
                        if args.aug_rc == 'rc':
                            labels = labels.flip(dims=(0,))
                            pred = pred.flip(dims=(0,))
                        preds_file['labels'][n_written:n_written + n_new, :, :] = labels.numpy()
                        preds_file['pred'][n_written:n_written + n_new, :, :] = pred.numpy()
                        n_written = n_written + n_new

                    # preds += [pred.cpu().numpy()]
                    # labelss += [labels.cpu().numpy()]

            corr_coef = corr_coef.compute()
            corr_coef_mean = corr_coef.mean().item()
            metrics[f'{sn_detailed}'] = {'pearson_corr_enformer': corr_coef_mean}
            metrics[f'{sn_detailed}']['pearson_corr_enformer_per_target'] = {}
            for i, v in enumerate(corr_coef):
                metrics[f'{sn_detailed}']['pearson_corr_enformer_per_target'][i] = v.item()
            logger.info(f'{sn_detailed} corr_coef: {corr_coef_mean}')

            if args.save_predictions is not None:
                print(f'saving predicions to {preds_filename} ...')
                print(f'written: {preds_file["labels"].shape}')
                preds_file.flush()
                preds_file.close()

    # save metrics into file
    for sn in metrics:
        for m in metrics[sn]:
            if m != 'pearson_corr_enformer_per_target':
                logger.info(f'{sn} {m}: {metrics[sn][m]}')
    # logger.info(f'metrics: {metrics}')
    if args.save_metrics:
        json.dump(metrics, metrics_file.open('w'), indent=4)
    print('done')

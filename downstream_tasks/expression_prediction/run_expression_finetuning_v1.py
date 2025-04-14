import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
import numpy as np

from lm_experiments_tools import Trainer, TrainerArgs, get_optimizer
from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration, get_git_diff, prepare_run
import lm_experiments_tools.optimizers as optimizers

from downstream_tasks.expression_prediction.expression_dataset import ExpressionDataset, worker_init_fn

import horovod.torch as hvd

from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate
from functools import partial

import time

timestamp = time.strftime("%Y%m%d-%H%M%S")

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger()

if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

hvd.init()

torch.set_num_threads(4)
torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--experiment_config', type=str, help='path to the experiment config') 
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--backbone_trainable', action='store_true', default=False,
                    help='make all model weights trainable, not only task-specific head.')

def main():
    args = parser.parse_args()
    experiment_config_path = Path(args.experiment_config).expanduser().absolute()

    with initialize_config_dir(str(experiment_config_path.parents[0])):
        experiment_config = compose(config_name=experiment_config_path.name)

    if "args_params" in experiment_config:
        trainer_kwargs = instantiate(experiment_config["args_params"])
        for k,v in trainer_kwargs.items():
            logger.info(f"Setting attr {k}:{v}")
            if hasattr(args,k):
                logger.warning(f"Conflicting setting for option {k} in args and config, overwritten by config option {v}")
            args.__setattr__(k,v)

    prepare_run(args, logger, logger_fmt)

    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')

    if args.resume is not None:
        logger.warning(f"Resuming from cpt {args.resume}. This will overwrite reset_lr, reset_optimizer, reset_iteration and init_cpt options")
        args.__setattr__("reset_lr", False)
        args.__setattr__("reset_optimizer", False)
        args.__setattr__("reset_iteration", False)
        args.__setattr__("init_checkpoint", args.model_path + "/" + args.resume)
        args.__setattr__("model_path", args.model_path + "/resume_" + args.resume + "/")

    if hvd.rank() == 0:
        if args.model_path is None:
            raise ValueError("Model path should not be None")
    args.model_path = os.path.join(args.model_path, timestamp)    
    logger.info(f"hvd.rank(): {hvd.rank()}, Model path: {args.model_path}")

    if hvd.rank() == 0 and args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    if hvd.rank() == 0 and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)
        open(model_path / 'git.diff', 'w').write(get_git_diff())

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    if args.model_path:
        if hvd.rank() == 0:
            content = "\n".join(open(experiment_config_path).readlines())
            with open(Path(args.model_path) / "experiment_config.yaml", "w") as fout:
                fout.write(content)
                
    tokenizer = AutoTokenizer.from_pretrained(args.gen_tokenizer)

    def collate_fn(batch):
        # Ключи, которые нужно паддить
        pad_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'labels_mask']
        # Ключи, которые передаются без изменений
        no_pad_keys = ['desc_vectors', 'tpm']
        

        pad_token_ids = {
            'input_ids': tokenizer.pad_token_id,  
            'token_type_ids': 0,
            'attention_mask': 0,
            'labels': 0,
            'labels_mask': 0
        }
        
        max_seq_len = max([len(sample['input_ids']) for sample in batch])
        batch_dict = {key: [] for key in pad_keys + no_pad_keys}
        
        for sample in batch:
            for key in pad_keys:
                seq = sample[key]
                pad_len = max_seq_len - len(seq)
                if pad_len > 0:
                    if key == 'labels' or key == 'labels_mask':
                        # labels имеет размер (seq_len, num_targets), паддим по seq_len
                        pad = torch.full((pad_len, seq.size(1)), pad_token_ids[key], dtype=seq.dtype)
                        padded_seq = torch.cat([seq, pad], dim=0)
                    else:
                        # Остальные ключи - одномерные тензоры
                        pad = torch.full((pad_len,), pad_token_ids[key], dtype=seq.dtype)
                        padded_seq = torch.cat([seq, pad], dim=0)
                else:
                    padded_seq = seq
                batch_dict[key].append(padded_seq)
            
            for key in no_pad_keys:
                batch_dict[key].append(sample[key])
        
        for key in pad_keys:
            batch_dict[key] = torch.stack(batch_dict[key])
        for key in no_pad_keys:
            batch_dict[key] = torch.stack(batch_dict[key])
        
        return batch_dict

    kwargs['collate_fn'] = collate_fn


    # get train datasets
    if hvd.rank() == 0:
        logger.info(f'preparing training data')
    
    # Instantiate training datasets
    train_datasets_configs = [v for k,v in experiment_config.items() if k.startswith('train_dataset')]
    if hvd.rank() == 0:
        train_datasets = [instantiate(config) for config in train_datasets_configs]
    hvd.barrier()
    train_datasets = [instantiate(config) for config in train_datasets_configs]
 
    if len(train_datasets) == 0:
        raise ValueError("No training datasets found")
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        # Combine datasets into one
        train_dataset = ConcatDataset(train_datasets)
    
    if hvd.rank() == 0:
        for i, dataset in enumerate(train_datasets):
            logger.info(f'dataset {i}: {dataset.describe()}')
        logger.info(f'total len(train_dataset): {len(train_dataset)}')
    
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)
    
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, 
                                  sampler=train_sampler, worker_init_fn=worker_init_fn, **kwargs)
    
    # get valid datasets if specified
    valid_datasets_configs = [v for k,v in experiment_config.items() if k.startswith('valid_dataset')]
    if len(valid_datasets_configs) > 0:
        if hvd.rank() == 0:
            logger.info(f'preparing validation data')
            valid_datasets = [instantiate(config) for config in valid_datasets_configs]
        hvd.barrier()
        valid_datasets = [instantiate(config) for config in valid_datasets_configs]
        if len(valid_datasets) == 1:
            valid_dataset = valid_datasets[0]
        else:
            valid_dataset = ConcatDataset(valid_datasets)
        
        for i, dataset in enumerate(valid_datasets):
            logger.info(f'dataset {i}: {dataset.describe()}')
        logger.info(f'total len(valid_dataset): {len(valid_dataset)}')
                
        valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, 
                                      sampler=valid_sampler, worker_init_fn=worker_init_fn, **kwargs)
        
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
        
    else:
        valid_dataloader = None
        if hvd.rank() == 0:
            logger.info('No validation data is used.')

    # define model
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)

#    model_cfg.add_head_dense = args.add_head_dense

    if "model_kwargs" in experiment_config:
        model_kwargs = instantiate(experiment_config["model_kwargs"])
    else:
        model_kwargs = {}

    if "config" in model_kwargs:
        for k,v in model_kwargs["config"].items():
            model_cfg.__setattr__(k,v)

    model_kwargs["config"] = model_cfg
  
    model_cls = get_cls_by_name(args.backbone_cls)
    if hvd.rank() == 0:
        logger.info(f'Using backbone model class: {model_cls}')
    model = model_cls(**model_kwargs)
    model_activation_fn = model.activation

    # Aydar # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        rmt_config = {
            'num_mem_tokens': args.num_mem_tokens,
            'max_n_segments': args.max_n_segments,
            'input_size': args.input_size,
            'bptt_depth': args.bptt_depth,
            'sum_loss': True,
            'tokenizer': tokenizer,
            'mixed_length_ratio': args.mixed_length_ratio,
        }
        rmt_cls = get_cls_by_name(args.model_cls)
        if hvd.rank() == 0:
            logger.info(f'Wrapping in: {rmt_cls}')
        model = rmt_cls(model, **rmt_config)

        # if args.init_checkpoint_l is not None:
        #     if hvd.rank() == 0:
        #         logger.info(f'loading pre-trained backbone from {args.init_checkpoint_l}')
        #     checkpoint = torch.load(args.init_checkpoint_l, map_location='cpu') 
        #     checkpoint.pop("model.classifier.weight", None)
        #     checkpoint.pop("model.classifier.bias", None)
        #     missing_k, unexpected_k = model.load_state_dict(checkpoint, strict=False)
        #     if len(missing_k) != 0 and hvd.rank() == 0:
        #         logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        #     if len(unexpected_k) != 0 and hvd.rank() == 0:
        #         logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')


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
        result = {
        'input_ids': batch['input_ids'], 
        'token_type_ids': batch['token_type_ids'],
        'attention_mask': batch['attention_mask'],
        'labels_mask': batch['labels_mask'],
        'desc_vectors': batch['desc_vectors'],
        'labels': batch['labels'],
        'tpm': batch['tpm'],
                        }
        return result
    
    def pearson_corr_coef(x_input, y_input):
        if isinstance(x_input, torch.Tensor):
            x = x_input
        else:
            x = torch.cat(x_input)
            
        if isinstance(y_input, torch.Tensor):
            y = y_input
        else:
            y = torch.cat(y_input)
        
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        x_centered = x_centered.unsqueeze(0)
        y_centered = y_centered.unsqueeze(0)
        
        corr = torch.nn.functional.cosine_similarity(x_centered, y_centered, dim=1)
        
        return corr.item()    

    def keep_for_metrics_fn(batch, output):
        predictions_segm = [[el.detach().cpu() for el in s] for s in output['logits_segm']]
        labels_segm = [[el.detach().cpu() for el in s] for s in output['labels_reshaped']]
        rmt_labels_masks_segm = [[el.detach().cpu().to(torch.bool) for el in s] for s in output['labels_mask_reshaped']]
        
        y_rmt, p_rmt = [], []
        for i in range(len(labels_segm)):
            labels_segm[i] = torch.stack(labels_segm[i])
            predictions_segm[i] = torch.stack(predictions_segm[i])
            rmt_labels_masks_segm[i] = torch.stack(rmt_labels_masks_segm[i])

            y_segm, p_segm = labels_segm[i], predictions_segm[i]        
            
            y_segm = y_segm[rmt_labels_masks_segm[i]]
            p_segm = p_segm[rmt_labels_masks_segm[i]]
            
            assert y_segm.shape == p_segm.shape
            y_rmt += [y_segm]
            p_rmt += [p_segm]

        if not y_rmt or not p_rmt:
            return {}

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
        labels_segm = [[el.detach().cpu() for el in s] for s in output['labels_reshaped']]
        rmt_labels_masks_segm = [[el.detach().cpu().to(torch.bool) for el in s] for s in output['labels_mask_reshaped']]

        y_rmt, p_rmt = [], []
        for i in range(len(labels_segm)):
            labels_segm[i] = torch.stack(labels_segm[i])
            predictions_segm[i] = torch.stack(predictions_segm[i])
            rmt_labels_masks_segm[i] = torch.stack(rmt_labels_masks_segm[i])

            y_segm, p_segm = labels_segm[i], predictions_segm[i]
            
            y_segm = y_segm[rmt_labels_masks_segm[i]]
            p_segm = p_segm[rmt_labels_masks_segm[i]]

            
            assert y_segm.shape == p_segm.shape
            y_rmt += [y_segm]
            p_rmt += [p_segm]

        if not y_rmt:
            return {}

        y_rmt = torch.cat(y_rmt)
        p_rmt = torch.cat(p_rmt)
        assert y_rmt.shape == p_rmt.shape


        metrics['pearson_corr'] = pearson_corr_coef(p_rmt, y_rmt)
        
        return metrics

    def metrics_fn(data):
        metrics = {}
        data['_product'] = torch.sum(data['_product'], dim=0)
        data['_true'] = torch.sum(data['_true'], dim=0)
        data['_true_squared'] = torch.sum(data['_true_squared'], dim=0)
        data['_pred'] = torch.sum(data['_pred'], dim=0)
        data['_pred_squared'] = torch.sum(data['_pred_squared'], dim=0)
        data['_count'] = torch.sum(data['_count'], dim=0)
        
        true_mean = data['_true'] / data['_count']
        pred_mean = data['_pred'] / data['_count']

        covariance = (data['_product'] - true_mean * data['_pred'] - pred_mean * data['_true'] + data['_count'] * true_mean * pred_mean)

        true_var = data['_true_squared'] - data['_count'] * torch.square(true_mean)
        pred_var = data['_pred_squared'] - data['_count'] * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        corr_coef = covariance / tp_var
        metrics['pearson_corr_statefull'] = corr_coef.item()
        return metrics

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader=valid_dataloader,
                      train_sampler=train_sampler, batch_transform_fn=batch_transform_fn) #,
                    #  batch_metrics_fn=batch_metrics_fn, metrics_fn=metrics_fn, keep_for_metrics_fn=keep_for_metrics_fn)
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

    # if args.test_data_path:
    #     # get test dataset
    #     if hvd.rank() == 0:
    #         logger.info(f'preparing test data from: {args.test_data_path}')
    #     test_data_path = Path(args.test_data_path).expanduser().absolute()
    #     test_dataset = EnformerDataset(tokenizer, test_data_path, max_seq_len=args.input_seq_len,
    #                                    bins_per_sample=args.bins_per_sample)
    #     test_sampler = DistributedSampler(test_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
    #     test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler, **kwargs)
    #     if hvd.rank() == 0:
    #         logger.info(f'len(test_dataset): {len(test_dataset)}')
    #         logger.info('Runnning validation on test data:')
    #     trainer.validate(test_dataloader, split='test', write_tb=True)

    trainer.save_metrics(args.model_path)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)

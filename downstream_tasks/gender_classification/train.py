import os
import importlib
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from gender_dataset import GenderDataChunkedDataset, worker_init_fn, collate_fn
from model import GenderChunkedClassifier

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def compute_metrics(p):
    probs, labels = p
    predictions = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': p,
        'recall': r,
        'f1': f1,
        'roc_auc': roc_auc_score(labels, probs)
    }


class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory,
                          worker_init_fn=worker_init_fn)

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory,
                          worker_init_fn=worker_init_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/20tb/ykuratov/gender_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_chunks", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=3072)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t')
    model = AutoModel.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t', trust_remote_code=True)
    model_module = importlib.import_module(model.__class__.__module__)
    cls = getattr(model_module, 'BertModel')
    model = cls.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t')

    args.data_path = Path(args.data_path)
    dataset = GenderDataChunkedDataset(args.data_path / 'train.h5', args.data_path / 'train.csv',
                                       n_chunks=args.n_chunks, chunk_size=args.chunk_size, seed=args.seed)
    valid_dataset = GenderDataChunkedDataset(args.data_path / 'valid.h5', args.data_path / 'valid.csv',
                                             n_chunks=args.n_chunks, chunk_size=args.chunk_size, max_n_samples=250,
                                             seed=args.seed)

    def preprocess_collate_fn(samples):
        batch = collate_fn(samples)

        batch['chunks'] = np.array(batch['chunks'])
        shape = batch['chunks'].shape
        batch['chunks'] = list(batch['chunks'].flatten())

        tokenized_batch = tokenizer(batch['chunks'], padding='longest', max_length=512,
                                    truncation=True, return_tensors='pt')

        for k in tokenized_batch:
            tokenized_batch[k] = tokenized_batch[k].reshape(*shape, -1)

        batch['labels'] = torch.Tensor(batch['labels'])

        return {
            'input_ids': tokenized_batch['input_ids'],
            'attention_mask': tokenized_batch['attention_mask'],
            'labels': batch['labels']
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gender_model = GenderChunkedClassifier(model).to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=1000,
        weight_decay=0.00,
        learning_rate=1e-4,
        logging_dir='./logs',
        report_to='tensorboard',
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        max_steps=5000,
        logging_steps=100,
        eval_steps=100,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        fp16=False,
        metric_for_best_model='roc_auc',
    )

    trainer = CustomTrainer(
        model=gender_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=preprocess_collate_fn
    )
    trainer.train()
    print('training done. running final evaluation...')
    trainer.evaluate(valid_dataset)

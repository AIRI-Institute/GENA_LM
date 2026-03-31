from TSSprediction.GENA_LM.downstream_tasks.TSSprediction.TSS_dataset_old import TSSDataset
import logging

TSSDataset(mapping_file='/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/dataset.csv',
            tokenizer='/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/data/tokenizers/t2t_1000h_multi_32k',
            cache_dir='/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/cache',
            num_before = 512,
            token_len_for_fetch = 10,
            max_seq_len = 1024,
            loglevel = logging.WARNING,
            data_workers = 10)

print('DONE')
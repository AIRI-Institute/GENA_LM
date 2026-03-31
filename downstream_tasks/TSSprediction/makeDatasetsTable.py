from pysam import FastaFile
import random
import copy
import polars as pl
import os



class DatasetList():
    
    def __init__(self, 
                    metadata:str|pl.DataFrame='metadata/condensed_metadata.csv', 
                    seed:int = 42,
                    valid_fraction:float = 0.2,
                    chrom_files_dir:str='/workspace-SR003.nfs2/estsoi/TSSprediction/GENA_LM/downstream_tasks/TSSprediction/data/chrom_files'):
        if isinstance(metadata, str):
            self.metadata = pl.read_csv(metadata)
        else:
            self.metadata = metadata
        self.chrom_files_dir = chrom_files_dir
        self.seed = seed
        self.valid_fraction = valid_fraction
        
    def dataset_name(self, genomePath:str):
        return os.path.basename(genomePath).split('.')[0]
        
    def write_chrom_files(self, genomePath:str):
        
        def add_separators(chroms, sep='\n'):
            return [f'{chrom}{sep}' for chrom in chroms]

        
        random.seed(self.seed)
        
        dataset_name = self.dataset_name(genomePath)
        
        with FastaFile(genomePath) as genome:
            
            chrom_list = genome.references
            chrom_total_count = len(chrom_list)
            chroms_valid_count = int(chrom_total_count * self.valid_fraction) + 1
            chroms_valid = random.sample(chrom_list, k=chroms_valid_count)
            
            chroms_train = list(set(chrom_list) - set(chroms_valid))
            
            chrom_valid_file = os.path.join(self.chrom_files_dir, f"{dataset_name}_valid.csv")
            with open(chrom_valid_file, 'w') as valid_handle:
                valid_handle.writelines(add_separators(chroms_valid))
            
            chrom_train_file = os.path.join(self.chrom_files_dir, f"{dataset_name}_train.csv")
            with open(chrom_train_file, 'w') as train_handle:
                train_handle.writelines(add_separators(chroms_train))
                
        return chrom_valid_file, chrom_train_file

    def update_metadata_with_chrom_files(self):
        updated_data = []
        for row in self.metadata.iter_rows(named=True):
            
            genomePath = row['genomePath']
            
            chrom_valid_file, chrom_train_file = self.write_chrom_files(genomePath=genomePath)
            
            train_dataset_entry = copy.deepcopy(row)
            valid_dataset_entry = row.copy()
            
            train_dataset_entry['chrom_file'] = os.path.realpath(chrom_train_file)
            train_dataset_entry['split'] = 'train'
            train_dataset_entry['dataset_name'] = self.dataset_name(genomePath=genomePath)
            valid_dataset_entry['chrom_file'] = os.path.relpath(chrom_valid_file)
            valid_dataset_entry['split'] = 'valid'
            valid_dataset_entry['dataset_name'] = self.dataset_name(genomePath=genomePath)
            
            updated_data.append(train_dataset_entry)
            updated_data.append(valid_dataset_entry)
            
        return DatasetList(metadata = pl.DataFrame(updated_data), seed=self.seed, chrom_files_dir=self.chrom_files_dir, valid_fraction=self.valid_fraction)
    
    def write_metadata(self, output_file:str):
        
        self.metadata.write_csv(output_file, include_header=True)
        

if __name__ == '__main__':
    DatasetList().update_metadata_with_chrom_files().write_metadata('metadata/config_entries.csv')
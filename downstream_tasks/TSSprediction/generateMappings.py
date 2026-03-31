import polars as pl
import numpy as np
import os
import json
from gtftools import gtftools


class Metadata():
    
    def __init__(self, 
                annotations:str = 'metadata/annotations.csv',
                sample2genome:str = 'metadata/file_mappings.csv',
                sample2taxon:str = 'metadata/sample2taxonType.json',
                genomesDIR:str = 'data/genomes/',
                annotationsDIR:str = 'data/annotations/',
                genomes_decompressed:bool = True,
                annotations_decompressed:bool = True
                ):
        
        self.annotationsDIR = annotationsDIR
        self.genomesDIR = genomesDIR
        
        for file, path in {
            'annotations': annotations, 
            'sample2genome': sample2genome,
            'sample2taxon': sample2taxon
        }.items():
            assert os.path.exists(path), f'File does not exist: {file}'
            
        for dir, path in {
            'genomesDIR' : genomesDIR,
            'annotationsDIR': annotationsDIR
        }.items():
            assert os.path.isdir(path), f'Dir does not exist: {dir}'
        
        
            
        self.annotations = pl.read_csv(annotations) \
                                .select('Sample','ScientificName','genome','gtf','liftOver_required') \
                                .filter((~pl.col('gtf').is_null()) & (~pl.col('liftOver_required'))) \
                                .with_columns(annotation = annotationsDIR + pl.col('gtf').str.split('/').list.gather([-1]).list.join('')) \
                                .drop('gtf', 'liftOver_required') \
                                .rename({'genome': 'genome_acc', 'Sample': 'id'})
        self.sample2genomePath = pl.read_csv(sample2genome) \
                                .select('id', 'genome') \
                                .with_columns(genomePath = genomesDIR + pl.col('genome').str.split('/').list.gather([-1]).list.join(''))
        
        with open(sample2taxon, 'r') as json_handle:
            self.sample2taxon = json.load(json_handle)
            
        self.final_metadata = self.annotations.join(self.sample2genomePath, on='id', how='left') \
                                                .with_columns(taxon = pl.col('id').replace_strict(self.sample2taxon)) #id, ScientificName, genome_acc, annotation, genomePath, taxon
        if genomes_decompressed:
            self.final_metadata = self.final_metadata.with_columns(pl.col('genomePath').str.strip_suffix('.gz'))
        if annotations_decompressed:
            self.final_metadata = self.final_metadata.with_columns(pl.col('annotation').str.strip_suffix('.gz'))
        
        
        self.condensed_metadata = self.final_metadata.unique('genome_acc').select('genome_acc', 'annotation', 'genomePath', 'taxon')
        
        
    def write(self, format:str, out_path:str='metadata/'):
        if format == 'full':
            self.final_metadata.write_csv(os.path.join(out_path, 'full_metadata.csv'))
        elif format == 'condensed':
            self.condensed_metadata.write_csv(os.path.join(out_path, 'condensed_metadata.csv'))
            
if __name__ == '__main__':
    Metadata().write(format='condensed')
            
        
            

        
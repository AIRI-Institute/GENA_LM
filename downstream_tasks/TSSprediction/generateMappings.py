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
                annotationsDIR:str = 'data/annotations/'
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
        
        
        self.condensed_metadata = self.final_metadata.unique('genome_acc').select('genome_acc', 'annotation', 'genomePath', 'taxon')
        
    
    
    @staticmethod
    def get_tss_region(GTFfile):

        TSSbed=[]

        f = open(GTFfile)
        for line in f:
            table = line.split('\t')
            if len(table) < 3:
                continue
            if table[2] == 'transcript':
                chrom  = table[0]
                strand = table[6] 
                tcx = line.split('transcript_id')[1].split('"')[1]
                geneid = line.split('gene_id')[1].split('"')[1]
                if "gene_name" in line:
                    genesymbol = line.split('gene_name')[1].split('"')[1]
                else:
                    genesymbol = geneid
                if  strand == "+":
                    iregion = {'chrom':chrom, 'TSS':int(table[3])-1, 'strand':strand, 'geneid':geneid, 'gene_symbol':genesymbol, 'annotation': GTFfile}
                elif strand == '-':
                    iregion = {'chrom':chrom, 'TSS':int(table[4]), 'strand':strand, 'geneid':geneid, 'gene_symbol':genesymbol, 'annotation': GTFfile}
                TSSbed.append(iregion)

        f.close()
        return pl.DataFrame(TSSbed)
    
        
    def write(self, format:str, out_path:str='metadata/'):
        if format == 'full':
            self.final_metadata.write_csv(os.path.join(out_path, 'full_metadata.csv'))
        elif format == 'condensed':
            self.condensed_metadata.write_csv(os.path.join(out_path, 'condensed_metadata.csv'))
            
if __name__ == '__main__':
    Metadata().write(format='condensed')
            
        
            

        
import pandas as pd
import os
import argparse
from tqdm.auto import tqdm
from pathlib import Path
import h5py
from Bio import SeqIO
import numpy as np
import re


dir = '/mnt/20tb/vsfishman/nn_interpretator/mammals_fasta_dataset/dataset'

TEST_SPECIES = ["Sarcophilus harrisii", "Monodelphis domestica", "Manis pentadactyla", "Equus asinus", "Sus scrofa", "Camelus dromedarius", 
                "Phyllostomus discolor", "Myotis daubentonii", "Suncus etruscus", "Lepus europaeus", "Ochotona princeps", 
                "Sciurus carolinensis", "Nycticebus coucang", "Lemur catta", "Cynocephalus volans", 'Elephas maximus indicus', 'Loxodonta africana',
                "Choloepus didactylus", "Ornithorhynchus anatinus", "Tachyglossus aculeatus"]

VALID_SPECIES = ["Lynx canadensis", "Neofelis nebulosa", "Eubalaena glacialis", "Balaenoptera musculus", "Cervus canadensis", "Dama dama",
                 "Meriones unguiculatus", "Chionomys nivalis", "Jaculus jaculus", "Callithrix jacchus"]



def convert_folders_to_hdf5(folders_paths, hdf5_save_path):
    
    with h5py.File(hdf5_save_path, 'w') as hdf5_file:
        for folder_path in tqdm(folders_paths, desc=f"Processing {hdf5_save_path.stem} data"):

            for file_path in tqdm(list(folder_path.glob('*.fasta')), desc=f"Processing {folder_path.stem} FASTA files"):

                with open(file_path) as handle:
                    for record in SeqIO.parse(handle, "fasta"):
                        # parsing descriptions
                        desc_line = record.description
                        chr, desc = re.findall("(chromosome [A-Z]?\d*)([^,]*)", desc_line)[0]
                        chr = chr.strip()
                        desc = desc.strip()
                        
                        # ingnore 'genomic patch of type FIX', 'genomic patch of type NOVEL',
                        if desc in ['', 'genomic scaffold', 'unlocalized genomic scaffold']:
                            seq = str(record.seq).strip('N').upper()

                            dataset_name = folder_path.name + '/' + record.id + " " + chr + " " + desc
                            # print(dataset_name)
                            dset = hdf5_file.create_dataset(
                                dataset_name,
                                shape=(len(seq),),
                                dtype='S1',
                                chunks=True,
                                compression='gzip'
                            )
                            dset[:] = np.frombuffer(seq.encode('utf-8'), dtype='S1')


def main(data_path, save_folder, sample_id_column):

    data_path = Path(data_path)
    save_folder = Path(save_folder)
    metadata_df = pd.read_csv(os.path.join(data_path, 'mammals_dataset_metadata3.txt'), sep='\t')

    # use only species with non-zero sex chromosomes
    metadata_df = metadata_df[(metadata_df['number_of_x'] != 0) & (metadata_df['number_of_y'] != 0)] 

    train_df = metadata_df[metadata_df['organism_name'].apply(lambda x: (x not in VALID_SPECIES) and (x not in TEST_SPECIES))]
    test_df = metadata_df[metadata_df['organism_name'].apply(lambda x: x in TEST_SPECIES)]
    valid_df = metadata_df[metadata_df['organism_name'].apply(lambda x: x in VALID_SPECIES)]

    data_splits = {
        "train": train_df, 
        "valid": valid_df,
        "test": test_df
        }

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    for split_name in tqdm(data_splits, desc="Creating HDF5 datasets"):
        hdf5_save_path = save_folder / f'{split_name}.h5'
        folders_paths = [data_path / sample_id for sample_id in data_splits[split_name][sample_id_column]]
        convert_folders_to_hdf5(folders_paths, hdf5_save_path)

"""
python convert_mammals_to_hdf5.py --data_path /mnt/20tb/vsfishman/nn_interpretator/mammals_fasta_dataset/dataset \
  --save_folder mammals_gender_data/ --sample_id_column assembly_accession

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data from fasta to HDF5 format")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--save_folder', type=str, required=True, help='Folder to save the HDF5 files')
    parser.add_argument('--sample_id_column', type=str, help='column name with sample id')

    args = parser.parse_args()

    main(args.data_path, args.save_folder, args.sample_id_column)

import argparse
from pathlib import Path
from Bio import SeqIO
from tqdm.auto import tqdm
import pandas as pd
import h5py
import numpy as np
import re

def stream_and_write_contigs_fasta(
    file_path, chr_name, sample_name, hdf5_file, min_gap=100):
    metadata = []

    def write_contig(seq, contig_id, start_pos, end_pos):
        contig_name = f"{chr_name}_contig_{contig_id}"
        dataset_name = f"{sample_name}/{contig_name}"
        dset = hdf5_file.create_dataset(
            dataset_name,
            shape=(len(seq),),
            dtype='S1',
            chunks=True,
            compression='gzip'
        )
        dset[:] = np.frombuffer(seq.encode('utf-8'), dtype='S1')
        metadata.append({
            "sample_id": sample_name,
            "chromosome": chr_name,
            "contig_name": contig_name,
            "contig_len": len(seq),
            "start": start_pos,
            "end": end_pos
        })

    seq_record = SeqIO.read(file_path, 'fasta')
    full_seq = str(seq_record.seq).upper()

    n_gap_regex = re.compile(f"N{{{min_gap},}}")
    matches = list(n_gap_regex.finditer(full_seq))

    contig_id = 0
    last_non_gapped_pos = 0

    for match in matches:
        # start and end of a gapped sequence
        gap_start, gap_end = match.span()
        # non-gapped sequence
        contig_seq = full_seq[last_non_gapped_pos:gap_start]
        if len(contig_seq) > 0:
            write_contig(contig_seq, contig_id, last_non_gapped_pos, gap_start)
            contig_id += 1
        last_non_gapped_pos = gap_end

    # Handle any remaining sequence after the last N-gap
    if last_non_gapped_pos < len(full_seq):
        write_contig(full_seq[last_non_gapped_pos:], contig_id, last_non_gapped_pos, len(full_seq))

    return metadata


def convert_folders_to_hdf5(hdf5_path, folders_paths):
    all_metadata = []
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for folder_path in tqdm(folders_paths, desc=f"Processing {hdf5_path.stem} data"):
            for file_path in tqdm(list(folder_path.glob('*.fasta')), desc=f"{folder_path.name} FASTA files"):
                sample_name = folder_path.name
                base_name = file_path.stem
                metadata = stream_and_write_contigs_fasta(file_path, base_name, sample_name, hdf5_file)
                all_metadata.extend(metadata)
    # Save metadata to CSV next to HDF5
    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = hdf5_path.with_suffix('.metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)

def main(data_path, train_csv, valid_csv, test_csv, save_folder, sample_id_column):
    data_path = Path(data_path)
    save_folder = Path(save_folder)

    data_splits = {}
    if train_csv is not None:
        data_splits['train'] = pd.read_csv(train_csv)
    if valid_csv is not None:
        data_splits['valid'] = pd.read_csv(valid_csv)
    if test_csv is not None:
        data_splits['test'] = pd.read_csv(test_csv)

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    for split_name in tqdm(data_splits, desc="Creating HDF5 datasets"):
        hdf5_path = save_folder / f'{split_name}.h5'
       
        folders_paths = [data_path / sample_id for sample_id in data_splits[split_name][sample_id_column]]
        convert_folders_to_hdf5(hdf5_path, folders_paths)


"""
python convert_to_hdf5.py --data_path /mnt/20tb/vsfishman/nn_interpretator/1000g_fasta_dataset/dataset/done \
  --train_csv /mnt/20tb/ykuratov/gender_data/train.csv \
  --valid_csv /mnt/20tb/ykuratov/gender_data/valid.csv \
  --test_csv /mnt/20tb/ykuratov/gender_data/test.csv \
  --save_folder /mnt/20tb/ykuratov/gender_data/

example how to read chunk from h5 data:
    def read_chunk_from_hdf5(hdf5_path, dataset_name, start, length):
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            dset = hdf5_file[dataset_name]
            chunk = dset[start:start + length].astype(str)
            return ''.join(chunk)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA genomes to contig-based HDF5 datasets")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--train_csv', type=str, help='Path to the train CSV file')
    parser.add_argument('--valid_csv', type=str, help='Path to the validation CSV file')
    parser.add_argument('--test_csv', type=str, help='Path to the test CSV file')
    parser.add_argument('--save_folder', type=str, required=True, help='Folder to save the HDF5 files')
    parser.add_argument('--sample_id_column', type=str, default='sample', help='Column with sample IDs')

    args = parser.parse_args()
    main(args.data_path, args.train_csv, args.valid_csv, args.test_csv, args.save_folder, args.sample_id_column)
#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd

import argparse
import glob
import h5py
import json
import functools
import os

from Bio import SeqIO, SeqRecord
from tqdm import tqdm


RECORDS_PER_TFRF = 256


def get_metadata(path: str) -> dict:
    """Open a file with metadata for a given dataset"""
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(path, 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def get_coordinates(path: str) -> pd.DataFrame:
    """Open a file with genomic coordinates for a given dataset"""
    data = pd.read_csv(os.path.join(path, 'sequences.bed'), sep='\t', 
                       names=('chr', 'start', 'end', 'subset'))
    return data


def tfrecord_files_by_subset(path: str, subset: str):
    """[from Enformer] Provide a list of TFRecordFiles for a dataset' subset"""
    # Sort the values by int(*).
    return sorted(tf.io.gfile.glob(os.path.join(
        path, 'tfrecords', f'{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def get_dataset_by_subset(path: str, subset: str, num_threads=2):
    """[from Enformer] Load whole dataset' subset"""
    metadata = get_metadata(path)
    dataset = tf.data.TFRecordDataset(
        tfrecord_files_by_subset(path, subset),
        compression_type='ZLIB', num_parallel_reads=1
        # num_parallel_reads>1 messes up order of records
    )
    dataset = dataset.map(
        functools.partial(deserialize, 
        metadata=metadata),
        num_parallel_calls=num_threads
    )
    return dataset


def get_dataset_from_tfr(path: str, tfr: str, num_threads=2):
    """Load records from one TFRecordFile"""
    metadata = get_metadata(path)
    dataset = tf.data.TFRecordDataset(
        tf.io.gfile.glob(tfr),
        compression_type='ZLIB', num_parallel_reads=1
    )
    dataset = dataset.map(
        functools.partial(deserialize, 
        metadata=metadata),
        num_parallel_calls=num_threads
    )
    return dataset


def deserialize(serialized_example, metadata):
    """[from Enformer] Deserialize bytes stored in TFRecordFile"""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)
  
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)
  
    return {'sequence': sequence,
            'target': target}


def process_record(record, 
                   index: int, 
                   coordinates: pd.DataFrame, 
                   genome: dict[str, SeqRecord], 
                   seq_len: int) -> dict:
    """Process a single record"""

    def extract_sequence(genome: dict[str, SeqRecord], 
                         chrom: str, start: int, end: int) -> str:
        """Retrieve sequence for a region in the genome"""
        if start < 0 or end > len(genome[chrom]):
            return ''
        return str(genome[chrom].seq[start:end]).upper()
    
    def around_center(start: int, end: int, length: int) -> list[int]:
        """Place new region at the center of a given region"""
        middle = (start + end) // 2
        new_start = middle - length//2
        new_end = middle + length//2
        return (new_start, new_end)

    target = record['target'].numpy()

    (chrom, start, end, split) = coordinates.iloc[index,:]
    (start, end) = around_center(start, end, seq_len)
    seq = extract_sequence(genome, chrom, start, end)

    if not seq:
        return dict()

    # we operate in BED coords (0-based, noninclusive end) 
    # but here need 1-based, inclusive end
    coords = f"{chrom}:{start+1}-{end}" 
    
    return {'seq': seq, 'target': target, 'coordinates': coords}


def process_subset(dataset, 
                   path: str, subset: str, 
                   genome: dict[str, SeqRecord], 
                   seq_len: int) -> list:
    """Process all records from a dataset' subset"""
    coordinates = get_coordinates(path)
    coordinates = coordinates[coordinates['subset'] == subset]

    metadata = get_metadata(path)
    if subset == 'train':
        key = 'train_seq'
    else:
        key = f'{subset}_seqs'
    total_records = metadata[key]

    for index, record in enumerate(tqdm(dataset, total=total_records)):
        processed = process_record(record, index, 
                                   coordinates, genome, seq_len)
        if processed:
            yield processed


def process_tfr(tfr: str, 
                path: str, subset: str, 
                genome: dict[str, SeqRecord], 
                seq_len: int) -> list:
    """Process all records from a TFRecordFile"""

    def get_tfr_offset(tfr: str) -> int:
        """Compute index offset in the dataset for a given TFRecordFile"""
        return int(tfr.split('-')[-1].split('.')[0]) * RECORDS_PER_TFRF
        
    dataset = get_dataset_from_tfr(path, tfr)
    
    coordinates = get_coordinates(path)
    coordinates = coordinates[coordinates['subset'] == subset]
    offset = get_tfr_offset(tfr)
    coordinates = coordinates.iloc[offset:offset+RECORDS_PER_TFRF,:]

    result = []
    for index, record in enumerate(tqdm(dataset, total=RECORDS_PER_TFRF)):
        processed = process_record(record, index, 
                                   coordinates, genome, seq_len)
        if processed:
            result.append(processed)

    return result


def dna_1hot_simple(seq: str) -> np.ndarray:
    """Get 1-hot-bool for a given DNA sequence"""
    seq = seq.upper()
    seq_len = len(seq)
    seq_code = np.zeros((seq_len, 4), dtype='bool')

    translate = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }

    for i in range(seq_len):
        try:
            seq_code[i, translate[seq[i]]] = 1
        except KeyError:
            pass

    return seq_code


def export_tfr(records: list, out_file: str) -> None:
    """Store records in a TFRecords file (of same structure)"""

    def feature_bytes(values):
        """[from Basenji] Convert numpy arrays to bytes features."""
        values = values.flatten().tobytes()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
    with tf.io.TFRecordWriter(out_file, tf_opts) as writer:
        for record in tqdm(records):
            seq_1hot = dna_1hot_simple(record['seq'])
            features_dict = {
                'sequence': feature_bytes(seq_1hot),
                'target': feature_bytes(record['target'])
            }
            example = tf.train.Example(features=tf.train.Features(feature=features_dict))
            writer.write(example.SerializeToString())


def export_hdf5(records: list, out_file: str) -> None:
    """Store records in a HDF5 file"""
    with h5py.File(out_file, 'w') as f:
        for index, record in enumerate(tqdm(records)):
            g = f.create_group(str(index))
            g.create_dataset('seq', shape=(), data=record['seq'])
            g.create_dataset('target', data=record['target'],
                             compression='gzip')
            g.attrs['coordinates'] = record['coordinates']


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base-path', type=str, required=True,
                        help='Path to the organism subfolder of Basenji dataset')
    parser.add_argument('--subset', choices=['train', 'test', 'valid'], required=True,
                        help='Name of data subset')
    parser.add_argument('--tfr', type=str,
                        help='Path to the one TFRecord file to convert')
    parser.add_argument('--genome', type=str, required=True,
                        help='Path to the corresponding reference genome (in fasta)')
    parser.add_argument('--sequence-length', type=int, default='4096',
                        help='Length of sequences to get from the reference')

    ogroup = parser.add_mutually_exclusive_group(required=True)
    ogroup.add_argument('--out-hdf5', type=str,
                        help='Path to the output HDF5 file')
    ogroup.add_argument('--out-tfr', type=str, 
                        help='Path to the output TFRecords file')

    return parser.parse_args()


def main():
    args = parse_arguments()

    print(f"Loading genome ({args.genome})...")
    genome = SeqIO.to_dict(SeqIO.parse(args.genome, 'fasta'))

    if args.tfr:
        print(f"Processing records from one TFR file ({args.tfr}):")
        records = process_tfr(args.tfr, args.base_path, args.subset,
                              genome, args.sequence_length)
    else:
        print(f"Processing whole subset ({args.subset}):")
        dataset = get_dataset_by_subset(args.base_path, args.subset)    
        records = process_subset(dataset, args.base_path, args.subset,
                                 genome, args.sequence_length)
    
    if args.out_hdf5:
        print(f"Writing output HDF5 ({args.out_hdf5}):")
        export_hdf5(records, args.out_hdf5)
    else:
        print(f"Writing output TFRecords ({args.out_tfr}):")
        export_tfr(records, args.out_tfr)
    print("Done!")


if __name__ == '__main__':
    main()

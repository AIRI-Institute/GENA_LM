# Enformer

Predict CAGE signal (=gene expression) based on DNA sequence.

Input sequence length: 196608 bps = left context (40960 bps) + 128 bps * 896 bins + right context (40960 bps).

Target: 896 x 5313. 5313 values for 896 bins. Each bin corresponds to 128bps.

## Resources and useful links:

1. Enformer paper: [https://www.nature.com/articles/s41592-021-01252-x](https://www.nature.com/articles/s41592-021-01252-x#Sec17)
2. Basenji github: [https://github.com/calico/basenji/tree/master/manuscripts/cross2020](https://github.com/calico/basenji/tree/master/manuscripts/cross2020)
3. Model usage example: [https://tfhub.dev/deepmind/enformer/1](https://tfhub.dev/deepmind/enformer/1)
4. enformer-pytorch: [https://github.com/lucidrains/enformer-pytorch](https://github.com/lucidrains/enformer-pytorch)

## Convert Basenji dataset

Enformer is trained and tested on the 
[Basenji dataset](https://console.cloud.google.com/storage/browser/basenji_barnyard/data).
It is in TensorFlow format and has 1-hot sequence encoding for genomic regions of length 131 072 bp.
Meanwhile [Enformer model published on TFhub](https://tfhub.dev/deepmind/enformer/1) 
requires length of 393 216 bp.

For GENA-LM we'd use input sequence as str (4096 bp for a start), and put records into HDF5.
```shell
# assuming already downloaded Basenji dataset
BASENJI_BASE_DIR=/path/to/basenji
OUTPUT_BASE_DIR=/path/to/export

HUMAN_REFERENCE=/path/to/hg38.fa
MOUSE_REFERENCE=/path/to/mm39.fa

echo "Convert for GENA-LM"
# ~8 GB RAM per process, 100-150% CPU usage
for SUBSET in train test valid ; do
  parallel -j 8 ./convert_basenji_dataset.py --base-path $BASENJI_BASE_DIR/human \ 
    --subset $SUBSET --genome $HUMAN_REFERENCE --tfr {} \
    --out-hdf5 $OUTPUT_BASE_DIR/human/gena-lm-4Kbp/{/.}.h5 \
    ::: $BASENJI_BASE_DIR/human/tfrecords/$SUBSET-*tfr
  parallel -j 8 ./convert_basenji_dataset.py --base-path $BASENJI_BASE_DIR/mouse \ 
    --subset $SUBSET --genome $MOUSE_REFERENCE --tfr {} \
    --out-hdf5 $OUTPUT_BASE_DIR/mouse/gena-lm-4Kbp/{/.}.h5 \
    ::: $BASENJI_BASE_DIR/mouse/tfrecords/$SUBSET-*tfr
done

echo "Convert for Enformer"
for SUBSET in train test valid ; do
  parallel -j 8 ./convert_basenji_dataset.py --base-path $BASENJI_BASE_DIR/human \ 
    --subset $SUBSET --genome $HUMAN_REFERENCE --tfr {} --sequence-length 393216 \
    --out-tfr $OUTPUT_BASE_DIR/human/enformer-393Kbp/{/.}.tfr \
    ::: $BASENJI_BASE_DIR/human/tfrecords/$SUBSET-*tfr
  parallel -j 8 ./convert_basenji_dataset.py --base-path $BASENJI_BASE_DIR/mouse \ 
    --subset $SUBSET --genome $MOUSE_REFERENCE --tfr {} --sequence-length 393216 \
    --out-tfr $OUTPUT_BASE_DIR/mouse/enformer-393Kbp/{/.}.tfr \
    ::: $BASENJI_BASE_DIR/mouse/tfrecords/$SUBSET-*tfr
done
```

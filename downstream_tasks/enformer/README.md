# Enformer

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

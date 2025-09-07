import tqdm
from simple_annotation_dataset import GenomicAnnotationDataset
from transformers import AutoTokenizer

def __main__():
	model_input_length_token = 512
	av_token_len = 9.0
	path_to_gff_db = "../../data/annotation/gencode.v48.annotation.gtf.db"
	path_to_fasta = "/mnt/nfs_dna/minja/hg38.fa"
	
	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t")
	
	# Create dataset
	dataset = GenomicAnnotationDataset(
		model_input_length_token=model_input_length_token,
		av_token_len=av_token_len,
		path_to_gff_db=path_to_gff_db,
		path_to_fasta=path_to_fasta,
		tokenizer=tokenizer,
		primary_transcript_types=["protein_coding"]
	)
	
	print(f"Dataset length: {len(dataset)}")
	for sample in tqdm.tqdm(dataset):
		pass
	print ("Done")

if __name__ == "__main__":
    __main__()
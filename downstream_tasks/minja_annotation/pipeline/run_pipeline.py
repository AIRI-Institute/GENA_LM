from hydra import initialize_config_dir, compose
from omegaconf import open_dict
from argparse import ArgumentParser
import logging
import os
from pathlib import Path
import torch
from pipeline_core import (
	run_inference,
    prepare_preds_for_peaks_onlyRC,
    peak_finding,
    find_tss_polya_pairs_right_left_only,
    filter_bed_by_intragenic,
	shift_bed_by_UCSC_chr_header,
	peaks2file
)
import numpy as np

parser = ArgumentParser()

# inference arguments
GENALM_HOME = os.environ.get('GENALM_HOME', '')

parser.add_argument('--cpt4', type=str,
 default=os.path.join(GENALM_HOME, 
 					  'runs/4class/MLGENX_modernGENA_rc_shift_mRNA_and_lncRNA_middle_pretrain_full_BCE/checkpoint-22750/model.safetensors'
					  ), 
 help='path to the 4-class model checkpoint')
parser.add_argument('--cpt6', type=str,
 default=os.path.join(GENALM_HOME, 
 					  'runs/6class/MLGENX_modernGENA_rc_shift_middle_pretrain_6_classes_with_intragenic_8192/checkpoint-15750/model.safetensors'
					  ), 
 help='path to the 6-class model checkpoint')
parser.add_argument('--cfg4', type=str, default="eval_modernGENA_4class.yaml", 
	help='path to the experiment config for 4-class model')
parser.add_argument('--cfg6', type=str, default="eval_modernGENA_6class.yaml", 
	help='path to the experiment config for 6-class model')
parser.add_argument('--batch_size', type=int, default=48, help='batch size') 
parser.add_argument('--fasta', type=str, default=None, help='path to the fasta file.')

# bigwig inputs
parser.add_argument('--bw_mode', action='store_true', 
	help='do not run inference, use provided bigwig files instead')
parser.add_argument('--chrom', type=str, default=None, help='chromosome name')

parser.add_argument('--bw4', type=str,
	default=os.path.join(GENALM_HOME, 'runs/annotation/modernGENAlarge_ep36_rc_shift/checkpoint-38750/eval/T2T-CHM13v2/NC_060944.1/'),
	help='path to 4-class eval BigWig directory for the chromosome')
parser.add_argument('--bw6', type=str,
	default=os.path.join(GENALM_HOME, 'runs/annotation/modernGENA_rc_shift_middle_pretrain_shawerma_6_classes_1024/checkpoint-268750/eval/T2T-CHM13v2/NC_060944.1'),
	help='directory containing 6-class intragenic BigWig files for BED filtering')

parser.add_argument('--bw_plus', type=str, default='intragenic_+.bw', 
	help='BigWig filename for plus strand')
parser.add_argument('--bw_minus', type=str, default='intragenic_-.bw', 
	help='BigWig filename for minus strand')
parser.add_argument('--bw_plus_rc', type=str, default='intragenic_+rev_comp_.bw', 
	help='BigWig filename for plus rev comp')
parser.add_argument('--bw_minus_rc', type=str, default='intragenic_-rev_comp_.bw', 
	help='BigWig filename for minus rev comp')

# postprocessing options
parser.add_argument('--lp_frac', type=float, default=0.05, help='peak finding LP_FRAC')
parser.add_argument('--pk_prom', type=float, default=0.1, help='peak finding PK_PROM')
parser.add_argument('--pk_dist', type=int, default=50, help='peak finding PK_DIST')
parser.add_argument('--pk_height', type=float, default=None, help='peak finding PK_HEIGHT')
parser.add_argument('--k', type=int, default=10, help='parameter k for find_tss_polya_pairs_right_left_only')
parser.add_argument('--prob_threshold', type=float, default=0.5, help='intragenic probability threshold for BED filtering')
parser.add_argument('--zero_fraction_drop_threshold', type=float, default=0.01,
	help='drop intervals with zero-fraction above this')

# output

parser.add_argument('--intermediate_files', action='store_true',
	help='Save intermediate files')

parser.add_argument('--bed_out', type=str, default=None,
	help='output BED file path')

# bed post-processing arguments
parser.add_argument('--shift', default=None, 
	help='Shift predictions by this number of bases. Useful when testing on reference chromosomal region with known start and end. If set to "UCSC", will determine shift from the FASTA header. Default is None - not shifting.')

# log arguments
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')

def main():	
	# Set up logging
	args = parser.parse_args()
	logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=args.log_level)
	logger = logging.getLogger()

	# create output directory if it doesn't exist
	if args.bed_out is None:
		if args.bw_mode:
			args.bed_out = args.bw4 + 'TSS_polya_filtered.bed'
		elif args.fasta is not None:
			args.bed_out = args.fasta + '.bed'
		else:
			raise ValueError("Either --bw_mode, --fasta, or --bed_out must be provided")

	if not os.path.exists(os.path.dirname(args.bed_out)):
		os.makedirs(os.path.dirname(args.bed_out))

	# run inference if not in bigwig mode
	if not args.bw_mode:
		logger.info("Running inference...")
		if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
			os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

		logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
		logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

		datahandlers = []
		for cfg, cpt in zip([args.cfg4, args.cfg6], [args.cpt4, args.cpt6]):
			logger.info(f"Running inference for {cfg} with checkpoint {cpt}")
			
			experiment_config_path = Path(cfg).expanduser().absolute()

			with initialize_config_dir(str(experiment_config_path.parents[0])):
				experiment_config = compose(config_name=experiment_config_path.name)

			experiment_config.eval_dataset["path_to_fasta"] = args.fasta
			with open_dict(experiment_config):
				experiment_config.batch_size = args.batch_size


			model_cpt = cpt
			datahandlers.append(run_inference(experiment_config, model_cpt, logger))
			num_processed_chromosomes = len(datahandlers[-1].chrom_data.keys())
			assert num_processed_chromosomes == 1, f"One and only one chromosome is supported for inference. Found {num_processed_chromosomes} chromosomes"
		
		chromosome = list(datahandlers[-1].chrom_data.keys())[0]
		bw4 = datahandlers[0]
		bw6 = datahandlers[1]
	else:
		chromosome = args.chrom
		bw4 = args.bw4
		bw6 = args.bw6
		logger.info("--bw was set. Skipping inference and using provided bigwig files...")


	tss_plus, polya_plus, tss_minus, polya_minus = prepare_preds_for_peaks_onlyRC(bw4, chromosome, logger)

	logger.info("Running peak finding...")
	X = np.array([
		tss_plus,
		polya_plus,
		tss_minus,
		polya_minus
	])

	arr = peak_finding(X, args.lp_frac, args.pk_prom, args.pk_dist, args.pk_height)
	if args.intermediate_files:
		peaks2file(arr, f"{args.bed_out}.unpaired_peaks.bed", chromosome, logger=logger)
		if args.shift == "UCSC":
			shift_bed_by_UCSC_chr_header(f"{args.bed_out}.unpaired_peaks.bed")

	pairs = find_tss_polya_pairs_right_left_only(arr, chrom_name=chromosome, 
													k=args.k, 
													out_bed_path=f"{args.bed_out}.no_intergagenic.filter.bed" if args.intermediate_files else None,
													progress_every=1000,
													logger=logger
												)
	if args.intermediate_files and args.shift == "UCSC":
		shift_bed_by_UCSC_chr_header(f"{args.bed_out}.no_intergagenic.filter.bed")

	filter_bed_by_intragenic(pairs, bw_dir = bw6, 
							BW_PLUS = args.bw_plus, BW_MINUS = args.bw_minus, 
							BW_PLUS_RC = args.bw_plus_rc, BW_MINUS_RC = args.bw_minus_rc,
							prob_threshold = args.prob_threshold, 
							zero_fraction_drop_threshold = args.zero_fraction_drop_threshold, 
							bed_out = args.bed_out,
							logger = logger
							)
	logger.info(f"Wrote {len(pairs)} intervals to file: {args.bed_out}")

	if args.shift == "UCSC":
		shift_bed_by_UCSC_chr_header(args.bed_out)

	logger.info("Pipeline completed successfully")

if __name__ == '__main__':
	main()
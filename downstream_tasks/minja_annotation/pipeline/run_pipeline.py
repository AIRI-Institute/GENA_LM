from hydra import initialize_config_dir, compose
from argparse import ArgumentParser
import logging
import os
from pathlib import Path
import torch
from pipeline_core import run_inference

parser = ArgumentParser()

# inference arguments
GENALM_HOME = os.environ.get('GENALM_HOME', '')
parser.add_argument('--cpt4', type=str,
 default=os.path.join(GENALM_HOME, 
 					  'runs/4class/MLGENX_modernGENA_rc_shift_mRNA_and_lncRNA_middle_pretrain_full_BCE/checkpoint-22750/model.safetensors'
					  ), 
 help='path to the model checkpoint')
parser.add_argument('--cpt6', type=str,
 default=os.path.join(GENALM_HOME, 
 					  'runs/6class/MLGENX_modernGENA_rc_shift_middle_pretrain_6_classes_with_intragenic_8192/checkpoint-15750/model.safetensors'
					  ), 
 help='path to the model checkpoint')
parser.add_argument('--cfg4', type=str, default="eval_modernGENA_4class.yaml", 
	help='path to the experiment config for 4-class model')
parser.add_argument('--cfg6', type=str, default="eval_modernGENA_6class.yaml", 
	help='path to the experiment config for 6-class model')
parser.add_argument('--batch_size', type=int, default=48, help='batch size') 
parser.add_argument('--fasta', type=str, required=True, default=None, help='path to the fasta file. Mutually exclusive with fasta_txt')

# bed post-processing arguments
parser.add_argument('--shift', default=None, help='Shift predictions by this number of bases. Useful when testing on reference chromosomal region with known start and end. Default is 0.')

# log arguments
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')

def main():	
	# Set up logging
	args = parser.parse_args()
	logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=args.log_level)
	logger = logging.getLogger()

	# Run inference for 4-class model

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
		from omegaconf import open_dict
		with open_dict(experiment_config):
			experiment_config.batch_size = args.batch_size

		model_cpt = cpt
		datahandlers.append(run_inference(experiment_config, model_cpt, logger))

if __name__ == '__main__':
	main()
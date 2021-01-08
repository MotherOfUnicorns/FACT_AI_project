import argparse
parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str, choices=['vanilla_lstm', 'ortho_lstm', 'diversity_lstm'], required=True)
parser.add_argument("--diversity",type=float,default=0)
parser.add_argument("--seed", type=int,default=0)

args, extras = parser.parse_known_args()
args.extras = extras
args.attention = 'tanh'

from Transparency.Trainers.DatasetBC import *
from Transparency.ExperimentsBC import *

import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	#torch.set_deterministic(True)
	torch.cuda.manual_seed_all(args.seed)
	#torch.backends.cudnn.deterministic = True
	#torch.backends.cudnn.benchmark = False
print('NEWJOB')
print('seed ',args.seed,', dataset ',args.dataset,', model ',args.encoder,', diversity ',args.diversity)

dataset = datasets[args.dataset](args)

if args.output_dir is not None :
    dataset.output_dir = args.output_dir

dataset.diversity = args.diversity
encoders = [args.encoder]

train_dataset_on_encoders(dataset, encoders)
generate_graphs_on_encoders(dataset, encoders)


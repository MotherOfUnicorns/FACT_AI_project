import argparse

parser = argparse.ArgumentParser(description="Run experiments on a dataset")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument(
    "--encoder",
    type=str,
    choices=["vanilla_lstm", "ortho_lstm", "diversity_lstm"],
    required=True,
)
parser.add_argument(
    "--attention", type=str, choices=["tanh", "dot", "all"], default="tanh"
)  # TODO: does attention work with dot/all? what are they?
parser.add_argument("--diversity", type=float, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--job_type", type=str, default="both", choices=["both", "train", "experiment"]
)

args, extras = parser.parse_known_args()
args.extras = extras

import random
import numpy as np
import torch
from Transparency.Trainers.DatasetBC import datasets
from Transparency.ExperimentsBC import (
    train_dataset,
    train_dataset_on_encoders,
    generate_graphs_on_encoders,
    run_experiments_on_latest_model,
    run_rationale_on_latest_model,
)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    # torch.set_deterministic(True)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

print("NEWJOB")
print(
    "seed ",
    args.seed,
    ", dataset ",
    args.dataset,
    ", model ",
    args.encoder,
    ", diversity ",
    args.diversity,
)

dataset = datasets[args.dataset](args)

if args.output_dir is not None:
    dataset.output_dir = args.output_dir

dataset.diversity = args.diversity
encoders = [args.encoder]

if args.job_type == "both":
    train_dataset_on_encoders(dataset, encoders)
    generate_graphs_on_encoders(dataset, encoders)
elif args.job_type == "train":
    # only train the (ortho/diverse)lstm+attention model, without other experiments
    train_dataset(dataset, args.encoder)
elif args.job_type == "experiment":
    # only run experiments using the latest trained model
    run_experiments_on_latest_model(dataset, args.encoder)
    run_rationale_on_latest_model(dataset, args.encoder)
    generate_graphs_on_encoders(dataset, encoders)

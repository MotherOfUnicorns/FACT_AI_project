import os

from Transparency.common_code.common import get_latest_model
from Transparency.configurations import configurations
from Transparency.Trainers.PlottingBC import generate_graphs
from Transparency.Trainers.TrainerBC import Evaluator, RationaleTrainer, Trainer


def update_config_with_args(dataset, args):

    config = configurations[args.encoder](dataset)
    config['model']['decoder']['attention']['type'] = args.attention
    return config


def train_dataset(dataset, args):
    print("STARTING TRAINING")

    config = update_config_with_args(dataset, args)
    trainer = Trainer(dataset, config=config, _type=dataset.trainer_type)
    if hasattr(dataset, "n_iter"):
        n_iters = dataset.n_iter
    else:
        n_iters = 8

    trainer.train(
        dataset.train_data,
        dataset.dev_data,
        n_iters=n_iters,
        save_on_metric=dataset.save_on_metric,
    )
    evaluator = Evaluator(dataset, trainer.model.dirname, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return trainer, evaluator


def run_rationale_on_latest_model(dataset, args):
    config = update_config_with_args(dataset, args)
    latest_model = get_latest_model(
        os.path.join(config["training"]["basepath"], config["training"]["exp_dirname"])
    )
    rationale_gen = RationaleTrainer(
        dataset, config, latest_model, _type=dataset.trainer_type
    )
    print("Training the Rationale Generator ...")
    _ = rationale_gen.train(dataset.train_data, dataset.dev_data)
    print("Running Exp to Compute Attention given to Rationales ...")
    rationale_gen.rationale_attn_experiment(dataset.test_data)
    return rationale_gen


def run_evaluator_on_latest_model(dataset, args):
    print("EVALUATING LATEST MODEL")

    config = update_config_with_args(dataset, args)
    latest_model = get_latest_model(
        os.path.join(config["training"]["basepath"], config["training"]["exp_dirname"])
    )
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return evaluator


def run_experiments_on_latest_model(dataset, args, force_run=True):
    evaluator = run_evaluator_on_latest_model(dataset, args)
    test_data = dataset.test_data
    print("RUNNING GRADIENT EXPERIMENT ON LATEST MODEL")
    evaluator.gradient_experiment(test_data, force_run=force_run)
    print("RUNNING QUANTITATIVE ANALYSIS EXPERIMENT ON LATEST MODEL")
    evaluator.quantitative_analysis_experiment(test_data, dataset, force_run=force_run)
    print("RUNNING IMPORTANCE RANKING EXPERIMENT ON LATEST MODEL")
    evaluator.importance_ranking_experiment(test_data, force_run=force_run)
    print("RUNNING CONICITY ANALYSIS EXPERIMENT ON LATEST MODEL")
    evaluator.conicity_analysis_experiment(test_data)
    print("RUNNING PERMUTATION EXPERIMENT ON LATEST MODEL")
    evaluator.permutation_experiment(test_data, force_run=force_run)
    print("RUNNING INTEGRATED GRADIENT EXPERIMENT ON LATEST MODEL")
    evaluator.integrated_gradient_experiment(dataset, force_run=force_run)


def generate_graphs_on_latest_model(dataset, args):
    print("GENERATING GRAPHS FOR EXPERIMENT ON LATEST MODEL")

    config = update_config_with_args(dataset, args)
    latest_model = get_latest_model(
        os.path.join(config["training"]["basepath"], config["training"]["exp_dirname"])
    )
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=False)
    generate_graphs(
        dataset,
        config["training"]["exp_dirname"],
        evaluator.model,
        test_data=dataset.test_data,
    )

from Transparency.common_code.common import *
from Transparency.Trainers.PlottingQA import generate_graphs
from Transparency.configurations import configurations_qa
from Transparency.Trainers.TrainerQA import Trainer, Evaluator


def train_dataset(dataset, config):
    print("STARTING TRAINING")

    config = configurations_qa[config](dataset)
    n_iters = dataset.n_iters if hasattr(dataset, "n_iters") else 25
    trainer = Trainer(dataset, config=config, _type=dataset.trainer_type)
    trainer.train(
        dataset.train_data,
        dataset.dev_data,
        n_iters=n_iters,
        save_on_metric=dataset.save_on_metric,
    )
    return trainer


def run_evaluator_on_latest_model(dataset, config):
    print("EVALUATING LATEST MODEL")

    config = configurations_qa[config](dataset)
    latest_model = get_latest_model(
        os.path.join(config["training"]["basepath"], config["training"]["exp_dirname"])
    )
    evaluator = Evaluator(dataset, latest_model)
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return evaluator


def run_experiments_on_latest_model(dataset, config, force_run=True):

    evaluator = run_evaluator_on_latest_model(dataset, config)
    test_data = dataset.test_data

    print("RUNNING GRADIENT EXPERIMENT ON LATEST MODEL")
    evaluator.gradient_experiment(test_data, force_run=force_run)
    print("RUNNING IMPORTANCE RANKING EXPERIMENT ON LATEST MODEL")
    evaluator.importance_ranking_experiment(test_data, force_run=force_run)
    print("RUNNING PERMUTATION EXPERIMENT ON LATEST MODEL")
    evaluator.permutation_experiment(test_data, force_run=force_run)
    print("RUNNING QUANTITATIVE ANALYSIS EXPERIMENT ON LATEST MODEL")
    evaluator.quantitative_analysis_experiment(test_data, dataset, force_run=force_run)
    print("RUNNING INTEGRATED GRADIENT EXPERIMENT ON LATEST MODEL")
    evaluator.integrated_gradient_experiment(
        test_data, force_run=force_run, no_of_instances=len(test_data.P)
    )


def generate_graphs_on_latest_model(dataset, config):
    print("GENERATING GRAPHS FOR EXPERIMENT ON LATEST MODEL")

    config = configurations_qa[config](dataset)
    latest_model = get_latest_model(
        os.path.join(config["training"]["basepath"], config["training"]["exp_dirname"])
    )
    if latest_model is not None:
        evaluator = Evaluator(dataset, latest_model)
        _ = evaluator.evaluate(dataset.test_data, save_results=True, is_embds=False)
        print("outside eval")
        generate_graphs(
            dataset,
            config["training"]["exp_dirname"],
            evaluator.model,
            test_data=dataset.test_data,
        )


def train_dataset_on_encoders(dataset, encoders):
    for e in encoders:
        train_dataset(dataset, e)
        run_experiments_on_latest_model(dataset, e)


def generate_graphs_on_encoders(dataset, encoders):
    for e in encoders:
        generate_graphs_on_latest_model(dataset, e)


def get_results(path):
    latest_model = get_latest_model(path)
    if latest_model is not None:
        evaluations = json.load(open(os.path.join(latest_model, "evaluate.json"), "r"))
        return evaluations
    else:
        raise LookupError("No Latest Model ... ")

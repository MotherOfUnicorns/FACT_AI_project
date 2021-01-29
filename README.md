# FACT AI project - University of Amsterdam 2021

This is a reproduction study for the paper [Towards Transparent and Explainable Attention Models](https://www.aclweb.org/anthology/2020.acl-main.387/) (ACL 2020)

This code is built on the [original authors' repository](https://github.com/akashkm99/Interpretable-Attention).

## Structure and overview

- The `job_scripts` directory contains some of the scripts we used to train the models on the [lisa cluster](https://userinfo.surfsara.nl/systems/lisa).
- The `Notes` directory contains our notes while working on this project.
- The `Transparency` directory on the `master` branch of this repository contains a duplication of the [original code](https://github.com/akashkm99/Interpretable-Attention) with some minor additions:
    *  We added a seeding mechanism
    *  We added the flexibility to separate the train phase and the experiment phase. This makes it easier to handle large datasets that take a long compute time.

### other branches

In the other branches we tested out some hypothesis and extensions of the models.
Here we provide an overview of the different branches, and details for running scripts in different branches are in the sections below:
- `biLSTM` contains our extension of using biLSTM as encoder instead of uni-directional LSTM
- `const_attention` enables forcing the attention weights of the model to be 1) equal on all hidden representations, or 2) all zeros except for the first hidden representation, or 3) all zeros except for the last hidden representation
- `embedding_params` do not fine-tune the pre-trained embeddings if they are used
- `lime` contains our experiments that compare attention weights with [LIME](https://github.com/marcotcr/lime) scores
- `Q_route_fix` contains our investigation about whether orthogonalisation should also be applied on the Q-route when training models
- `dataset_analysis` was use to run additional experiments.


## Installation 

- Clone this repository
  `git clone git@github.com:MotherOfUnicorns/FACT_AI_project.git`
- Move to the project directory
  `cd FACT_AI_project`
- Add the parent directory of `Transparency` directory (which should be your current directory) to your python path
  `export PYTHONPATH=$PYTHONPATH:$(pwd)`


## Requirements 

- Python 3.6 or 3.7
- Install all required packages as specified:
    ```
    pip install -r requirements.txt
    ```
- To run the LIME experiments, also
    ```
    pip install lime
    ```
- Install the required packages and download the spacy en model:
    ```
    python -m spacy download en
    ```
- Download some nltk taggers needed for the experiments:
    ```
    python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('universal_tagset')"
    ```


## Preparing the Datasets 

Each dataset has a separate jupyter notebook in the `Transparency/preprocess` folder.
Follow the instructions in the jupyter notebooks to download and preprocess the datasets.

Alternatively, we have made our pre-processed datasets available on the lisa cluster at `/home/lgpu0136/project/Transparency/preprocess`


## Training & Running Experiments (on the `master` branch)

How to train and experiment with models using different encoders, including `vanilla_lstm, ortho_lstm, diversity_lstm`:

### single input sequence tasks

Datasets available are `sst, imdb, yelp, amazon, 20News_sports, tweet`.

An example to train and test the orthogonal LSTM model on `imdb` dataset:
```
dataset_name=imdb
model_name=ortho_lstm
output_path=./experiments
python Transparency/train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir Transparency --output_dir ${output_path} --encoder ${model_name}
```

To use the `diversity_lstm` model, an additional `--diversity` flag is needed to specify the diversity weight:
```
python Transparency/train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir Transparency --output_dir ${output_path} --encoder diversity_lstm --diversity 0.5
```

### dual input sequences tasks

Datasets available are `snli, qqp, babi_1, babi_2, babi_3, cnn`.

An example to train and test the orthogonal LSTM model on `babi_1` dataset:
```
dataset_name=babi_1
model_name=ortho_lstm
output_path=./experiments
python Transparency/train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir Transparency --output_dir ${output_path} --encoder ${model_name}
```

Similarly, to use the `diversity_lstm` model, an additional `--diversity` flag is needed to specify the diversity weight:
```
python Transparency/train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir Transparency --output_dir ${output_path} --encoder ${model_name} --diversity 0.5
```


### other arguments

Using the `--job_type` flag, you can specify whether you want to only train the model:
```
--job_type train
```
or to only run the experiments when you already have a trained model:
```
--job_type experiment
```
The default behaviour if you don't specify this flag is performing both train and experiment, i.e. equivalent to
```
--job_type both
```

Using the `--seed` flag, you can manually set a random seed.
If unspecified, the default seed is zero, i.e. equivalent to
```
--seed 0
```

### branch-specific tasks

The following are only available in the specified branches.
To use them, first move to the correct branch:
```
git checkout [branchname]
```


#### `biLSTM` branch

The `--encoder` flag now can accept either one of these arguments:
- `vanilla_lstm`
- `ortho_lstm`
- `diversity_lstm`
- `bi_lstm`
- `ortho_bi_lstm`
- `diversity_bi_lstm`

#### `const_attention` branch

Using the `--attention` flag, you can switch between different attention weight distributions:
- `tanh`: normal unconstrained attention weight with tanh activation
- `equal`: equal attention weights to all words in the sentence
- `first_only`: all attention weights are concentrated on the first word
- `last_only`: all attention weights are concentrated on the last word (equivalent to LSTM only without attention)


#### `embedding_params` branch

The utilities in the `const_attention` branch are also available here.

#### `lime` branch

First make sure you have a trained model,
then use the `Transparency/lime_explanation_bc.py` or `Transparency/lime_explanation_qa.py` script to compare attention weights with LIME scores,
depending on the specific dataset.

##### single input sequence tasks

Datasets available are `sst, imdb, yelp, amazon, 20News_sports, tweet`.

An example to run the LIME experiment on the `imdb` dataset with `ortho_lstm` encoder:
```
dataset_name=imdb
model_name=ortho_lstm
output_path=./experiments # make sure this the same directory as when you trained your model
python Transparency/lime_explanation_bc.py --dataset ${dataset_name} --data_dir Transparency --output_dir ${output_path} --encoder ${model_name}
```

The LIME outputs will also be found in `${output_path}`.

When using `diversity_lstm` as the encoder, an additional `--diversity` flag is needed to specify the diversity weight.

##### dual input sequences tasks

Datasets available are `snli, qqp, babi_1, babi_2, babi_3, cnn`.

An example to run the LIME experiment on the `babi_1` dataset with `ortho_lstm` encoder:
```
dataset_name=babi_1
model_name=ortho_lstm
output_path=./experiments # make sure this the same directory as when you trained your model
python Transparency/lime_explanation_qa.py --dataset ${dataset_name} --data_dir Transparency --output_dir ${output_path} --encoder ${model_name}
```

The LIME outputs will also be found in `${output_path}`.

When using `diversity_lstm` as the encoder, an additional `--diversity` flag is needed to specify the diversity weight.

#### `Q_route_fix` branch

Same arguments as the `master` branch are accepted here.
But when the `ortho_lstm` encoder is used for any dual input sequence tasks, only the P-path is orthogonalised, and ***not*** the Q-path.



## Our Results

For as long as they persist on the lisa cluster, our results (trained models + various experiments) are available to view at:
- `/home/lgpu0136/experiments`
- `/home/lgpu0136/experiments_attentions`
- `/home/lgpu0136/experiments_bilstm`
- `/home/lgpu0136/experiments_fixed_embeddings`
- `/home/lgpu0136/experiments_q_route`

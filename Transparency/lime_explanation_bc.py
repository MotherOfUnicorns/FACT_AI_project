import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from lime.lime_text import LimeTextExplainer
from scipy.stats import pearsonr

from Transparency.common_code.common import get_latest_model
from Transparency.configurations import configurations
from Transparency.model import Binary_Classification as BC
from Transparency.Trainers.DatasetBC import datasets

parser = argparse.ArgumentParser(
    description="Compare attention weights and lime results on a dataset"
)
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
)
parser.add_argument("--diversity", type=float, default=0)

args, extras = parser.parse_known_args()
args.extras = extras


DATASET = datasets[args.dataset](args)

if args.output_dir is not None:
    DATASET.output_dir = args.output_dir
DATASET.diversity = args.diversity

CONFIG = configurations[args.encoder](DATASET)
TEST_DATA_SENTENCES = [
    " ".join(DATASET.vec.map2words(sentence_indices)[1:-1])
    # taking [1:-1] to remove the <SOS>/<EOS> tags (because they'll be added back later)
    for sentence_indices in DATASET.test_data.X
]

LATEST_MODEL_NAME = get_latest_model(
    os.path.join(CONFIG["training"]["basepath"], CONFIG["training"]["exp_dirname"])
)

MODEL = BC.Model.init_from_config(LATEST_MODEL_NAME, load_gen=False)
MODEL.dirname = LATEST_MODEL_NAME

CLASS_NAMES = ["0", "1"]
EXPLAINER = LimeTextExplainer(class_names=CLASS_NAMES)


def predict_proba(sentences):
    """takes a list of sentences as input (each sentence being a string without <SOS>/<EOS>,
    and returns the predicted probabilities for both classes as well as the attention weights"""

    data = DATASET.vec.get_seq_for_docs(sentences)
    predictions, attns, conicities = MODEL.evaluate(data)

    predictions = np.array(predictions)

    # concatenate predictions for the other class
    predictions = np.concatenate([1 - predictions, predictions], axis=1)

    # remove attention on <SOS>/<EOS> (they are equal to 0)
    attns = [a[1:-1] for a in attns]
    return predictions, attns


def get_attn_and_lime(sentence, explainer=EXPLAINER):
    """takes a list of sentences as input (each sentence being a string without <SOS>/<EOS>,
    and returns the model attention weight as well as the lime scores"""

    num_words = len(sentence.split(" "))
    exp = explainer.explain_instance(
        sentence, lambda x: predict_proba(x)[0], num_features=num_words
    )
    exp_dict = {exp[0] if exp[0] != "UNK" else "<UNK>": exp[1] for exp in exp.as_list()}

    predictions, attns = predict_proba([sentence])
    predicted_class = int(predictions[0][1] > 0.5)

    score_multiplier = 1 if predicted_class == 1 else -1
    lime_scores = [exp_dict.get(w, 0) * score_multiplier for w in sentence.split(" ")]

    return attns, lime_scores


def calc_attn_lime_correlation(sentence, explainer=EXPLAINER):
    attns, lime_scores = get_attn_and_lime(sentence, explainer)

    r, p_val = pearsonr(attns[0], lime_scores)
    return r, p_val


if __name__ == "__main__":
    import pandas as pd

    print("CALCULATING CORRELATION WITH LIME......")
    print("dataset={args.dataset}, encoder={args.encoder}, attention={args.attention}")
    r_p_vals = [calc_attn_lime_correlation(s, EXPLAINER) for s in TEST_DATA_SENTENCES]
    r_vals = [x[0] for x in r_p_vals]
    p_vals = [x[1] for x in r_p_vals]
    sentence_lengths = [len(s.split(" ")) for s in TEST_DATA_SENTENCES]

    df = pd.DataFrame()
    df["r_val"] = r_vals
    df["p_val"] = p_vals
    df["sentence_length"] = sentence_lengths

    df.to_csv(os.path.join(LATEST_MODEL_NAME, "lime_correlations.csv"))
    df.mean().to_csv(os.path.join(LATEST_MODEL_NAME, "lime_correlations_avg.csv"))

    fig, ax = plt.subplots()
    ax.scatter(df.sentence_length, df.r_val)
    ax.set_xlabel("sentence length")
    ax.set_ylabel("correlation(attn weights, lime)")
    fig.savefig(os.path.join(LATEST_MODEL_NAME, "lime_correlations_sentence_len.png"))

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
EXPLAINER = LimeTextExplainer(class_names=CLASS_NAMES, bow=False)


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

    predictions, attns = predict_proba([sentence])
    predicted_class = int(predictions[0][1] > 0.5)

    score_multiplier = 1 if predicted_class == 1 else -1
    lime_scores = [exp[1] * score_multiplier for exp in sorted(exp.as_map()[1])]

    return attns[0], np.array(lime_scores)


# def calc_attn_lime_correlation(sentence, explainer=EXPLAINER):
#     attns, lime_scores = get_attn_and_lime(sentence, explainer)
#
#     r, p_val = pearsonr(attns[0], lime_scores)
#     return r, p_val


if __name__ == "__main__":
    import csv
    import pandas as pd
    from tqdm import tqdm
    from common_code.common import jsd

    print("CALCULATING CORRELATION WITH LIME......")
    print(f"dataset={args.dataset}, encoder={args.encoder}, attention={args.attention}")

    with open(os.path.join(LATEST_MODEL_NAME, "lime_attn_scores.csv"), "w") as f:
        score_writer = csv.writer(f, delimiter=",")
        score_writer.writerow(["idx", "sentence_length", "attention", "lime"])

        r_vals, p_vals, jsd_vals, sentence_lengths = [], [], [], []
        for idx, sentence in tqdm(enumerate(TEST_DATA_SENTENCES)):
            attns, lime_scores = get_attn_and_lime(sentence, EXPLAINER)
            l = len(sentence.split(" "))
            score_writer.writerow(
                [idx, l, np.array2string(attns), np.array2string(lime_scores)]
            )

            r, p = pearsonr(attns, lime_scores)
            lime_min, lime_max = lime_scores.min(), lime_scores.max()
            lime_rescaled = (lime_scores - lime_min) / (lime_max - lime_min)
            lime_rescaled /= lime_rescaled.sum()
            d = jsd(attns, lime_rescaled)

            r_vals.append(r)
            p_vals.append(p)
            jsd_vals.append(d)
            sentence_lengths.append(l)

    df = pd.DataFrame()
    df["r_val"] = r_vals
    df["p_val"] = p_vals
    df["jsd_val"] = jsd_vals
    df["sentence_length"] = sentence_lengths

    df.to_csv(os.path.join(LATEST_MODEL_NAME, "lime_corr_jsd.csv"))
    df.mean().to_csv(os.path.join(LATEST_MODEL_NAME, "lime_corr_jsd_avg.csv"))
    df.std().to_csv(os.path.join(LATEST_MODEL_NAME, "lime_corr_jsd_std.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(df.sentence_length, df.r_val)
    axes[0].set_xlabel("sentence length")
    axes[0].set_ylabel("correlation(attn weights, lime)")
    axes[1].scatter(df.sentence_length, df.jsd_val)
    axes[1].set_xlabel("sentence length")
    axes[1].set_ylabel("JSD(attn weights, lime)")
    fig.savefig(os.path.join(LATEST_MODEL_NAME, "lime_corr_jsd_sentence_len.png"))

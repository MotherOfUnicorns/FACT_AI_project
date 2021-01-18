import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from lime.lime_text import LimeTextExplainer
from scipy.stats import pearsonr

from Transparency.common_code.common import get_latest_model
from Transparency.configurations import configurations
from Transparency.model import Question_Answering as QA
from Transparency.Trainers.DatasetQA import datasets

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
    for sentence_indices in DATASET.test_data.P
]


LATEST_MODEL_NAME = get_latest_model(
    os.path.join(CONFIG["training"]["basepath"], CONFIG["training"]["exp_dirname"])
)

MODEL = QA.Model.init_from_config(LATEST_MODEL_NAME, load_gen=False)
MODEL.dirname = LATEST_MODEL_NAME
# questions = DATASET.test_data.Q[:2]
# entity_masks = DATASET.test_data.E[:2]

ENTITY_LIST = sorted(list(DATASET.vec.idx2entity.items()))
CLASS_NAMES = [ent[1] for ent in ENTITY_LIST]
EXPLAINER = LimeTextExplainer(class_names=CLASS_NAMES)


def predict_proba(sentences, questions, entity_masks):
    """takes a list of sentences as input (each sentence being a string without <SOS>/<EOS>,
    with corresponding questions and entity masks,
    and returns the predicted probabilities for both classes as well as the attention weights"""

    # lime may use multiple mutated sentences as input, so we need to pad the
    # corresponding questions and entity_masks
    len_s, len_q, len_e = len(sentences), len(questions), len(entity_masks)
    if (len_s > len_q) and (len_q == 1):
        questions *= len_s // len_q
    if (len_s > len_e) and (len_e == 1):
        entity_masks *= len_s // len_e

    data = {
        "P": DATASET.vec.get_seq_for_docs(sentences),
        "Q": questions,
        "E": entity_masks,
    }
    predictions, attns, conicities, entropies = MODEL.evaluate(
        data, data_as_dict=True, return_softmax_prob=True
    )

    predictions = np.array(predictions)

    attns = [
        a[1:-1] for a in attns
    ]  # remove attention on <SOS>/<EOS> (they are equal to 0)
    return predictions, attns


def get_attn_and_lime(sentence, explainer, questions, entity_masks):
    """takes a list of sentences as input (each sentence being a string without <SOS>/<EOS>,
    as well as the corresponding questions, entity_masks,
    and returns the model attention weight as well as the lime scores"""

    predictions, attns = predict_proba([sentence], questions, entity_masks)
    predicted_class = np.argmax(predictions)
    # classes_to_explain = list(set([predicted_class, true_class]))
    classes_to_explain = [predicted_class]

    num_words = len(sentence.split())
    exp = explainer.explain_instance(
        sentence,
        lambda x: predict_proba(x, questions, entity_masks)[0],
        num_features=num_words,
        labels=classes_to_explain,
    )
    exp_dict = {
        exp[0] if exp[0] != "UNK" else "<UNK>": exp[1]
        for exp in exp.as_list(label=predicted_class)
    }

    lime_scores = [exp_dict.get(w, 0) for w in sentence.split()]

    return attns, lime_scores


def calc_attn_lime_correlation(idx, explainer):

    questions = [DATASET.test_data.Q[idx]]
    entity_masks = [DATASET.test_data.E[idx]]
    sentence = TEST_DATA_SENTENCES[idx]

    attns, lime_scores = get_attn_and_lime(sentence, explainer, questions, entity_masks)

    r, p_val = pearsonr(attns[0], lime_scores)
    return r, p_val


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    print("CALCULATING CORRELATION WITH LIME......")
    print(f"dataset={args.dataset}, encoder={args.encoder}, attention={args.attention}")

    r_vals = []
    p_vals = []
    sentence_lengths = []
    for idx, sentence in tqdm(enumerate(TEST_DATA_SENTENCES)):
        r, p = calc_attn_lime_correlation(idx, EXPLAINER)
        r_vals.append(r)
        p_vals.append(p)
        sentence_lengths.append(len(sentence.split(" ")))

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

from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from configs.config import SHAPConfig
from const.path import DATASET_TSV_PATH
from utils.notify import notify_slack
from utils.random_seed import set_random_seed
from utils.text import preprocess_batch


def generate_explianability_by_shap(
    pretrained: str,
    path_to_trained_model: Path,
    max_length: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    # define model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    # load & define dataset
    df = pd.read_csv(DATASET_TSV_PATH, sep="\t", encoding="utf-8")
    texts = df[df["label"] == 1].text.tolist()

    # Load Model
    net = AutoModelForSequenceClassification.from_pretrained(
        pretrained,
        num_labels=2,
        label2id={"non-darkpattern": 0, "darkpattern": 1},
        id2label={0: "non-darkpattern", 1: "darkpattern"},
    ).to(device)
    net.load_state_dict(torch.load(path_to_trained_model, map_location=device))
    net.eval()

    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    predictor = pipeline(
        "text-classification", model=net, tokenizer=tokenizer, device=0
    )

    explainer = shap.Explainer(predictor)
    texts = list(preprocess_batch(texts))

    shap_values = explainer(texts)

    values, data = (
        shap_values[:, :, "darkpattern"].values,
        shap_values[:, :, "darkpattern"].data,
    )

    word_to_scores = defaultdict(list)
    for i in range(len(data)):
        for j in range(len(data[i])):
            word_to_scores[data[i][j]].append(abs(values[i][j]))

    word_to_score = {}
    for k, v in word_to_scores.items():
        word_to_score[k] = sum(v) / len(v)

    words, scores = np.array(list(word_to_score.keys())), np.array(
        list(word_to_score.values())
    )
    sorted_words, sorted_scores = (
        words[np.argsort(-scores)],
        scores[np.argsort(-scores)],
    )
    notify_slack(word_to_score)
    print(word_to_score)
    notify_slack(f"{sorted_words}, {sorted_scores}")
    print(f"{sorted_words}, {sorted_scores}")
    notify_slack(f"{sorted_words[:50]}, {sorted_scores[:50]}")
    print(f"{sorted_words[:50]}, {sorted_scores[:50]}")


@hydra.main(config_name="shap_config")
def main(cfg: SHAPConfig) -> None:
    set_random_seed(cfg.random_seed)
    pretrained: str = cfg.pretrained
    max_length = cfg.max_length
    generate_explianability_by_shap(
        pretrained, Path(cfg.path_to_trained_model), max_length
    )


if __name__ == "__main__":
    main()

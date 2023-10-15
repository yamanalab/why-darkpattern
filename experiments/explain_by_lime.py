import re
from pathlib import Path
from typing import Callable, Generator, List

import hydra
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from configs.config import LimeConfig
from const.path import DATASET_TSV_PATH, PROJECT_ROOT
from utils.loggings import logging
from utils.random_seed import set_random_seed


def convert_num_to_label(text: str) -> str:
    num_pattern = re.compile("[0-9]+")
    return num_pattern.sub(r"number", text)


def preprocess(text: str) -> str:
    text = convert_num_to_label(text)
    return text


def preprocess_batch(texts: List[str]) -> Generator[str, None, None]:
    for text in texts:
        yield preprocess(text)


def create_batch_transform_fn_from_pretrained_tokenizer(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    padding: bool = True,
    preprocess_text: bool = False,
) -> Callable[[List[str]], torch.Tensor]:
    def transform(texts: List[str]) -> torch.Tensor:
        if preprocess_text:
            texts = list(preprocess_batch(texts))
        return torch.Tensor(
            tokenizer.batch_encode_plus(texts, max_length=max_length, padding=padding)[
                "input_ids"
            ]
        ).to(torch.long)

    return transform


def create_lime_text_predictor_fn(
    net: nn.Module,
    batch_transform: Callable[[List[str]], torch.Tensor],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Callable[[List[str]], List[List[float]]]:
    def lime_text_predictor(texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            text_tensor = batch_transform(texts).to(device)
            output = net(text_tensor).logits
            prob = F.softmax(output, dim=1)
        print(prob.cpu().numpy())
        return prob.cpu().numpy()

    return lime_text_predictor


def generate_explianability_by_lime(
    pretrained: str,
    num_features: int,
    max_length: int,
    path_to_trained_models: Path,
    top_labels: int,
    num_samples: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    df = pd.read_csv(DATASET_TSV_PATH, sep="\t", encoding="utf-8")
    texts, labels = df["text"].to_list(), df["label"].to_list()
    transform = create_batch_transform_fn_from_pretrained_tokenizer(
        AutoTokenizer.from_pretrained(pretrained), max_length
    )
    explainer = LimeTextExplainer(class_names=["non-darkpattern", "darkpatterns"])

    skf = StratifiedKFold(n_splits=5)

    for fold, (train_idxes, val_idxes) in enumerate(skf.split(texts, labels)):
        """
        1. Load Trained Model
        """
        net = AutoModelForSequenceClassification.from_pretrained(
            pretrained, num_labels=top_labels
        ).to(device)
        net.load_state_dict(torch.load(path_to_trained_models, map_location=device))
        net.eval()

        """
        2. Generate LIME Explaination for all val data
        """
        lime_text_predictor_fn = create_lime_text_predictor_fn(net, transform, device)

        logging.info("Generate Explaination for Validation Data")
        logging.info(f"fold: {fold}, pretrained: {pretrained}")
        for val_idx in val_idxes:
            text, label = texts[val_idx], labels[val_idx]
            exp = explainer.explain_instance(
                text,
                lime_text_predictor_fn,
                num_features=num_features,
                num_samples=3000,
                top_labels=top_labels,
            )
            input_test = transform([text]).to(device)

            # inference
            with torch.no_grad():
                output = net(input_test).logits  # [batch_size,label_size]

            print(exp.predict_proba, exp.local_pred)
            pred = output.argmax(dim=-1)

            logging.info(f"texts: '{text}'")

            logging.info(f"True Label: {label}, Pred Label: {int(pred)}, ")

            logging.info(exp.as_list())


@hydra.main(config_name="lime_config")
def main(cfg: LimeConfig) -> None:
    set_random_seed(cfg.random_seed)
    pretrained: str = cfg.pretrained
    max_length: int = cfg.max_length
    num_features: int = cfg.num_features
    top_labels: int = 2
    num_samples: int = cfg.num_samples
    path_to_trained_models: Path = Path(PROJECT_ROOT) / cfg.path_to_trained_model

    generate_explianability_by_lime(
        pretrained,
        num_features,
        max_length,
        path_to_trained_models,
        top_labels,
        num_samples,
    )


if __name__ == "__main__":
    main()

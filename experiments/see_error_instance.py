import json
from os.path import join
from typing import Dict, List

import hydra
import pandas as pd
import torch
from const.path import CONFIG_PATH, DATASET_TSV_PATH, NN_MODEL_PICKLES_PATH, OUTPUT_DIR
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.random_seed import set_random_seed
from utils.text import text_to_tensor as _text_to_tensor
from utils.uuid import generate_uuid


def cross_validation(
    n_fold: int,
    pretrained: str,
    batch_size: int,
    lr: float,
    start_factor: float,
    max_length: int,
    dropout: float,
    epochs: int,
    save_model: bool,
    experiment_id: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_labels: int = 2,
) -> None:
    # define model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def text_to_tensor(text: str) -> torch.Tensor:
        return _text_to_tensor(text, tokenizer, max_length)

    def tensor_to_text(tensor: torch.Tensor) -> str:
        return tokenizer.decode(tensor)

    # load & define dataset
    df = pd.read_csv(DATASET_TSV_PATH, sep="\t", encoding="utf-8")
    texts = df.text.tolist()
    labels = df.label.tolist()
    skf = StratifiedKFold(n_splits=n_fold)

    for fold, (_, test_idxes) in enumerate(skf.split(texts, labels)):

        # load model
        net = AutoModelForSequenceClassification.from_pretrained(
            pretrained, num_labels=num_labels
        ).to(device)
        path_to_model = join(NN_MODEL_PICKLES_PATH, f"{pretrained}_{fold}.pth")
        net.load_state_dict(torch.load(path_to_model, map_location=device))
        net.eval()

        error_instances: List[Dict] = []

        for i, (test_idx) in enumerate(test_idxes):
            text = texts[test_idx]
            input_test = torch.unsqueeze(text_to_tensor(text).to(device), 0)
            target = torch.tensor([labels[test_idx]]).to(device)
            # inference
            with torch.no_grad():
                output = net(input_test).logits  # [batch_size,label_size]

            pred = output.argmax(dim=-1)
            if pred != target:
                error_instance = {
                    "idx": int(test_idx),
                    "model": pretrained,
                    "text": text,
                    "pred": int(pred),
                    "target": int(target),
                }
                print(error_instance)
                error_instances.append(error_instance)

    with open(join(OUTPUT_DIR, "output.json"), "w") as f:
        json.dump(error_instances, f)
        f.write(json.dumps(error_instances))
        print(error_instances)


@hydra.main(config_path=CONFIG_PATH, config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    n_fold = cfg.train.n_fold
    pretrained = cfg.model.pretrained
    batch_size = cfg.train.batch_size
    lr = cfg.train.lr
    max_length = cfg.preprocess.max_length
    dropout = cfg.model.dropout
    epochs = cfg.train.epochs
    start_factor = cfg.train.start_factor
    save_model = cfg.train.save_model

    set_random_seed(cfg.random.seed)
    experiment_id = generate_uuid()

    cross_validation(
        n_fold=n_fold,
        pretrained=pretrained,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        dropout=dropout,
        epochs=epochs,
        start_factor=start_factor,
        save_model=save_model,
        experiment_id=experiment_id,
        device=torch.device("cpu"),
    )


if __name__ == "__main__":
    main()

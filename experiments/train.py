from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from configs.config import TrainConfig
from const.path import DATASET_TSV_PATH, NN_MODEL_PICKLES_PATH
from const.wandb import WANDB_PROJECT_NAME
from utils.dataset import DarkpatternDataset
from utils.notify import notify_slack
from utils.random_seed import set_random_seed
from utils.text import preprocess
from utils.text import text_to_tensor as _text_to_tensor
from utils.uuid import generate_uuid


def notify_error(
    e: Exception,
    pretrained: str,
    batch_size: int,
    lr: float,
    max_length: int,
) -> None:
    text = f"""
                ```
                Exception Occured in :
                    parameters:
                    pretrained: {pretrained}
                    batch_size: {batch_size}
                    lr: {lr}
                    max_length: {max_length}                    
                Exception :
                    {e}
                ```
                """
    notify_slack(text=text)


def custom_compute_metrics(res: EvalPrediction) -> dict:
    pred = res.predictions.argmax(axis=1)
    prob = F.softmax(torch.Tensor(res.predictions), dim=1)[:, 1].cpu().numpy()
    target = res.label_ids
    return {
        "accuracy_score": metrics.accuracy_score(target, pred),
        "f1_score": metrics.f1_score(target, pred),
        "precision_score": metrics.precision_score(target, pred),
        "recall_score": metrics.recall_score(target, pred),
        "roc_auc_score": metrics.roc_auc_score(target, prob),
    }


def cross_validation(
    n_fold: int,
    pretrained: str,
    batch_size: int,
    lr: float,
    start_factor: float,
    max_length: int,
    gradient_accumulation_steps: int,
    dropout: float,
    epochs: int,
    experiment_id: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_labels: int = 2,
) -> None:
    # define model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def text_to_tensor(text: str) -> torch.Tensor:
        text = preprocess(text)
        return _text_to_tensor(text, tokenizer, max_length)

    # load & define dataset
    df = pd.read_csv(DATASET_TSV_PATH, sep="\t", encoding="utf-8")
    texts = df.text.tolist()
    labels = df.label.tolist()
    ds = DarkpatternDataset(texts, labels, text_to_tensor)
    skf = StratifiedKFold(n_splits=n_fold)

    # n fold cross validation

    accuracy_scores: List[float] = []
    f1_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    roc_auc_scores: List[float] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):

        # define train & test dataset
        train_ds = Subset(ds, train_idx)
        test_ds = Subset(ds, test_idx)

        net = AutoModelForSequenceClassification.from_pretrained(
            pretrained, num_labels=num_labels
        ).to(device)

        optimizer = AdamW(net.parameters(), lr=lr)
        lr_scheduler = LinearLR(
            optimizer, start_factor=start_factor, total_iters=epochs
        )

        training_args = TrainingArguments(
            output_dir=NN_MODEL_PICKLES_PATH / experiment_id,
            logging_strategy="steps",
            save_total_limit=5,
            lr_scheduler_type="constant",
            weight_decay=0.0,
            evaluation_strategy="no",
            save_strategy="epoch",
            label_names=["labels"],
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            remove_unused_columns=False,
            logging_steps=1,
            report_to="none",
        )

        trainer = Trainer(
            model=net,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            optimizers=(optimizer, lr_scheduler),
            compute_metrics=custom_compute_metrics,
        )
        trainer.train()
        evaluate_metrics = trainer.evaluate(eval_dataset=test_ds)

        accuracy_scores.append(evaluate_metrics["eval_accuracy_score"])
        f1_scores.append(evaluate_metrics["eval_f1_score"])
        precision_scores.append(evaluate_metrics["eval_precision_score"])
        recall_scores.append(evaluate_metrics["eval_recall_score"])
        roc_auc_scores.append(evaluate_metrics["eval_roc_auc_score"])

    f1_score_average = np.mean(f1_scores)
    accuracy_score_average = np.mean(accuracy_scores)
    precision_score_average = np.mean(precision_scores)
    recall_score_average = np.mean(recall_scores)
    roc_auc_score_average = np.mean(roc_auc_scores)
    wandb.log(
        {
            "accuracy_scores": accuracy_scores,
            "f1_scores": f1_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "roc_auc_scores": roc_auc_scores,
            "f1_score_average": f1_score_average,
            "accuracy_score_average": accuracy_score_average,
            "precision_score_average": precision_score_average,
            "recall_score_average": recall_score_average,
            "roc_auc_score_average": roc_auc_score_average,
        }
    )

    # send result to slack
    text = f"""
    ```parameters:
        pretrained: {pretrained}
        batch_size: {batch_size}
        lr: {lr}
        max_length: {max_length}
        dropout: {dropout}
        epochs: {epochs}
        device: {device}
        num_labels: {num_labels}
    metrics for test:
         f1_score_average:{f1_score_average}
         accuracy_score_average:{accuracy_score_average}
         precision_score_average:{precision_score_average}
         recall_score_average:{recall_score_average}
         roc_auc_score_average:{roc_auc_score_average}
    ```
    """
    notify_slack(text)


@hydra.main(config_name="train_config")
def main(cfg: TrainConfig) -> None:

    pretrained = cfg.pretrained
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    epochs = cfg.epochs
    learning_rate = cfg.learning_rate
    dropout = cfg.dropout
    max_len = cfg.max_len
    start_factor = cfg.start_factor

    set_random_seed(cfg.random.seed)
    experiment_id = generate_uuid()

    config = {
        "pretrained": pretrained,
        "batch_size": batch_size,
        "lr": learning_rate,
        "max_length": max_len,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "dropout": dropout,
        "epochs": epochs,
        "start_factor": start_factor,
    }

    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        config=config,
        reinit=True,
        name=experiment_id,
    )

    assert run is not None, "wandb.init() failed"

    cross_validation(
        n_fold=5,
        pretrained=pretrained,
        batch_size=batch_size,
        lr=learning_rate,
        max_length=max_len,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dropout=dropout,
        epochs=epochs,
        start_factor=start_factor,
        experiment_id=experiment_id,
    )

    run.finish()


if __name__ == "__main__":
    main()

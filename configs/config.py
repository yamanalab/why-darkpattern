from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    random_seed: int = 42
    pretrained: str = "bert-base-uncased"
    batch_size: int = 128
    gradient_accumulation_steps: int = 1  # batch_size = 16 x 8 = 128
    epochs: int = 5
    learning_rate: float = 1e-5
    dropout: float = 0.1
    max_len: int = 32
    start_factor: float = 0.5


@dataclass
class SHAPConfig:
    pretrained: str = "bert-base-uncased"
    max_length: int = 16
    preprocess: bool = False
    path_to_trained_model: str = ""


@dataclass
class LimeConfig:
    random_seed: int = 42
    pretrained: str = "roberta-large"
    max_length: int = 16
    num_features: int = 8
    num_samples: int = 100
    path_to_trained_model: str = ""


cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainConfig)

cs.store(name="lime_config", node=LimeConfig)

cs.store(name="shap_config", node=SHAPConfig)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent / ".."

DATASET_PATH = PROJECT_ROOT / "dataset"

DATASET_TSV_PATH = DATASET_PATH / "dataset.tsv"

PICKLES_PATH = PROJECT_ROOT / "pickles"

NN_MODEL_PICKLES_PATH = PICKLES_PATH / "models/nn"

FIGURE_PATH = PROJECT_ROOT / "figures"

CONFIG_PATH = PROJECT_ROOT / "configs"

OUTPUT_DIR = PROJECT_ROOT / "output"

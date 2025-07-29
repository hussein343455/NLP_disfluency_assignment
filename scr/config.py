import torch
from click.testing import Result

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Configuration
FILE_PATHS = {
    'train': '../Data/raw/train.json',
    'val': '../Data/raw/dev.json',
    'test': '../Data/raw/test.json'
}
PREFIX = "correct disfluency: "
MAX_SOURCE_LENGTH = 32
MAX_TARGET_LENGTH = 40
SEED = 42

# Training Configuration
BASE_MODEL_CHECKPOINT_T5 = "google-t5/t5-small"
BASE_TOKENIZER_NAME_T5 = "google-t5/t5-small"
TRAIN_MODEL_VERSION = "v1"
OUTPUT_DIR = f"../models/t5-disfluency-correction_{TRAIN_MODEL_VERSION}"
FINAL_MODEL_NAME = F"disfl-qa-t5-final_{TRAIN_MODEL_VERSION}"
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 8
WEIGHT_DECAY = 0.01

# Evaluation Configuration
EVAL_MODEL_VERSION = "v1"
EVAL_MODEL_NAME = F"Adam5151/disfl-qa-t5-final_{EVAL_MODEL_VERSION}"
EVAL_TOKENIZER_NAME = F"Adam5151/disfl-qa-t5-final_{EVAL_MODEL_VERSION}"
# pick between "test", "train", "val" than run evaluate_model.py to save the results
# on the EVAL_MODEL_NAME
EVAL_DATA= 'test'
RESULT_PATH = f"../results/t5-disfluency-correction_{EVAL_MODEL_VERSION}/{EVAL_DATA}"
EVAL_BATCH_SIZE = 8

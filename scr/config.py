import torch

MODEL_CHECKPOINT_T5 = "google-t5/t5-small"
TOKENIZER_NAME_T5 = "google-t5/t5-small"

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
OUTPUT_DIR = "./models/t5-disfluency-correction"
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
WEIGHT_DECAY = 0.01

# Generation Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
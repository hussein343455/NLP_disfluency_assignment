import torch
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
import config
from data_processing import load_dataset_from_raw, clean_dataset_dict, tokenize_dataset_for_t5
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 1. Load Datasets
print("Loading datasets...")
full_dataset = load_dataset_from_raw(config.FILE_PATHS)
full_dataset_cleaned = clean_dataset_dict(full_dataset)
print("datasets loaded successfully")
device = config.DEVICE
print(device)
#%%

# 2. Load Model
print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(config.MODEL_CHECKPOINT_T5)
print("model loaded successfully")
model.to(device)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME_T5)
print("tokenizer loaded successfully")

#%%
tokenized_training_data = tokenize_dataset_for_t5(tokenizer,
                                                  dataset_dict = full_dataset_cleaned,
                                                  prefix=config.PREFIX,
                                                  max_source_length=config.MAX_SOURCE_LENGTH,
                                                  max_target_length=config.MAX_TARGET_LENGTH)
#%%

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

#%%
import evaluate
import numpy as np

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    """
    Helper function to strip whitespace from predictions and labels.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    """
    This function is called by the Trainer to compute metrics during evaluation.
    It decodes predictions and labels, post-processes them, and calculates the BLEU score.
    """
    preds, labels = eval_preds

    # The trainer output is a tuple, we only need the predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 (padding token) with the actual pad_token_id for decoding.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-process the text (stripping whitespace)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    # Add a measure of prediction length for debugging.
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    # Round the results to 4 decimal places.
    result = {k: round(v, 4) for k, v in result.items()}
    return result
#%%

# Define Training Arguments
# These arguments control the fine-tuning process.
training_args = Seq2SeqTrainingArguments(
    output_dir=config.OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=config.LEARNING_RATE,
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
    weight_decay=config.WEIGHT_DECAY,
    save_total_limit=3,
    num_train_epochs=config.NUM_TRAIN_EPOCHS,
    # load_best_model_at_end=True,
    predict_with_generate=True,
    logging_dir=f"{config.OUTPUT_DIR}/logs",
    logging_steps=40,
    fp16=True,
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training_data["train"],
    eval_dataset=tokenized_training_data["val"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()



import torch
import config
from data_processing import load_dataset_from_raw, clean_dataset_dict, tokenize_dataset_for_t5
import numpy as np
from transformers import set_seed

# Set the seed for reproducibility
set_seed(config.SEED)

# ============================================
# Load and clean Data
# ============================================

print("Loading datasets...")
full_dataset = load_dataset_from_raw(config.FILE_PATHS)
full_dataset = clean_dataset_dict(full_dataset)
print("datasets loaded successfully")
device = config.DEVICE

#%%
# ============================================
# Load the fine tuned model and tokenizer from the hub or locally
# ============================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config

model = AutoModelForSeq2SeqLM.from_pretrained(config.EVAL_MODEL_NAME) #f"{config.OUTPUT_DIR}/{config.FINAL_MODEL_NAME}
tokenizer = AutoTokenizer.from_pretrained(config.BASE_TOKENIZER_NAME_T5)
model.to(device)

#%%
# ============================================
# Tokenizer the data
# ============================================

tokenized_data = tokenize_dataset_for_t5(tokenizer,
                                                  dataset_dict = full_dataset,
                                                  prefix=config.PREFIX,
                                                  max_source_length=config.MAX_SOURCE_LENGTH,
                                                  max_target_length=config.MAX_TARGET_LENGTH)
#%%
# ============================================
# Configure data collator and Dataloader
# ============================================

from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

# Select just the tokenized test set for evaluation
tokenized_test_dataset = tokenized_data[config.EVAL_DATA]

# Set the format and create the DataLoader
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

test_dataloader = DataLoader(tokenized_test_dataset, batch_size=config.EVAL_BATCH_SIZE, collate_fn=data_collator)

#%%
# ============================================
# Generate Predictions
# ==================================
from tqdm import tqdm

print("Generating predictions from the test set...")
model.eval()
all_predictions = []

# Disable gradient calculations to save memory and speed up inference
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        # Move batch to the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Generate predictions using beam search for higher quality results
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.MAX_TARGET_LENGTH,
            num_beams=5,
            early_stopping=True
        )

        # Decode the generated tokens back to text
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_predictions.extend(decoded_preds)

#%%
# ==================================
# Compute BLEU Score
# ==================================

import evaluate

print("\nComputing sacrebleu score...")

# Load the BLEU metric
bleu_metric = evaluate.load("sacrebleu")

# Get the ground truth labels from the original (non-tokenized) dataset
ground_truth_labels = full_dataset[config.EVAL_DATA]['original']

# Prepare data for the metric:
# Predictions should be a list of strings.
# References (labels) should be a list of lists of strings.
cleaned_preds = [pred.strip() for pred in all_predictions]
cleaned_labels = [[label.strip()] for label in ground_truth_labels]

# Compute the score
results = bleu_metric.compute(predictions=cleaned_preds, references=cleaned_labels)

#%%
# ==================================
# Compute METEOR Score
# ==================================

print("\nComputing METEOR score...")
meteor_metric = evaluate.load('meteor')
# Compute the score
meteor_results = meteor_metric.compute(predictions=cleaned_preds, references=cleaned_labels)

#%%
# ==================================
# Compute BERTScore
# ==================================

print("\nComputing BERTScore...")
bertscore_metric = evaluate.load('bertscore')

# BERTScore expects references as a list of strings, not a list of lists.
# flatten the 'cleaned_labels' list for this metric.
bert_references = [label[0] for label in cleaned_labels]

# Compute the score
bert_results = bertscore_metric.compute(
    predictions=cleaned_preds,
    references=bert_references,
    lang='en',
    model_type='microsoft/deberta-xlarge-mnli' # Recommended model for high correlation
)

#%%
# ==================================
# Display and Save Results
# ==================================

import os

print("\n--- ✅ Evaluation Complete ---")

# Calculate BERTScore Averages
# calculates the mean of the list of scores provided by BERTScore
avg_precision = np.mean(bert_results['precision'])
avg_recall = np.mean(bert_results['recall'])
avg_f1 = np.mean(bert_results['f1'])

# create directory if not exists and Define the output filename
if not os.path.exists(config.RESULT_PATH):
      os.makedirs(config.RESULT_PATH)
output_filename = f"{config.RESULT_PATH}/evaluation_results.txt"

# Write results to a file
try:
    with open(output_filename, 'w') as f:
        f.write("--- Evaluation Results ---\n\n")

        # Write BLEU Score
        f.write("Metric: BLEU\n")
        f.write("="*20 + "\n")
        f.write(f"Score: {results['score']:.4f}\n")
        f.write(f"Precisions (1-4 grams): {[round(p, 4) for p in results['precisions']]}\n\n")

        # Write METEOR Score
        f.write("Metric: METEOR\n")
        f.write("="*20 + "\n")
        f.write(f"Score: {meteor_results['meteor']:.4f}\n\n")

        # Write BERTScore
        f.write("Metric: BERTScore\n")
        f.write("="*20 + "\n")
        f.write(f"Precision: {avg_precision:.4f}\n")
        f.write(f"Recall:    {avg_recall:.4f}\n")
        f.write(f"F1 Score:  {avg_f1:.4f}\n")

    print(f"✅ Successfully saved results to '{output_filename}'")

except IOError as e:
    print(f"❌ Error: Could not write to file {output_filename}. Reason: {e}")


# print results to the console
print("\n--- Summary ---")
print(f"BLEU Score: {results['score']:.4f}")
print(f"METEOR Score: {meteor_results['meteor']:.4f}")
print(f"BERTScore F1: {avg_f1:.4f}")
print("-----------------\n")


#%%
# ==================================
# Display a few examples
# ==================================

import random
# Display a few examples for qualitative analysis
print("--- Example Predictions ---")
random_indices = random.sample(range(len(cleaned_preds)), 5)
for i in random_indices:
    print(f"Input:        '{full_dataset[config.EVAL_DATA]['disfluent'][i]}'")
    print(f"Prediction:   '{cleaned_preds[i]}'")
    print(f"Ground Truth: '{cleaned_labels[i][0]}'")
    print("-" * 25)
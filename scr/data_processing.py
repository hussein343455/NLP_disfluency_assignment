import json
from datasets import Dataset, DatasetDict
from transformers import T5TokenizerFast

def reorder_json(data_old_structure, key_name):
    """Converts a dictionary of records into a Hugging Face Dataset object."""

    # Check the structure of each item in the dictionary
    for item_id, content in data_old_structure.items():
        # Check if the content is a dictionary and has the required keys
        if not (isinstance(content, dict) and 'original' in content and 'disfluent' in content):
            print(f"Validation failed: Item '{item_id}' has a un expected structure and will not be loaded.")
            return None

    data_new_structure = []
    for item_id, content in data_old_structure.items():
        record = {
            'id': item_id,
            'disfluent': content.get('disfluent'),
            'original': content.get('original')
        }
        data_new_structure.append(record)

    hugging_dataset = Dataset.from_list(data_new_structure)
    dataset_dict = DatasetDict({
        key_name: hugging_dataset
    })
    return dataset_dict


def load_dataset_from_raw(all_file_paths):
    """Load the dataset, """
    full_dataset = DatasetDict()
    for key_name, file_path in all_file_paths.items():
        try:
            print(f"Processing '{key_name}' split from {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_file = json.load(f)

            single_reordered_dict = reorder_json(loaded_file, key_name=key_name)
            full_dataset[key_name] = single_reordered_dict[key_name]

        except FileNotFoundError:
            print(f" Error: The file '{file_path}' was not found. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    print("\n✅ Raw Data Loaded Successfully:")
    return full_dataset

def clean_dataset_dict(dataset_dict: DatasetDict) -> DatasetDict:
    """
    Cleans a DatasetDict by removing rows where the 'disfluent' column
    is null or empty across all splits.
    """
    cleaned_dict = DatasetDict()
    for split_name, dataset_split in dataset_dict.items():
        initial_rows = len(dataset_split)
        print(f"\n--- Cleaning Split: '{split_name}' ---")
        print(f"Initial number of rows: {initial_rows}")

        # Filter out invalid examples.
        # The filter keeps examples where the lambda function returns True.
        cleaned_split = dataset_split.filter(
            lambda example: (
                    example.get('disfluent')
                    and example['disfluent'].strip()
                    and example['disfluent'] != '#VALUE!'
                    and len(example['disfluent'].split()) > 2
            )
        )

        final_rows = len(cleaned_split)
        rows_removed = initial_rows - final_rows
        print(f"Removed {rows_removed} null/empty rows.")
        print(f"Final number of rows: {final_rows}")
        cleaned_dict[split_name] = cleaned_split

    print("\n--- ✅ DatasetDict Cleaning Complete ---")
    return cleaned_dict

def tokenize_dataset_for_t5(
        used_tokenizer: T5TokenizerFast,
        dataset_dict: DatasetDict,
        prefix: str,
        max_source_length: int,
        max_target_length: int,
):
    """
    Tokenizes a DatasetDict for a sequence-to-sequence task, applying a task prefix.

    Args:
        used_tokenizer (T5TokenizerFast): the Tokenizer in use.
        dataset_dict (datasets.DatasetDict): The dataset to tokenize.
        prefix (str): The task-specific prefix to add to source texts.
        max_source_length (int): Maximum length for the tokenized source text.
        max_target_length (int): Maximum length for the tokenized target text.

    Returns:
        datasets.DatasetDict: The tokenized dataset, ready for the Trainer API.
    """

    def preprocess_function(examples):
        inputs = [prefix + str(doc) for doc in examples['disfluent']]

        # tokenize inputs and outputs
        model_inputs = used_tokenizer(inputs, max_length=max_source_length, truncation=True)
        labels = used_tokenizer(examples['original'], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply the preprocessing function to all splits in the dataset
    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict["train"].column_names
    )

    print("Tokenization complete. Sample of tokenized data:")
    print(tokenized_datasets)
    return tokenized_datasets
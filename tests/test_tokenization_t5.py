import unittest
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from src.data_processing import tokenize_dataset_for_t5
from src import config

class TestTokenization(unittest.TestCase):
    """
    Unit tests for the tokenize_dataset_for_t5 function.
    """

    def setUp(self):
        """
        Set up a mock dataset and tokenizer for testing.
        This method is called before each test function.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_TOKENIZER_NAME_T5)
        self.prefix = config.PREFIX
        self.max_source_length = config.MAX_SOURCE_LENGTH
        self.max_target_length = config.MAX_TARGET_LENGTH

        # Create a dummy dataset
        dummy_data = {
            "train": Dataset.from_dict({
                "id": ["5a835482e60761001a2eb598", "5ad4f0675b96ef001a10a6ea"],
                "disfluent": [
                    "What kind of organs do ctenophores or no make that do some sponges have?",
                    "What are or actually who were the main detractors of the humoral theory of immunity?"
                ],
                "original": [
                    "What kind of organs do some sponges have?",
                    "Who were the main detractors of the humoral theory of immunity?"
                ]
            }),
            "validation": Dataset.from_dict({
                "id": ["5a8923a23b2508001a72a4c9"],
                "disfluent": ["What is a fundamental no is another function that primes have that the number 15 does not?"],
                "original": ["What is another function that primes have that the number 15 does not?"]
            })
        }
        self.mock_dataset_dict = DatasetDict(dummy_data)

    def test_output_structure_and_keys(self):
        """
        Tests if the tokenized dataset has the correct structure and required keys.
        """
        tokenized_datasets = tokenize_dataset_for_t5(
            self.tokenizer,
            self.mock_dataset_dict,
            self.prefix,
            self.max_source_length,
            self.max_target_length
        )

        # Check if the output is still a DatasetDict
        self.assertIsInstance(tokenized_datasets, DatasetDict)
        self.assertIn("train", tokenized_datasets)
        self.assertIn("validation", tokenized_datasets)

        # Check for essential keys in the tokenized output
        train_features = tokenized_datasets["train"].features
        self.assertIn("input_ids", train_features)
        self.assertIn("attention_mask", train_features)
        self.assertIn("labels", train_features)

    def test_prefix_addition(self):
        """
        Tests if the prefix is correctly added to the start of each input.
        """
        tokenized_datasets = tokenize_dataset_for_t5(
            self.tokenizer,
            self.mock_dataset_dict,
            self.prefix,
            self.max_source_length,
            self.max_target_length
        )

        # Get the first tokenized input from the training set
        first_input_ids = tokenized_datasets["train"][0]["input_ids"]

        # Decode the tokenized input
        decoded_input = self.tokenizer.decode(first_input_ids, skip_special_tokens=True)

        # Check if the decoded string starts with the prefix
        self.assertTrue(decoded_input.strip().startswith(self.prefix.strip()))

    def test_sequence_truncation(self):
        """
        Tests if the input_ids and labels are truncated to their max lengths.
        """
        tokenized_datasets = tokenize_dataset_for_t5(
            self.tokenizer,
            self.mock_dataset_dict,
            self.prefix,
            self.max_source_length,
            self.max_target_length
        )

        for split in ["train", "validation"]:
            for example in tokenized_datasets[split]:
                # The length includes the special end-of-sequence token
                self.assertLessEqual(len(example["input_ids"]), self.max_source_length)
                self.assertLessEqual(len(example["labels"]), self.max_target_length)

    def test_column_removal(self):
        """
        Tests if the original columns are removed after tokenization.
        """
        tokenized_datasets = tokenize_dataset_for_t5(
            self.tokenizer,
            self.mock_dataset_dict,
            self.prefix,
            self.max_source_length,
            self.max_target_length
        )

        # Check that the original columns are gone
        remaining_columns = tokenized_datasets["train"].column_names
        self.assertNotIn("id", remaining_columns)
        self.assertNotIn("disfluent", remaining_columns)
        self.assertNotIn("original", remaining_columns)

    # def test_handles_empty_and_null_inputs(self):
    #     """
    #     Tests if the function handles empty strings and None values gracefully.
    #     """
    #     messy_data = {
    #         "train": Dataset.from_dict({
    #             "id": ["1", "2", "3"],
    #             "disfluent": ["a valid question", "", None],
    #             "original": ["valid answer", None, ""]
    #         })
    #     }
    #     messy_dataset_dict = DatasetDict(messy_data)
    #
    #     try:
    #         tokenized_datasets = tokenize_dataset_for_t5(
    #             messy_dataset_dict,
    #             self.tokenizer_name,
    #             self.prefix,
    #             self.max_source_length,
    #             self.max_target_length
    #         )
    #         # Check that the output for the empty/None inputs is valid
    #         # The tokenizer adds a single end-of-sequence token (id=1) for empty strings.
    #         # For the input, it will be prefix + empty string, so it will have a few tokens.
    #         self.assertGreater(len(tokenized_datasets["train"][1]["input_ids"]), 1)  # prefix + eos
    #         self.assertEqual(len(tokenized_datasets["train"][1]["labels"]), 1)  # just eos
    #         self.assertEqual(tokenized_datasets["train"][1]["labels"][0], self.tokenizer.eos_token_id)
    #
    #         # The third example should behave identically to the second
    #         self.assertEqual(tokenized_datasets["train"][1]['input_ids'], tokenized_datasets["train"][2]['input_ids'])
    #         self.assertEqual(tokenized_datasets["train"][1]['labels'], tokenized_datasets["train"][2]['labels'])
    #
    #     except Exception as e:
    #         self.fail(f"Function crashed with empty or None inputs: {e}")

# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

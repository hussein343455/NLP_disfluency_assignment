import unittest
import json
from unittest.mock import patch, mock_open
from datasets import Dataset, DatasetDict
from src.data_processing import reorder_json, load_dataset_from_raw, clean_dataset_dict

class TestDatasetLoading(unittest.TestCase):

    def setUp(self):
        """Set up mock data for tests."""
        self.mock_train_data = {
            "item1": {"disfluent": "uh this is a test", "original": "this is a test"},
            "item2": {"disfluent": "and um another one", "original": "and another one"}
        }
        self.mock_validation_data = {
            "item3": {"disfluent": "so like a validation example", "original": "a validation example"}
        }
        self.mock_test_data = {
            "item4": {"disfluent": "finally the test case you know", "original": "finally the test case"}
        }

    def test_reorder_json(self):
        """Test the reorder_json function."""
        print("\n--- Running test_reorder_json ---")
        key_name = 'train'
        result_dataset_dict = reorder_json(self.mock_train_data, key_name)

        # Check if the result is a DatasetDict
        self.assertIsInstance(result_dataset_dict, DatasetDict)

        # Check if the key is present
        self.assertIn(key_name, result_dataset_dict)

        # Check the content of the dataset
        dataset = result_dataset_dict[key_name]
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['id'], 'item1')
        self.assertEqual(dataset[0]['disfluent'], 'uh this is a test')
        self.assertEqual(dataset[0]['original'], 'this is a test')
        self.assertEqual(dataset[1]['id'], 'item2')
        print("✅ test_reorder_json passed.")

    def test_load_dataset_from_raw_success(self):
        """Test load_dataset_from_raw with mock files."""
        print("\n--- Running test_load_dataset_from_raw_success ---")
        mock_file_paths = {
            'train': 'dummy/path/train.json',
            'validation': 'dummy/path/dev.json'
        }

        # Mock file content
        mock_files_content = {
            'dummy/path/train.json': json.dumps(self.mock_train_data),
            'dummy/path/dev.json': json.dumps(self.mock_validation_data)
        }

        # Use mock_open to simulate file reading
        # The lambda function decides which mock content to return based on the file path
        m = mock_open(read_data=json.dumps(self.mock_train_data))
        m.side_effect = lambda file, *args, **kwargs: mock_open(read_data=mock_files_content[file]).return_value

        with patch('builtins.open', m):
            full_dataset = load_dataset_from_raw(mock_file_paths)

            # Verify the structure and content
            self.assertIn('train', full_dataset)
            self.assertIn('validation', full_dataset)
            self.assertEqual(len(full_dataset['train']), 2)
            self.assertEqual(len(full_dataset['validation']), 1)
            self.assertEqual(full_dataset['train'][1]['original'], 'and another one')
            self.assertEqual(full_dataset['validation'][0]['id'], 'item3')
            print("✅ test_load_dataset_from_raw_success passed.")

    def test_load_dataset_from_raw_file_not_found(self):
        """Test load_dataset_from_raw handles FileNotFoundError."""
        print("\n--- Running test_load_dataset_from_raw_file_not_found ---")
        mock_file_paths = {
            'train': 'real/path/train.json',
            'test': 'non_existent/path/test.json'
        }

        # Mock only the 'train.json' file to exist
        m = mock_open(read_data=json.dumps(self.mock_train_data))

        # Raise FileNotFoundError for any other path
        def open_side_effect(file, *args, **kwargs):
            if file == 'real/path/train.json':
                return mock_open(read_data=json.dumps(self.mock_train_data)).return_value
            else:
                raise FileNotFoundError

        m.side_effect = open_side_effect

        with patch('builtins.open', m):
            full_dataset = load_dataset_from_raw(mock_file_paths)

            self.assertIn('train', full_dataset)
            self.assertNotIn('test', full_dataset)
            self.assertEqual(len(full_dataset), 1)
            print("✅ test_load_dataset_from_raw_file_not_found passed.")

    def test_load_dataset_from_raw_with_empty_filepath_dict(self):
        """Test calling the loader with no files."""
        print("\n--- Running test_load_dataset_from_raw_with_empty_filepath_dict ---")
        full_dataset = load_dataset_from_raw({})

        # Should return an empty DatasetDict
        self.assertIsInstance(full_dataset, DatasetDict)
        self.assertEqual(len(full_dataset), 0)
        print("✅ test_load_dataset_from_raw_with_empty_filepath_dict passed.")

    def test_clean_dataset_dict_normal_case_with_mixed_validity(self):
        """
        Normal Case: Tests standard cleaning of nulls, empty strings,
        whitespace, and valid data all in one go.
        """
        print("\nRunning: test_clean_dataset_dict_normal_case_with_mixed_validity")
        # Create a mock dataset with a mix of valid and invalid rows
        data = {
            'disfluent': [
                'This is a perfectly valid sentence.', # Keep
                None,                                # Remove
                '',                                  # Remove
                '   ',                               # Remove
                'This one is also quite good.',      # Keep
            ],
            'original': ['' for _ in range(5)] # Dummy data for the other column
        }
        dataset = Dataset.from_dict(data)
        dataset_dict = DatasetDict({'train': dataset})

        # Run the cleaning function
        cleaned_dict = clean_dataset_dict(dataset_dict)

        # Assert that only the 2 valid rows remain
        self.assertEqual(len(cleaned_dict['train']), 2)

    def test_clean_dataset_dict_edge_case_all_rows_invalid(self):
        """
        Edge Case: Tests if the function correctly handles a split where all rows are invalid,
        resulting in an empty dataset.
        """
        print("\nRunning: test_edge_case_all_rows_invalid")
        data = {
            'disfluent': [None, '', '  ', '#VALUE!', 'word1 word2'],
            'original': ['' for _ in range(5)]
        }
        dataset = Dataset.from_dict(data)
        dataset_dict = DatasetDict({'validation': dataset})

        cleaned_dict = clean_dataset_dict(dataset_dict)

        # Assert that the resulting dataset for the split is empty
        self.assertEqual(len(cleaned_dict['validation']), 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

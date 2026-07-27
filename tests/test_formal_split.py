import unittest

import numpy as np

from common.data_utils import FORMAL_SPLIT_PATH, build_main_sample_metadata, load_formal_split


class FormalSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)

    def test_split_sizes_and_exclusivity(self):
        train = self.split["train_indices"]
        validation = self.split["validation_indices"]
        test = self.split["test_indices"]
        self.assertEqual((len(train), len(validation), len(test)), (2435, 305, 305))
        combined = np.concatenate([train, validation, test])
        self.assertEqual(len(np.unique(combined)), 3045)
        np.testing.assert_array_equal(np.sort(combined), np.arange(3045))

    def test_five_geometry_strata_are_preserved(self):
        expected = {
            "train_indices": {
                "upward_crack_1mm": 232,
                "upward_crack_5mm": 232,
                "downward_crack_5mm": 232,
                "circular_hole": 840,
                "double_crack_1mm": 899,
            },
            "validation_indices": {
                "upward_crack_1mm": 29,
                "upward_crack_5mm": 29,
                "downward_crack_5mm": 29,
                "circular_hole": 105,
                "double_crack_1mm": 113,
            },
            "test_indices": {
                "upward_crack_1mm": 29,
                "upward_crack_5mm": 29,
                "downward_crack_5mm": 29,
                "circular_hole": 105,
                "double_crack_1mm": 113,
            },
        }
        types = self.split["geometry_types"]
        for key, expected_counts in expected.items():
            values, counts = np.unique(types[self.split[key]], return_counts=True)
            self.assertEqual(dict(zip(values.tolist(), counts.tolist())), expected_counts)

    def test_metadata_and_fingerprint_match_current_excel_data(self):
        current = build_main_sample_metadata()
        self.assertEqual(str(self.split["dataset_fingerprint"].item()), current["dataset_fingerprint"])
        np.testing.assert_array_equal(self.split["sample_ids"], current["sample_ids"])
        np.testing.assert_array_equal(self.split["source_files"], current["source_files"])
        np.testing.assert_array_equal(self.split["geometry_types"], current["geometry_types"])
        np.testing.assert_allclose(
            self.split["geometry_parameters"], current["geometry_parameters"], equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()

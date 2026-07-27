import unittest

import numpy as np

from baselines.nearest_neighbor.evaluate import find_nearest_neighbors


class NearestNeighborBaselineTests(unittest.TestCase):
    def test_retrieval_returns_training_row_and_euclidean_distance(self):
        training = np.asarray([[0.0, 0.0], [3.0, 4.0], [10.0, 10.0]])
        queries = np.asarray([[2.9, 4.1], [0.0, 1.0]])

        positions, distances = find_nearest_neighbors(training, queries)

        np.testing.assert_array_equal(positions, [1, 0])
        np.testing.assert_allclose(distances, [np.sqrt(0.02), 1.0])

    def test_retrieval_is_deterministic_for_equal_distances(self):
        training = np.asarray([[-1.0], [1.0]])
        queries = np.asarray([[0.0]])

        positions, _distances = find_nearest_neighbors(training, queries)

        self.assertEqual(int(positions[0]), 0)


if __name__ == "__main__":
    unittest.main()

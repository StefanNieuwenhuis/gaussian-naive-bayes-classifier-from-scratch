import pytest
import numpy as np

from numpy.ma.testutils import assert_array_almost_equal

from model.gaussian_naive_bayes import GaussianNaiveBayes

DEFAULT_PRECISION = 8

class TestGaussianNaiveBayesClassifier:
    @pytest.fixture
    def dummy_data(self):
        """
        Generate six distinguishable points that represent two Iris Flower Species ("1", and "0").
               ^
              | x
              | x x
        <-----+----->
          o o |
            o |
              v

        x - Denotes Iris flower species "1"
        o - Denotes Iris flower species "0"

        Returns
        -------
        C: union of shape [(n_samples, n_features,), (n_classes,)]
            (n_samples x n_features) Feature matrix, and
            (n_classes,) Target vector
        """
        X = np.array(
            [
                [-2, -1],
                [-1, -1],
                [-1, -2],
                [1, 1],
                [1, 2],
                [2, 1],
            ]
        )

        y = np.array([0, 0, 0, 1, 1, 1])

        return [X, y]

    def test_log_prior_probabilities(self, dummy_data) -> None:
        """
        It should correctly compute the log prior probabilities from the given test dataset
        """

        # Arrange
        X, y = dummy_data
        clf = GaussianNaiveBayes()

        classes, class_count = np.unique(y, return_counts=True)
        expected_prior_probabilities = class_count / class_count.sum()
        expected_log_prior_probabilities = np.log(expected_prior_probabilities)

        # Act
        clf.fit(X, y)

        # Assert
        assert_array_almost_equal(expected_log_prior_probabilities, clf.class_prior_, DEFAULT_PRECISION)
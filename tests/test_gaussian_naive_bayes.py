import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal

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
        assert_array_almost_equal(expected_prior_probabilities, np.exp(clf.class_prior_), DEFAULT_PRECISION)

    def test_test_log_likelihood_constants(self, dummy_data) -> None:
        """
        It should correctly compute the log likelihood constants (curvature, mean pull & constants) from the given test dataset
        """

        # Arrange
        X, y = dummy_data
        clf = GaussianNaiveBayes()

        expected_class_curvature = [[-2.24999999, -2.24999999],[-2.24999999, -2.24999999]]
        expected_mean_pull = [[-5.99999997, -5.99999997],[ 5.99999997,  5.99999997]]
        expected_ll_constants = [[-4.16689982, -4.16689982], [-4.16689982, -4.16689982]]

        # Act
        clf.fit(X, y)

        # Assert
        assert_array_almost_equal(expected_class_curvature, clf.class_curvature_, DEFAULT_PRECISION)
        assert_array_almost_equal(expected_mean_pull, clf.class_mean_pull_, DEFAULT_PRECISION)
        assert_array_almost_equal(expected_ll_constants, clf.class_log_likelihood_constants_, DEFAULT_PRECISION)

    def test_fit_empty_feature_matrix(self, dummy_data) -> None:
        """
        It should throw a ValueError when the provided feature matrix is empty
        """

        # Arrange
        X = []
        _, y = dummy_data
        clf = GaussianNaiveBayes()

        # Act & Assert
        with pytest.raises(ValueError, match="Feature Matrix cannot be empty"):
            clf.fit(X, y)

    def test_fit_empty_target_vector(self, dummy_data) -> None:
        """
        It should throw a ValueError when the provided target vector is empty
        """

        # Arrange
        y = []
        X, _ = dummy_data
        clf = GaussianNaiveBayes()

        # Act & Assert
        with pytest.raises(ValueError, match="Target Vector cannot be empty"):
            clf.fit(X, y)

    def test_predict(self, dummy_data) -> None:
        """
        It should correctly classify flower species
        X_train, y_train == X_test, y_test
        """

        # Arrange
        X, y = dummy_data
        clf = GaussianNaiveBayes()
        clf.fit(X,y)

        # Act
        predicted_labels = clf.predict(X)

        # Assert
        assert_array_equal(y, predicted_labels)
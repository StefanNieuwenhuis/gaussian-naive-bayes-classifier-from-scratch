import re

import pytest
import numpy as np
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal, assert_equal

from data.outlier_remover import OutlierRemover

DEFAULT_PRECISION = 8

class TestOutlierRemover:
    """
    Unit tests to test Outline Remover Transformer
    """

    @pytest.fixture
    def dummy_data(self):
        """
        Generate six distinguishable points that represent two Iris Flower Species.

        Returns
        -------
        C: union of shape [(n_samples, n_features,), (n_classes,)]
            (n_samples x n_features) Feature matrix, and
            (n_classes,) Target vector
        """
        X = np.array([
            [1, 2],         # class 0 - normal
            [2, 3],         # class 0 - normal
            [100, 200],     # class 0 - outlier

            [10, 20],       # class 1 - normal
            [11, 19],       # class 1 - normal
            [999, 888],     # class 1 - outlier
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        return [X, y]

    def test_fit(self, dummy_data) -> None:
        """
        It should correctly compute Q1 and Q3 bounds
        """

        # Arrange
        X, y = dummy_data

        classes, class_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        n_features = X.shape[1]

        expected_q1_ = np.zeros((n_classes, n_features))
        expected_q3_ = np.zeros((n_classes, n_features))

        for class_id in range(n_classes):
            mask = class_indices == class_id
            subset = X[mask]

            expected_q1_[class_id] = np.percentile(subset, .25, axis=0)
            expected_q3_[class_id] = np.percentile(subset, .75, axis=0)

        outlier_remover = OutlierRemover()

        # Act
        outlier_remover.fit(X, y)

        # Assert
        assert_array_almost_equal(expected_q1_, outlier_remover.q1_, DEFAULT_PRECISION)
        assert_array_almost_equal(expected_q3_, outlier_remover.q3_, DEFAULT_PRECISION)

    def test_fit_empty_feature_matrix(self, dummy_data) -> None:
        """
        It should throw a ValueError when the provided target vector is empty
        """

        # Arrange
        X = []
        _, y = dummy_data
        outlier_remover = OutlierRemover()

        # Act & Assert
        with pytest.raises(ValueError, match="OutlierRemover requires a Feature matrix to learn IQR bounds"):
            outlier_remover.fit(X, y)

    def test_fit_empty_target_vector(self, dummy_data) -> None:
        """
        It should throw a ValueError when the provided target vector is empty
        """

        # Arrange
        X, _ = dummy_data
        y = []
        outlier_remover = OutlierRemover()

        # Act & Assert
        with pytest.raises(ValueError, match=re.escape("OutlierRemover requires a target column (e.g., 'species') for group-wise IQR.")):
            outlier_remover.fit(X, y)

    def test_transform(self, dummy_data) -> None:
        """
        It should correctly transform input data with learned IQR bounds
        """

        # Arrange
        X, y = dummy_data

        # we deliberately downgraded the threshold to actually remove outliers in such a small dataset as used in this test
        outlier_remover = OutlierRemover(threshold=0.5)
        outlier_remover.fit(X, y)

        expected_X = np.array([
            [1, 2],
            [2, 3],
            [10, 20],
            [11, 19],
        ])
        expected_y = np.array([0, 0, 1, 1])

        # Act
        transformed_X, transformed_y = outlier_remover.transform(X, y)

        # Assert
        assert_array_equal(expected_X, transformed_X)
        assert_array_equal(expected_y, transformed_y)

    def test_transform_not_fitted(self, dummy_data) -> None:
        """
        It should throw a ValueError when OutlierRemover is not fitted
        """

        # Arrange
        X, y = dummy_data
        outlier_remover = OutlierRemover()

        # Act & Assert
        with pytest.raises(ValueError, match=re.escape("OutlierRemover instance is not fitted yet. Call 'fit' first.")):
            outlier_remover.transform(X, y)

    def test_fit_transform(self, dummy_data) -> None:
        """
        It should correctly transform input data with learned IQR bounds
        """

        # Arrange
        X, y = dummy_data

        # we deliberately downgraded the threshold to actually remove outliers in such a small dataset as used in this test
        outlier_remover = OutlierRemover(threshold=0.5)
        outlier_remover.fit_transform(X, y)

        expected_X = np.array([
            [1, 2],
            [2, 3],
            [10, 20],
            [11, 19],
        ])
        expected_y = np.array([0, 0, 1, 1])

        # Act
        transformed_X, transformed_y = outlier_remover.transform(X, y)

        # Assert
        assert_array_equal(expected_X, transformed_X)
        assert_array_equal(expected_y, transformed_y)

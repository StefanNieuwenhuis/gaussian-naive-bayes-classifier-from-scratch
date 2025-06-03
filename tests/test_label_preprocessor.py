import re

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from data.label_preprocessor import LabelPreProcessor


class TestLabelProcessor():
    """
    Unit tests to test Label Preprocessor Transformer
    """

    @pytest.fixture
    def sample_labels(self):
        return np.array(['cat', 'dog', 'cat', 'bird'])

    def test_fit_transform(self, sample_labels) -> None:
        """
        It should correctly fit and transform class labels
        """

        # Arrange
        lp = LabelPreProcessor()
        expected_preprocessed_labels = np.array([1, 2, 1, 0])

        # Act
        y_transformed = lp.fit_transform(X=None, y=sample_labels)

        # Assert
        assert_array_equal(y_transformed, expected_preprocessed_labels)

    def test_inverse_transform(self, sample_labels) -> None:
        """
        It should correctly inverse-transform encoded class labels from a target vector y
        """

        # Arrange
        lp = LabelPreProcessor()
        y_transformed = lp.fit_transform(X=None, y=sample_labels)

        # Act
        y_inverse = lp.inverse_transform(y_transformed)

        # Assert
        assert_array_equal(y_inverse, sample_labels)

    def test_transform_requires_fit_first(self, sample_labels) -> None:
        """
        It should raise an error when LabelPreProcessor is not initialized/fitted yet
        """

        # Arrange
        lp = LabelPreProcessor()

        # Act & Assert
        with pytest.raises(ValueError, match=re.escape("This LabelEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")):
            _ = lp.transform(X=None, y=sample_labels)

    def test_fit_raises_on_none_y(self) -> None:
        """
        It should raise a ValueError when no target vector is provided
        """

        # Arrange
        lp = LabelPreProcessor()

        # Act & Assert
        with pytest.raises(ValueError):
            lp.fit(X=None, y=None)

    def test_transform_raises_on_none_y(self, sample_labels) -> None:
        """
        It should raise a ValueError when no target vector is provided
        """

        # Arrange
        lp = LabelPreProcessor()

        # Act & Assert
        with pytest.raises(ValueError):
            lp.transform(X=None, y=None)

    def test_fit_transform_idempotency(self, sample_labels) -> None:
        """
        Encoded class labels should be idempotent when two class instances are fitted on the same target vector
        """

        # Arrange
        lp1 = LabelPreProcessor()
        lp2 = LabelPreProcessor()

        # Act
        y1 = lp1.fit_transform(X=None, y=sample_labels)
        y2 = lp2.fit_transform(X=None, y=sample_labels)

        # Assert
        assert_array_equal(y1, y2)
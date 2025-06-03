import pytest
import numpy as np

from collections import Counter
from typing import Tuple

from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from data.data_splitter import DataSplitter


class TestDataSplitter:
    """
    Unit test for DataSplitter Class
    """

    @pytest.fixture
    def dummy_data(self) -> Tuple[NDArray, ...]:
        # Create a balanced binary classification dataset
        X = np.array([[i, i + 1] for i in range(100)])
        y = np.array([0] * 50 + [1] * 50)
        return X, y

    def test_split_shapes(self, dummy_data) -> None:
        """
        It should split provided feature matrix and target vector according to provided settings
        """

        # Arrange
        X, y = dummy_data
        splitter = DataSplitter()

        # Act
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        # Assert
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
        assert X_train.shape[1] == X.shape[1]

    def test_stratification(self, dummy_data) -> None:
        """
        It should (approx.) preserve relative class frequencies in each train and validation fold with stratification
        """

        # Arrange
        X, y = dummy_data
        splitter = DataSplitter(test_size=0.3, stratify=y)

        # Act
        _, _, y_train, y_test = splitter.split(X, y)

        train_counts = Counter(y_train)
        test_counts = Counter(y_test)

        # Since we have balanced classes, test/train splits should reflect that
        assert train_counts[0] == train_counts[1]
        assert test_counts[0] == test_counts[1]

    def test_non_stratified_split(self, dummy_data) -> None:
        """
        It should be roughly balanced without stratification enabled
        """

        # Arrange
        X, y = dummy_data
        splitter = DataSplitter(test_size=0.25)

        # Act
        _, _, y_train, y_test = splitter.split(X, y)

        # Not guaranteed to be perfectly balanced
        train_counts = Counter(y_train)
        test_counts = Counter(y_test)

        # Assert
        assert abs(train_counts[0] - train_counts[1]) <= 10  # Should still be roughly balanced
        assert abs(test_counts[0] - test_counts[1]) <= 10

    def test_random_state_reproducibility(self, dummy_data) -> None:
        """
        It should return the same data sets with the same random state
        """

        # Arrange
        X, y = dummy_data
        splitter1 = DataSplitter(test_size=0.25, random_state=42)
        splitter2 = DataSplitter(test_size=0.25, random_state=42)

        # Act
        X_train_1, X_test_1, y_train_1, y_test_1 = splitter1.split(X, y)
        X_train_2, X_test_2, y_train_2, y_test_2 = splitter2.split(X, y)

        # Assert
        assert_array_equal(X_train_1, X_train_2)
        assert_array_equal(X_test_1, X_test_2)
        assert_array_equal(y_train_1, y_train_2)
        assert_array_equal(y_test_1, y_test_2)

    def test_different_random_states_yield_different_splits(self, dummy_data) -> None:
        """
        It should yield different data sets with different random states
        """

        # Arrange
        X, y = dummy_data
        splitter1 = DataSplitter(test_size=0.25, random_state=1)
        splitter2 = DataSplitter(test_size=0.25, random_state=99)

        # Act
        X_train_1, X_test_1, _, _ = splitter1.split(X, y)
        X_train_2, X_test_2, _, _ = splitter2.split(X, y)

        # Assert
        with pytest.raises(AssertionError):
            assert_array_equal(X_train_1, X_train_2)

        with pytest.raises(AssertionError):
            assert_array_equal(X_test_1, X_test_2)
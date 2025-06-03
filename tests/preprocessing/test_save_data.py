import pytest
import numpy as np

from pathlib import Path
from numpy.ma.testutils import assert_equal

from data.save_data import SaveData

DEFAULT_PRECISION = 8

class TestSaveData:
    """
    Unit tests to test SaveData class
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
            [1, 2],  # class 0 - normal
            [2, 3],  # class 0 - normal
            [100, 200],  # class 0 - outlier

            [10, 20],  # class 1 - normal
            [11, 19],  # class 1 - normal
            [999, 888],  # class 1 - outlier
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        return [X, y]

    def test_save_path(self) -> None:
        """
        It should correctly set the save path
        """

        # Arrange
        expected_save_path = Path("/path/to/data.csv")

        # Act
        data_saver = SaveData(expected_save_path)

        # Assert
        assert_equal(expected_save_path, data_saver.save_path_)

    def test_save_creates_file(self, tmp_path) -> None:
        """
        It should save the dataset to the given path
        """

        # Arrange
        data_saver = SaveData(save_path=str(tmp_path))
        X = np.array([
            [1, 2],  # class 0 - normal
            [2, 3],  # class 0 - normal
            [100, 200],  # class 0 - outlier

            [10, 20],  # class 1 - normal
            [11, 19],  # class 1 - normal
            [999, 888],  # class 1 - outlier
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        X_path = tmp_path / "X_test.csv"
        y_path = tmp_path / "y_test.csv"

        # Act
        data_saver.save(X, y, 'test')

        # Assert
        assert X_path.exists()
        assert y_path.exists()

    def test_download_raises_if_file_exists(self, tmp_path, dummy_data) -> None:
        """
        It should raise an error if the dataset already exists
        """

        # Arrange
        X, y = dummy_data
        X_path = tmp_path / "X_test.csv"
        y_path = tmp_path / "y_test.csv"
        X_path.write_text("dummy content")
        y_path.write_text("dummy content")

        data_saver = SaveData(save_path=str(tmp_path))

        # Act & Assert
        with pytest.raises(FileExistsError):
            data_saver.save(X, y, 'test')


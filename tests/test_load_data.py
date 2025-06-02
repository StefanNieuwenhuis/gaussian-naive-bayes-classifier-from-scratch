import pytest

from numpy.ma.testutils import assert_equal

from data.load_data import IrisDataDownloader

DEFAULT_PRECISION = 8

class TestIrisDataDownloader:
    """
    Unit tests to test Loading Iris Data
    """

    @pytest.fixture
    def temp_file(self, tmp_path):
        return tmp_path / "iris.csv"

    def test_save_path(self) -> None:
        """
        It should correctly set the save path
        """

        # Arrange
        expected_save_path = "/path/to/data.csv"

        # Act
        downloader = IrisDataDownloader(expected_save_path)

        # Assert
        assert_equal(expected_save_path, downloader.save_path_)

    def test_download_creates_file(self, temp_file) -> None:
        """
        It should download the Iris dataset to the given path
        """

        # Arrange
        downloader = IrisDataDownloader(save_path=str(temp_file))

        # Act
        downloader.download()

        # Assert
        assert temp_file.exists()

    def test_download_raises_if_file_exists(self, temp_file) -> None:
        """
        It should raise an error if the dataset already exists
        """

        # Arrange
        temp_file.write_text("dummy content")
        downloader = IrisDataDownloader(save_path=str(temp_file))

        # Act & Assert
        with pytest.raises(FileExistsError, match="already exists"):
            downloader.download()


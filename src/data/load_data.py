from typing import Tuple, Self

import pandas as pd
import seaborn as sns
from pathlib import Path
from numpy.typing import NDArray

from utils.logging_factory import LoggerFactory

class IrisDataDownloader:
    """
    Load the Iris Flowers Species dataset from Seaborn
    """

    def __init__(self, save_path: str = 'data/raw/iris.csv') -> None:
        self.save_path_ = Path(save_path)
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

        self.logger.info(f"Set save path: {save_path}")

    def download(self, force: bool = False) -> Self:
        """
        Downloads the Iris dataset using Seaborn and saves it to CSV

        Parameters
        ----------
        force: boolean, default False
            If True, forces redownload and overwrite.

        Returns
        -------
        self: class-instance

        Raises
        ------
        FileExistsError if file exists and force is False.
        """

        if self.save_path_.exists() and not force:
            message = f"File {self.save_path_} already exists. Use force=True to overwrite."
            self.logger.warning(message)
            raise FileExistsError(message)

        # Ensure the parent directories exist
        self.save_path_.parent.mkdir(parents=True, exist_ok=True)

        iris_data = sns.load_dataset("iris")
        iris_data.to_csv(self.save_path_, index=False)
        self.logger.info(f"Iris dataset saved to {self.save_path_}")

        return self

    def load(self, as_numpy: bool = True) ->  Tuple[NDArray, NDArray]:
        """
        Load the Iris Dataset

        Parameters
        ----------
        as_numpy: boolean, default True
            If True, returns (X, y) as NumPy arrays.
            If False, returns the full DataFrame

        Returns
        -------
        Tuple of (X, y) as NumPy arrays if as_numpy=True,
        else returns the full DataFrame.

        Raises
        ------
        FileNotFoundError if the CSV file is not found.
        """

        if not self.save_path_.exists():
            error_message = f"File {self.save_path_} does not exist. Please download first."
            self.logger.error(error_message)
            raise FileNotFoundError(error_message)

        iris_df = pd.read_csv(self.save_path_)
        if as_numpy:
            X = iris_df.drop(columns=["species"]).values
            y = iris_df["species"].values
            self.logger.info(f"Loaded data as NumPy arrays from {self.save_path_}")
            return X, y
        else:
            self.logger.info(f"Loaded data as DataFrame from {self.save_path_}")
            return iris_df
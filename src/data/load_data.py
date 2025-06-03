from pathlib import Path
from typing import Self

import seaborn as sns

from utils.logging_factory import LoggerFactory


class IrisDataDownloader:
    """
    Load the Iris Flowers Species dataset from Seaborn
    """

    def __init__(self, save_path: str = 'data/raw/iris.csv') -> None:
        self.save_path_ = Path(save_path)
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

        self.logger.info(f"Set save path: {save_path}")

    def download(self) -> Self:
        """
        Downloads the Iris dataset using Seaborn and saves it to CSV

        Returns
        -------
        self: class-instance
        """
        if self.save_path_.exists():
            error_message = f"File {self.save_path_} already exists. Aborting download to avoid overwrite."
            self.logger.error(error_message)
            raise FileExistsError(error_message)

        # Ensure the parent directories exist
        self.save_path_.parent.mkdir(parents=True, exist_ok=True)

        iris_data = sns.load_dataset("iris")
        iris_data.to_csv(self.save_path_, index=False)
        self.logger.info(f"Iris dataset saved to {self.save_path_}")
